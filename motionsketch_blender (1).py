bl_info = {
    "name": "MotionSketch",
    "author": "MotionSketch",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar (N) > MotionSketch",
    "description": "Draw the motion with Annotation strokes and turn it into keyframes. "
                   "Draw your own shapes, mark keyframes with strokes, export an MP4.",
    "category": "Animation",
}

import bisect
import math
import os

import bmesh
import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from mathutils import Matrix, Vector

LAYER_SHAPE = "MS Shape"
LAYER_MOTION = "MS Motion"
LAYER_KEYS = "MS Keys"
LAYER_COLORS = {
    LAYER_SHAPE: (0.10, 0.40, 0.90),
    LAYER_MOTION: (0.10, 0.10, 0.10),
    LAYER_KEYS: (0.90, 0.20, 0.10),
}

# Outlines (percent coordinates, y down) shared with the web version
SHAPE_POLY = {
    "TRIANGLE": [(50, 0), (100, 100), (0, 100)],
    "STAR": [(50, 0), (61, 35), (98, 35), (68, 57), (79, 95),
             (50, 72), (21, 95), (32, 57), (2, 35), (39, 35)],
}


# --------------------------------------------------------------------------
# Annotation helpers
# --------------------------------------------------------------------------

def get_annotation(context, create=False):
    """Return the scene's Annotation datablock (legacy Grease Pencil or the
    new Annotation type from Blender 4.3+)."""
    gp = getattr(context, "annotation_data", None)
    if gp is None:
        gp = context.scene.grease_pencil
    if gp is None and create:
        coll = getattr(bpy.data, "annotations", None)
        if coll is None:
            coll = bpy.data.grease_pencils
        gp = coll.new("Annotations")
        context.scene.grease_pencil = gp
    return gp


def find_layer(gp, name):
    if gp is None:
        return None
    for layer in gp.layers:
        if (getattr(layer, "info", None) or getattr(layer, "name", None)) == name:
            return layer
    return None


def _set_active_layer(gp, layer):
    """Make `layer` the active annotation layer. The API differs between
    Blender versions, so try each known way. Returns True on success."""
    layers = gp.layers
    for attr in ("active", "active_layer"):
        try:
            setattr(layers, attr, layer)
            return True
        except Exception:
            pass
    try:                                           # index based
        layers.active_index = list(layers).index(layer)
        return True
    except Exception:
        pass
    try:
        gp.layers.active_note = layer.info         # very old builds
        return True
    except Exception:
        return False


def ensure_layer(gp, name):
    layer = find_layer(gp, name)
    if layer is None:
        layer = gp.layers.new(name, set_active=True)   # new layers become active
        for attr, value in (("color", LAYER_COLORS[name]), ("thickness", 3)):
            try:
                setattr(layer, attr, value)
            except Exception:
                pass
        return layer
    if not _set_active_layer(gp, layer):
        # Could not switch to the existing layer. If it is empty we can simply
        # recreate it (a new layer is always active); otherwise keep the strokes.
        if not layer_strokes(layer):
            try:
                gp.layers.remove(layer)
                return ensure_layer(gp, name)
            except Exception:
                pass
    return layer


def layer_strokes(layer):
    """Strokes of the most recent frame that has any."""
    if layer is None:
        return []
    for frame in reversed(list(layer.frames)):
        if len(frame.strokes):
            return list(frame.strokes)
    return []


def stroke_points(stroke):
    return [p.co.copy() for p in stroke.points]


def clear_layer(layer):
    if layer is None:
        return
    try:
        layer.clear()
    except Exception:
        for frame in list(layer.frames):
            try:
                layer.frames.remove(frame)
            except Exception:
                pass


def find_view3d(context):
    areas = []
    if context.area and context.area.type == 'VIEW_3D':
        areas.append(context.area)
    areas += [a for a in context.screen.areas if a.type == 'VIEW_3D']
    for area in areas:
        for region in area.regions:
            if region.type == 'WINDOW':
                return area, region, area.spaces.active
    return None, None, None


# --------------------------------------------------------------------------
# Path maths
# --------------------------------------------------------------------------

def clean_path(pts, min_step):
    out = [pts[0]]
    for p in pts[1:]:
        if (p - out[-1]).length >= min_step:
            out.append(p)
    if (pts[-1] - out[-1]).length > 1e-9:
        out.append(pts[-1])
    return out


def smooth_path(pts, win=2):
    if len(pts) < 5:
        return pts
    out = []
    for i in range(len(pts)):
        w = min(win, i, len(pts) - 1 - i)
        acc = Vector((0.0, 0.0, 0.0))
        for j in range(i - w, i + w + 1):
            acc += pts[j]
        out.append(acc / (2 * w + 1))
    return out


def polyline_length(pts):
    return sum((pts[i] - pts[i - 1]).length for i in range(1, len(pts)))


class SketchPath:
    """A 3D polyline parametrised by arc length (s in 0..1)."""

    def __init__(self, pts):
        self.pts = pts
        self.L = [0.0]
        for i in range(1, len(pts)):
            self.L.append(self.L[-1] + (pts[i] - pts[i - 1]).length)
        self.length = self.L[-1]

    def at(self, s):
        d = max(0.0, min(1.0, s)) * self.length
        i = bisect.bisect_left(self.L, d)
        if i <= 0:
            return self.pts[0].copy()
        if i >= len(self.pts):
            return self.pts[-1].copy()
        span = self.L[i] - self.L[i - 1]
        f = (d - self.L[i - 1]) / span if span > 0 else 0.0
        return self.pts[i - 1].lerp(self.pts[i], f)

    def nearest(self, p):
        """Return (s, distance) of the closest point on the path."""
        best_d, best_along = 1e30, 0.0
        for i in range(len(self.pts) - 1):
            a, b = self.pts[i], self.pts[i + 1]
            ab = b - a
            l2 = ab.length_squared
            f = max(0.0, min(1.0, (p - a).dot(ab) / l2)) if l2 > 0 else 0.0
            d = (p - (a + ab * f)).length
            if d < best_d:
                best_d, best_along = d, self.L[i] + math.sqrt(l2) * f
        return best_along / self.length, best_d


def ease(u):
    return 2 * u * u if u < 0.5 else 1 - 2 * (1 - u) * (1 - u)


def unease(s):
    return math.sqrt(s / 2) if s < 0.5 else 1 - math.sqrt((1 - s) / 2)


def stroke_hit_s(path, pts, tolerance):
    """Where does a (keyframe mark) stroke touch the path? -> s or None."""
    best_s, best_d = None, 1e30
    for p in pts:
        s, d = path.nearest(p)
        if d < best_d:
            best_s, best_d = s, d
    return best_s if best_d <= tolerance else None


# --------------------------------------------------------------------------
# Animation helpers
# --------------------------------------------------------------------------

def set_world_location(ob, world):
    if ob.parent is None:
        ob.location = world
    else:
        ob.location = ob.matrix_parent_inverse.inverted() @ (ob.parent.matrix_world.inverted() @ world)


def clear_location_fcurves(ob):
    ad = ob.animation_data
    if not ad or not ad.action:
        return
    act = ad.action
    fcurves = None
    try:
        fcurves = act.fcurves                      # Blender <= 4.4
    except Exception:
        fcurves = None
    if fcurves is None:
        try:                                       # Blender 4.4+ slotted actions
            from bpy_extras import anim_utils
            bag = anim_utils.action_get_channelbag_for_slot(act, ad.action_slot)
            fcurves = bag.fcurves if bag else None
        except Exception:
            fcurves = None
    if fcurves is None:
        return
    for fc in [f for f in fcurves if f.data_path == "location"]:
        try:
            fcurves.remove(fc)
        except Exception:
            pass


def animated_range(scene):
    lo = hi = None
    for ob in scene.objects:
        if ob.get("ms_path"):
            continue
        ad = ob.animation_data
        if ad and ad.action:
            a, b = ad.action.frame_range
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
    if lo is None:
        return None
    lo, hi = int(math.floor(lo)), int(math.ceil(hi))
    return lo, max(hi, lo + 1)


def select_only(context, ob):
    for o in context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    context.view_layer.objects.active = ob


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------

def signed_area(pts):
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                     for i in range(len(pts)))


def shape_outline(kind, size):
    if kind == "SQUARE":
        pts = [(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)]
    elif kind == "CIRCLE":
        pts = [(.5 * math.cos(2 * math.pi * i / 48), .5 * math.sin(2 * math.pi * i / 48)) for i in range(48)]
    else:
        pts = [(x / 100 - .5, .5 - y / 100) for x, y in SHAPE_POLY[kind]]
    if signed_area(pts) < 0:
        pts.reverse()
    return [(x * size, y * size) for x, y in pts]


def make_flat_object(context, name, outline, matrix, color, depth):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    verts = [bm.verts.new((x, y, 0.0)) for x, y in outline]
    try:
        bm.faces.new(verts)
    except ValueError:
        pass
    if bm.faces:
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.matrix_world = matrix
    ob.color = (color[0], color[1], color[2], 1.0)
    context.collection.objects.link(ob)
    if depth > 0:
        mod = ob.modifiers.new("Thickness", 'SOLIDIFY')
        mod.thickness = depth
        mod.offset = 0.0
    select_only(context, ob)
    # show the object colour in solid mode
    _, _, space = find_view3d(context)
    if space and space.shading.type == 'SOLID':
        space.shading.color_type = 'OBJECT'
    return ob


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

class MotionSketchSettings(bpy.types.PropertyGroup):
    shape_size: FloatProperty(name="Size", default=1.0, min=0.01, soft_max=10.0)
    shape_color: FloatVectorProperty(name="Colour", subtype='COLOR', size=3,
                                     default=(0.97, 0.96, 0.92), min=0.0, max=1.0)
    shape_depth: FloatProperty(name="Thickness", default=0.0, min=0.0, soft_max=1.0,
                               description="0 = flat shape, above 0 adds a Solidify modifier")

    duration: IntProperty(name="Duration", default=48, min=2, description="Length of the motion in frames")
    key_mode: EnumProperty(
        name="Keyframes",
        items=[
            ('AUTO', "Auto", "Evenly spaced keyframes along the sketch"),
            ('MARKS', "From marks", "Keyframes where your red keyframe-mark strokes cross the path"),
            ('EVERY_FRAME', "Every frame", "A keyframe on every frame (exact, but dense)"),
        ],
        default='AUTO')
    key_count: IntProperty(name="Count", default=6, min=2, max=50)
    timing: EnumProperty(
        name="Timing",
        items=[('UNIFORM', "Uniform", "Constant speed along the sketch"),
               ('EASE', "Ease in/out", "Start slow, speed up, finish slow")],
        default='UNIFORM')
    interp: EnumProperty(
        name="Interpolation",
        items=[('LINEAR', "Linear", "Straight lines between keyframes"),
               ('BEZIER', "Smooth", "Smooth Bezier curves between keyframes")],
        default='BEZIER')
    relative: BoolProperty(
        name="Relative to object", default=True,
        description="Object travels the sketched shape from where it is now. "
                    "Off: object jumps onto the sketch itself")
    keep_curve: BoolProperty(name="Keep path curve", default=True,
                             description="Leave a curve object showing the path after baking")
    clear_after: BoolProperty(name="Clear sketch after", default=True)

    export_path: StringProperty(name="File", subtype='FILE_PATH', default="//motionsketch.mp4")
    paper_look: BoolProperty(name="Paper look", default=True,
                             description="Flat, outlined Workbench render on a paper-coloured background")
    show_paths: BoolProperty(name="Show pencil path", default=False,
                             description="Render the path curves as thin lines in the video")
    scale_pct: IntProperty(name="Resolution %", default=100, min=10, max=200, subtype='PERCENTAGE')


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------

class MOTIONSKETCH_OT_add_shape(bpy.types.Operator):
    bl_idname = "motionsketch.add_shape"
    bl_label = "Add Shape"
    bl_description = "Add a flat shape at the 3D cursor, facing the viewport"
    bl_options = {'REGISTER', 'UNDO'}

    kind: EnumProperty(items=[('SQUARE', "Square", ""), ('CIRCLE', "Circle", ""),
                              ('TRIANGLE', "Triangle", ""), ('STAR', "Star", "")])

    def execute(self, context):
        ms = context.scene.motionsketch
        _, _, space = find_view3d(context)
        rot = Matrix.Identity(4)
        if space and space.region_3d:
            rot = space.region_3d.view_rotation.to_matrix().to_4x4()
        matrix = Matrix.Translation(context.scene.cursor.location) @ rot
        make_flat_object(context, self.kind.title(), shape_outline(self.kind, ms.shape_size),
                         matrix, ms.shape_color, ms.shape_depth)
        return {'FINISHED'}


class MOTIONSKETCH_OT_sketch(bpy.types.Operator):
    bl_idname = "motionsketch.sketch"
    bl_label = "Sketch"
    bl_description = "Switch to the Annotate tool on the matching layer. Then draw in the viewport"
    bl_options = {'REGISTER'}

    mode: EnumProperty(items=[('SHAPE', "Shape", ""), ('MOTION', "Motion", ""), ('KEYS', "Keys", "")])

    def execute(self, context):
        layer_name = {'SHAPE': LAYER_SHAPE, 'MOTION': LAYER_MOTION, 'KEYS': LAYER_KEYS}[self.mode]
        gp = get_annotation(context, create=True)
        ensure_layer(gp, layer_name)
        try:
            context.scene.tool_settings.annotation_stroke_placement_view3d = 'CURSOR'
        except Exception:
            pass
        try:
            bpy.ops.wm.tool_set_by_id(name="builtin.annotate")
        except Exception:
            self.report({'WARNING'}, "Pick the Annotate tool in the toolbar, then draw")
        hints = {
            'SHAPE': "Draw a closed outline, then press 'Make Shape'",
            'MOTION': "Draw the path of the motion, then press 'Bake Motion'",
            'KEYS': "Draw short strokes across the path where you want keyframes",
        }
        self.report({'INFO'}, hints[self.mode])
        return {'FINISHED'}


class MOTIONSKETCH_OT_shape_from_sketch(bpy.types.Operator):
    bl_idname = "motionsketch.shape_from_sketch"
    bl_label = "Make Shape"
    bl_description = "Turn the last stroke on the 'MS Shape' layer into a filled mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ms = context.scene.motionsketch
        layer = find_layer(get_annotation(context), LAYER_SHAPE)
        strokes = layer_strokes(layer)
        if not strokes:
            self.report({'ERROR'}, "Draw an outline first (Sketch Shape)")
            return {'CANCELLED'}
        pts = stroke_points(strokes[-1])
        if len(pts) < 5:
            self.report({'ERROR'}, "Outline is too short")
            return {'CANCELLED'}

        total = polyline_length(pts)
        pts = smooth_path(clean_path(pts, max(total / 120.0, 1e-5)), 2)
        if len(pts) > 3 and (pts[0] - pts[-1]).length < total / 60.0:
            pts.pop()                              # drop the duplicate closing point
        if len(pts) < 3:
            self.report({'ERROR'}, "Outline is too small")
            return {'CANCELLED'}

        centre = sum(pts, Vector((0.0, 0.0, 0.0))) / len(pts)
        normal = Vector((0.0, 0.0, 0.0))
        for i, a in enumerate(pts):
            b = pts[(i + 1) % len(pts)]
            normal.x += (a.y - b.y) * (a.z + b.z)
            normal.y += (a.z - b.z) * (a.x + b.x)
            normal.z += (a.x - b.x) * (a.y + b.y)
        if normal.length < 1e-9:
            self.report({'ERROR'}, "Could not work out the drawing plane, draw a bigger loop")
            return {'CANCELLED'}
        normal.normalize()
        u = pts[0] - centre
        u -= normal * u.dot(normal)
        if u.length < 1e-9:
            u = normal.orthogonal()
        u.normalize()
        v = normal.cross(u)

        outline = [((p - centre).dot(u), (p - centre).dot(v)) for p in pts]
        rot = Matrix((u, v, normal)).transposed().to_4x4()
        make_flat_object(context, "Sketch Shape", outline, Matrix.Translation(centre) @ rot,
                         ms.shape_color, ms.shape_depth)
        if ms.clear_after:
            clear_layer(layer)
        return {'FINISHED'}


class MOTIONSKETCH_OT_bake_motion(bpy.types.Operator):
    bl_idname = "motionsketch.bake_motion"
    bl_label = "Bake Motion"
    bl_description = "Turn the last stroke on the 'MS Motion' layer into location keyframes on the selected objects"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        ms = sc.motionsketch
        gp = get_annotation(context)
        motion_layer = find_layer(gp, LAYER_MOTION)
        strokes = layer_strokes(motion_layer)
        if not strokes:
            self.report({'ERROR'}, "Draw a motion first (Sketch Motion)")
            return {'CANCELLED'}
        pts = stroke_points(strokes[-1])
        if len(pts) < 3:
            self.report({'ERROR'}, "Stroke is too short")
            return {'CANCELLED'}
        total = polyline_length(pts)
        if total < 1e-4:
            self.report({'ERROR'}, "Stroke is too short")
            return {'CANCELLED'}
        path = SketchPath(smooth_path(clean_path(pts, max(total / 400.0, 1e-5)), 2))

        objs = [o for o in context.selected_objects if not o.get("ms_path")]
        if not objs and context.active_object and not context.active_object.get("ms_path"):
            objs = [context.active_object]
        if not objs:
            self.report({'ERROR'}, "Select the object(s) to animate")
            return {'CANCELLED'}

        # ---- where along the path do keyframes go? ----
        if ms.key_mode == 'EVERY_FRAME':
            n = ms.duration + 1
            s_list = [i / (n - 1) for i in range(n)]
        elif ms.key_mode == 'MARKS':
            s_list = [0.0, 1.0]
            for st in layer_strokes(find_layer(gp, LAYER_KEYS)):
                s = stroke_hit_s(path, stroke_points(st), path.length * 0.15)
                if s is not None:
                    s_list.append(s)
            s_list.sort()
            dedup = [s_list[0]]
            for s in s_list[1:]:
                if s - dedup[-1] > 0.01:
                    dedup.append(s)
            if dedup[-1] < 1.0:
                dedup[-1] = 1.0
            s_list = dedup
            if len(s_list) == 2:
                self.report({'WARNING'}, "No keyframe marks found on the path, only start and end were keyed")
        else:
            n = ms.key_count
            s_list = [i / (n - 1) for i in range(n)]

        start = sc.frame_current
        frames = []
        for s in s_list:
            tau = unease(s) if ms.timing == 'EASE' else s
            f = start + int(round(tau * ms.duration))
            if frames and f <= frames[-1][0]:
                continue
            frames.append((f, s))
        if frames[-1][0] != start + ms.duration:
            frames[-1] = (start + ms.duration, 1.0)
        if sc.frame_end < start + ms.duration:
            sc.frame_end = start + ms.duration

        # ---- insert keyframes ----
        edit = context.preferences.edit
        old_interp, old_handle = edit.keyframe_new_interpolation_type, edit.keyframe_new_handle_type
        edit.keyframe_new_interpolation_type = ms.interp
        edit.keyframe_new_handle_type = 'AUTO_CLAMPED'
        p0 = path.at(0.0)
        curve_points = None
        curve_owner = None
        try:
            for ob in objs:
                base = ob.matrix_world.translation.copy()
                clear_location_fcurves(ob)
                for f, s in frames:
                    p = path.at(s)
                    set_world_location(ob, base + (p - p0) if ms.relative else p)
                    ob.keyframe_insert("location", frame=f)
                if ob == (context.active_object or objs[0]) or curve_points is None:
                    curve_points = [(base + (path.at(i / 199) - p0)) if ms.relative else path.at(i / 199)
                                    for i in range(200)]
                    curve_owner = ob.name
        finally:
            edit.keyframe_new_interpolation_type = old_interp
            edit.keyframe_new_handle_type = old_handle
        sc.frame_set(sc.frame_current)

        if ms.keep_curve and curve_points:
            self.make_curve(context, curve_owner, curve_points)

        if ms.clear_after:
            clear_layer(motion_layer)
            clear_layer(find_layer(gp, LAYER_KEYS))

        self.report({'INFO'}, f"Baked {len(frames)} keyframes over {ms.duration} frames. "
                              "Edit them in the Dope Sheet or Graph Editor.")
        return {'FINISHED'}

    @staticmethod
    def make_curve(context, owner_name, points):
        name = f"{owner_name} path"
        old = bpy.data.objects.get(name)
        if old:
            data = old.data
            bpy.data.objects.remove(old)
            if data and data.users == 0:
                bpy.data.curves.remove(data)
        cu = bpy.data.curves.new(name, 'CURVE')
        cu.dimensions = '3D'
        spline = cu.splines.new('POLY')
        spline.points.add(len(points) - 1)
        for i, p in enumerate(points):
            spline.points[i].co = (p.x, p.y, p.z, 1.0)
        co = bpy.data.objects.new(name, cu)
        co["ms_path"] = True
        co.hide_render = True
        co.show_in_front = True
        co.color = (0.1, 0.1, 0.1, 1.0)
        context.collection.objects.link(co)


class MOTIONSKETCH_OT_keys_from_marks(bpy.types.Operator):
    bl_idname = "motionsketch.keys_from_marks"
    bl_label = "Add Keys from Marks"
    bl_description = ("Add a keyframe on the active object wherever your red keyframe-mark strokes cross "
                      "its existing motion. The motion itself does not change")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        ob = context.active_object
        if ob is None or not ob.animation_data or not ob.animation_data.action:
            self.report({'ERROR'}, "Select an object that already has animation")
            return {'CANCELLED'}
        marks = layer_strokes(find_layer(get_annotation(context), LAYER_KEYS))
        if not marks:
            self.report({'ERROR'}, "Draw keyframe marks first (Sketch Keyframe Marks)")
            return {'CANCELLED'}

        f0, f1 = (int(ob.animation_data.action.frame_range[0]),
                  int(math.ceil(ob.animation_data.action.frame_range[1])))
        current = sc.frame_current
        samples = []
        for f in range(f0, f1 + 1):
            sc.frame_set(f)
            dg = context.evaluated_depsgraph_get()
            samples.append((f, ob.evaluated_get(dg).matrix_world.translation.copy()))
        trail = sum((samples[i][1] - samples[i - 1][1]).length for i in range(1, len(samples)))
        tolerance = max(trail * 0.15, 1e-4)

        added = []
        for st in marks:
            best_f, best_d = None, 1e30
            for p in stroke_points(st):
                for f, pos in samples:
                    d = (pos - p).length
                    if d < best_d:
                        best_f, best_d = f, d
            if best_f is not None and best_d <= tolerance:
                added.append(best_f)

        for f in sorted(set(added)):
            sc.frame_set(f)
            ob.keyframe_insert("location", frame=f)
        sc.frame_set(current)

        if context.scene.motionsketch.clear_after:
            clear_layer(find_layer(get_annotation(context), LAYER_KEYS))
        if added:
            self.report({'INFO'}, f"Added {len(set(added))} keyframe(s) at frames {sorted(set(added))}")
        else:
            self.report({'WARNING'}, "No marks were close enough to the object's path")
        return {'FINISHED'}


class MOTIONSKETCH_OT_clear_sketch(bpy.types.Operator):
    bl_idname = "motionsketch.clear_sketch"
    bl_label = "Clear Sketches"
    bl_description = "Remove all MotionSketch annotation strokes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        gp = get_annotation(context)
        for name in (LAYER_SHAPE, LAYER_MOTION, LAYER_KEYS):
            clear_layer(find_layer(gp, name))
        return {'FINISHED'}


class MOTIONSKETCH_OT_camera_from_view(bpy.types.Operator):
    bl_idname = "motionsketch.camera_from_view"
    bl_label = "Camera from View"
    bl_description = "Create a camera if needed and snap it to the current viewport view"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        area, region, _ = find_view3d(context)
        if area is None:
            self.report({'ERROR'}, "No 3D viewport found")
            return {'CANCELLED'}
        if sc.camera is None:
            cd = bpy.data.cameras.new("MS Camera")
            cam = bpy.data.objects.new("MS Camera", cd)
            context.collection.objects.link(cam)
            sc.camera = cam
        with context.temp_override(area=area, region=region):
            bpy.ops.view3d.camera_to_view()
        return {'FINISHED'}


class MOTIONSKETCH_OT_play(bpy.types.Operator):
    bl_idname = "motionsketch.play"
    bl_label = "Play"
    bl_description = "Jump to the first frame and play the animation"

    def execute(self, context):
        sc = context.scene
        rng = animated_range(sc)
        if rng:
            sc.frame_start, sc.frame_end = rng
        sc.frame_set(sc.frame_start)
        bpy.ops.screen.animation_play()
        return {'FINISHED'}


class MOTIONSKETCH_OT_export_video(bpy.types.Operator):
    bl_idname = "motionsketch.export_video"
    bl_label = "Export Video"
    bl_description = "Render the animation to an MP4 file (Blender is busy until it finishes)"

    def execute(self, context):
        sc = context.scene
        ms = sc.motionsketch
        r = sc.render

        if sc.camera is None:
            self.report({'ERROR'}, "The scene has no camera. Press 'Camera from View' first")
            return {'CANCELLED'}
        rng = animated_range(sc)
        if rng is None:
            self.report({'ERROR'}, "Nothing is animated yet. Bake a motion first")
            return {'CANCELLED'}

        raw = ms.export_path
        if raw.startswith("//") and not bpy.data.filepath:
            raw = os.path.join(os.path.expanduser("~"), raw[2:])
        target = os.path.abspath(bpy.path.abspath(raw))
        if not target.lower().endswith(".mp4"):
            target += ".mp4"
        root = target[:-4]
        folder = os.path.dirname(target)
        os.makedirs(folder, exist_ok=True)

        shading = sc.display.shading
        snap = [(sc, 'frame_start'), (sc, 'frame_end'), (r, 'filepath'), (r, 'engine'),
                (r, 'resolution_percentage')]
        if hasattr(r.image_settings, 'media_type'):          # Blender 5.0+
            snap.append((r.image_settings, 'media_type'))
        snap += [(r.image_settings, 'file_format'), (r.ffmpeg, 'format'), (r.ffmpeg, 'codec'),
                 (r.ffmpeg, 'constant_rate_factor'),
                 (shading, 'light'), (shading, 'color_type'), (shading, 'show_object_outline')]
        if sc.world:
            snap.append((sc.world, 'color'))
        saved = [(o, a, getattr(o, a)) for o, a in snap]

        path_objs = [o for o in sc.objects if o.get("ms_path") and o.type == 'CURVE']
        saved_paths = [(o, o.hide_render, o.data.bevel_depth, o.data.bevel_resolution) for o in path_objs]

        try:
            sc.frame_start, sc.frame_end = rng
            r.resolution_percentage = ms.scale_pct
            r.filepath = root
            if hasattr(r.image_settings, 'media_type'):
                r.image_settings.media_type = 'VIDEO'
            r.image_settings.file_format = 'FFMPEG'
            r.ffmpeg.format = 'MPEG4'
            r.ffmpeg.codec = 'H264'
            r.ffmpeg.constant_rate_factor = 'HIGH'
            if ms.paper_look:
                r.engine = 'BLENDER_WORKBENCH'
                shading.light = 'FLAT'
                shading.color_type = 'OBJECT'
                shading.show_object_outline = True
                if sc.world:
                    sc.world.color = (0.91, 0.90, 0.87)
            if ms.show_paths:
                for o in path_objs:
                    o.hide_render = False
                    o.data.bevel_depth = 0.012 * max(ms.shape_size, 0.1)
                    o.data.bevel_resolution = 0
            bpy.ops.render.render(animation=True)
        except RuntimeError as exc:
            self.report({'ERROR'}, f"Render failed: {exc}")
            return {'CANCELLED'}
        finally:
            for o, a, v in saved:
                try:
                    setattr(o, a, v)
                except Exception:
                    pass
            for o, hr, bd, br in saved_paths:
                o.hide_render = hr
                o.data.bevel_depth = bd
                o.data.bevel_resolution = br

        # Blender appends the frame range to movie names; give it the exact name asked for
        prefix = os.path.basename(root)
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.startswith(prefix) and f.lower().endswith(".mp4")]
        if files:
            newest = max(files, key=os.path.getmtime)
            if os.path.abspath(newest) != target:
                try:
                    if os.path.exists(target):
                        os.remove(target)
                    os.replace(newest, target)
                except OSError:
                    target = newest
        self.report({'INFO'}, f"Saved {target}")
        return {'FINISHED'}


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

class MOTIONSKETCH_PT_main(bpy.types.Panel):
    bl_label = "MotionSketch"
    bl_idname = "MOTIONSKETCH_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MotionSketch"

    def draw(self, context):
        col = self.layout.column(align=True)
        col.label(text="Draw the motion.")
        row = col.row(align=True)
        row.operator("motionsketch.play", icon='PLAY')
        row.operator("motionsketch.clear_sketch", icon='TRASH', text="Clear")


class MOTIONSKETCH_PT_shapes(bpy.types.Panel):
    bl_label = "1 · Shapes"
    bl_idname = "MOTIONSKETCH_PT_shapes"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MotionSketch"
    bl_parent_id = "MOTIONSKETCH_PT_main"

    def draw(self, context):
        ms = context.scene.motionsketch
        layout = self.layout
        row = layout.row(align=True)
        for kind, icon in (("SQUARE", 'MESH_PLANE'), ("CIRCLE", 'MESH_CIRCLE'),
                           ("TRIANGLE", 'TRIA_UP'), ("STAR", 'SOLO_ON')):
            row.operator("motionsketch.add_shape", text="", icon=icon).kind = kind
        col = layout.column(align=True)
        col.prop(ms, "shape_size")
        col.prop(ms, "shape_color")
        col.prop(ms, "shape_depth")
        layout.label(text="Or draw your own:")
        col = layout.column(align=True)
        op = col.operator("motionsketch.sketch", text="Sketch Shape", icon='GREASEPENCIL')
        op.mode = 'SHAPE'
        col.operator("motionsketch.shape_from_sketch", text="Make Shape", icon='MESH_DATA')


class MOTIONSKETCH_PT_motion(bpy.types.Panel):
    bl_label = "2 · Motion"
    bl_idname = "MOTIONSKETCH_PT_motion"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MotionSketch"
    bl_parent_id = "MOTIONSKETCH_PT_main"

    def draw(self, context):
        ms = context.scene.motionsketch
        layout = self.layout
        col = layout.column(align=True)
        col.operator("motionsketch.sketch", text="Sketch Motion", icon='GREASEPENCIL').mode = 'MOTION'
        col.operator("motionsketch.sketch", text="Sketch Keyframe Marks", icon='KEYFRAME_HLT').mode = 'KEYS'

        layout.separator()
        col = layout.column(align=True)
        col.prop(ms, "duration")
        col.prop(ms, "key_mode")
        if ms.key_mode == 'AUTO':
            col.prop(ms, "key_count")
        col.prop(ms, "timing")
        col.prop(ms, "interp")
        col.prop(ms, "relative")
        col.prop(ms, "keep_curve")
        col.prop(ms, "clear_after")

        layout.separator()
        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator("motionsketch.bake_motion", icon='KEYFRAME')
        col = layout.column(align=True)
        col.operator("motionsketch.keys_from_marks", icon='KEYFRAME_HLT')


class MOTIONSKETCH_PT_export(bpy.types.Panel):
    bl_label = "3 · Export"
    bl_idname = "MOTIONSKETCH_PT_export"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MotionSketch"
    bl_parent_id = "MOTIONSKETCH_PT_main"

    def draw(self, context):
        ms = context.scene.motionsketch
        layout = self.layout
        col = layout.column(align=True)
        col.prop(ms, "export_path", text="")
        col.prop(ms, "scale_pct")
        col.prop(ms, "paper_look")
        col.prop(ms, "show_paths")
        layout.operator("motionsketch.camera_from_view", icon='CAMERA_DATA')
        row = layout.row()
        row.scale_y = 1.3
        row.operator("motionsketch.export_video", icon='FILE_MOVIE')


classes = (
    MotionSketchSettings,
    MOTIONSKETCH_OT_add_shape,
    MOTIONSKETCH_OT_sketch,
    MOTIONSKETCH_OT_shape_from_sketch,
    MOTIONSKETCH_OT_bake_motion,
    MOTIONSKETCH_OT_keys_from_marks,
    MOTIONSKETCH_OT_clear_sketch,
    MOTIONSKETCH_OT_camera_from_view,
    MOTIONSKETCH_OT_play,
    MOTIONSKETCH_OT_export_video,
    MOTIONSKETCH_PT_main,
    MOTIONSKETCH_PT_shapes,
    MOTIONSKETCH_PT_motion,
    MOTIONSKETCH_PT_export,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.motionsketch = PointerProperty(type=MotionSketchSettings)


def unregister():
    del bpy.types.Scene.motionsketch
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
