"""
Stage 3: texture the mesh with the real product photos and render a 360 orbit.

Runs inside Blender's bundled Python, NOT the project venv:

    LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
    ./blender/blender --background --python blender_turntable.py -- \
        --mesh output/mesh.glb --front output/front_cutout.png --back output/back_cutout.png \
        --outdir output/frames --res 720 --frames 120

Two design decisions worth knowing about:

1. Texture coordinates come from *world position*, not a UV unwrap. Because the
   cutouts were trimmed to the object's bounding square (see isolate.py), an
   orthographic front projection is just `u = x/ext + 0.5, v = z/ext + 0.5`, which
   a few shader nodes compute directly. That avoids UV unwrapping, the UV Project
   modifier, and projector cameras entirely. The back photo uses the same mapping
   with u mirrored. A mask built from the surface normal's dot product with the
   front axis chooses between them, with a soft blend band across the +/-90 sides.

2. The camera AND the lights orbit together on one pivot, rather than spinning the
   object. Spinning the object would drag it through the world-space projection and
   smear the texture; orbiting a rigid camera+light rig keeps the projection locked
   while still looking like a turntable (lighting stays fixed relative to the view).
"""

import argparse
import math
import sys

import bpy
from mathutils import Matrix, Vector

FRONT_AXIS = Vector((0.0, -1.0, 0.0))  # Blender is Z-up; the camera starts on -Y


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render a 360 turntable of a photo-textured mesh")
    parser.add_argument("--mesh", required=True, help="Input .glb from stage 2")
    parser.add_argument("--front", help="Front cutout PNG (omit to render untextured)")
    parser.add_argument("--back", help="Back cutout PNG")
    parser.add_argument("--outdir", required=True, help="Directory for rendered PNG frames")
    parser.add_argument("--res", type=int, default=720)
    parser.add_argument("--frames", type=int, default=120, help="Frames for a full 360")
    parser.add_argument("--frame-step", type=int, default=1, help=">1 renders a sparse subset (calibration)")
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples per pixel")
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.0,
        help="Degrees about X applied before yaw, to fix the up axis. 0 is correct for "
        "Hunyuan3D output (Blender's glTF importer already rights it). Meshes authored "
        "Z-up and exported by trimesh without axis conversion need -90.",
    )
    parser.add_argument("--yaw-offset", type=float, default=0.0, help="Degrees to spin the mesh so its front faces -Y")
    parser.add_argument("--elevation", type=float, default=12.0, help="Camera elevation in degrees")
    parser.add_argument("--lens", type=float, default=50.0, help="Focal length in mm")
    parser.add_argument("--fit-margin", type=float, default=1.25, help="Framing headroom; >1 pulls the camera back")
    parser.add_argument("--blend", type=float, default=0.25, help="Half-width of the front/back blend band (dot units)")
    parser.add_argument(
        "--projection",
        default="cylindrical",
        choices=["cylindrical", "flat"],
        help="How photos map onto the surface. 'cylindrical' wraps them by angle and keeps "
        "the sides usable; 'flat' is a true orthographic slide projection, more faithful "
        "head-on but it smears badly at +/-90 degrees.",
    )
    parser.add_argument("--tex-scale", type=float, default=1.0, help="Nudge projection scale if the texture misaligns")
    parser.add_argument("--roughness", type=float, default=0.6)
    parser.add_argument(
        "--specular",
        type=float,
        default=0.0,
        help="Specular IOR Level. 0 by default: the photos already contain their own "
        "highlights, so adding more only puts a milky sheen over the material.",
    )
    parser.add_argument("--light-energy", type=float, default=1.0, help="Multiplier on all light power")
    parser.add_argument(
        "--ambient",
        type=float,
        default=0.85,
        help="World strength. Kept high and the lamps kept low on purpose — under near-uniform "
        "light a diffuse surface renders close to its base colour, so the product ends up "
        "looking like the photograph instead of the photograph lit a second time.",
    )
    parser.add_argument("--transparent", action="store_true", help="Render with an alpha background")
    parser.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
    return parser.parse_args(argv)


def socket(node, name, kind):
    """Fetch an input socket by name AND type.

    ShaderNodeMix exposes several same-named sockets (A/B for VALUE, VECTOR, RGBA,
    ROTATION), so looking up by name alone is ambiguous and index-based access is
    brittle across versions.
    """
    for sock in node.inputs:
        if sock.name == name and sock.type == kind:
            return sock
    raise KeyError(f"{node.bl_idname} has no {kind} input named {name!r}")


def reset_scene():
    # Must happen before configuring the GPU: this resets in-memory preferences.
    bpy.ops.wm.read_factory_settings(use_empty=True)


def configure_gpu(scene, engine):
    scene.render.engine = engine
    if engine != "CYCLES":
        return "n/a (EEVEE)"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    # OptiX is preferred but is unavailable under WSL2 on this box (it reports
    # "OptiX initialization failed with error code 7805" even with libnvoptix.so.1
    # on LD_LIBRARY_PATH), so fall through to CUDA and finally CPU.
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.refresh_devices()
        if any(d.type == backend for d in prefs.devices):
            for device in prefs.devices:
                device.use = device.type == backend
            scene.cycles.device = "GPU"
            return backend

    scene.cycles.device = "CPU"
    return "CPU"


def import_mesh(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    imported = [o for o in set(bpy.data.objects) - before if o.type == "MESH"]
    if not imported:
        sys.exit(f"no mesh objects found in {path}")

    for obj in bpy.data.objects:
        obj.select_set(obj in imported)
    bpy.context.view_layer.objects.active = imported[0]
    if len(imported) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def world_bounds(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return lo, hi


def normalize(obj, pitch_degrees, yaw_degrees):
    """Stand the mesh upright, center it on the origin, scale its longest axis to 1.0."""
    lo, hi = world_bounds(obj)
    center = (lo + hi) / 2.0
    longest = max(hi - lo)
    if longest <= 0:
        sys.exit("mesh has zero extent")

    # Pitch first (fix the up axis), then yaw about the now-correct vertical.
    obj.matrix_world = (
        Matrix.Rotation(math.radians(yaw_degrees), 4, "Z")
        @ Matrix.Rotation(math.radians(pitch_degrees), 4, "X")
        @ Matrix.Scale(1.0 / longest, 4)
        @ Matrix.Translation(-center)
        @ obj.matrix_world
    )
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    lo, hi = world_bounds(obj)
    return hi - lo


def map_range(tree, source, from_min, from_max, to_min, to_max, clamp=False):
    """A MapRange node wired to `source`, returning its Result socket."""
    node = tree.nodes.new("ShaderNodeMapRange")
    node.clamp = clamp
    for name, value in (
        ("From Min", from_min),
        ("From Max", from_max),
        ("To Min", to_min),
        ("To Max", to_max),
    ):
        socket(node, name, "VALUE").default_value = value
    tree.links.new(source, socket(node, "Value", "VALUE"))
    return node.outputs["Result"]


def math_node(tree, operation, first, second=None):
    """A Math node; `first`/`second` may be sockets or plain floats."""
    node = tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    for index, operand in enumerate((first, second)):
        if operand is None:
            continue
        if isinstance(operand, float):
            node.inputs[index].default_value = operand
        else:
            tree.links.new(operand, node.inputs[index])
    return node.outputs["Value"]


def turn_fraction(tree, separate):
    """Angle around the vertical axis as a 0-1 fraction of a turn, 0.5 dead ahead."""
    angle = math_node(
        tree, "ARCTAN2", separate.outputs["X"], math_node(tree, "MULTIPLY", separate.outputs["Y"], -1.0)
    )
    return map_range(tree, angle, -math.pi, math.pi, 0.0, 1.0)


def horizontal_uv(tree, separate, turn, half, flip):
    """Texture u for one of the two projectors."""
    if turn is None:
        # Flat orthographic slide projection: world X maps straight to image X.
        return map_range(tree, separate.outputs["X"], -half, half, 1.0 if flip else 0.0, 0.0 if flip else 1.0)

    # Cylindrical: spread the image around the turn angle instead. A flat projection
    # crushes every side-facing surface into the image's outermost pixel column, which
    # is exactly what smears the +/-90 degree sides; by angle, each azimuth gets its
    # own column. The back image is offset half a turn so it centres behind the object.
    source = math_node(tree, "FRACT", math_node(tree, "ADD", turn, 0.25)) if flip else turn
    span = (0.0, 0.5) if flip else (0.25, 0.75)
    return map_range(tree, source, span[0], span[1], 0.0, 1.0)


def build_material(obj, front_png, back_png, extent, args):
    material = bpy.data.materials.new("ProductProjection")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()

    out = tree.nodes.new("ShaderNodeOutputMaterial")
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    socket(bsdf, "Roughness", "VALUE").default_value = args.roughness
    socket(bsdf, "Specular IOR Level", "VALUE").default_value = args.specular
    tree.links.new(bsdf.outputs["BSDF"], socket(out, "Surface", "SHADER"))

    geometry = tree.nodes.new("ShaderNodeNewGeometry")

    if not front_png:
        socket(bsdf, "Base Color", "RGBA").default_value = (0.55, 0.55, 0.58, 1.0)
        obj.data.materials.clear()
        obj.data.materials.append(material)
        return material

    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(geometry.outputs["Position"], socket(separate, "Vector", "VECTOR"))

    half = extent / 2.0
    v_coord = map_range(tree, separate.outputs["Z"], -half, half, 0.0, 1.0)
    turn = turn_fraction(tree, separate) if args.projection == "cylindrical" else None

    def image_node(path, flip):
        combine = tree.nodes.new("ShaderNodeCombineXYZ")
        tree.links.new(horizontal_uv(tree, separate, turn, half, flip), socket(combine, "X", "VALUE"))
        tree.links.new(v_coord, socket(combine, "Y", "VALUE"))

        tex = tree.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(path)
        tex.extension = "EXTEND"
        tree.links.new(combine.outputs["Vector"], socket(tex, "Vector", "VECTOR"))
        return tex

    front_tex = image_node(front_png, flip=False)
    # Seen from behind, the object's +X side appears on the left, which is exactly how
    # the back photo was shot — so the back projection is the front mapping mirrored.
    back_tex = image_node(back_png or front_png, flip=True)

    # Pick front vs back from which way each surface faces.
    dot = tree.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    tree.links.new(geometry.outputs["Normal"], dot.inputs[0])
    dot.inputs[1].default_value = FRONT_AXIS
    # Facing away -> 1 -> back photo; facing the viewer -> 0 -> front photo.
    mask = map_range(tree, dot.outputs["Value"], -args.blend, args.blend, 1.0, 0.0, clamp=True)

    mix = tree.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    tree.links.new(mask, socket(mix, "Factor", "VALUE"))
    tree.links.new(front_tex.outputs["Color"], socket(mix, "A", "RGBA"))
    tree.links.new(back_tex.outputs["Color"], socket(mix, "B", "RGBA"))
    tree.links.new(mix.outputs["Result"], socket(bsdf, "Base Color", "RGBA"))

    obj.data.materials.clear()
    obj.data.materials.append(material)
    return material


def build_rig(scene, dims, args):
    """Camera + lights on a single pivot, keyframed for one full revolution."""
    half_fov = math.atan((36.0 / 2.0) / args.lens)  # 36mm is Blender's default sensor width
    # Orbiting about Z, the widest horizontal half-extent is the XY diagonal.
    horizontal = 0.5 * math.hypot(dims.x, dims.y)
    needed = max(horizontal, dims.z / 2.0)
    distance = needed / math.tan(half_fov) * args.fit_margin

    pivot = bpy.data.objects.new("Pivot", None)
    scene.collection.objects.link(pivot)

    elevation = math.radians(args.elevation)
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = args.lens
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, -distance * math.cos(elevation), distance * math.sin(elevation))
    camera.rotation_euler = (math.pi / 2.0 - elevation, 0.0, 0.0)
    camera.parent = pivot
    scene.camera = camera

    # Softbox-ish three-point rig, scaled with the framing distance so exposure holds.
    # Deliberately gentle: these only add shape-revealing gradients on top of the
    # ambient fill, since the projected photos supply the actual lighting.
    falloff = distance**2
    for name, location, energy, size in (
        ("Key", (-1.0, -1.1, 1.2), 55.0, 2.2),
        ("Fill", (1.3, -0.9, 0.2), 22.0, 2.6),
        ("Rim", (0.4, 1.3, 1.0), 35.0, 1.8),
    ):
        light_data = bpy.data.lights.new(name, type="AREA")
        light_data.energy = energy * falloff * args.light_energy
        light_data.size = size * distance / 2.5
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = Vector(location) * distance
        # Aim each light at the origin.
        direction = -light.location
        light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        light.parent = pivot

    # LINEAR interpolation via preferences: Blender 5.0 removed the legacy
    # Action.fcurves API that the usual post-hoc fixup relies on.
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    pivot.rotation_euler = (0.0, 0.0, 0.0)
    pivot.keyframe_insert(data_path="rotation_euler", frame=1)
    # Key a full turn one frame PAST the last rendered frame so the loop closes
    # without duplicating the first frame at the end.
    pivot.rotation_euler = (0.0, 0.0, 2.0 * math.pi)
    pivot.keyframe_insert(data_path="rotation_euler", frame=args.frames + 1)

    return distance


def setup_world(scene, ambient):
    world = bpy.data.worlds.new("Studio")
    scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    # Look the Background node up by type; its name is conventional, not guaranteed.
    background = next((n for n in tree.nodes if n.type == "BACKGROUND"), None)
    if background is None:
        background = tree.nodes.new("ShaderNodeBackground")
        output = next((n for n in tree.nodes if n.type == "OUTPUT_WORLD"), None) or tree.nodes.new(
            "ShaderNodeOutputWorld"
        )
        tree.links.new(background.outputs["Background"], socket(output, "Surface", "SHADER"))
    socket(background, "Color", "RGBA").default_value = (0.95, 0.95, 0.96, 1.0)
    socket(background, "Strength", "VALUE").default_value = ambient


def configure_output(scene, args):
    render = scene.render
    render.resolution_x = render.resolution_y = args.res
    render.resolution_percentage = 100
    render.fps = 30
    scene.frame_start = 1
    scene.frame_end = args.frames
    scene.frame_step = args.frame_step
    render.film_transparent = args.transparent

    # PNG frames rather than Blender's FFmpeg muxer: resumable after a crash, and
    # encoding happens in the venv where imageio-ffmpeg already lives.
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA" if args.transparent else "RGB"
    render.filepath = args.outdir.rstrip("/") + "/frame_"

    if scene.render.engine == "CYCLES":
        scene.cycles.samples = args.samples
        if hasattr(scene.cycles, "use_denoising"):
            scene.cycles.use_denoising = True


def main():
    args = parse_args()
    reset_scene()
    scene = bpy.context.scene

    backend = configure_gpu(scene, args.engine)
    obj = import_mesh(args.mesh)
    dims = normalize(obj, args.pitch, args.yaw_offset)
    extent = max(dims.x, dims.z) * args.tex_scale
    build_material(obj, args.front, args.back, extent, args)
    setup_world(scene, args.ambient)
    distance = build_rig(scene, dims, args)
    configure_output(scene, args)

    print(
        f"[turntable] engine={scene.render.engine} device={backend} "
        f"dims=({dims.x:.3f},{dims.y:.3f},{dims.z:.3f}) extent={extent:.3f} "
        f"cam_dist={distance:.3f} frames={args.frames} step={args.frame_step} -> {args.outdir}"
    )
    bpy.ops.render.render(animation=True)
    print("[turntable] done")


if __name__ == "__main__":
    main()
