"""
Stage 2: reconstruct a 3D mesh from the front and back cutouts.

Uses Hunyuan3D-2's multiview shape model, which is the reason this pipeline can
use *both* photos: `MVImageProcessorV2` maps view names to conditioning slots
({'front': 0, 'left': 1, 'back': 2, 'right': 3}) and iterates whatever subset of
that dict you hand it, so front+back is valid (upstream's own example passes only
three views).

Shape generation only — no AI texture. That keeps peak VRAM around 6GB (fits an
8GB card) and avoids compiling `custom_rasterizer`/`differentiable_renderer`,
which the texture stage needs and which has no proven build on Blackwell/CUDA 13.
Texturing happens in Blender instead, by projecting the real photos.

Usage:
    python image_to_3d.py --front output/front_cutout.png --back output/back_cutout.png \
        --output output/mesh.glb
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent / "Hunyuan3D-2"
# Where hy3dgen's smart_load_model looks before falling back to the network.
LOCAL_MODEL_ROOT = Path(os.environ.get("HY3DGEN_MODELS", "~/.cache/hy3dgen")).expanduser()

DEFAULT_MODEL = "tencent/Hunyuan3D-2mv"
DEFAULT_SUBFOLDER = "hunyuan3d-dit-v2-mv-turbo"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", required=True, help="Front cutout (RGBA, transparent background)")
    parser.add_argument("--back", help="Back cutout (RGBA). Omit to reconstruct from the front only.")
    parser.add_argument("--left", help="Optional left view cutout")
    parser.add_argument("--right", help="Optional right view cutout")
    parser.add_argument("--output", default="output/mesh.glb", help="Where to write the mesh")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--subfolder",
        default=DEFAULT_SUBFOLDER,
        help="Checkpoint variant. '-turbo' is step-distilled and fastest; drop to "
        "'hunyuan3d-dit-v2-mv' with ~30-50 steps if the mesh comes out mushy.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Denoising steps. 5 suits the turbo variant; the undistilled model wants 30-50.",
    )
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument(
        "--octree-resolution",
        type=int,
        default=256,
        help="Marching-cubes resolution. 256 keeps VRAM/time down; upstream uses 380.",
    )
    parser.add_argument("--num-chunks", type=int, default=20000, help="Lower this if you hit OOM")
    parser.add_argument("--max-faces", type=int, default=40000, help="Face budget after decimation")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--no-clean", action="store_true", help="Skip floater/degenerate/decimate cleanup")
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Let hy3dgen download the subfolder itself (also pulls a redundant 4.6GB .ckpt)",
    )
    parser.add_argument(
        "--low-ram",
        dest="low_ram",
        action="store_true",
        default=True,
        help="Build the model directly in fp16 to halve host RAM during load (default)",
    )
    parser.add_argument("--no-low-ram", dest="low_ram", action="store_false")
    return parser.parse_args()


def prefetch(model, subfolder, variant="fp16"):
    """Fetch just the config and safetensors weights.

    Left to itself, hy3dgen's smart_load_model snapshot-downloads the whole
    subfolder — which ships model.fp16.ckpt *and* an identical
    model.fp16.safetensors, so 4.6GB of the 9.2GB is pure waste. It checks a local
    directory first, so populating that directory sidesteps the download entirely.
    Symlinks keep only one copy on disk.
    """
    from huggingface_hub import hf_hub_download

    dest = LOCAL_MODEL_ROOT / model / subfolder
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", f"model.{variant}.safetensors"):
        link = dest / name
        if link.exists():
            continue
        blob = Path(hf_hub_download(repo_id=model, filename=f"{subfolder}/{name}")).resolve()
        link.symlink_to(blob)
    return dest


def report_vram(label):
    import torch

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  [{label}] peak VRAM {peak:.2f} GB")


def main():
    args = parse_args()
    if not REPO_DIR.is_dir():
        sys.exit(f"Hunyuan3D-2 repo not found at {REPO_DIR}\nClone it: git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git")
    # Import by path rather than `pip install -e .`: hy3dgen is a plain package with no
    # build step, and its setup.py would drag in gradio/xatlas/onnxruntime we don't need.
    sys.path.insert(0, str(REPO_DIR))

    import torch
    from PIL import Image
    from hy3dgen.shapegen import (
        DegenerateFaceRemover,
        FaceReducer,
        FloaterRemover,
        Hunyuan3DDiTFlowMatchingPipeline,
    )

    views = {"front": args.front, "left": args.left, "back": args.back, "right": args.right}
    images = {}
    for name, path in views.items():
        if not path:
            continue
        img = Image.open(path)
        if img.mode != "RGBA":
            sys.exit(f"{path} is {img.mode}, expected RGBA — run isolate.py first")
        images[name] = img
    print(f"Conditioning views: {', '.join(images)}")

    if not args.no_prefetch:
        print(f"Fetching weights for {args.model} / {args.subfolder}")
        print(f"  cached at {prefetch(args.model, args.subfolder)}")

    print(f"Loading {args.model} / {args.subfolder} (fp16)")
    # hy3dgen builds the model at torch's default dtype and only casts to fp16
    # afterwards, so a 1.1B model transiently needs ~4.4GB of fp32 parameters on top
    # of the ~4.9GB checkpoint — enough to get OOM-killed on a 16GB box. Making fp16
    # the default dtype halves the model allocation. The pipeline casts to fp16
    # anyway, so this only removes a wasted intermediate.
    previous_dtype = torch.get_default_dtype()
    if args.low_ram:
        torch.set_default_dtype(torch.float16)
    try:
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args.model, subfolder=args.subfolder, variant="fp16"
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    print(f"Generating shape: {args.steps} steps, octree {args.octree_resolution}")
    started = time.time()
    mesh = pipeline(
        image=images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        generator=torch.manual_seed(args.seed),
        output_type="trimesh",
    )[0]
    print(f"  done in {time.time() - started:.1f}s -> {len(mesh.faces)} faces")
    report_vram("shape")

    if not args.no_clean:
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=args.max_faces)
        print(f"Cleaned -> {len(mesh.faces)} faces, {len(mesh.vertices)} verts")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out)
    print(f"Saved {out} ({out.stat().st_size / 1024**2:.1f} MB)")
    print(f"  bounds min={mesh.bounds[0].round(3).tolist()} max={mesh.bounds[1].round(3).tolist()}")


if __name__ == "__main__":
    main()
