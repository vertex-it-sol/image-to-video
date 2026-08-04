"""
Run the whole pipeline: photos -> cutouts -> mesh -> rendered frames -> MP4.

    python run_pipeline.py --front inputs/front.png --back inputs/back.png --output spin.mp4

Stages 1 and 2 run in this interpreter. Stage 3 runs in Blender's own Python via
subprocess (Blender bundles its own interpreter and cannot import this venv), and
stage 4 encodes the frames back here with imageio-ffmpeg.

Skip stages you've already done with --skip, e.g. while tuning the render:

    python run_pipeline.py --front inputs/front.png --back inputs/back.png --skip isolate mesh
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLENDER = HERE / "blender" / "blender"
STAGES = ("isolate", "mesh", "render", "encode")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--front", default="inputs/front.png", help="Front product photo")
    parser.add_argument("--back", default="inputs/back.png", help="Back product photo")
    parser.add_argument("--output", default="spin.mp4")
    parser.add_argument("--workdir", default="output")
    parser.add_argument("--res", type=int, default=720)
    parser.add_argument("--frames", type=int, default=120, help="120 frames at 30fps = a 4s revolution")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples per pixel")
    parser.add_argument("--steps", type=int, default=5, help="Shape-model denoising steps")
    parser.add_argument("--yaw-offset", type=float, default=0.0, help="Rotate the mesh so its front faces the camera")
    parser.add_argument("--elevation", type=float, default=12.0)
    parser.add_argument("--blend", type=float, default=0.25, help="Front/back texture blend band")
    parser.add_argument("--calibrate", action="store_true", help="Render only 4 stills (0/90/180/270) to check orientation")
    parser.add_argument("--transparent", action="store_true", help="Alpha background (frames stay PNG; MP4 gets white)")
    parser.add_argument("--untextured", action="store_true", help="Render a plain material instead of the photos")
    parser.add_argument("--skip", nargs="*", default=[], choices=STAGES, help="Stages to skip")
    return parser.parse_args()


def run(cmd, env=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run([str(c) for c in cmd], env=env)
    if result.returncode != 0:
        sys.exit(f"stage failed with exit code {result.returncode}")


def stage_isolate(args, work):
    run([sys.executable, HERE / "isolate.py", "--inputs", args.front, args.back, "--outdir", work])


def stage_mesh(args, work, mesh):
    run(
        [
            sys.executable, HERE / "image_to_3d.py",
            "--front", work / f"{Path(args.front).stem}_cutout.png",
            "--back", work / f"{Path(args.back).stem}_cutout.png",
            "--output", mesh,
            "--steps", args.steps,
        ]
    )


def stage_render(args, work, mesh, frames_dir):
    if not BLENDER.exists():
        sys.exit(f"Blender not found at {BLENDER} — run ./setup_blender.sh first")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    cmd = [
        BLENDER, "--background", "--python", HERE / "blender_turntable.py", "--",
        "--mesh", mesh,
        "--outdir", frames_dir,
        "--res", args.res,
        "--frames", args.frames,
        "--samples", args.samples,
        "--yaw-offset", args.yaw_offset,
        "--elevation", args.elevation,
        "--blend", args.blend,
    ]
    if not args.untextured:
        # The edge-bled _tex.png variants, not the transparent cutouts: see isolate.py.
        cmd += [
            "--front", work / f"{Path(args.front).stem}_tex.png",
            "--back", work / f"{Path(args.back).stem}_tex.png",
        ]
    if args.transparent:
        cmd += ["--transparent"]
    if args.calibrate:
        # Every 90 degrees: frames 1, 31, 61, 91 of a 120-frame revolution.
        cmd += ["--frame-step", max(1, args.frames // 4)]

    # OptiX/CUDA live in /usr/lib/wsl/lib, which WSL2 puts on PATH but not on
    # LD_LIBRARY_PATH, so Blender can't find them without this.
    env = dict(os.environ)
    wsl_libs = "/usr/lib/wsl/lib"
    if Path(wsl_libs).is_dir():
        env["LD_LIBRARY_PATH"] = f"{wsl_libs}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    run(cmd, env=env)


def stage_encode(args, frames_dir, output):
    import imageio.v2 as imageio
    import numpy as np

    paths = sorted(frames_dir.glob("frame_*.png"))
    if not paths:
        sys.exit(f"no frames found in {frames_dir}")

    writer = imageio.get_writer(
        output, fps=args.fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1
    )
    try:
        for path in paths:
            frame = imageio.imread(path)
            if frame.ndim == 3 and frame.shape[2] == 4:
                # MP4 has no alpha; composite transparent frames onto white.
                rgb = frame[:, :, :3].astype(np.float32)
                alpha = frame[:, :, 3:4].astype(np.float32) / 255.0
                frame = (rgb * alpha + 255.0 * (1.0 - alpha)).round().astype(np.uint8)
            writer.append_data(frame)
    finally:
        writer.close()
    print(f"\nWrote {output}: {len(paths)} frames @ {args.fps}fps = {len(paths) / args.fps:.2f}s")


def main():
    args = parse_args()
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    mesh = work / "mesh.glb"
    frames_dir = work / "frames"
    output = Path(args.output)

    if "isolate" not in args.skip:
        stage_isolate(args, work)
    if "mesh" not in args.skip:
        stage_mesh(args, work, mesh)
    if "render" not in args.skip:
        stage_render(args, work, mesh, frames_dir)
    if "encode" not in args.skip:
        if args.calibrate:
            stills = sorted(frames_dir.glob("frame_*.png"))
            print(f"\nCalibration stills ({len(stills)}): {[p.name for p in stills]}")
            print("Check that frame_0001 shows the product's front; if not, re-run with --yaw-offset.")
        else:
            stage_encode(args, frames_dir, output)


if __name__ == "__main__":
    main()
