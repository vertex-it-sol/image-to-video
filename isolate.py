"""
Stage 1: isolate the product from its background.

Both the 3D reconstruction and the texture projection depend on a clean matte,
so this is the cheapest place to catch a problem — always eyeball the output.

The cutout is trimmed to the object's alpha bounding box and padded to a square
canvas. That normalizes scale between the front and back views (the two source
photos may frame the product slightly differently) and gives the Blender stage a
predictable mapping: the square image covers exactly the object's bounding
square, so it can derive texture coordinates from world position without
needing UV unwrapping.

Two files come out per input:

  <stem>_cutout.png  RGBA with a transparent background, for the 3D reconstruction.
  <stem>_tex.png     Opaque RGB whose background is filled by inpainting outward from
                     the product edge, for Blender's projection texture.

The second exists because a projected texture gets sampled slightly outside the
silhouette on surfaces facing away from the camera — near the +/-90 degree sides of
the object. With a transparent background those samples come back as black (Blender
premultiplies alpha), painting an ugly dark smear down the object's sides. Extending
the product's own colors outward makes those samples read as plausible material.

Usage:
    python isolate.py --inputs inputs/front.png inputs/back.png --outdir output
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Source product photos")
    parser.add_argument("--outdir", default="output", help="Where to write the cutouts")
    parser.add_argument("--model", default="u2net", help="rembg model name")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Extra padding as a fraction of the object's longest side. Keep at 0 so the "
        "square canvas matches the object's bounding square exactly — the Blender "
        "stage assumes this when projecting.",
    )
    parser.add_argument("--no-trim", action="store_true", help="Skip trim-and-square, keep original framing")
    parser.add_argument("--bleed-blur", type=float, default=4.0, help="Blur applied to the _tex.png background fill")
    parser.add_argument(
        "--bleed-fade",
        type=float,
        default=0.06,
        help="Distance (as a fraction of image width) over which the background fill fades "
        "from the nearest edge colour to the product's median colour",
    )
    return parser.parse_args()


def cut_out(path, session):
    """Remove the background, returning RGBA with a fully transparent backdrop."""
    # .convert("RGB") first is deliberate: these PNGs are already RGBA-mode but with
    # a solid off-white background, so rembg must see them as RGB to actually run.
    # (The upstream shape_gen_multiview.py example converts to RGBA and *then* tests
    # `if image.mode == 'RGB'`, which can never be true — its rembg call is dead code.)
    src = Image.open(path).convert("RGB")
    return remove(src, session=session, bgcolor=[255, 255, 255, 0])


def trim_to_square(rgba, margin):
    """Crop to the visible object, then pad to a centered square canvas."""
    bbox = rgba.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError("matte is fully transparent — background removal removed the product too")
    obj = rgba.crop(bbox)
    side = round(max(obj.size) * (1 + 2 * margin))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(obj, ((side - obj.width) // 2, (side - obj.height) // 2))
    return canvas


def bleed_edges(rgba, blur, fade_px):
    """Return opaque RGB with the background replaced by the nearest product colour.

    cv2.inpaint is the obvious tool but collapses to black on a mask this large (the
    background is ~25% of the frame in one blob). A distance transform is exact and
    cheap instead: every background pixel simply takes the colour of the nearest
    foreground pixel, extending the product's edge outward like a Voronoi fill.
    """
    array = np.array(rgba)
    rgb = array[:, :, :3]
    hole = (array[:, :, 3] == 0).astype(np.uint8)
    if not hole.any():
        return Image.fromarray(rgb)

    # labels index the nearest zero (i.e. foreground) pixel, numbered in raster order.
    distance, labels = cv2.distanceTransformWithLabels(
        hole, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    foreground = np.column_stack(np.nonzero(hole == 0))
    nearest = foreground[labels[hole == 1] - 1]
    edge_colour = rgb[nearest[:, 0], nearest[:, 1]].astype(np.float32)

    # Nearest-edge colour alone is wrong far from the silhouette: around a mug's handle
    # the closest edge pixel is deep shadow, so whole regions fill in near-black and
    # then land on side-facing geometry as dark blobs. Fade toward the product's median
    # colour with distance — continuous at the silhouette, plain material further out.
    median = np.median(rgb[hole == 0].reshape(-1, 3), axis=0).astype(np.float32)
    fade = max(1.0, fade_px)
    weight = np.clip(distance[hole == 1] / fade, 0.0, 1.0)[:, None]

    filled = rgb.copy()
    filled[hole == 1] = (edge_colour * (1.0 - weight) + median * weight).round().astype(np.uint8)
    if blur > 0:
        # Soften the radial streaks the Voronoi fill leaves behind, but only outside
        # the product so its own detail stays sharp.
        smoothed = cv2.GaussianBlur(filled, (0, 0), blur)
        filled[hole == 1] = smoothed[hole == 1]
    return Image.fromarray(filled)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    session = new_session(args.model)
    for path in args.inputs:
        src = Path(path)
        cutout = cut_out(src, session)
        if not args.no_trim:
            cutout = trim_to_square(cutout, args.margin)

        dest = outdir / f"{src.stem}_cutout.png"
        cutout.save(dest)
        texture = outdir / f"{src.stem}_tex.png"
        bleed_edges(cutout, args.bleed_blur, args.bleed_fade * cutout.width).save(texture)

        histogram = cutout.getchannel("A").histogram()
        coverage = sum(histogram[1:]) / (cutout.width * cutout.height)
        print(f"{src.name} -> {dest} + {texture.name}  {cutout.width}x{cutout.height}  object covers {coverage:.0%}")


if __name__ == "__main__":
    main()
