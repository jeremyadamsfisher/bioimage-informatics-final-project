import subprocess
import csv
import argparse
import tempfile
import itertools
import PIL
import openslide
import math

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--manifest-fp", required=True)
parser.add_argument("--outdir", required=True, type=Path)
opt = parser.parse_args()

opt.outdir.mkdir(exist_ok=True)

def chunks(iterable, size):
    """https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks#434328"""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it,size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it,size))


def svs2scaled_PIL(slide_path, scale_factor):
    """Convert a WSI training slide to a scaled-down PIL image.
    Args:
      slide_path: The slide number.
    Returns:
      Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """

    slide = openslide.open_slide(str(slide_path))

    og_width, og_height = slide.dimensions
    ds_width = math.floor(og_width / scale_factor)
    ds_height = math.floor(og_height / scale_factor)
    level = slide.get_best_level_for_downsample(scale_factor)
    whole_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_image = whole_image.convert("RGB")
    img = whole_image.resize((ds_width, ds_height), PIL.Image.BILINEAR)

    return img


with open(opt.manifest_fp) as f:
    manifest = csv.DictReader(
        f,
        fieldnames=["id", "filename", "md5", "size", "state"],
        delimiter="\t"
    )
    next(manifest) # skip header
    manifest = list(manifest)  # random access
    chunk_size = 5
    for i, imgs in enumerate(chunks(manifest, chunk_size)):
        with tempfile.TemporaryDirectory() as t_dir:
            print(f"Downloading image ({i+1:,} to {i+chunk_size+1:,}) of {len(manifest):,} to {t_dir}...")
            subprocess.check_output([
                "gdc-client",
                "download",
                *[img["id"] for img in imgs]
            ], cwd=t_dir)
            new_images = list(Path(t_dir).glob("*/*.svs"))
            print(f"Converting {' '.join(map(str, new_images))}")
            for img_fp in new_images:
                img = svs2scaled_PIL(img_fp, 16)
                img.save(img_fp.with_suffix(".png"))