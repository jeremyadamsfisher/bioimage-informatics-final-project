"""download from TCGA, downsampled and upload to a bucket"""
import subprocess
import os
import csv
import argparse
import tempfile
import itertools
import PIL
import openslide
import math

from pathlib import Path
from google.cloud import storage

parser = argparse.ArgumentParser()
parser.add_argument("--bucket-name", required=True)
parser.add_argument("--manifest-fp", required=True, type=os.path.abspath)
parser.add_argument("--gcloud-credentials", required=True, type=os.path.abspath)
opt = parser.parse_args()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = opt.gcloud_credentials
storage_client = storage.Client()

def chunks(iterable, size):
    """https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks#434328"""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it,size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it,size))


def svs2pil(slide_path, scale_factor):
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


def determine_undownloaded_objects(manifest):
    """determine which SVS files have not been downloaded"""
    fnames = {blob.name for blob in storage_client.list_blobs(opt.bucket_name)}
    manifest_remaining = []
    for entry in manifest:
        if entry["filename"] not in fnames:
            manifest_remaining.append(entry)
    return manifest_remaining

with open(opt.manifest_fp) as f:
    manifest = csv.DictReader(
        f,
        fieldnames=["id", "filename", "md5", "size", "state"],
        delimiter="\t"
    )
    next(manifest) # skip header
    manifest = list(manifest)  # random access

manifest_remaining = determine_undownloaded_objects(manifest)

chunk_size = 2
for i, imgs in enumerate(chunks(manifest_remaining, chunk_size)):
    with tempfile.TemporaryDirectory() as t_dir:
        print(
            f"Downloading images ({(i*chunk_size)+1:,} to {((i+1)*chunk_size)+1:,}) "
            f"of {len(manifest):,} to {t_dir}..."
        )
        subprocess.check_output([
            "gdc-client",
            "download",
            *[img["id"] for img in imgs]
        ], cwd=t_dir)
        new_images = list(Path(t_dir).glob("*/*.svs"))
        for i, img_fp in enumerate(new_images):
            print(f"({i}/{len(new_images)})")

            print(f"\tConverting...")

            # check to see if magnification = 40
            slide_properties = dict(openslide.open_slide(str(img_fp)).properties)
            if int(slide_properties['aperio.AppMag']) != 40:
                print(f"rejected image {img_fp.name} with magnification {int(slide_properties['aperio.AppMag'])}")
                continue

            img = svs2pil(img_fp, 32)

            # check to see if aspect ratio is too crazy
            x, y = sorted(img.size)
            if x * 2 <= y:
                print(f"rejected image {img_fp.name} with aspect ratio {x}/{y}")
                continue

            img_fp_converted = Path(t_dir)/f"{img_fp.stem}.png"
            img.save(img_fp_converted)

            print(f"\tUploading...")
            bucket = storage_client.get_bucket(opt.bucket_name)
            blob = bucket.blob(img_fp_converted.name)
            blob.upload_from_filename(img_fp_converted)

            print(f"\tCleaning...")
            img_fp.unlink()
            img_fp_converted.unlink()
