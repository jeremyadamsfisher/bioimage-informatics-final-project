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
import json
import time

from pathlib import Path
from google.cloud import storage

parser = argparse.ArgumentParser()
parser.add_argument("--bucket-name", required=True)
parser.add_argument("--manifest-fp", required=True, type=os.path.abspath)
parser.add_argument("--gcloud-credentials", required=True, type=os.path.abspath)
opt = parser.parse_args()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = opt.gcloud_credentials
storage_client = storage.Client()

blacklist_fp = Path("blacklist.json")

def chunks(iterable, size):
    """https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks#434328"""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it,size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it,size))


def gdc_client_download(img_ids, t_dir):
    """wrapper for gdc-client with some error handling"""
    try:
        subprocess.call([
            "gdc-client",
            "download",
            *img_ids
        ], cwd=t_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return gdc_client_download(img_ids)
    else:
        return list(Path(t_dir).glob("*/*.svs"))


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


def blacklist_img(img_fp: Path):
    img_fp = img_fp.with_suffix(".svs")
    if not blacklist_fp.exists():
        blacklist = []
    else:
        with blacklist_fp.open() as f:
            blacklist = json.load(f)
    blacklist.append(img_fp.name)
    with blacklist_fp.open("w") as f_out:
        json.dump(blacklist, f_out)

def determine_undownloaded_objects(manifest):
    """determine which SVS files have not been downloaded"""
    previously_downloaded_fnames = {blob.name.replace(".png", ".svs") for blob in storage_client.list_blobs(opt.bucket_name)}
    if blacklist_fp.exists():
        with blacklist_fp.open() as f:
            blacklist = json.load(f)
    else:
        blacklist = []
    manifest_remaining = []
    for entry in manifest:
        if entry["filename"] not in previously_downloaded_fnames \
            and entry["filename"] not in blacklist:
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

chunk_size = 5
for i, imgs in enumerate(chunks(manifest_remaining, chunk_size)):
    with tempfile.TemporaryDirectory() as t_dir:
        print(
            f"Downloading images ({(i*chunk_size)+1:,} to {((i+1)*chunk_size)+1:,}) "
            f"of {len(manifest_remaining):,} to {t_dir}..."
        )
        new_images = gdc_client_download([img["id"] for img in imgs], t_dir)
        for i, img_fp in enumerate(new_images):
            print(f"({i}/{len(new_images)})")

            print(f"\tConverting...")

            # check to see if magnification = 40
            slide_properties = dict(openslide.open_slide(str(img_fp)).properties)
            if float(slide_properties['aperio.AppMag']) != 40:
                blacklist_img(img_fp)
                print(f"rejected image {img_fp.name} with magnification {int(slide_properties['aperio.AppMag'])}")
                img_fp.unlink()
                continue

            img = svs2pil(img_fp, 32)

            # check to see if aspect ratio is too crazy
            x, y = sorted(img.size)
            if x * 2 <= y:
                blacklist_img(img_fp)
                print(f"rejected image {img_fp.name} with aspect ratio {x}/{y}")
                img_fp.unlink()
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
