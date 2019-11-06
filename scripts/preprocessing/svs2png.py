import glob
import math
import multiprocessing as mp
import os
import argparse

import openslide
import PIL
import numpy as np

from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source-dir", dest="source_dir", default="./")
parser.add_argument("-o", "--output-dir", dest="outdir", default="./")
parser.add_argument("--scale_factor", default=32, type=int)
opt = parser.parse_args()

source_dir = opt.source_dir
out_dir = opt.outdir
scale_factor = opt.scale_factor


def svs2png(slide_path):
    """Convert a WSI training slide to a saved scaled-down image in a format such as jpg or png.
    Args:
      slide_number: The slide number.
    """
    img = svs2scaled_PIL(slide_path)

    img_outpath = out_dir + slide_path.split('/')[-1] + '.png'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img.save(img_outpath)


def svs2scaled_PIL(slide_path):
    """Convert a WSI training slide to a scaled-down PIL image.
    Args:
      slide_path: The slide number.
    Returns:
      Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """

    slide = openslide.open_slide(slide_path)

    og_width, og_height = slide.dimensions
    ds_width = math.floor(og_width / scale_factor)
    ds_height = math.floor(og_height / scale_factor)
    level = slide.get_best_level_for_downsample(scale_factor)
    whole_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_image = whole_image.convert("RGB")
    img = whole_image.resize((ds_width, ds_height), PIL.Image.BILINEAR)
    return img

for fp in Path(source_dir).glob("*.svs"):
    svs2png(str(fp))
