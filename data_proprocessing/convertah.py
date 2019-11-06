import glob
import math
import multiprocessing as mp
import numpy as np
import openslide
import os
import PIL
from PIL import Image

source_dir = ''
out_dir = ''
scale_factor = 32


def svs2png(pool):
    """
  Convert a WSI training slide to a saved scaled-down image in a format such as jpg or png.
  Args:
    slide_number: The slide number.
  """

    slide_path = pool.get()
    if slide_path is None:
        return
    img = svs2scaled_PIL(slide_path)

    img_outpath = out_dir + slide_path.split('/')[-1] + '.png'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img.save(img_outpath)


def svs2scaled_PIL(slide_path):
    """
  Convert a WSI training slide to a scaled-down PIL image.
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


def mp_convert(file_paths):
    """
  Convert all WSI training slides to smaller images using multiple processes (one process per core).
  Each process will process a range of slide numbers.
  """

    pool = mp.Pool(len(file_paths))

    workers = [mp.Process(svs2png, args=pool) for i in range(mp.cpu_count())]
    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


if __name__ == "__main__":

    file_paths = []
    for root, dirs, files in os.walk(os.path.abspath(source_dir)):
        for file in files:
            file_paths.append(os.path.join(root, file))

    mp_convert(file_paths)
