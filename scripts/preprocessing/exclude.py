import argparse

from PIL import Image
from pathlib import Path

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-x",   type=float)
    parser.add_argument("--max-x",   type=float)
    parser.add_argument("--min-y",   type=float)
    parser.add_argument("--max-y",   type=float)
    parser.add_argument("--img-dir", type=Path)
    parser.add_argument("--excluded-img-dir", type=Path)
    return parser.parse_args().__dict__

def main(min_x, max_x, min_y, max_y, img_dir, excluded_img_dir):
    excluded_img_dir.mkdir(exist_ok=True)

    for img_fp in list(img_dir.glob("*")):
        img = Image.open(img_fp)
        x, y = img.size
        if max_x < x or max_y < y:
            img_fp.rename(excluded_img_dir/img_fp.name)

if __name__ == "__main__":
    main(**cli())