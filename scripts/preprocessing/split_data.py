from typing import Tuple

import os, sys
sys.path.append(os.path.join(os.getcwd(), "keras-autoencoder"))

import argparse
import random
from pathlib import Path

random.seed(42)

class img_data_type:
    valid=0
    train=1
    test=2

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_test_split",
        help="train, test, validation split; formatted as comma-delimited string, e.g.; 0.5,0.5",
        type=lambda s: [float(ss) for ss in s.split(",")]
    )
    parser.add_argument(
        "-i", "--histology-dir",
        dest="histology_dir",
        type=Path
    )
    parser.add_argument(
        "-o", "--outdir", dest="outdir", type=Path
    )
    args = parser.parse_args().__dict__
    return args

def main(train_test_split: Tuple[float,float,float],
         histology_dir: Path,
         outdir: Path):
    train_prop, test_prop = train_test_split
    train_dir = outdir/"train"; train_dir.mkdir(exist_ok=True, parents=True)
    test_dir = outdir/"test"; test_dir.mkdir(exist_ok=True, parents=True)
    
    weighted = [img_data_type.train] * int(train_prop * 10000) \
             + [img_data_type.test]  * int(test_prop  * 10000)

    for img_fp in list(histology_dir.glob("*.png")):
        d_type = random.choice(weighted)
        c_outdir = {
            img_data_type.train: train_dir,
            img_data_type.test: test_dir,
        }[d_type]
        img_fp.replace(c_outdir/img_fp.name)

if __name__ == "__main__":
    main(**cli())