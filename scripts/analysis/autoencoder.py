from typing import Tuple

import argparse

from pathlib import Path

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_test_valid_split",
        help="train, test, validation split; formatted as comma-delimited string, e.g.; 0.5,0.3,0.2",
        type=lambda s: [float(ss) for ss in s.split(",")]
    )
    parser.add_argument(
        "-i", "--histology-dir",
        dest="histology_dir",
        type=Path
    )
    parser.add_argument(
        "-o", "--outfp"
    )
    args = parser.parse_args().__dict__
    train_prop, test_prop, valid_prop = args.pop("train_test_valid_split")
    return args

def main(train_test_valid_split: Tuple[float,float,float],
         histology_dir: Path,
         outfp: Path):
    train_prop, test_prop, valid_prop = train_test_valid_split
    raise NotImplementedError

if __name__ == "__main__":
    main(**cli())