from typing import Tuple

import os, sys
sys.path.append(os.path.join(os.getcwd(), "scripts", "analysis"))

import argparse
import csv
import torch
from pathlib import Path

from autoencoders import convolutional

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--valid-dir", type=Path, required=True)
    parser.add_argument("--epochs", help="number of epochs to train", type=int, default=100)
    parser.add_argument("-o", "--outfp", type=Path, required=True)
    args = parser.parse_args().__dict__
    return args

def main(train_dir: Path, test_dir: Path, valid_dir: Path, outfp: Path, epochs):
    model, model_weights_fp, valid_dataset = convolutional.train_model(train_dir, test_dir, valid_dir, epochs)
    latent_df = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for img, (img_fp, *_) in valid_dataset:
        x, x_latent = model(img.to(device))
        l1, l2, l3, l4 = tuple(x_latent.detach().numpy())
        latent_df.append({"l1": l1, "l2": l2, "l3": l3, "l4":l4, "img_fp": str(img_fp)})
    
    with open(outfp, "w") as f:
        w = csv.DictWriter(f, fieldnames=["l1", "l2", "l3", "l4", "img_fp"])
        w.writeheader()
        w.writerows(latent_df)

if __name__ == "__main__":
    main(**cli())