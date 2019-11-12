from typing import Tuple

import os, sys
sys.path.append(os.path.join(os.getcwd(), "scripts", "analysis"))

import argparse
import csv
import torch
import torchvision.transforms.functional as TF
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



def preprocess_histology_img(img) -> torch.Tensor:
    """regardless of the autoencoder, we preprocess images
    the same way"""

    bg_color = (244,244,244)
    resize_to = (2**10, 2**10)
    height, width = img.size
    padding = abs(height - width)
    if height > width:
        img = TF.pad(img, (0,padding), fill=bg_color)
    elif height < width:
        img = TF.pad(img, (padding,0), fill=bg_color)
    img = TF.resize(img, resize_to)

    return TF.to_tensor(img)

def main(train_dir: Path, test_dir: Path, valid_dir: Path, outfp: Path, epochs):
    model, model_weights_fp, test_dataset = convolutional.train_model(
        train_dir, test_dir, valid_dir, epochs, preprocess_histology_img
    )
    latent_df = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for img, (img_fp, *_) in test_dataset:
        x, x_latent = model(img.to(device))
        x_latent = list(x_latent.cpu().detach().numpy())
        latent_df.append({
            "img_fp": str(img_fp),
            **{f"l{i}": latent_dim for i, latent_dim in enumerate(x_latent)}
        })
    
    with open(outfp, "w") as f:
        fieldnames = sorted(latent_df[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(latent_df)

if __name__ == "__main__":
    main(**cli())