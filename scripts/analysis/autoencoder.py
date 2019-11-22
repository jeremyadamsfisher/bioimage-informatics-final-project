from typing import Tuple, List

import os, sys
sys.path.append(os.path.join(os.getcwd(), "scripts", "analysis"))

import argparse
import csv
import ast
import torch
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image

from autoencoders import convolutional

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size-max", type=int, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--epochs", help="number of epochs to train", type=int, default=100)
    parser.add_argument("-o", "--outfp", type=Path, required=True)
    args = parser.parse_args().__dict__
    return args

# preprocessing constants

def gen_preprocessing_callable(img_size_max):
    """clojure to store dataset specific preprocessing stuff"""

    tsfms = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
    ])
    bg_color = (244,244,244)
    resize_to = (1024, 1024)

    def preprocess_histology_img(img) -> torch.Tensor:
        """regardless of the autoencoder, we preprocess images
        the same way"""

        height, width = img.size
        
        img = img.rotate(random.randint(0,180), fillcolor=bg_color)
        img_padded = Image.new(img.mode, (img_size_max, img_size_max), bg_color)
        img_padded.paste(img, (1500-height//2, 1500-width//2))
        img_padded.thumbnail(resize_to, Image.ANTIALIAS)

        return tsfms(img_padded)
    
    return preprocess_histology_img

def main(train_dir: Path, test_dir: Path, outfp: Path, epochs, img_size_max: int):
    preprocess_histology_img = gen_preprocessing_callable(img_size_max)

    print("Training...")
    model, model_weights_fp, test_dataset = convolutional.train_model(
        train_dir, test_dir, epochs, preprocess_histology_img
    )
    print("...done")

    print("Running inference")
    latent_df = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for img, (img_fp, *_) in test_dataset:
        with torch.no_grad():
            x, x_latent = model(img.to(device))
        x_latent = list(x_latent.cpu().numpy())
        latent_df.append({
            "img_fp": str(img_fp),
            **{f"l{i}": latent_dim for i, latent_dim in enumerate(x_latent)}
        })
    with open(outfp, "w") as f:
        fieldnames = sorted(latent_df[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(latent_df)
    print("...done")

if __name__ == "__main__":
    main(**cli())