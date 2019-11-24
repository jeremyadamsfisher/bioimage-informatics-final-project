from typing import Tuple, List

import os, sys
sys.path.append(os.path.join(os.getcwd(), "scripts", "analysis"))

import argparse
import csv
import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from pathlib import Path
from PIL import Image

from autoencoders import autoencoder1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size-max", type=int, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--epochs", help="number of epochs to train", type=int, default=100)
    parser.add_argument("-o", "--outfp", type=Path, required=True)
    args = parser.parse_args().__dict__
    return args


class HistologyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir,
                       max_image_size=5000,
                       bg_color=(244, 244, 244),
                       resize_to=(1024, 1024),
                       tsfms=None):

        self.img_fps = list(Path(img_dir).glob("*.png"))
        self.max_image_size = max_image_size
        self.bg_color = bg_color
        self.resize_to = resize_to
        if not tsfms:
            self.tsfms = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor(),
            ])
        else:
            self.tsfms = tsfms

    def __len__(self):
        return len(self.img_fps)

    def __getitem__(self, idx):
        img_fp = self.img_fps[idx]

        img = Image.open(img_fp)
        height, width = img.size
        img = img.rotate(random.randint(0,180), fillcolor=self.bg_color)
        img_padded = Image.new(img.mode, (self.max_image_size, self.max_image_size), self.bg_color)
        img_padded.paste(img, (1500-height//2, 1500-width//2))
        img_padded.thumbnail(self.resize_to, Image.ANTIALIAS)

        return self.tsfms(img_padded), str(img_fp)


def train_model(model, train_iterator, epochs):
    """train an arbitrary autoencoder model"""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2,
        weight_decay=1e-5,
    )
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        losses = []
        for i, (x, _) in enumerate(train_iterator):
            x = x.to(device)
            optimizer.zero_grad()
            fx, _ = model(x)
            loss = criterion(fx, x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = sum(losses) / len(train_iterator)
        print(f"| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} |")

    return model


def main(train_dir: Path, test_dir: Path, outfp: Path, epochs, img_size_max: int):
    train_dataset = DataLoader(HistologyDataset(train_dir))
    test_dataset = DataLoader(HistologyDataset(test_dir))

    model_outdir = Path("models")
    model_outdir.mkdir(exist_ok=True)

    model = autoencoder1.Autoencoder().to(device)

    print("Training...")
    model = train_model(model, train_dataset, epochs)
    torch.save(model.state_dict(), model_outdir/"autoencoder1.pth")
    print("...done")

    print("Running inference")
    model.eval()
    latent_df = []
    for img, img_fp in test_dataset:
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