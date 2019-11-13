import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HistologyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, preprocess_callable):
        self.img_fps = list(Path(img_dir).glob("*.png"))
        self.preproc = preprocess_callable

    def __len__(self):
        return len(self.img_fps)

    def __getitem__(self, idx):
        img_fp = self.img_fps[idx]
        return self.preproc(Image.open(img_fp)), img_fp.name


def load_dataset(train_dir, test_dir, valid_dir, preprocess_callable):
    datasets = {}
    for data_type, data_path in [("train", train_dir),
                                 ("test",  test_dir),
                                 ("valid", valid_dir)]:
        datasets[data_type] = DataLoader(
            HistologyDataset(data_path, preprocess_callable),
            batch_size=1
        )
    return datasets["train"], datasets["test"], datasets["valid"]
    
    
def train(model, device, iterator, optimizer, criterion):
    epoch_loss = 0
    model.train()

    for i, (x, _) in enumerate(iterator):
        if i % 10 == 0:
            print(f"\t{i}/{len(iterator)}...")
        x = x.to(device)
        optimizer.zero_grad()
        fx, _ = model(x)
        loss = criterion(fx, x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
    
    
def evaluate(model, device, iterator, criterion):
    epoch_loss = 0
    model.eval()
    
    with torch.no_grad():
        for (x, _) in iterator:
            x = x.to(device)
            fx, _ = model(x)
            loss = criterion(fx, x)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


class Autoencoder(torch.nn.Module):
    def __init__(self, layer_spec=((3, 8), (8, 3), (3, 1)), kern_size=5):
        super(Autoencoder, self).__init__()

        self.encoder_layers = []
        self.decoder_layers = []
        for in_features, out_features in layer_spec:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_features, out_features, kern_size),
                    nn.ReLU()
                )
            )
            self.decoder_layers.append(
                nn.Sequential(
                    torch.nn.ConvTranspose2d(out_features, in_features, kern_size),
                    torch.nn.Sigmoid()
                )
            )

        self.pooler   = nn.MaxPool2d(4, stride=4, return_indices=True)
        self.unpooler = nn.MaxUnpool2d(4, stride=4)
        
        self.nn_layers = nn.ModuleList()
        self.nn_layers.extend(self.encoder_layers)
        self.nn_layers.extend(self.decoder_layers)

    def forward(self, x):
        pool_results = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            x_size_orig = x.size()
            x, pooling_idxs = self.pooler(x)
            pool_results.append((x_size_orig, pooling_idxs))
        x_latent = x.reshape(-1)
        for decoder_layer, (out_size, unpool_idx) in zip(reversed(self.decoder_layers),
                                                         reversed(pool_results)):
            x = self.unpooler(x, unpool_idx, output_size=out_size)
            x = decoder_layer(x)
        return x, x_latent


def train_model(train_dir, test_dir, valid_dir, epochs, preprocess_callable):
    train_iterator, test_iterator, valid_iterator = load_dataset(train_dir, test_dir, valid_dir, preprocess_callable)        
                        
    print("Initializing model")
    model = Autoencoder().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2,
        weight_decay=1e-5,
    )
    criterion = nn.MSELoss()
    print("...done")

    ### Training ###
    SAVE_DIR = Path("./models")
    SAVE_DIR.mkdir(exist_ok=True)
    MODEL_SAVE_PATH = SAVE_DIR/"autoencoder.pt"

    best_valid_loss = float("inf")

    for epoch in range(epochs):
        print(f"starting epoch {epoch+1}...")
        train_loss = train(model, device, train_iterator, optimizer, criterion)
        valid_loss = evaluate(model, device, valid_iterator, criterion)
        print(f"| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} |")
        
    # save model for later if desired
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    ### Testing ###
    test_loss = evaluate(model, device, test_iterator, criterion)
    print(f"| Test Loss: {test_loss:.3f} |")

    return model, MODEL_SAVE_PATH, test_iterator
