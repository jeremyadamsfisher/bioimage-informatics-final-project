import torch.nn as nn
from collections import deque

class Autoencoder(nn.Module):

    def __init__(self, n_latent_dimensions):
        super(Autoencoder, self).__init__()

        self.n_latent_dimensions = n_latent_dimensions

        layer_spec = [
            [(1,64), (64, 64)],  # 1024 -> 256
            [(64,128), (128, 128)],  # 256 -> 64
            [(128, 256), (256,256)],  # 64 -> 16
            [(256,256), (256,256)],  # 4 -> 1
            [(256,512), (512,512)],  # ???
        ]

        self.encoder_layers = deque()
        self.decoder_layers = deque()

        for block in layer_spec:
            for in_features, out_features in block:
                encoder_block = nn.Sequential(
                    nn.Conv2d(in_features, out_features, 3, padding=1),
                    nn.ReLU()
                )
                self.encoder_layers.append(encoder_block)
                decoder_block = nn.Sequential(
                    nn.ConvTranspose2d(out_features, in_features, 3, padding=1),
                    nn.ReLU()
                )
                self.decoder_layers.appendleft(decoder_block)
            self.encoder_layers.append(None)
            self.decoder_layers.appendleft(None)

        self.pooler   = nn.MaxPool2d(4, stride=4, return_indices=True)
        self.unpooler = nn.MaxUnpool2d(4, stride=4)

        self.to_bottleneck   = nn.ConvTranspose2d(512, self.n_latent_dimensions, 5, padding=2)
        self.from_bottleneck = nn.ConvTranspose2d(self.n_latent_dimensions, 512, 5, padding=2)

        self.nn_layers = nn.ModuleList()
        self.nn_layers.extend(self.encoder_layers)
        self.nn_layers.extend(self.decoder_layers)

    def forward(self, x):
        idxs = []
        for encoder_layer in self.encoder_layers:
            if encoder_layer:
                x = encoder_layer(x)
            else:
                x, idx = self.pooler(x)
                idxs.append(idx)

        x_latent = self.to_bottleneck(x)
        x = self.from_bottleneck(x_latent)
        
        for decoder_layer in self.decoder_layers:
            if decoder_layer:
                x = decoder_layer(x)
            else:
                x = self.unpooler(x, idxs.pop())

        return x, x_latent.view(-1)