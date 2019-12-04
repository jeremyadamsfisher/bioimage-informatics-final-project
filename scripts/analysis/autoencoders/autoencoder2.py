import torch.nn as nn
from collections import deque

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        layer_spec = [
            [(1,64),    (64, 64)],    # 512 -> 256
            [(64,128),  (128, 128)],  # 256 -> 128
            [(128,256), (256,256)],   # 128 -> 64
            [(256,256), (256,256)],   # 64 -> 32
            [(256,256), (256,256)],   # 32 -> 16
        ]

        self.encoder_layers = list()
        self.decoder_layers = deque()

        for i, block in enumerate(layer_spec):
            for j, (in_features, out_features) in enumerate(block):
                encoder_block = nn.Sequential(
                    nn.Conv2d(in_features, out_features, 5, padding=5//2),
                    nn.ReLU()
                )
                self.encoder_layers.append(encoder_block)
                decoder_block = nn.Sequential(
                    nn.ConvTranspose2d(out_features, in_features, 5, padding=5//2),
                    nn.Tanh() if (i == j == 0) else nn.ReLU()
                )
                self.decoder_layers.appendleft(decoder_block)
            self.encoder_layers.append(None)
            self.decoder_layers.appendleft(None)

        self.pooler   = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpooler = nn.MaxUnpool2d(2, stride=2)

        n_latent_dimensions = 100
        self.final_conv = nn.Conv2d(256, n_latent_dimensions, 16)
        self.first_deconv = nn.ConvTranspose2d(n_latent_dimensions, 256, 16)

        self.nn_layers = nn.ModuleList()
        self.nn_layers.extend(self.encoder_layers)
        self.nn_layers.extend(self.decoder_layers)
        self.nn_layers.extend([self.final_conv,
                               self.first_deconv,
                               self.pooler,
                               self.unpooler])

    def forward(self, x):
        idxs = []
        for encoder_layer in self.encoder_layers:
            if encoder_layer:
                x = encoder_layer(x)
            else:
                x, idx = self.pooler(x)
                idxs.append(idx)
        x_latent = self.final_conv(x)
        x = self.first_deconv(x_latent)
        for decoder_layer in self.decoder_layers:
            if decoder_layer:
                x = decoder_layer(x)
            else:
                x = self.unpooler(x, idxs.pop())
        return x, x_latent.view(-1)