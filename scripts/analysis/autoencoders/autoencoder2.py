"""adapted from FCN8s layers: https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn8s.py"""

import torch.nn as nn
from collections import deque

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        layer_spec = [
            [(1,64), (64, 64)],  # 1024 -> 512
            [(64,128), (128, 128)],  # 512 -> 256
            [(128, 256), (256,256), (256,256)],  # 256 -> 128
            [(256, 512), (512,512), (512,512)],  # 128 -> 64
            [(512, 512), (512,512), (512,512)],  # 64 -> 32
        ]

        self.encoder_layers = list()
        self.decoder_layers = deque()

        for block in layer_spec:
            for in_features, out_features in block:
                encoder_block = nn.Sequential(
                    nn.Conv2d(in_features, out_features, 3, padding=1),
                    nn.ReLU()
                )
                self.encoder_layers.append(encoder_block)
                decoder_block = nn.Sequential(
                    nn.ConvTranspose2d(out_features, in_features, 3),
                    nn.ReLU()
                )
                self.decoder_layers.appendleft(decoder_block)
            self.encoder_layers.append(nn.MaxPool2d(2, stride=2))
            self.decoder_layers.appendleft(nn.MaxUnpool2d(2, stride=2))

        n_latent_dimensions = 100
        self.final_conv = nn.ConvTranspose2d(512, n_latent_dimensions, 32)
        self.first_deconv = nn.ConvTranspose2d(n_latent_dimensions, 512, 32)

        self.nn_layers = nn.ModuleList()
        self.nn_layers.extend(self.encoder_layers)
        self.nn_layers.extend(self.decoder_layers)

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            
        x_latent = self.final_conv(x)
        x = self.first_deconv(x_latent)

        for decoder_layer in self.decoder_layers:
            x = self.unpooler(x)
            x = decoder_layer(x)
        return x, x_latent