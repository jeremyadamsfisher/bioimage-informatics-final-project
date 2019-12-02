import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, layer_spec=((1, 32),   # 1024x1024x1 -> 256x256x32
                                   (32, 16),  # -> 64x64
                                   (16, 8),   # -> 16x16x8
                                   ), kern_size=5):
        super(Autoencoder, self).__init__()

        self.encoder_layers = []
        self.decoder_layers = []
        for in_features, out_features in layer_spec:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_features, out_features, kern_size, padding=2),
                    nn.ReLU()
                )
            )
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(out_features, in_features, kern_size, padding=2),
                    nn.ReLU()
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



