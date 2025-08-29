import numpy as np
import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import (
    Linear,
    ReLU,
    Sigmoid,
    ConstantPad1d,
    Identity,
    ELU,
    Tanh,
    Softmax,
    SiLU,
)
from src import data_utils as du

"""
Convolutional NN Autoencoder; can operate on multiple channels of input (such as number and mass densities)
"""


class CNNEncoder(torch.nn.Module):
    def __init__(self, n_channels=2, n_bins=35, n_latent=10):
        super(CNNEncoder, self).__init__()
        self.n_bins = n_bins
        self.n_channels = n_channels
        self.conv1 = Conv1d(
            in_channels=n_channels,
            out_channels=n_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.activation1 = ReLU()
        self.conv2 = Conv1d(
            in_channels=n_channels * 2,
            out_channels=n_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.activation2 = ReLU()
        self.conv3 = Conv1d(
            in_channels=n_channels * 4,
            out_channels=n_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.activation3 = ReLU()
        self.lin1 = Linear(int(2 * n_channels * np.floor(n_bins / 8)), n_latent)

        self.layers = [self.conv1, self.conv2, self.conv3, self.lin1]

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)

    def forward(self, x):
        n_bins = self.n_bins
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = x.view(-1, 1, int(2 * self.n_channels * np.floor(n_bins / 8)))
        x = self.lin1(x)

        return x

    def get_weights(self):
        weights = []
        biases = []
        for i, layer in enumerate(self.layers):
            weights.append(layer.weight)
            biases.append(layer.bias)

        return (weights, biases)

    def set_weights(self, weights, biases):
        for i, layer in enumerate(self.layers):
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]


class CNNDecoder(torch.nn.Module):
    def __init__(self, n_channels=2, n_bins=100, n_latent=10, distribution=False):
        super(CNNDecoder, self).__init__()

        self.n_latent = n_latent
        self.n_channels = n_channels
        self.n_bins = n_bins

        self.n_bins = n_bins
        self.lin = Linear(n_latent, int(2 * n_channels * np.floor(n_bins / 8)))
        self.conv1 = ConvTranspose1d(
            in_channels=n_channels * 2,
            out_channels=n_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.activation1 = ReLU()
        self.constantpad1d1 = ConstantPad1d((1, 0), 0)
        self.conv2 = ConvTranspose1d(
            in_channels=n_channels * 4,
            out_channels=n_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.activation2 = ReLU()
        self.conv3 = ConvTranspose1d(
            in_channels=n_channels * 2,
            out_channels=n_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.activation3 = ReLU()
        self.lin2 = Linear(n_bins, n_bins)
        if distribution:
            self.activation4 = Softmax(dim=2)
        else:
            self.activation4 = Sigmoid()

        self.layers = [self.lin, self.conv1, self.conv2, self.conv3, self.lin2]

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)

    def forward(self, x):
        inp = x
        x = self.lin(inp)
        x = x.reshape(-1, self.n_channels * 2, int(np.floor(self.n_bins / 8)))
        x = self.conv1(x)
        x = self.activation1(x)
        if self.n_bins % 8 != 0:
            x = self.constantpad1d1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        if self.n_bins % 8 != 0:
            x = self.constantpad1d1(x)
        x = self.conv3(x)
        x = self.activation3(x)
        if self.n_bins % 8 != 0:
            x = self.constantpad1d1(x)
        x = self.lin2(x)
        x = self.activation4(x)

        return x

    def get_weights(self):
        weights = []
        biases = []
        for i, layer in enumerate(self.layers):
            weights.append(layer.weight)
            biases.append(layer.bias)

        return (weights, biases)

    def set_weights(self, weights, biases):
        for i, layer in enumerate(self.layers):
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]


class CNNAutoEncoder(torch.nn.Module):
    def __init__(self, n_channels=2, n_bins=100, n_latent=10):
        super(CNNAutoEncoder, self).__init__()

        self.encoder = CNNEncoder(
            n_channels=n_channels, n_bins=n_bins, n_latent=n_latent
        )
        self.decoder = CNNDecoder(
            n_channels=n_channels, n_bins=n_bins, n_latent=n_latent
        )

    def forward(self, x):
        latent = self.encoder(x)

        reconstruction = self.decoder(latent)

        return reconstruction


"""
Feed-forward neural network autoencoder
"""


class FFNNEncoder(torch.nn.Module):
    def __init__(self, n_bins=64, n_latent=3):
        super(FFNNEncoder, self).__init__()
        self.n_bins = n_bins
        self.layer1 = Linear(n_bins, int(n_bins / 2))
        self.activation1 = ReLU()
        self.layer2 = Linear(int(n_bins / 2), int(n_bins / 4))
        self.activation2 = ReLU()
        self.layer3 = Linear(int(n_bins / 4), int(n_bins / 8))
        self.activation3 = ReLU()
        self.layer4 = Linear(int(n_bins / 8), n_latent)
        self.activation4 = Identity()

        self.apply(self.init_weights)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.act = [
            self.activation1,
            self.activation2,
            self.activation3,
            self.activation4,
        ]

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)
        x = self.activation4(x)

        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def get_weights(self):
        weights = []
        biases = []
        for i, layer in enumerate(self.layers):
            weights.append(layer.weight)
            biases.append(layer.bias)

        return (weights, biases)

    def set_weights(self, weights, biases):
        for i, layer in enumerate(self.layers):
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]


class FFNNDecoder(torch.nn.Module):
    def __init__(self, n_bins=64, n_latent=3, distribution=True):
        super(FFNNDecoder, self).__init__()

        self.n_bins = n_bins
        self.layer1 = Linear(n_latent, int(n_bins / 8))
        self.layer2 = Linear(int(n_bins / 8), int(n_bins / 4))
        self.layer3 = Linear(int(n_bins / 4), int(n_bins / 2))
        self.layer4 = Linear(int(n_bins / 2), n_bins)
        self.activation1 = ReLU()
        self.activation2 = ReLU()
        self.activation3 = ReLU()
        if distribution:
            self.activation4 = Softmax(dim=-1)
        else:
            self.activation4 = Sigmoid()

        self.apply(self.init_weights)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.act = [
            self.activation1,
            self.activation2,
            self.activation3,
            self.activation4,
        ]

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)
        x = self.activation4(x)

        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def get_weights(self):
        weights = []
        biases = []
        for i, layer in enumerate(self.layers):
            weights.append(layer.weight)
            biases.append(layer.bias)

        return (weights, biases)

    def set_weights(self, weights, biases):
        for i, layer in enumerate(self.layers):
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]


class FFNNAutoEncoder(torch.nn.Module):
    def __init__(self, n_bins=100, n_latent=10):
        super(FFNNAutoEncoder, self).__init__()

        self.encoder = FFNNEncoder(n_bins=n_bins, n_latent=n_latent)
        self.decoder = FFNNDecoder(n_bins=n_bins, n_latent=n_latent)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)

        return reconstruction


"""
Pseudo-SINDy network for time derivatives
"""


class SINDyDeriv(torch.nn.Module):
    def __init__(self, n_latent=10, poly_order=2, use_thresholds=False):
        super(SINDyDeriv, self).__init__()
        self.library_size = du.library_size(n_latent, poly_order)
        self.n_latent = n_latent
        self.poly_order = poly_order

        self.sindy_coeffs = torch.nn.Linear(
            self.library_size, self.n_latent, bias=False
        )
        self.use_thresholds = use_thresholds
        if use_thresholds:
            self.mask = torch.ones_like(self.sindy_coeffs.weight.data, dtype=bool)

        self.apply(self.init_weights)

    def forward(self, z, M=None):
        if M is not None:
            latent = torch.cat([z, M], dim=-1)
        else:
            latent = z
        library = du.sindy_library_tensor(latent, self.n_latent, self.poly_order)
        if self.use_thresholds:
            self.sindy_coeffs.weight.data = self.sindy_coeffs.weight.data * self.mask
        dldt = self.sindy_coeffs(library)
        return dldt

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def get_coeffs(self):
        return self.sindy_coeffs.weight.data * self.mask

    def update_mask(self, new_mask):
        self.mask = self.mask * new_mask
        self.sindy_coeffs.weight.data = self.mask * self.sindy_coeffs.weight.data


"""
Black-box network for predicting time derivatives
"""


class NNDerivatives(torch.nn.Module):
    def __init__(self, n_latent=3, layer_size=None):
        super(NNDerivatives, self).__init__()
        self.n_latent = n_latent
        if layer_size is None:
            layer_size = (n_latent, n_latent, n_latent)
        else:
            assert len(layer_size) == 3

        self.layer1 = Linear(n_latent, layer_size[0])
        self.layer2 = Linear(layer_size[0], layer_size[1])
        self.layer3 = Linear(layer_size[1], layer_size[2])
        self.layer4 = Linear(layer_size[2], n_latent)
        self.activation1 = SiLU()
        self.activation2 = SiLU()
        self.activation3 = SiLU()

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.act = [
            self.activation1,
            self.activation2,
            self.activation3,
        ]

        self.initialize_network()

    def forward(self, z, M=None):
        if M is not None:
            x = torch.cat([z, M], dim=-1)
        else:
            x = z
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)

        return x

    def get_weights(self):
        weights = []
        biases = []
        for i, layer in enumerate(self.layers):
            weights.append(layer.weight)
            biases.append(layer.bias)

        return (weights, biases)

    def set_weights(self, weights, biases):
        for i, layer in enumerate(self.layers):
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]

    def initialize_network(self):
        for i, module in enumerate(self.layers):
            if isinstance(module, nn.Linear):
                if i < len(self.layers) - 1:  # Hidden layers with SiLU
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_in", nonlinearity="relu"
                    )
                    nn.init.constant_(module.bias, 0.0)
                else:  # Output layer (no activation)
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    nn.init.constant_(module.bias, 0.0)


"""
Black-box network for predicting states
"""


class Autoregressive(torch.nn.Module):
    def __init__(self, n_bins=3, n_bins_in=None, layer_size=None):
        super(Autoregressive, self).__init__()
        self.n_bins = n_bins
        if layer_size is None:
            layer_size = (10, 20, 10)
        if n_bins_in is None:
            n_bins_in = self.n_bins

        self.layer1 = Linear(n_bins_in, layer_size[0])
        self.layer2 = Linear(layer_size[0], layer_size[1])
        self.layer3 = Linear(layer_size[1], layer_size[2])
        self.layer4 = Linear(layer_size[2], self.n_bins)
        self.activation1 = ELU()
        self.activation2 = ELU()
        self.activation3 = ELU()
        self.activation4 = Sigmoid()

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.act = [
            self.activation1,
            self.activation2,
            self.activation3,
            self.activation4,
        ]

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)
        x = self.activation4(x)

        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


"""
Utility functions
"""


def get_latent_var(model, dataloader, device, n_latent):
    dataset = dataloader.dataset
    latents = np.zeros((len(dataset), n_latent))

    jj = 0
    for data in dataloader:
        bin0 = data
        bin0 = bin0.to(device)
        latent = model.encoder(bin0.float())
        bs = latent.shape[0]

        latents[jj : jj + bs, :] = latent.detach().cpu().numpy().reshape(bs, n_latent)
        jj += bs

    return latents


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
