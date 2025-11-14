import torch
from torch import nn
from torch.nn import ELU, Identity, Linear, ReLU, Sigmoid, SiLU, Softmax

from src import data_utils as du


class FFNNEncoder(torch.nn.Module):
    def __init__(self, n_bins=64, n_latent=3):
        """
        Pytorch model for the feed-forward neural network encoder part of an
        autoencoder. Used in multiple other models.

        :param n_bins: Number of bins for the droplet size distributions
        :param n_latent: Number of latent variables
        """
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
        """
        Pytorch model for the feed-forward neural network decoder part of an
        autoencoder. Used in multiple other models.

        :param n_bins: Number of bins for the droplet size distributions
        :param n_latent: Number of latent variables
        :param distribution: Flag to indicate whether output is a true
                             distribution (area under curve is 1) or
                             not normalized.
        """
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
        """
        Combines FFNNEncoder and FFNNDecoder into a single autoencoder model

        :param n_bins: Number of bins for the droplet size distributions
        :param n_latent: Number of latent variables
        """
        super(FFNNAutoEncoder, self).__init__()

        self.encoder = FFNNEncoder(n_bins=n_bins, n_latent=n_latent)
        self.decoder = FFNNDecoder(n_bins=n_bins, n_latent=n_latent)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)

        return reconstruction


class SINDyDeriv(torch.nn.Module):
    def __init__(self, n_latent=10, poly_order=2, use_thresholds=False):
        """
        Pytorch SINDy model that is to be paired with autoencoder. Works directly
        from latent variables.

        :param n_latent: Number of latent variables
        :param poly_order: SINDy polynomial order, should typically be 2 or 3
        :param use_thresholds: Flag to indicate whether coefficients are thresholded
                               during training.
        """
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


class NNDerivatives(torch.nn.Module):
    def __init__(self, n_latent=3, layer_size=None):
        """
        Pytorch black box model to predict time derivatives directly  of droplet
        size distributions directly (while SINDy predicts a simplified equation form
        of time derivatives). Paired with autoencoder.

        :param n_latent: Number of latent variables
        :param layer_size: Number of layers and sizes used in network, e.g. [40, 45, 35]
                           is a three layer network with 40, 45, and 35 nodes for each
                           hidden layer.
        """
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


class Autoregressive(torch.nn.Module):
    def __init__(self, n_bins=3, n_bins_in=None, layer_size=None):
        """
        Pytorch model to predict the next droplet size distribution time step directly.
        SINDy and dzdt predict time derivatives of droplet size dsitributions, whereas
        this model is an autoregressive model that only works with the DSDs. Paired
        with autoencoder.

        :param n_bins:
        :param n_bins_in:
        :param layer_size:
        """
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


def count_parameters(model):
    """
    Count the number of parameters in a model

    :param model: Pytorch model
    :return: Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
