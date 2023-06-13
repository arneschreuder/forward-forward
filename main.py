from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Params
seed = 1
batch_size = 1024
shuffle = True
hidden_layers = 2
hidden_units = 128
activation = nn.ReLU()
lr = 0.001
threshold = 2.0
num_epochs = 10
p_dropout = None

# Variables
classes = 10
width, height, channels = 28, 28, 1
features_in = width * height * channels


if seed is not None:
    torch.random.manual_seed(seed)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]"""
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Dataset:
    def __init__(self, batch_size: int = 32, shuffle: bool = True):
        # Define the transformations: convert to tensor and normalize
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: torch.flatten(x)),
            ]
        )

        # Data
        self.train = DataLoader(
            datasets.MNIST(
                root="data",
                train=True,
                download=True,
                transform=self.transform,  # Use the defined transformations
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.test = DataLoader(
            datasets.MNIST(
                root="data",
                train=False,
                download=True,
                transform=self.transform,  # Use the defined transformations
            ),
            batch_size=batch_size,
            shuffle=False,
        )


class DenseLayer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: torch.optim.Optimizer = None,
        lr: float = 0.001,
        threshold: float = 2.0,
        p_dropout=0.2,
        num_epochs: int = 1000,
    ):
        super().__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.lr = lr
        self.threshold = threshold
        self.optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.p_dropout = p_dropout
        self.dropout = nn.Dropout(p_dropout) if p_dropout is not None else None

    def _layer_goodness(self, x: torch.Tensor):
        # Goodness is calculated as the Sum of Squared Activations (SSA)
        return x.pow(2).sum(dim=1)

    def forward(self, x: torch.Tensor):
        # Calculate norm
        x_norm = x.norm(p=2, dim=1, keepdim=True)
        # Apply L2 layer normalisation, add small non-zero number to avoid div by 0 error
        x = x / (x_norm + 1e-8)
        # Apply linear transformation
        x = self.linear(x)
        # x = torch.mm(x, self.weight.T) + self.bias.unsqueeze(0)
        # Apply non-linearity
        x = self.activation(x) if self.activation else x
        # Apply dropout
        x = self.dropout(x) if self.dropout and self.training else x
        # Compute layer goodness
        layer_goodness = self._layer_goodness(x)
        return x, layer_goodness

    def loss(self, g_pos, g_neg):
        # We want to have g_pos >> threshold and g_neg << threshold
        m_pos = (
            g_pos - self.threshold
        )  # -(threshold - g_pos) + (g_neg - threshold) # (g_pos - threshold) + (g_neg - threshold)
        m_neg = g_neg - self.threshold
        # TODO: This is unnecessary, but added to illustrate the point of a probability being optimised
        m_pos = self.sigmoid(m_pos)
        m_neg = self.sigmoid(m_neg)
        m_sum = -m_pos + m_neg
        # Ensure that the loss value is positive
        cost = torch.log(1 + torch.exp(m_sum))
        # Get the batch mean
        cost = cost.mean()
        return cost

    def train(self, x_pos: torch.Tensor, x_neg: torch.Tensor):
        # Calculate positive and negative goodness
        # TODO: Ensure this does not get the gradients of any previous layer
        h_pos, g_pos = self.forward(x_pos)
        h_neg, g_neg = self.forward(x_neg)
        loss = self.loss(g_pos, g_neg)
        self.optimiser.zero_grad()
        # Compute loss
        # this backward just compute the derivative and hence
        # is not considered backpropagation.
        loss.backward()
        self.optimiser.step()

        return (h_pos, g_pos), (h_neg, g_neg), loss


# Feed Forward Forward Neural Network (pun intended)
class FFFNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        classes: int,
        hidden_layers: int = 4,
        hidden_units_per_layer: int = 128,
        activation: Callable = nn.ReLU(),
        lr: float = 0.001,
        threshold: float = 2.0,
        p_dropout: float = 0.2,
    ):
        super(FFFNN, self).__init__()
        self.in_features = in_features
        self.classes = classes
        self.hidden_layers = hidden_layers
        self.hidden_units_per_layer = hidden_units_per_layer
        self.activation = activation
        self.layers = []
        for i in range(hidden_layers):
            in_features = self.in_features if i == 0 else self.hidden_units_per_layer
            out_features = self.hidden_units_per_layer
            layer = DenseLayer(
                in_features, out_features, activation, lr, threshold, p_dropout
            )
            layer = layer.to(device)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor):
        goodness = []
        for layer in self.layers:
            x, layer_goodness = layer(x)
            goodness += [layer_goodness]
        return x, goodness

    def predict(self, x: torch.Tensor):
        goodness_per_label = []
        for y in range(self.classes):
            h = overlay_y_on_x(x, y)
            _, goodness = self.forward(h)
            # Only use the goodness from all the layers except the first hidden
            # print(goodness)
            # goodness = goodness[1:]
            # Get the sum of the layer goodnesses
            goodness = sum(goodness)
            # Add dimension and restructure
            goodness = goodness.unsqueeze(dim=1)
            goodness_per_label += [goodness]
        # Concatenate the goodnesses per label forming a new tensor
        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        # Get the label with the highest goodness
        label = goodness_per_label.argmax(dim=1)
        return label

    def train(self, x_pos: torch.Tensor, x_neg: torch.Tensor, layer_idx):
        h_pos, h_neg = x_pos, x_neg
        g_pos, g_neg, loss = None, None, None

        for l, layer in enumerate(self.layers):
            if l < layer_idx:
                # with torch.no_grad():
                h_pos, g_pos = layer.forward(h_pos)
                h_neg, g_neg = layer.forward(h_neg)
            if l == layer_idx:
                return layer.train(h_pos, h_neg)

        return (h_pos, g_pos), (h_neg, g_neg), loss


# def get_derangement(n):
#     permutation = torch.randperm(n)
#     match = torch.arange(n, dtype=torch.long)
#     derangement = (permutation + (permutation == match).long()) % n
#     return derangement


def train(model, dataset):
    for l, layer in enumerate(model.layers):
        for e in (pbar := tqdm(range(num_epochs))):
            mean_train_loss = 0.0
            mean_train_err = 0.0
            mean_test_err = 0.0
            for x, y in dataset.train:
                x, y = x.to(device), y.to(device)
                x_pos = overlay_y_on_x(x, y)
                rnd = torch.randperm(y.size(0))
                x_neg = overlay_y_on_x(x, y[rnd])
                _, _, loss = model.train(x_pos, x_neg, l)
                mean_train_loss += loss
                mean_train_loss /= 2

                pred_y = model.predict(x)
                train_err = 1.0 - pred_y.eq(y).float().mean().item()
                mean_train_err += train_err
                mean_train_err /= 2

                pbar.set_description(
                    f"layer: {l}, epoch: {e}, avg train loss: {mean_train_loss:.5f}, avg train err: {mean_train_err:.5f}"
                )


def test(model, dataset):
    test_errs = []
    for x, y in dataset.test:
        x, y = x.to(device), y.to(device)
        pred_y = model.predict(x)
        test_err = 1.0 - pred_y.eq(y).float().mean().item()
        test_errs.append(test_err)
    print(f"{(sum(test_errs)/len(test_errs)):.5f}")


if __name__ == "__main__":
    dataset = Dataset(batch_size, shuffle=shuffle)
    # dataset = dataset.to(device)

    model = FFFNN(
        features_in,
        classes,
        hidden_layers,
        hidden_units,
        activation,
        lr,
        threshold,
    )
    model = model.to(device)
    train(model, dataset)
    model.training = False
    for layer in model.layers:
        layer.training = False
    test(model, dataset)
