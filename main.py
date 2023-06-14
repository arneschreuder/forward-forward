# Loosely based on:
# - https://github.com/mohammadpz/pytorch_forward_forward
# - https://github.com/pytorch/examples/blob/main/mnist_forward_forward/main.py

import argparse
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import wandb


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Invalid activation function.")


def parse_args():
    global args

    parser = argparse.ArgumentParser(description="Forward-Forward parameters")

    # Params
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--shuffle", type=bool, default=True, help="Shuffle the dataset"
    )
    parser.add_argument(
        "--hidden_layers", type=int, default=4, help="Number of hidden layers"
    )
    parser.add_argument(
        "--hidden_units", type=int, default=256, help="Number of units in hidden layer"
    )
    parser.add_argument(
        "--activation", type=str, default="relu", help="Activation function"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--threshold", type=float, default=2.0, help="Threshold value")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--p_dropout", type=float, default=None, help="Dropout probability"
    )

    args = parser.parse_args()

    # Convert string to actual activation function
    args.activation = get_activation(args.activation)
    return args


# CONSTS
CLASSES = 10
WIDTH, HEIGHT, CHANNELS = 28, 28, 1
FEATURES_IN = WIDTH * HEIGHT * CHANNELS


def embed_label(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]"""
    x_ = x.clone()
    x_[:, :10] *= x.min()
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
        lr: float = 0.01,
        threshold: float = 2.0,
        p_dropout=0.2,
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
        m_pos = g_pos - self.threshold
        m_neg = g_neg - self.threshold
        # Maximise positive and minimise negative vectors
        m_sum = -m_pos + m_neg
        # Ensure that the loss value is positive
        cost = self.softplus(m_sum)
        # Get the batch mean
        cost = cost.mean()
        return cost

    def train_layer(self, x_pos: torch.Tensor, x_neg: torch.Tensor):
        self.training = True

        # Calculate positive and negative goodness
        h_pos, g_pos = self.forward(x_pos)
        h_neg, g_neg = self.forward(x_neg)

        # Calculate loss
        loss = self.loss(g_pos, g_neg)

        # Optimise
        self.optimiser.zero_grad()
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
        p_dropout: float = 0.1,
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
            h = embed_label(x, y)
            _, goodness = self.forward(h)
            # From original paper
            # Only use the goodness from the second layer onwards
            # However, commenting out for now,
            # since then you technically can not have a single layer network
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

    def train_model(self, x_pos: torch.Tensor, x_neg: torch.Tensor, layer_idx):
        h_pos, h_neg = x_pos, x_neg
        g_pos, g_neg, loss = None, None, None
        (h_pos, g_pos), (h_neg, g_neg), loss = (x_pos, None), (x_neg, None), None

        for l, layer in enumerate(self.layers):
            if l < layer_idx:
                # Do not compute gradients here, since it is not necessary
                with torch.no_grad():
                    h_pos, g_pos = layer.forward(h_pos)
                    h_neg, g_neg = layer.forward(h_neg)
            if l == layer_idx:
                return layer.train_layer(h_pos, h_neg)

        return (h_pos, g_pos), (h_neg, g_neg), loss


def train(model, dataset):
    # Set model in training mode
    model.train()

    # Main training loop
    for l, _ in enumerate(model.layers):
        for e in range(args.num_epochs):
            mean_train_err = 0.0
            mean_test_err = 0.0

            for x, y in (pbar := tqdm(dataset.train)):
                x, y = x.to(device), y.to(device)
                x_pos = embed_label(x, y)
                rnd = torch.randperm(y.size(0))
                x_neg = embed_label(x, y[rnd])
                _, _, loss = model.train_model(x_pos, x_neg, l)
                pbar.set_description(
                    f"layer: {l+1}/{len(model.layers)}, epoch: {e+1}/{args.num_epochs}, loss: {loss:.5f}"
                )
                wandb.log({"loss": loss})

            for x, y in dataset.test:
                x, y = x.to(device), y.to(device)
                pred_y = model.predict(x)
                train_err = 1.0 - pred_y.eq(y).float().mean().item()
                mean_train_err += train_err
                mean_train_err /= 2

            wandb.log({"mean train error": mean_train_err})
            print(f"mean train error: {mean_train_err:.5f}")

            for x, y in dataset.test:
                x, y = x.to(device), y.to(device)
                pred_y = model.predict(x)
                test_err = 1.0 - pred_y.eq(y).float().mean().item()
                mean_test_err += test_err

            wandb.log({"mean test error": mean_test_err})
            print(f"mean test err: {mean_test_err:.5f}")


def test(model, dataset):
    # Set model in testing mode
    model.eval()

    # Run tests against test set
    test_errs = []
    for x, y in dataset.test:
        x, y = x.to(device), y.to(device)
        pred_y = model.predict(x)
        test_err = 1.0 - pred_y.eq(y).float().mean().item()
        test_errs.append(test_err)
    print(f"{(sum(test_errs)/len(test_errs)):.5f}")


def setup():
    global args
    global device

    if args.seed is not None:
        torch.random.manual_seed(args.seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Log metrics with Weights & Biases
    wandb.init(project="forward-forward", entity="arneschreuder", config=args)


if __name__ == "__main__":
    global args

    parse_args()

    print("Arguments:")
    print(args)

    setup()

    dataset = Dataset(args.batch_size, shuffle=args.shuffle)

    model = FFFNN(
        FEATURES_IN,
        CLASSES,
        args.hidden_layers,
        args.hidden_units,
        args.activation,
        args.lr,
        args.threshold,
    )
    model = model.to(device)

    train(model, dataset)
    test(model, dataset)
