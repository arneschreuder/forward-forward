from typing import Any, List

import torch
from torch import nn


class FFLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.1,
        threshold: float = 2.0,
        optimiser: torch.optim.Optimizer = torch.optim.Adam,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.dropout = dropout
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.linear_layer = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        self.dropout_layer = (
            nn.Dropout(self.dropout)
            if self.dropout is not None and self.dropout > 0
            else None
        )
        self.optimiser = optimiser(self.parameters(), lr=self.learning_rate)

    def forward(self, x: torch.Tensor):
        x_norm = x.norm(p=2, dim=1, keepdim=True)
        x = x / (x_norm + 1e-8)
        x = self.linear_layer(x)
        x = self.activation(x) if self.activation is not None else x
        x = (
            self.dropout_layer(x)
            if self.dropout_layer is not None and self.training
            else x
        )
        return x


class FFHiddenLayer(FFLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.1,
        threshold: float = 2.0,
        optimiser: torch.optim.Optimizer = torch.optim.Adam,
        learning_rate: float = 0.001,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            activation=activation,
            dropout=dropout,
            threshold=threshold,
            optimiser=optimiser,
            learning_rate=learning_rate,
        )
        self.softplus = nn.Softplus()

    def goodness(self, x: torch.Tensor):
        return x.pow(2).sum(dim=1)

    def loss(self, g_pos, g_neg):
        m_pos = g_pos - self.threshold
        m_neg = g_neg - self.threshold
        m_sum = -m_pos + m_neg
        loss = self.softplus(m_sum)
        loss = loss.mean()
        return loss

    def train(self, x_pos, x_neg):
        super().train()

        # Calculate positive and negative goodness
        h_pos = self.forward(x_pos)
        h_neg = self.forward(x_neg)

        # Goodness
        g_pos = self.goodness(h_pos)
        g_neg = self.goodness(h_neg)

        # Calculate loss
        loss = self.loss(g_pos, g_neg)

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss


class FFSoftmaxLayer(FFLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        optimiser: torch.optim.Optimizer = torch.optim.Adam,
        learning_rate: float = 0.001,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            optimiser=optimiser,
            learning_rate=learning_rate,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = super().forward(x)
        x = self.softmax(x)
        return x


class FFFNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_layers: int = 4,
        hidden_units: int = 2000,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.1,
        threshold: float = 2.0,
        optimiser: torch.optim.Optimizer = torch.optim.Adam,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.threshold = threshold
        self.optimiser = optimiser
        self.learning_rate = learning_rate

        self.layers: List[FFHiddenLayer] = []
        for i in range(hidden_layers):
            in_features = in_features if i == 0 else hidden_units
            out_features = hidden_units
            layer = FFHiddenLayer(
                in_features=in_features,
                out_features=out_features,
                activation=self.activation,
                dropout=self.dropout,
                threshold=self.threshold,
                optimiser=self.optimiser,
                learning_rate=self.learning_rate,
            )
            self.layers.append(layer)

    def embed_label(self, x: torch.Tensor, y: torch.Tensor):
        x_ = x.clone()
        x_[:, :10] *= x.min()
        x_[range(x.shape[0]), y] = x.max()
        return x_

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x: torch.Tensor):
        return self.forward(x)


class FFClassifier(FFFNN):
    def __init__(
        self,
        in_features: int,
        classes: int = 10,
        hidden_layers: int = 4,
        hidden_units: int = 2000,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.1,
        threshold: float = 2.0,
        optimiser: torch.optim.Optimizer = torch.optim.Adam,
        learning_rate: float = 0.001,
    ):
        super().__init__(
            in_features=in_features,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            activation=activation,
            dropout=dropout,
            threshold=threshold,
            optimiser=optimiser,
            learning_rate=learning_rate,
        )
        self.classes = classes
        self.out_layer = FFSoftmaxLayer(
            in_features=self.hidden_units,
            out_features=self.classes,
            optimiser=optimiser,
            learning_rate=learning_rate,
        )

    def predict(self, x: torch.Tensor):
        goodness_per_label = []
        for y in range(self.classes):
            h = self.embed_label(x, y)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                layer_goodness = layer.goodness(h)
                goodness.append(layer_goodness)
            goodness = sum(goodness)
            # Add dimension and restructure
            goodness = goodness.unsqueeze(dim=1)
            goodness_per_label += [goodness]
        # Concatenate the goodnesses per label forming a new tensor
        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        # Get the label with the highest goodness
        label = goodness_per_label.argmax(dim=1)
        return label
