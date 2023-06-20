from typing import Any, Callable

import torch
import wandb
from tqdm import tqdm

from src.datasets import Dataset
from src.ff.models import FFFNN
from src.metrics import accuracy


class GreedyTrainer:
    def __init__(
        self, epochs: int = 10, seed: Any | int = None, metrics: list[Callable] = []
    ):
        self.epochs = epochs
        self.metrics = metrics

    def train(self, model: FFFNN, dataset: Dataset, device: Any | str = "cpu"):
        # Set model in training mode
        model.to(device)
        model.train()

        # Main training loop
        for train_layer, _ in enumerate(model.layers):
            for epoch in range(self.epochs):
                for x, y in (pbar := tqdm(dataset.train)):
                    x, y = x.to(device), y.to(device)

                    # Generate positive and negative samples
                    x_pos = model.embed_label(x, y)
                    rnd = torch.randperm(y.size(0))
                    x_neg = model.embed_label(x, y[rnd])

                    h_pos = x_pos
                    h_neg = x_neg

                    for forward_layer, layer in enumerate(model.layers):
                        if forward_layer < train_layer:
                            with torch.no_grad():
                                h_pos = layer.forward(h_pos)
                                h_neg = layer.forward(h_neg)
                        elif forward_layer == train_layer:
                            loss = layer.train(h_pos, h_neg)
                            pbar.set_description(
                                f"layer: {train_layer+1}/{len(model.layers)}, epoch: {epoch+1}/{self.epochs}, loss: {loss:.5f}"
                            )
                            wandb.log({f"layer {train_layer+1} loss": loss})

        # Train Evaluation
        train_accuracies = []
        for x, y in dataset.train:
            x, y = x.to(device), y.to(device)
            pred_y = model.predict(x)
            acc = accuracy(pred_y, y)
            train_accuracies.append(acc)
        mean_train_accuracy = torch.mean(torch.tensor(train_accuracies))
        mean_train_err = 1.0 - mean_train_accuracy
        print(
            f"train accuracy: {mean_train_accuracy*100:.2f}%, train error: {(1.0 - mean_train_accuracy)*100:.2f}%"
        )
        wandb.log({"mean train error": mean_train_err})

        # Test Evaluation
        test_accuracies = []
        for x, y in dataset.test:
            x, y = x.to(device), y.to(device)
            pred_y = model.predict(x)
            acc = accuracy(pred_y, y)
            test_accuracies.append(acc)
        mean_test_accuracy = torch.mean(torch.tensor(test_accuracies))
        mean_test_err = 1.0 - mean_test_accuracy
        print(
            f"test accuracy: {mean_test_accuracy*100:.2f}%, test error: {(1.0 - mean_test_accuracy)*100:.2f}%"
        )
        wandb.log({"mean test error": mean_test_err})
