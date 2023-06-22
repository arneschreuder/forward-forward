import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.ff.models import FFFNN
from src.util.config import instantiate_class_from_config, load_config
from src.util.util import get_device


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Load config file")

    # Add the arguments
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="The path to the config file",
    )

    # Parse the arguments
    args = parser.parse_args()
    return args


def visualize_weights(model: FFFNN):
    num_layers = len(model.layers)
    plt.figure(
        figsize=(10 * num_layers, 10)
    )  # Adjust figure size to accomodate all subplots
    for r in range(4):
        for c, layer in enumerate(model.layers):
            params = [x for x in layer.parameters()]
            weights = params[0].data[:4, :]

            # Iterate over the first 4 weight vectors
            weight_vector = weights[r, :]

            # Reshape the weight vector into a 28x28 array
            size = int(np.sqrt(weight_vector.size(0)))
            weight_image = np.reshape(weight_vector, (size, size))

            # Create a subplot for each weight vector
            plt.subplot(4, num_layers, r * num_layers + c + 1)
            plt.title(f"l{c+1}i{r+1}")

            # Use imshow to show the weights as an image.
            plt.imshow(weight_image, cmap="gray")

    # Show the figure
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    # Seed
    seed = config.seed
    if seed is not None:
        torch.random.manual_seed(seed)

    dataset = instantiate_class_from_config(config.dataset)

    model = instantiate_class_from_config(config.model)
    params = torch.load(config.model.location)
    model.load_state_dict(params)

    # Call the function at the end of training
    visualize_weights(model)
