import argparse

import torch

import wandb
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


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    # Seed
    seed = config.seed
    if seed is not None:
        torch.random.manual_seed(seed)

    wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config)

    dataset = instantiate_class_from_config(config.dataset)

    model = instantiate_class_from_config(config.model)
    params = torch.load(config.model.location)
    model.load_state_dict(params)

    tester = instantiate_class_from_config(config.tester)
    tester.test(model=model, dataset=dataset, device=device)
