import torch


def accuracy(pred_y: torch.Tensor, true_y: torch.Tensor):
    return pred_y.eq(true_y).float().mean().item()


def accuracy_error(pred_y: torch.Tensor, true_y: torch.Tensor):
    return 1.0 - accuracy(pred_y, true_y)


def mse(pred_y: torch.Tensor, true_y: torch.Tensor):
    return torch.nn.functional.mse_loss(pred_y, true_y)


def rmse(pred_y: torch.Tensor, true_y: torch.Tensor):
    return torch.sqrt(torch.nn.functional.mse_loss(pred_y, true_y))
