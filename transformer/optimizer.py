import torch
from torch.optim.optimizer import Optimizer


def get_optimizer(name: str) -> Optimizer:
    """
    Get PyTorch optimizer class from optimizer name

    Parameters
    ----------
    name: str
        Name of the optimizer (e.g., 'adam', 'sgd', 'adamw')

    Returns
    -------
    OptimizerA
        PyTorch optimizer class
    """
    if not isinstance(name, str):
        raise ValueError("optimizer must be a string")

    optimizers = {
        "adadelta": torch.optim.Adadelta,
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sparse_adam": torch.optim.SparseAdam,
        "adamax": torch.optim.Adamax,
        "asgd": torch.optim.ASGD,
        "lbfgs": torch.optim.LBFGS,
        "nadam": torch.optim.NAdam,
        "radam": torch.optim.RAdam,
        "rmsprop": torch.optim.RMSprop,
        "rprop": torch.optim.Rprop,
        "sgd": torch.optim.SGD,
    }

    if name.lower() not in optimizers:
        raise ValueError(f"Optimizer '{name.lower()}' is not supported")

    return optimizers[name.lower()]
