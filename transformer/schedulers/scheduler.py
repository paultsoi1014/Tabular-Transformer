import torch.optim as optim
from typing import Optional, Type

from .cosine_warmup import CosineWarmupScheduler


def get_scheduler(
    scheduler_name: Optional[str] = None,
) -> Optional[Type[optim.lr_scheduler.LRScheduler]]:
    """
    Get a PyTorch learning rate scheduler class based on the scheduler name

    Parameters
    ----------
    scheduler_name: Optional[str]
        Name of the scheduler to retrieve. If None, returns None. Supported
        values: "step", "multi_step", "constant", "linear", "exponential",
        "polynomial", "cosine_annealing", "reduce_on_plateau", "cyclic",
        "one_cycle", "cosine_warm_restarts"

    Returns
    -------
    Optional[Type[optim.lr_scheduler.LRScheduler]]
        The corresponding PyTorch lr_scheduler class, or None if scheduler_name
        is None
    """
    if scheduler_name is None:
        return None

    schedulers = {
        "constant": optim.lr_scheduler.ConstantLR,
        "cosine_annealing": optim.lr_scheduler.CosineAnnealingLR,
        "cosine_warmup": CosineWarmupScheduler,
        "cosine_warm_restarts": optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "cyclic": optim.lr_scheduler.CyclicLR,
        "exponential": optim.lr_scheduler.ExponentialLR,
        "linear": optim.lr_scheduler.LinearLR,
        "multi_step": optim.lr_scheduler.MultiStepLR,
        "one_cycle": optim.lr_scheduler.OneCycleLR,
        "polynomial": optim.lr_scheduler.PolynomialLR,
        "reduce_on_plateau": optim.lr_scheduler.ReduceLROnPlateau,
        "step": optim.lr_scheduler.StepLR,
    }

    if scheduler_name not in schedulers:
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}. Must be one of {list(schedulers.keys())}"
        )

    return schedulers[scheduler_name]
