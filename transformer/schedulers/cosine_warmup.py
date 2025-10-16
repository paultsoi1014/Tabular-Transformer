import torch

import numpy as np
from typing import List


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int):
        """
        Cosine annealing learning rate scheduler with linear warmup

        This scheduler combines a linear warmup phase with cosine annealing
        decay. During warmup, the learning rate increases linearly from 0 to the
        base learning rate. After warmup, it follows a cosine annealing schedule,
        decreasing smoothly to near zero by the end of training

        Parameters
        ----------
        optimizer: torch.optim.Optimizer
            Wrapped optimizer
        warmup: int
            Number of warmup iterations/epochs during which the learning rate
            increases linearly from 0 to base_lr
        max_iters: int
            Maximum number of iterations/epochs for the entire training schedule.
            The cosine annealing is computed over this full period

        Attributes
        ----------
        warmup: int
            Number of warmup iterations
        max_iters: int
            Maximum number of iterations for the schedule
        """
        self.warmup = warmup
        self.max_num_iters = max_iters
        super(CosineWarmupScheduler, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler

        Returns
        -------
        List[float]
            List of learning rates for each parameter group
        """
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        """
        Calculate the learning rate scaling factor for a given epoch.

        Parameters
        ----------
        epoch: int
            Current epoch/iteration number

        Returns
        -------
        float
            Learning rate scaling factor to multiply with base learning rate
        """
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup

        return lr_factor
