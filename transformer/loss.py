import torch.nn as nn
from typing import Any, Dict, Optional


def get_loss_function(
    name: str, loss_kwargs: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Get PyTorch loss function by name

    Parameters
    ----------
    name: str
        Name of the loss function (e.g., 'mse', 'cross_entropy', 'bce')
    loss_kwargs: Optional[Dict[str, Any]]
        Keyword arguments to pass to the loss function constructor. Default is
        None

    Returns
    -------
    nn.Module
        Instantiated PyTorch loss function
    """
    if not isinstance(name, str):
        raise ValueError("Loss function name must be a string")

    if loss_kwargs is None:
        loss_kwargs = {}
    elif not isinstance(loss_kwargs, dict):
        raise TypeError("loss_kwargs must be a dictionary or None")

    loss_functions = {
        "l1": nn.L1Loss,
        "mse": nn.MSELoss,
        "cross_entropy": nn.CrossEntropyLoss,
        "ctc": nn.CTCLoss,
        "nll": nn.NLLLoss,
        "poisson_nll": nn.PoissonNLLLoss,
        "gaussian_nll": nn.GaussianNLLLoss,
        "kl_div": nn.KLDivLoss,
        "bce": nn.BCELoss,
        "bce_with_logits": nn.BCEWithLogitsLoss,
        "margin_ranking": nn.MarginRankingLoss,
        "hinge_embedding": nn.HingeEmbeddingLoss,
        "multi_label_margin": nn.MultiLabelMarginLoss,
        "huber": nn.HuberLoss,
        "smooth_l1": nn.SmoothL1Loss,
        "soft_margin": nn.SoftMarginLoss,
        "multi_label_soft_margin": nn.MultiLabelSoftMarginLoss,
        "cosine_embedding": nn.CosineEmbeddingLoss,
        "multi_margin": nn.MultiMarginLoss,
        "triplet_margin": nn.TripletMarginLoss,
        "triplet_margin_with_distance": nn.TripletMarginWithDistanceLoss,
    }

    if name not in loss_functions:
        raise ValueError(f"Loss function '{name}' is not supported.")

    try:
        return loss_functions[name](**loss_kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid arguments for {name}: {e}")
