import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .base import Activation, ColumnEmbedding


class CategoricalFeaturesEncoder(nn.Module):
    def __init__(
        self,
        vocabulary: Dict[str, Dict[str, int]],
        embedding_dim: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        """
        Transformer-based encoder for categorical features

        Encodes categorical features using learned embeddings followed by a
        Transformer encoder. Each categorical feature is embedded and processed
        through multi-head self-attention layers

        Attributes
        ----------
        vocabulary: Dict[str, Dict[str, int]]
            Vocabulary mappings for categorical features. Format:
            {feature_name: {feature_value: index}}
        model: nn.ModuleDict
            Dictionary containing column_embedding and transformer_encoder
            modules

        Parameters
        ----------
        vocabulary: Dict[str, Dict[str, int]]
            Vocabulary mappings for categorical features. Format:
            {feature_name: {feature_value: index}}
        embedding_dim: int
            Dimension of feature embeddings and transformer hidden states
        n_heads: int
            Number of attention heads in transformer layers
        n_layers: int
            Number of transformer encoder layers
        dim_feedforward: int
            Dimension of feedforward network in transformer layers
        dropout: float
            Dropout probability for transformer layers
        """
        super(CategoricalFeaturesEncoder, self).__init__()

        self.vocabulary = vocabulary

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        layer_norm = nn.LayerNorm([embedding_dim])
        self.model = nn.ModuleDict(
            {
                "column_embedding": ColumnEmbedding(vocabulary, embedding_dim),
                "transformer_encoder": nn.TransformerEncoder(
                    encoder_layer=encoder_layer, num_layers=n_layers, norm=layer_norm
                ),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through categorical encoder

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, n_categorical_features)
            containing encoded categorical values

        Returns
        -------
        torch.Tensor
            Encoded features of shape
            (batch_size, embedding_dim * n_categorical_features)
        """
        batch_size = x.size(0)
        x = [
            self.model["column_embedding"](x[:, i], col)
            for i, col in enumerate(self.vocabulary)
        ]
        x = torch.stack(x, dim=1)
        x = self.model["transformer_encoder"](x).view(batch_size, -1)

        return x


class NumericalFeaturesEncoder(nn.Module):
    def __init__(self, n_features: int):
        """
        Layer normalization encoder for numerical/continuous features

        Attributes
        ----------
        layer_norm: nn.LayerNorm
            Layer normalization module

        Parameters
        ----------
        n_features: int
            Number of continuous features
        """
        super(NumericalFeaturesEncoder, self).__init__()
        self.layer_norm = nn.LayerNorm([n_features])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through numerical encoder

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, n_features)

        Returns
        -------
        torch.Tensor
            Normalized features of shape (batch_size, n_features)
        """
        return self.layer_norm(x)


class MLPBlock(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, activation: str, dropout: float
    ):
        """
        Single MLP block with linear layer, normalization, activation, and
        dropout

        Attributes
        ----------
        model: nn.Sequential
            Sequential container of linear, normalization, activation, and
            dropout layers

        Parameters
        ----------
        input_dim: int
            The input dimension
        output_dim: int
            The output dimension
        activation: str
            The activation function
        dropout: float
            The dropout rate
        """
        super(MLPBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm([output_dim]),
            Activation(activation),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP block

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        return self.model(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str,
        dropout: float,
    ):
        """
        Multi-layer perceptron with configurable hidden layers

        Attributes
        ----------
        model: nn.Sequential
            Sequential container of MLP blocks and final linear layer

        Parameters
        ----------
        input_dim: int
            The input dimension
        output_dim: int
            The output dimension
        hidden_dims: List[int]
            List of hidden layer dimensions
        activation: str
            Activation function
        dropout: float
            The dropout rate
        """
        super(MLP, self).__init__()

        dims = [input_dim] + hidden_dims
        self.model = nn.Sequential(
            *(
                [
                    MLPBlock(
                        input_dim=dims[i],
                        output_dim=dims[i + 1],
                        activation=activation,
                        dropout=dropout,
                    )
                    for i in range(len(dims) - 1)
                ]
                + [nn.Linear(dims[-1], output_dim)]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        return self.model(x)


class TabTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        vocabulary: Dict[str, Dict[str, int]],
        n_continuous_features: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        attn_dropout: float,
        mlp_hidden_dims: List[int],
        activation: str,
        ffn_dropout: float,
    ):
        """
        TabTransformer model for tabular data with mixed feature types

        Combines transformer-based encoding for categorical features with simple
        normalization for continuous features, followed by an MLP classifier

        Attributes
        ----------
        encoders: nn.ModuleDict
            Dictionary containing categorical_encoder and continuous_encoder
        classifier: MLP
            MLP classifier head

        Parameters
        ----------
        embedding_dim: int
            Dimension of categorical feature embeddings
        output_dim: int
            Number of output classes/dimensions
        vocabulary: Dict[str, Dict[str, int]]
            Vocabulary mappings for categorical features
        n_continuous_features: int
            Number of continuous/numerical features
        n_heads: int
            Number of attention heads in transformer
        n_layers: int
            Number of transformer encoder layers
        dim_feedforward: int
            Dimension of feedforward network in transformer
        attn_dropout: float
            Dropout probability for attention layers
        mlp_hidden_dims: List[int]
            Hidden layer dimensions for MLP classifier
        activation: str
            Activation function for MLP
        ffn_dropout: float
            Dropout probability for feedforward layers
        """
        super(TabTransformer, self).__init__()

        self.encoders = nn.ModuleDict(
            {
                "categorical_encoder": CategoricalFeaturesEncoder(
                    vocabulary=vocabulary,
                    embedding_dim=embedding_dim,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=attn_dropout,
                ),
                "continuous_encoder": NumericalFeaturesEncoder(n_continuous_features),
            }
        )
        self.classifier = MLP(
            input_dim=embedding_dim * len(vocabulary) + n_continuous_features,
            output_dim=output_dim,
            hidden_dims=mlp_hidden_dims,
            activation=activation,
            dropout=ffn_dropout,
        )

    def forward(
        self, x_categorical: torch.Tensor, x_continuous: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through TabTransformer

        Parameters
        ----------
        x_categorical: torch.Tensor
            Categorical features of shape (batch_size, n_categorical_features)
        x_continuous: torch.Tensor
            Continuous features of shape (batch_size, n_continuous_features)

        Returns
        -------
        torch.Tensor
            Model predictions of shape (batch_size, output_dim)
        """
        x_categorical = self.encoders["categorical_encoder"](x_categorical)
        x_continuous = self.encoders["continuous_encoder"](x_continuous)

        x = torch.cat([x_categorical, x_continuous], dim=-1)
        x = self.classifier(x)
        return x


class LightningTabTransformer(pl.LightningModule):
    def __init__(
        self,
        model: TabTransformer,
        output_dim: int,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        custom_metric: Optional[Callable] = None,
        scheduler_custom_metric: Optional[bool] = False,
        use_wandb: Optional[bool] = False,
    ):
        """
        PyTorch Lightning wrapper for TabTransformer model

        Handles training, validation, and optimizer configuration with optional
        learning rate scheduling and custom metrics

        Attributes
        ----------
        model: TabTransformer
            The TabTransformer model
        output_dim: int
            Number of output dimensions (1 for regression, >1 for classification)
        criterion: nn.Module
            Loss function
        scheduler: torch.optim.lr_scheduler._LRScheduler
            Learning rate scheduler to use
        custom_metric: Callable
            Custom evaluation metric function that takes (y_true, y_pred) and
            returns a scalar
        scheduler_custom_metric: bool
            If True, use custom_metric for scheduler monitoring instead of
            val_loss
        val_y_true: List
            Accumulated validation ground truth labels
        val_y_pred: List
            Accumulated validation predictions
        use_wandb: bool
            Whether to use Weights & Biases logging

        Parameters
        ----------
        model: TabTransformer
            TabTransformer model instance
        output_dim: int
            Number of output dimensions (1 for regression, >1 for classification)
        criterion: nn.Module
            Loss function
        optimizer: torch.optim.Optimizer
            Optimizer instance (not class)
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
            Learning rate scheduler. Default is None
        custom_metric: Optional[Callable]
            Custom evaluation metric function that takes (y_true, y_pred) and
            returns a scalar. Default is None
        scheduler_custom_metric: Optional[bool]
            If True, use custom_metric for scheduler monitoring instead of
            val_loss. Default is False
        use_wandb: Optional[bool]
            If True, enable Weights & Biases logging. Default is False
        """
        super(LightningTabTransformer, self).__init__()

        self.model = model
        self.output_dim = output_dim
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.custom_metric = custom_metric
        self.scheduler_custom_metric = scheduler_custom_metric

        self.training_loss = []
        self.validation_loss = []
        self.val_y_true = []
        self.val_y_pred = []
        self.use_wandb = use_wandb

        self.save_hyperparameters(
            ignore=["model", "criterion", "optimizer", "scheduler"]
        )

    def forward(
        self, categorical_data: torch.Tensor, continuous_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model

        Parameters
        ----------
        categorical_data: torch.Tensor
            Categorical features of shape (batch_size, n_categorical_features)
        continuous_data: torch.Tensor
            Continuous features of shape (batch_size, n_continuous_features)

        Returns
        -------
        torch.Tensor
            Model predictions of shape (batch_size, output_dim)
        """
        return self.model(categorical_data, continuous_data)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step for a single batch

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Batch containing (categorical_data, continuous_data, target)
        batch_idx: int
            Index of the current batch

        Returns
        -------
        torch.Tensor
            Training loss for the batch
        """
        categorical_data, continuous_data, target = batch
        if self.output_dim == 1:
            target = target.unsqueeze(1)

        output = self(categorical_data, continuous_data)
        loss = self.criterion(output, target)
        self.training_loss.append(loss)

        self.log("train_loss", loss, prog_bar=True)

        if self.use_wandb:
            self.log("train/loss", loss, on_step=True, on_epoch=True)
            self.log("train/epoch", float(self.current_epoch), on_step=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step for a single batch

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Batch containing (categorical_data, continuous_data, target)
        batch_idx: int
            Index of the current batch

        Returns
        -------
        torch.Tensor
            Validation loss for the batch
        """
        categorical_data, continuous_data, target = batch

        if self.output_dim == 1:
            self.val_y_true.extend(target.cpu().numpy().reshape(-1).tolist())
            target = target.unsqueeze(1)
        else:
            self.val_y_true.append(target.cpu().numpy())

        output = self(categorical_data, continuous_data)

        if self.output_dim == 1:
            self.val_y_pred.extend(output.cpu().numpy().reshape(-1).tolist())
        else:
            self.val_y_pred.append(torch.argmax(output, dim=1).cpu().numpy())

        loss = self.criterion(output, target)
        self.validation_loss.append(loss)

        self.log("val_loss", loss, prog_bar=True)

        if self.use_wandb:
            self.log("val/loss", loss, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        """
        Called at the end of validation epoch

        Computes custom metric if provided and resets accumulated predictions
        and ground truth
        """
        if self.output_dim > 1:
            y_true = np.concatenate(self.val_y_true)
            y_pred = np.concatenate(self.val_y_pred)
        else:
            y_true = self.val_y_true
            y_pred = self.val_y_pred

        if self.custom_metric is not None:
            val_metric = self.custom_metric(y_true, y_pred)
            self.log("val_metric", val_metric, prog_bar=True)

            if self.use_wandb:
                self.log("val/custom_metric", val_metric)

        if self.use_wandb:

            if len(self.training_loss) > 0:
                avg_train_loss = torch.stack(self.training_loss).mean()
            else:
                avg_train_loss = torch.tensor(0.0)

            if len(self.validation_loss) > 0:
                avg_val_loss = torch.stack(self.validation_loss).mean()
            else:
                avg_val_loss = torch.tensor(0.0)

            self.log("train/avg_loss_epoch", avg_train_loss)
            self.log("val/avg_loss_epoch", avg_val_loss)

        self.val_y_true = []
        self.val_y_pred = []
        self.training_loss = []
        self.validation_loss = []

        return None

    def configure_optimizers(self) -> dict:
        """
        Configure optimizers and learning rate schedulers

        Returns
        -------
        dict
            If scheduler is provided, returns dict with optimizer and scheduler
            configuration. Otherwise returns optimizer only
        """
        if self.scheduler is not None:
            monitor = "val_metric" if self.scheduler_custom_metric else "val_loss"

            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {"scheduler": self.scheduler, "monitor": monitor},
            }

        return self.optimizer
