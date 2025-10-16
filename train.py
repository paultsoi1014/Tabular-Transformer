import os
import wandb

from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config import Config
from tabular_data.datamodule import TabularDataModule
from transformer.models import TabTransformer, LightningTabTransformer
from transformer.loss import get_loss_function
from transformer.optimizer import get_optimizer
from transformer.schedulers import get_scheduler


class TabularTransformerTrainer:
    def __init__(self):
        """
        Trainer class for tabular transformer with WanDB integration

        Handles the complete training pipeline including data preparation, model
        initialization, training configuration, and optional WandB logging

        Attributes
        ----------
        config: Config
            Configuration object containing all hyperparameters
        use_wandb: bool
            Indicates whether WandB logging is enabled
        accelerator: str
            The device used for training, either `'gpu'` or `'cpu'`
        datamodule: TabularDataModule
            Data module handling data loading and splitting
        model: LightningTabTransformer
            The initialized TabTransformer model wrapped in a PyTorch Lightning
            module
        callbacks: list
            List of training callbacks including early stopping and
            checkpointing
        trainer: pl.Trainer
            PyTorch Lightning trainer responsible for managing the training loop
        """
        self.config = Config.from_env()

        # Initialize WandB if enabled
        self.use_wandb = self.config.wandb.use_wandb
        if self.use_wandb:
            self._initialize_wandb(
                self.config.wandb.api_key,
                project_name=self.config.wandb.project_name,
                run_name=self.config.wandb.run_name,
                config={
                    "model": self.config.model.__dict__,
                    "training": self.config.training.__dict__,
                    "data": self.config.data.__dict__,
                },
            )

        # Define accelerator
        self.accelerator = "gpu" if torch.cuda.is_available() else "cpu"

        # Initialize components
        self.datamodule = self._setup_datamodule()
        self.model = self._setup_model()
        self.callbacks = self._setup_callbacks()
        self.trainer = self._setup_trainer()

    def _initialize_wandb(
        self,
        api_key: str,
        project_name: str,
        run_name: str,
        config: Optional[dict] = None,
    ) -> None:
        """
        Initialize Weights & Biases logging

        Parameters
        ----------
        api_key: str
            WandB API key
        project: str
            WandB project name
        run_name: str
            Name for the WandB run
        config: Optional[dict]
            Configuration dictionary to log to WandB. Default is None
        """
        # wandb login
        wandb.login(key=api_key)

        # Create WandB logger
        self.wandb_logger = WandbLogger(
            project=project_name, name=run_name, config=config
        )

        return None

    def _setup_datamodule(self) -> TabularDataModule:
        """
        Set up the data module

        Returns
        -------
        TabularDataModule
            Configured data module
        """
        datamodule = TabularDataModule(
            data_dir=self.config.data.data_dir,
            data_files=self.config.data.data_files,
            val_size=self.config.data.val_size,
            test_size=self.config.data.test_size,
            batch_size=self.config.data.batch_size,
            target_name=self.config.data.target_name,
            output_dim=self.config.model.output_dim,
            categorical_features=self.config.data.categorical_features,
            continuous_features=self.config.data.continuous_features,
        )
        return datamodule

    def _setup_model(self) -> LightningTabTransformer:
        """
        Set up the TabTransformer model with optimizer, criterion, and scheduler

        Returns
        -------
        LightningTabTransformer
            Configured Lightning model
        """
        # Initialize model architecture
        tab_transformer = TabTransformer(
            embedding_dim=self.config.model.embedding_dim,
            output_dim=self.config.model.output_dim,
            vocabulary=self.datamodule.get_vocabulary(),
            n_continuous_features=len(self.config.data.continuous_features),
            n_heads=self.config.model.n_heads,
            n_layers=self.config.model.n_layers,
            dim_feedforward=self.config.model.dim_feedforward,
            attn_dropout=self.config.model.dropout,
            mlp_hidden_dims=self.config.model.mlp_hidden_dims,
            activation=self.config.model.activation,
            ffn_dropout=self.config.model.dropout,
        )

        # Criterion, Optimizer and Scheduler
        optimizer = get_optimizer(self.config.training.optimizer)(
            tab_transformer.parameters(), **self.config.training.optimizer_kwargs
        )
        criterion = get_loss_function(
            self.config.training.loss_function,
            loss_kwargs=self.config.training.loss_kwargs,
        )
        scheduler = get_scheduler(self.config.training.scheduler)(
            optimizer, **self.config.training.scheduler_kwargs
        )

        # Initialize the lightning model
        model = LightningTabTransformer(
            tab_transformer,
            output_dim=self.config.model.output_dim,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            use_wandb=self.use_wandb,
        )

        return model

    def _setup_callbacks(self) -> list:
        """
        Set up training callbacks (early stopping, model checkpointing)

        Returns
        -------
        list
            List of configured callbacks
        """
        callbacks = []

        # Early stopping
        if self.config.training.early_stopping:
            early_stopping_config = self.config.training.early_stopping_config
            callbacks.append(
                EarlyStopping(
                    monitor=early_stopping_config.get("monitor", "val_loss"),
                    patience=early_stopping_config.get("patience", 10),
                    mode=early_stopping_config.get("mode", "min"),
                    min_delta=0,
                    verbose=True,
                )
            )

        # Model checkpointing
        if self.config.training.save_model:
            save_model_config = self.config.training.save_model_config
            dirpath = save_model_config.get("path", "models/")

            os.makedirs(dirpath, exist_ok=True)

            callbacks.append(
                ModelCheckpoint(
                    dirpath=dirpath,
                    save_top_k=1,
                    monitor=save_model_config.get("monitor", "val_loss"),
                    mode=save_model_config.get("mode", "min"),
                )
            )

        return callbacks

    def _setup_trainer(self) -> pl.Trainer:
        """
        Set up PyTorch Lightning trainer

        Returns
        -------
        pl.Trainer
            Configured PyTorch Lightning trainer
        """
        trainer_kwargs = {
            "max_epochs": self.config.training.epochs,
            "accelerator": self.accelerator,
            "callbacks": self.callbacks,
        }

        # Add WandB logger if enabled
        if self.use_wandb and self.wandb_logger:
            trainer_kwargs["logger"] = self.wandb_logger

        trainer = pl.Trainer(**trainer_kwargs)
        return trainer

    def fit(self):
        """Train the model"""
        self.trainer.fit(self.model, self.datamodule)

        if self.use_wandb:
            wandb.finish()

        return None

    def test(self):
        """
        Test the model on test dataset
        """
        self.trainer.test(self.model, self.datamodule)

    def predict(self, datamodule: Optional[TabularDataModule] = None):
        """
        Make predictions using the trained model

        Parameters
        ----------
        datamodule: Optional[TabularDataModule]
            Data module for predictions. If None, uses the training datamodule

        Returns
        -------
        Predictions from the model
        """
        if datamodule is None:
            datamodule = self.datamodule

        return self.trainer.predict(self.model, datamodule)


if __name__ == "__main__":
    trainer = TabularTransformerTrainer()
    trainer.fit()
