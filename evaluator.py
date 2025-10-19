import logging
import os
import wandb

import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from config import Config
from tabular_data.datamodule import TabularDataModule, TabularPredictDataModule
from transformer.models import TabTransformer, LightningTabTransformer
from transformer.loss import get_loss_function
from transformer.optimizer import get_optimizer
from transformer.schedulers import get_scheduler

logger = logging.getLogger(__name__)


class TabularTransformerEvaluator:
    def __init__(self, ckpt_path: str):
        """
        Predictor class for TabularTransformer specialized for inference

        This class is designed specifically for making predictions with trained
        TabularTransformer models. It loads a model from a checkpoint file and
        provides convenient methods for batch and single-sample predictions

        Attributes
        ----------
        ckpt_path: str
            Path to the model checkpoint file (.ckpt)
        config: Config
            Configuration object loaded from environment or config file
        accelerator: str
            Device to use for inference ('gpu' or 'cpu')
        model: LightningTabTransformer
            Loaded model from checkpoint
        trainer: pl.Trainer
            PyTorch Lightning trainer for prediction

        Parameters
        ----------
        ckpt_path: str
            Path to the trained model checkpoint file (.ckpt format)
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        self.config = Config.from_env()
        self.ckpt_path = ckpt_path

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

        # Set accelerator
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        if accelerator == "gpu":
            self.map_location = torch.device("cuda:0")
        else:
            self.map_location = torch.device("cpu")

        # Load model
        self.model = self._load_model()

        # Extract vocabulary
        self.vocabulary = self._extract_vocabulary()

        # Create trainer for prediction
        self.trainer = pl.Trainer(
            accelerator=self.accelerator,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )

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

    def _load_model(self) -> LightningTabTransformer:
        """
        Load the model from checkpoint

        Returns
        -------
        LightningTabTransformer
            Loaded model ready for inference
        """
        # setup datamodule to get vocabulary
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

        # Initialize model architecture
        tab_transformer = TabTransformer(
            embedding_dim=self.config.model.embedding_dim,
            output_dim=self.config.model.output_dim,
            vocabulary=datamodule.get_vocabulary(),
            n_continuous_features=len(self.config.data.continuous_features),
            n_heads=self.config.model.n_heads,
            n_layers=self.config.model.n_layers,
            transformer_activation=self.config.model.transformer_activation,
            dim_feedforward=self.config.model.dim_feedforward,
            attn_dropout=self.config.model.dropout,
            mlp_hidden_dims=self.config.model.mlp_hidden_dims,
            mlp_activation=self.config.model.mlp_activation,
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

        model = LightningTabTransformer(
            tab_transformer,
            output_dim=self.config.model.output_dim,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            use_wandb=self.use_wandb,
        ).load_from_checkpoint(self.ckpt_path, map_location=self.map_location)
        model.eval()

        return model

    def _extract_vocabulary(self) -> Dict[str, Dict[str, int]]:
        """
        Extract vocabulary from the loaded model

        Returns
        -------
        Dict[str, Dict[str, int]]
            Vocabulary mappings for categorical features
        """
        return self.model.model.encoders["categorical_encoder"].vocabulary

    def predict_from_datamodule(
        self, datamodule: TabularPredictDataModule, return_numpy: Optional[bool] = True
    ) -> Union[List[torch.Tensor], np.ndarray]:
        """
        Make predictions using a TabularPredictDataModule

        Parameters
        ----------
        datamodule: TabularPredictDataModule
            DataModule containing the data to predict on
        return_numpy: bool
            If True, returns predictions as numpy array. If False, returns list
            of torch tensors. Default is True

        Returns
        -------
        Union[List[torch.Tensor], np.ndarray]
            Predictions from the model
        """
        predictions = self.trainer.predict(self.model, datamodule=datamodule)

        if return_numpy:
            predictions = torch.cat(predictions, dim=0).cpu().numpy()

        return predictions

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        return_numpy: Optional[bool] = True,
    ) -> Union[List[torch.Tensor], np.ndarray]:
        """
        Make predictions from a pandas DataFrame

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing the features to predict on. Must contain all
            categorical and continuous features specified in the configuration
        batch_size: Optional[int]
            Batch size for prediction. If None, uses the batch size from config.
            Default is None
        return_numpy: bool
            If True, returns predictions as numpy array. If False, returns list
            of torch tensors. Default is True

        Returns
        -------
        Union[List[torch.Tensor], np.ndarray]
            Predictions from the model
        """
        features_required = (
            self.config.data.categorical_features + self.config.data.continuous_features
        )
        missing_features = [col for col in features_required if col not in df.columns]
        if missing_features:
            raise ValueError(
                f"Missing required features in DataFrame: {missing_features}"
            )

        # Creeat datamlodule
        if self.config.data.target_name not in df.columns:
            df = df.copy()
            df[self.config.data.target_name] = 0

        datamodule = TabularPredictDataModule(
            df=df,
            vocabulary=self.vocabulary,
            batch_size=batch_size or self.config.data.batch_size,
            output_dim=self.config.model.output_dim,
            categorical_features=self.config.data.categorical_features,
            continuous_features=self.config.data.continuous_features,
        )
        datamodule.setup()

        # Predict
        predictions = self.trainer.predict(
            self.model, dataloaders=datamodule.predict_dataloader()
        )

        if return_numpy:
            predictions = torch.cat(predictions, dim=0).cpu().numpy()

        return predictions

    def predict_from_csv(
        self,
        file_path: str,
        batch_size: Optional[int] = None,
        return_numpy: Optional[bool] = True,
        save_to_csv: Optional[bool] = True,
    ):
        """
        Make predictions from a CSV file

        Parameters
        ----------
        file_path: str
            Path to the CSV file containing features
        batch_size: Optional[int]
            Batch size for prediction. If None, uses config batch size.
            Default is None
        return_numpy: bool
            If True, returns predictions as numpy array. Default is True
        save_to_csv: Optional[bool]
            If True, saves predictions to the path as the csv located. Default
            is True

        Returns
        -------
        Union[List[torch.Tensor], np.ndarray]
            Predictions from the model
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # Get file path directory
        file_dir = os.path.dirname(file_path)

        # Load csv into dataframe
        df = pd.read_csv(file_path)

        # Predict
        predictions = self.predict_from_dataframe(
            df, batch_size=batch_size, return_numpy=return_numpy
        )

        if save_to_csv:
            # Save predictions to CSV
            df_predict = pd.DataFrame(predictions)
            predictions_file_path = os.path.join(file_dir, "predictions.csv")
            df_predict.to_csv(predictions_file_path, index=False)
            logger.info(f"Predictions saved to: {predictions_file_path}")

        return predictions
