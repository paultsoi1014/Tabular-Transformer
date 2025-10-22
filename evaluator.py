import logging
import os

import pandas as pd
from typing import Optional

import torch

from config import Config
from tabular_data.dataset import TabularDataset
from tabular_data.utils import build_vocabulary
from transformer.models import TabTransformer, LightningTabTransformer

logger = logging.getLogger(__name__)


class TabularTransformerEvaluator:
    def __init__(self, ckpt_path: str):
        """
        Inference engine for trained TabTransformer models. This class loads a
        trained TabTransformer model from a checkpoint and provides methods for
        making predictions on new data

        Attributes
        ----------
        ckpt_path: str
            Stored checkpoint path of the trained model
        config: Config
            Configuration object containing all hyperparameters which are loaded
            from environment variables
        device: torch.device
            Device to run inference on (CUDA if available, else CPU)
        model: LightningTabTransformer
            Loaded TabTransformer model wrapped in a PyTorch Lightning module

        Parameters
        ----------
        ckpt_path: str
            Path to the trained model checkpoint file (.ckpt format)
        """
        self.ckpt_path = ckpt_path
        self.config = Config.from_env()

        # Define device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = self._load_model()

    def _load_model(self) -> TabTransformer:
        """
        Load the TabTransformer model from checkpoint

        This method:
        1. Loads training data to rebuild the vocabulary
        2. Reconstructs the model architecture
        3. Loads trained weights from checkpoint
        4. Sets model to evaluation mode

        Returns
        -------
        LightningTabTransformer
            Loaded model in inference mode, ready for predictions
        """
        # Load training data
        df_train = pd.read_csv(
            os.path.join(
                self.config.data.data_dir, "train", self.config.data.data_files
            )
        )

        # get vocabulary
        vocabulary = build_vocabulary(
            df=df_train, categorical_features=self.config.data.categorical_features
        )

        # load table transformer
        tab_transformer = TabTransformer(
            embedding_dim=self.config.model.embedding_dim,
            output_dim=self.config.model.output_dim,
            vocabulary=vocabulary,
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
        model = LightningTabTransformer.load_from_checkpoint(
            checkpoint_path=self.ckpt_path,
            model=tab_transformer,
            inference=True,
        )
        model.eval()

        return model

    def predict_from_dataframe(
        self, df: pd.DataFrame, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate predictions for data in a pandas DataFrame

        Parameters
        ----------
        df: pd.DataFrame
            Input dataframe containing features for prediction
        batch_size: Optional[int]
            Batch size for inference. If None, uses default from config

        Returns
        -------
        torch.Tensor
            Predictions tensor of shape (n_samples, output_dim)
            For regression (output_dim=1): raw predictions
            For classification (output_dim>1): logits (apply softmax for
            probabilities)
        """
        if batch_size is None:
            batch_size = self.config.data.batch_size

        if self.config.data.target_name not in df.columns:
            df = df.copy()
            df[self.config.data.target_name] = 0  # Dummy target

        dataset = TabularDataset(
            df=df,
            target=self.config.data.target_name,
            output_dim=self.config.model.output_dim,
            categorical_features=self.config.data.categorical_features,
            continuous_features=self.config.data.continuous_features,
        )

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        # Generate predictions batch by batch
        predictions = []
        with torch.no_grad():
            for categorical_data, continuous_data, _ in dataloader:
                categorical_data = categorical_data.to(self.device)
                continuous_data = continuous_data.to(self.device)

                output = self.model(categorical_data, continuous_data)
                predictions.append(output.cpu())

        predictions = torch.cat(predictions, dim=0)

        return predictions

    def predict_from_csv(
        self, file_path: str, batch_size: Optional[int] = None
    ) -> None:
        """
        Generate predictions from a CSV file and save results.

        This method:
        1. Reads data from the CSV file
        2. Generates predictions
        3. Adds prediction columns to the original data
        4. Saves results to a new CSV file with '_predictions' suffix

        Parameters
        ----------
        file_path: str
            Path to the input CSV file
        batch_size: Optional[int]
            Batch size for inference. If None, uses the batch size from config
        """
        # Load data from CSV
        df = pd.read_csv(file_path)

        # Generate predictions
        predictions = self.predict_from_dataframe(df, batch_size=batch_size)

        # Format predictions based on task type
        if self.config.model.output_dim == 1:
            df["prediction"] = predictions.squeeze().numpy()
        else:
            # For multi-class, add predicted class and probabilities
            df["predicted_class"] = torch.argmax(predictions, dim=1).numpy()

            probs = torch.softmax(predictions, dim=1)
            for i in range(self.config.model.output_dim):
                df[f"prob_class_{i}"] = probs[:, i].numpy()

        # Save predictions to CSV
        output_file = file_path.replace(".csv", "_predictions.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")

        return None
