import os
import pandas as pd

from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import TabularDataset


class TabularDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        data_files: str,
        val_size: float,
        test_size: float,
        batch_size: int,
        target_name: str,
        output_dim: int,
        categorical_features: Optional[List[str]] = None,
        continuous_features: Optional[List[str]] = None,
        random_state: Optional[int] = 42,
    ):
        """
        PyTorch Lightning DataModule for tabular data

        Handles loading, splitting, and preparing tabular data for training,
        validation, and testing. Automatically creates train/val/test splits
        and provides DataLoaders for each split

        Attributes
        ----------
        target_name: str
            Name of the target column in the dataframe
        output_dim: int
            Number of output dimensions. Use 1 for regression or binary
            classification, >1 for multi-class classification
        categorical_features: Optional[List[str]]
            List of categorical feature column names. Default is None
        continuous_features: Optional[List[str]]
            List of continuous feature column names. Default is None
        batch_size: int
            Batch size for DataLoaders
        random_state: Optional[int]
            Random seed for reproducible train/val/test splits. Default is 42
        df_train: pd.DataFrame
            Training dataframe
        df_val: pd.DataFrame
            Validation dataframe
        df_test: pd.DataFrame
            Test dataframe
        train_dataset: TabularDataset
            Training dataset (created in setup())
        val_dataset: TabularDataset
            Validation dataset (created in setup())
        test_dataset: TabularDataset
            Test dataset (created in setup())

        Parameters
        ----------
        data_dir: str
            Directory containing the data file
        data_files: str
            Name of the CSV file to load
        val_size: float
            Fraction of data to allocate to validation set (0-1)
        test_size: float
            Fraction of data to allocate to test set (0-1)
        batch_size: int
            Batch size for DataLoaders
        target_name: str
            Name of the target column in the dataframe
        output_dim: int
            Number of output dimensions. Use 1 for regression or binary
            classification, >1 for multi-class classification
        categorical_features: Optional[List[str]]
            List of categorical feature column names. Default is None
        continuous_features: Optional[List[str]]
            List of continuous feature column names. Default is None
        random_state: Optional[int]
            Random seed for reproducible train/val/test splits. Default is 42
        """
        super(TabularDataModule, self).__init__()

        self.target_name = target_name
        self.output_dim = output_dim
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features

        self.batch_size = batch_size
        self.random_state = random_state

        # Load dataset
        self.df_train, self.df_val, self.df_test = self.load_data(
            data_dir=data_dir,
            data_files=data_files,
            val_size=val_size,
            test_size=test_size,
        )

    def get_vocabulary(self) -> Dict[str, Dict[str, int]]:
        """
        Get vocabulary mappings from the training dataset

        Returns
        -------
        Dict[str, Dict[str, int]]
            Dictionary mapping each categorical feature to its value-to-index
            mapping. Format: {feature_name: {feature_value: index}}
        """
        train_dataset = TabularDataset(
            df=self.df_train,
            target=self.target_name,
            output_dim=self.output_dim,
            categorical_features=self.categorical_features,
            continuous_features=self.continuous_features,
        )

        return train_dataset.get_vocabulary()

    def load_data(
        self, data_dir: str, data_files: str, val_size: int, test_size: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and split data from CSV file

        Reads the CSV file and splits it into train, validation, and test sets
        using the specified proportions

        Parameters
        ----------
        data_dir: str
            Directory containing the data file
        data_files: str
            Name of the CSV file to load
        val_size: float
            Fraction of data to allocate to validation set (0-1)
        test_size: float
            Fraction of data to allocate to test set (0-1)

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Tuple containing the training, validation and testing dataset
        """
        # Load training data
        df = pd.read_csv(os.path.join(data_dir, "train", data_files))

        # get training dataset, validation dataset, and test set
        df_train, df_val, df_test = self.train_test_val_split(
            df, val_size=val_size, test_size=test_size
        )

        return df_train, df_val, df_test

    def train_test_val_split(
        self,
        data: pd.DataFrame,
        val_size: float,
        test_size: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into training, validation, and test sets

        Parameters
        ----------
        data: pd.DataFrame
            2D numpy array of bilingual sentence pairs
        val_size: float
            Fraction of data to allocate to validation set (0–1)
        test_size: float
            Fraction of data to allocate to test set (0-1)

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Train, validation, and test subsets of the dataset
        """
        # Separate test set
        df_train_val, df_test = train_test_split(
            data, test_size=test_size, random_state=self.random_state
        )

        # separate validation from training
        val_ratio = val_size / (1 - test_size)
        df_train, df_val = train_test_split(
            df_train_val,
            test_size=val_ratio,
            random_state=self.random_state,
        )

        return df_train, df_val, df_test

    def setup(self, stage=None) -> None:
        """
        Prepare datasets for training, validation, and testing

        Parameters
        ----------
        stage: Optional[str]
            Training stage ("fit", "validate", "test", "predict"). Unused in
            this implementation. Default is None
        """
        self.train_dataset = TabularDataset(
            df=self.df_train,
            target=self.target_name,
            output_dim=self.output_dim,
            categorical_features=self.categorical_features,
            continuous_features=self.continuous_features,
        )

        self.val_dataset = TabularDataset(
            df=self.df_val,
            target=self.target_name,
            output_dim=self.output_dim,
            categorical_features=self.categorical_features,
            continuous_features=self.continuous_features,
        )

        self.test_dataset = TabularDataset(
            df=self.df_test,
            target=self.target_name,
            output_dim=self.output_dim,
            categorical_features=self.categorical_features,
            continuous_features=self.continuous_features,
        )

        return None

    def train_dataloader(self) -> DataLoader:
        """
        Create DataLoader for the training dataset

        Returns
        -------
        DataLoader
            Training dataloader with shuffling enabled
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """
        Create DataLoader for the validation dataset

        Returns
        -------
        DataLoader
            Validation dataloader without shuffling
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """
        Create DataLoader for the test dataset

        Returns
        -------
        DataLoader
            Test dataloader without shuffling
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
