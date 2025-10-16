import pandas as pd
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        output_dim: int,
        categorical_features: Optional[List[str]] = None,
        continuous_features: Optional[List[str]] = None,
    ):
        """
        PyTorch Dataset for tabular data with categorical and continuous features

        This dataset handles mixed-type tabular data by separating categorical
        and continuous features, automatically creating vocabularies for
        categorical features, and converting data to appropriate tensor formats

        Attributes
        ----------
        categorical_data: Union[pd.DataFrame, None]
            DataFrame containing categorical features, or None if not provided
        continuous_data: Union[np.ndarray, None]
            NumPy array containing continuous features, or None if not provided
        df_length: int
            Number of samples in the dataset
        target: str
            Name of the target column
        target_dtype: torch.dtype
            Data type for target tensor (float for regression, long for
            classification)
        vocabulary: Dict[str, Dict]
            Mapping of categorical features to their integer encodings

        Parameters
        ----------
        df: pd.DataFrame
            The input dataframe
        target: str
            Name of the target column
        output_dim: int
            Number of output dimensions. Use 1 for regression or binary
            classification, >1 for multi-class classification. Determines target
            dtype (float for 1, long otherwise)
        categorical_features: Optional[List[str]]
            List of categorical feature column names. Default is None
        continuous_features: Optional[List[str]]
            List of continuous feature column names. Default is None
        """
        super(TabularDataset, self).__init__()
        self.df_length = len(df)

        if categorical_features is None and continuous_features is None:
            raise ValueError(
                "At least one of categorical_features and continuous_features "
                "must be provided"
            )

        if not isinstance(output_dim, int):
            raise ValueError("The output dimension (output_dim) must be an integer")
        elif output_dim <= 0:
            raise ValueError(
                "The output dimension (output_dim) must be a positive integer"
            )
        elif output_dim == 1:
            self.target_dtype = torch.float
        else:
            self.target_dtype = torch.long

        if categorical_features is not None:
            self._check_feature_existance(df, categorical_features)
            self.categorical_data = df[categorical_features]
        else:
            self.categorical_data = None

        if continuous_features is not None:
            self._check_feature_existance(df, continuous_features)
            self.continuous_data = df[continuous_features].to_numpy()
        else:
            self.continuous_data = None

        # Build vocabulary
        self.vocabulary = self._build_vocabulary(df, categorical_features)

        # Set target
        self.target = torch.tensor(df[target].to_numpy(), dtype=self.target_dtype)

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset

        Returns
        -------
        int
            Number of samples
        """
        return self.df_length

    def __getitem__(
        self, idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Get a single sample from the dataset

        Parameters
        ----------
        idx: int
            Index of the sample to retrieve

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing:
            - categorical_data: Encoded categorical features (empty tensor if none)
            - continuous_data: Continuous features (empty tensor if none)
            - target: Target value
        """
        if self.categorical_data is None:
            categorical_data = torch.empty(0, dtype=torch.long)
        else:
            categorical_data = [
                self.vocabulary[col][self.categorical_data[col].iloc[idx]]
                for col in self.categorical_data.columns
            ]
            categorical_data = torch.tensor(categorical_data)

        if self.continuous_data is None:
            continuous_data = torch.empty(0, dtype=torch.float32)
        else:
            continuous_data = torch.tensor(
                self.continuous_data[idx], dtype=torch.float32
            )

        return categorical_data, continuous_data, self.target[idx]

    def _check_feature_existance(self, df: pd.DataFrame, features: List[str]) -> None:
        """
        Validate that all specified features exist in the dataframe

        Parameters
        ----------
        df: pd.DataFrame
            Input dataframe
        features: List[str]
            List of feature column names to check

        Raises
        ------
        ValueError
            If any feature is not found in the dataframe
        """
        missing_features = [col for col in features if col not in df.columns]
        if missing_features:
            raise ValueError(
                f"Features not found in dataframe: {', '.join(missing_features)}"
            )

        return None

    def _build_vocabulary(
        self, df: pd.DataFrame, categorical_features: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Build vocabulary mappings for categorical features

        Parameters
        ----------
        df: pd.DataFrame
            Input dataframe
        categorical_features: Optional[List[str]]
            List of categorical feature column names

        Returns
        -------
        Dict[str, Dict[str, int]]
            Dictionary mapping each categorical feature to its value-to-index
            mapping
        """
        vocabulary = {}

        if categorical_features is None:
            return vocabulary

        # Check feature existance
        for col in categorical_features:
            unique_values = sorted(df[col].unique().tolist())
            vocabulary[col] = {value: i for i, value in enumerate(unique_values)}

        return vocabulary

    def get_vocabulary(self) -> dict:
        """
        Get the vocabulary mappings for categorical features

        Returns
        -------
        dict
            Dictionary mapping each categorical feature to its value-to-index
            mapping
        """
        return self.vocabulary
