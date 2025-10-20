import pandas as pd
from typing import Dict, Optional, List


def build_vocabulary(
    df: pd.DataFrame, categorical_features: Optional[List[str]] = None
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
