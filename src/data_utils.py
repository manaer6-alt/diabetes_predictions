from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_TARGET = "Outcome"


def load_data(data_path: str | Path) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    return pd.read_csv(data_path)


def split_features_target(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features X and target y."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    X = df.drop(columns=target)
    y = df[target]

    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )