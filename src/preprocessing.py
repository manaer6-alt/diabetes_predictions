from typing import Iterable

import numpy as np
import pandas as pd


SUSPICIOUS_ZERO_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

DEFAULT_INDICATOR_COLS = [
    "Insulin",
    "SkinThickness",
]


def replace_suspicious_zeros(
    X: pd.DataFrame,
    zero_cols: Iterable[str] = SUSPICIOUS_ZERO_COLS,
) -> pd.DataFrame:
    """Replace suspicious zero values with NaN.

    This function is intended to be used inside an sklearn Pipeline
    before SimpleImputer.
    """
    X = X.copy()

    existing_zero_cols = [col for col in zero_cols if col in X.columns]
    X[existing_zero_cols] = X[existing_zero_cols].replace(0, np.nan)

    return X


def replace_suspicious_zeros_with_indicators(
    X: pd.DataFrame,
    zero_cols: Iterable[str] = SUSPICIOUS_ZERO_COLS,
    indicator_cols: Iterable[str] = DEFAULT_INDICATOR_COLS,
) -> pd.DataFrame:
    """Add missing indicators and replace suspicious zero values with NaN.

    For selected columns, creates binary indicators showing whether
    the original value was zero before replacement.
    """
    X = X.copy()

    existing_indicator_cols = [col for col in indicator_cols if col in X.columns]

    for col in existing_indicator_cols:
        X[f"{col}_missing"] = (X[col] == 0).astype(int)

    existing_zero_cols = [col for col in zero_cols if col in X.columns]
    X[existing_zero_cols] = X[existing_zero_cols].replace(0, np.nan)

    return X