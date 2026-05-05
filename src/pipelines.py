from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.preprocessing import (
    replace_suspicious_zeros,
    replace_suspicious_zeros_with_indicators,
)


def build_classification_pipeline(
    model: BaseEstimator,
    use_scaler: bool,
    use_indicators: bool = True,
) -> Pipeline:
    """Build sklearn classification pipeline.

    Pipeline steps:
    1. Replace suspicious zero values with NaN.
    2. Optionally add missing indicators for selected columns.
    3. Impute missing values using median strategy.
    4. Optionally apply StandardScaler.
    5. Fit the model.

    Parameters
    ----------
    model:
        Sklearn-compatible estimator.
    use_scaler:
        Whether to add StandardScaler before the model.
    use_indicators:
        Whether to add missing indicators for selected suspicious-zero columns.

    Returns
    -------
    Pipeline
        Sklearn Pipeline ready for cross-validation or fitting.
    """
    zero_handler_func = (
        replace_suspicious_zeros_with_indicators
        if use_indicators
        else replace_suspicious_zeros
    )

    steps = [
        ("zero_handler", FunctionTransformer(zero_handler_func, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))

    return Pipeline(steps=steps)