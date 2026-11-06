import numpy as np
from .base import BaseModel

# Trivial baseline: predicts training mean (regression) or class prior (classification)
class MeanPredictor(BaseModel):

    def __init__(self, seed: int = 42, task_type: str = "regression"):
        self._task_type          = task_type
        self._mean: float | None = None

    # Short identifier used in result tables and file names
    @property
    def name(self) -> str:
        return "MeanPredictor"

    # Fit the model to the training data
    def fit(self, X_train, y_train: np.ndarray, X_val=None, y_val=None, log_fn=None) -> None:
        self._mean = float(np.mean(y_train))

    # Predict the mean (regression) or majority class (classification)
    def predict(self, X) -> np.ndarray:
        assert self._mean is not None, "Call fit() before predict()."
        n     = len(X)
        value = round(self._mean) if self._task_type == "classification" else self._mean
        return np.full(n, value, dtype=np.float32)

    # Return class prior probability for all inputs (classification only)
    def predict_proba(self, X) -> np.ndarray | None:
        if self._task_type != "classification":
            return None
        assert self._mean is not None, "Call fit() before predict_proba()."
        return np.full(len(X), self._mean, dtype=np.float32)
