from abc import ABC, abstractmethod
import numpy as np
import torch

# Common interface for all CycPepNet models
class BaseModel(ABC):

    # Short identifier used in result tables and file names
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    # Resolve device string to torch.device
    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    # Train the model
    @abstractmethod
    def fit(
        self,
        X_train,
        y_train: np.ndarray,
        X_val=None,
        y_val: np.ndarray | None = None,
        log_fn=None,
    ) -> None:
        ...

    # Return predictions as a 1-D float array aligned with X
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        ...

    # Return predicted probabilities for the positive class (classification only)
    def predict_proba(self, X) -> np.ndarray | None:
        return None

    # Extra metadata to persist in logs/results when a model needs it
    def benchmark_metadata(self) -> dict[str, object]:
        return {}
