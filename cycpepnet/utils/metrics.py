import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

# Compute root mean squared error (RMSE)
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# Compute QF3² = 1 - (PRESS / n_ext) / (SD / n_tr), normalized by the TRAINING-set variance
def qf3_squared(y_true_ext: np.ndarray, y_pred_ext: np.ndarray, y_train: np.ndarray) -> float:

    n_ext = len(y_true_ext)
    n_tr  = len(y_train)

    y_mean_train = np.mean(y_train)

    press = np.sum((y_true_ext - y_pred_ext) ** 2)
    sd    = np.sum((y_train - y_mean_train) ** 2)

    if sd == 0:
        return 0.0

    return float(1.0 - (press * n_tr) / (sd * n_ext))

# Return both RMSE and QF3²
def evaluate(y_true_ext: np.ndarray, y_pred_ext: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true_ext, y_pred_ext),
        "qf3" : qf3_squared(y_true_ext, y_pred_ext, y_train),
    }

# Return AUROC, TPR, TNR, Precision, and Balanced Accuracy for binary classification.
def evaluate_classification(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:

    nan = {
        "auroc"       : float("nan"),
        "tpr"         : float("nan"),
        "tnr"         : float("nan"),
        "precision"   : float("nan"),
        "balanced_acc": float("nan"),
    }

    if len(np.unique(y_true)) < 2:
        return nan

    y_pred = (y_score >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    tnr  = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    bal  = (tpr + tnr) / 2 if not (np.isnan(tpr) or np.isnan(tnr)) else float("nan")

    return {
        "auroc"       : float(roc_auc_score(y_true, y_score)),
        "tpr"         : float(tpr),
        "tnr"         : float(tnr),
        "precision"   : float(prec),
        "balanced_acc": float(bal),
    }
