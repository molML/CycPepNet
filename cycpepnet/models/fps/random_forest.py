import optuna
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from ..base import BaseModel

# Random Forest regressor or classifier for fingerprint vectors
class RandomForestModel(BaseModel):

    def __init__(
        self,
        n_estimators:     int        = 500,          # Number of trees in the forest
        max_features:     str        = "sqrt",       # Feature sampling strategy ("sqrt", "log2", float)
        min_samples_leaf: int        = 2,            # Minimum samples per leaf (regularisation)
        n_jobs:           int        = -1,           # Number of parallel jobs (-1 = use all cores)
        seed:             int        = 42,           # Random seed for reproducibility
        n_trials:         int        = 0,            # Optuna tuning trials
        hpo_timeout:      int | None = None,         # Max tuning time in seconds
        task_type:        str        = "regression", # Task type (regression/classification)
        **kwargs,
    ):
        self._seed           = seed
        self._n_jobs         = n_jobs
        self._n_trials       = n_trials
        self._hpo_timeout    = hpo_timeout
        self._task_type      = task_type
        self._default_kwargs = kwargs

        model_cls = RandomForestClassifier if task_type == "classification" else RandomForestRegressor

        self._model = model_cls(
            n_estimators     = n_estimators,
            max_features     = max_features,
            min_samples_leaf = min_samples_leaf,
            random_state     = seed,
            n_jobs           = n_jobs,
            **kwargs,
        )

    # Short identifier used in result tables and file names
    @property
    def name(self) -> str:
        return "RandomForest"

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    # Optuna TPE on fixed validation split
    def _tune(

        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        log_fn               = None,

    ) -> dict:

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        seed      = self._seed
        n_jobs    = self._n_jobs
        kwargs    = self._default_kwargs
        is_cls    = self._task_type == "classification"
        model_cls = RandomForestClassifier if is_cls else RandomForestRegressor

        def objective(trial: optuna.Trial) -> float:

            params = {
                "n_estimators"    : trial.suggest_int("n_estimators", 200, 1000, step=100),
                "max_features"    : trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_depth"       : trial.suggest_categorical("max_depth", [None, 20, 30, 40]),
                "random_state"    : seed,
                "n_jobs"          : n_jobs,
            }

            m = model_cls(**params, **kwargs)
            m.fit(X_train, y_train)

            if is_cls:
                if len(np.unique(y_val)) < 2:
                    raise optuna.TrialPruned()
                val_metric = -roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

            else:
                val_metric = np.sqrt(np.mean((m.predict(X_val) - y_val) ** 2))

            return float(val_metric)

        sampler = optuna.samplers.TPESampler(seed=seed)
        study   = optuna.create_study(direction="minimize", sampler=sampler)

        callbacks = []

        if log_fn is not None:

            metric_key = "hpo/val_auroc" if is_cls else "hpo/val_rmse"
            best_key   = "hpo/best_val_auroc" if is_cls else "hpo/best_val_rmse"

            def _hpo_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                if trial.value is not None:
                    v = -trial.value if is_cls else trial.value
                    b = -study.best_value if is_cls else study.best_value
                    log_fn({"hpo/trial": trial.number, metric_key: v, best_key: b})

            callbacks.append(_hpo_cb)

        study.optimize(
            objective,
            n_trials          = self._n_trials,
            timeout           = self._hpo_timeout,
            show_progress_bar = False,
            callbacks         = callbacks,
        )

        self.hpo_metadata = {"best_params": study.best_params, "best_value": study.best_value, "n_trials": len(study.trials), "trial_states": [t.state.name for t in study.trials]}
        return study.best_params

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    # Fit the model to the training data
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, log_fn=None) -> None:

        if self._n_trials > 0:

            if X_val is None or y_val is None:
                raise ValueError("n_trials > 0 requires X_val")

            best      = self._tune(X_train, y_train, X_val, y_val, log_fn=log_fn)
            extra     = {k: v for k, v in self._default_kwargs.items() if k not in best}
            model_cls = RandomForestClassifier if self._task_type == "classification" else RandomForestRegressor

            self._model = model_cls(
                random_state = self._seed,
                n_jobs       = self._n_jobs,
                **best,
                **extra,
            )

        self._model.fit(X_train, y_train)

    # Predict values (regression) or binary labels (classification)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    # Return positive class probabilities (classification only)
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self._task_type != "classification":
            return None
        return self._model.predict_proba(X)[:, 1]
