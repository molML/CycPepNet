import optuna
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import roc_auc_score

from ..base import BaseModel

# XGBoost regressor or classifier for fingerprint vectors
class XGBoostModel(BaseModel):

    def __init__(
        self,
        n_estimators:          int        = 500,          # Number of boosting rounds
        max_depth:             int        = 4,            # Maximum tree depth
        learning_rate:         float      = 0.05,         # Step size shrinkage (eta)
        subsample:             float      = 0.8,          # Row subsampling ratio per tree
        colsample_bytree:      float      = 0.8,          # Column subsampling ratio per tree
        min_child_weight:      int        = 1,            # Minimum sum of instance weight in a child
        early_stopping_rounds: int        = 20,           # Stop if val loss doesn't improve for N rounds
        n_jobs:                int        = -1,           # Parallel threads (-1 = use all cores)
        seed:                  int        = 42,           # Random seed for reproducibility
        n_trials:              int        = 0,            # Optuna tuning trials
        hpo_timeout:           int | None = None,         # Max tuning time in seconds
        task_type:             str        = "regression", # Task type (regression/classification)
        **kwargs,
    ):

        self._seed                  = seed
        self._n_jobs                = n_jobs
        self._n_trials              = n_trials
        self._hpo_timeout           = hpo_timeout
        self._early_stopping_rounds = early_stopping_rounds
        self._task_type             = task_type
        self._default_kwargs        = kwargs

        model_cls    = XGBClassifier if task_type == "classification" else XGBRegressor
        self._model  = model_cls(
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            learning_rate    = learning_rate,
            subsample        = subsample,
            colsample_bytree = colsample_bytree,
            min_child_weight = min_child_weight,
            random_state     = seed,
            n_jobs           = n_jobs,
            verbosity        = 0,
            **kwargs,
        )

    # Short identifier used in result tables and file names
    @property
    def name(self) -> str:
        return "XGBoost"

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

        seed                  = self._seed
        n_jobs                = self._n_jobs
        kwargs                = self._default_kwargs
        early_stopping_rounds = self._early_stopping_rounds
        is_cls                = self._task_type == "classification"
        model_cls             = XGBClassifier if is_cls else XGBRegressor

        def objective(trial: optuna.Trial) -> float:

            params = {
                "n_estimators"         : trial.suggest_int("n_estimators", 200, 800, step=100),
                "max_depth"            : trial.suggest_int("max_depth", 3, 9),
                "learning_rate"        : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample"            : trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree"     : trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight"     : trial.suggest_int("min_child_weight", 1, 7),
                "reg_alpha"            : trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
                "reg_lambda"           : trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
                "random_state"         : seed,
                "n_jobs"               : n_jobs,
                "verbosity"            : 0,
                "early_stopping_rounds": early_stopping_rounds,
            }

            m = model_cls(**params, **kwargs)
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

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
            model_cls = XGBClassifier if self._task_type == "classification" else XGBRegressor

            self._model = model_cls(
                random_state          = self._seed,
                n_jobs                = self._n_jobs,
                verbosity             = 0,
                early_stopping_rounds = self._early_stopping_rounds,
                **best,
                **extra,
            )

        fit_kwargs: dict = {"verbose": False}

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self._model.fit(X_train, y_train, **fit_kwargs)

        if log_fn is not None and X_val is not None:

            is_cls    = self._task_type == "classification"
            metric    = "logloss" if is_cls else "rmse"
            log_key   = "boost/val_loss" if is_cls else "boost/val_rmse"
            rounds    = self._model.evals_result_.get("validation_0", {}).get(metric, [])

            for val_metric in rounds:
                log_fn({log_key: val_metric})

    # Predict values (regression) or binary labels (classification)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    # Return positive class probabilities (classification only)
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self._task_type != "classification":
            return None
        return self._model.predict_proba(X)[:, 1]
