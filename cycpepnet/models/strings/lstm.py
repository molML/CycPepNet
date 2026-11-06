import time
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ..base import BaseModel
from .tokenizer import SMILESTokenizer

# Prune any HPO trial that exceeds this wallclock budget
_TRIAL_TIMEOUT_SEC = 1800


# =============================================
#              NETWORK DEFINITION
# =============================================

# Bidirectional LSTM
class _BiLSTMNet(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layers: int, dropout: float, pad_idx: int):

        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm  = nn.LSTM(
            embed_dim, hidden_dim, n_layers,
            batch_first   = True,
            dropout       = dropout if n_layers > 1 else 0.0,
            bidirectional = True,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x         = self.embed(x)
        _, (h, _) = self.lstm(x)
        h         = torch.cat([h[-2], h[-1]], dim=-1)

        return self.head(h).squeeze(-1)


# =============================================
#              MODEL WRAPPER
# =============================================

# LSTM regressor for SMILES strings
class LSTMModel(BaseModel):

    def __init__(
        self,
        embed_dim:     int         = 128,              # Token embedding dimension
        hidden_dim:    int         = 256,              # LSTM hidden state dimension (per direction)
        n_layers:      int         = 2,                # Number of stacked LSTM layers
        dropout:       float       = 0.2,              # Dropout probability
        max_len:       int         = 1024,             # Maximum sequence token length (longer sequences are truncated)
        lr:            float       = 1e-3,             # Adam learning rate
        weight_decay:  float       = 0.0,              # L2 regularization strength
        batch_size:    int         = 64,               # Mini-batch size
        max_epochs:    int         = 200,              # Maximum training epochs
        patience:      int         = 15,               # Early stopping patience (epochs without val improvement)
        device:        str         = "auto",           # Device to use
        seed:          int         = 42,               # Random seed for reproducibility
        n_trials:      int         = 0,                # Optuna tuning trials
        hpo_timeout:   int | None  = None,             # Max tuning time in seconds
        tokenizer_cls: type        = SMILESTokenizer,  # Tokenizer class to use
        task_type:     str         = "regression",     # Task type (regression/classification)
    ):
        self.embed_dim               = embed_dim
        self.hidden_dim              = hidden_dim
        self.n_layers                = n_layers
        self.dropout                 = dropout
        self.max_len                 = max_len
        self.lr                      = lr
        self.weight_decay            = weight_decay
        self.batch_size              = batch_size
        self.max_epochs              = max_epochs
        self.patience                = patience
        self.seed                    = seed
        self.n_trials                = n_trials
        self.hpo_timeout             = hpo_timeout
        self.task_type               = task_type
        self.device                  = self._resolve_device(device)
        self._tokenizer_cls          = tokenizer_cls
        self.tokenizer               = tokenizer_cls(max_len=max_len)
        self._net: _BiLSTMNet | None = None
        self._eff_bs: int | None     = None

    @property
    def name(self) -> str:
        return "LSTM"

    # Effective batch size adapted to dataset size
    @staticmethod
    def _effective_bs(batch_size: int, n_samples: int) -> int:
        return min(batch_size, max(8, n_samples // 4))

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    # Run Optuna TPE search using the validation set with a reduced epoch budget
    def _tune(self, X_train: list[str], y_train: np.ndarray, X_val: list[str], y_val: np.ndarray, log_fn=None) -> None:

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        seed   = self.seed
        device = self.device

        n_train = len(X_train)

        if n_train < 100:
            budget = max(10, self.max_epochs // 10)
        elif n_train < 500:
            budget = max(15, self.max_epochs // 5)
        else:
            budget = max(20, self.max_epochs // 3)

        tokenizer = self._tokenizer_cls(max_len=self.max_len)
        tokenizer.build_vocab(list(X_train))

        X_enc_train = torch.tensor(tokenizer.encode_batch(X_train), dtype=torch.long)
        y_train_t   = torch.tensor(y_train, dtype=torch.float32)
        X_enc_val   = torch.tensor(tokenizer.encode_batch(X_val),   dtype=torch.long)
        y_val_t     = torch.tensor(y_val,   dtype=torch.float32)

        def objective(trial: optuna.Trial) -> float:

            embed_dim    = trial.suggest_categorical("embed_dim",  [64, 128, 256])
            hidden_dim   = trial.suggest_categorical("hidden_dim", [128, 256, 512])
            n_layers     = trial.suggest_int("n_layers", 1, 3)
            dropout      = trial.suggest_float("dropout", 0.1, 0.5)
            lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])

            torch.manual_seed(seed)
            net = _BiLSTMNet(
                tokenizer.vocab_size, embed_dim, hidden_dim, n_layers, dropout, tokenizer.pad_idx
            ).to(device)

            eff_bs       = self._effective_bs(batch_size, n_train)
            gen          = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(TensorDataset(X_enc_train, y_train_t), batch_size=eff_bs, shuffle=True, generator=gen)
            val_loader   = DataLoader(TensorDataset(X_enc_val,   y_val_t),   batch_size=eff_bs, shuffle=False)
            optimizer    = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
            criterion    = nn.BCEWithLogitsLoss() if self.task_type == "classification" else nn.MSELoss()

            best_val_loss = float("inf")
            trial_start   = time.time()

            for step in range(budget):

                net.train()

                for xb, yb in train_loader:

                    if time.time() - trial_start > _TRIAL_TIMEOUT_SEC:
                        raise optuna.TrialPruned()

                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()

                    try:
                        criterion(net(xb), yb).backward()
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        raise optuna.TrialPruned()

                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()

                net.eval()

                total, n = 0.0, 0

                with torch.no_grad():
                    for xb, yb in val_loader:

                        if time.time() - trial_start > _TRIAL_TIMEOUT_SEC:
                            raise optuna.TrialPruned()

                        xb, yb  = xb.to(device), yb.to(device)
                        total  += criterion(net(xb), yb).item() * xb.size(0)
                        n      += xb.size(0)

                val_loss      = total / max(n, 1)
                best_val_loss = min(best_val_loss, val_loss)

                trial.report(val_loss, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return best_val_loss

        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

        callbacks = []
        if log_fn is not None:
            def _hpo_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                if trial.value is not None:
                    log_fn({"hpo/trial": trial.number, "hpo/val_loss": trial.value, "hpo/best_val_loss": study.best_value})
            callbacks.append(_hpo_cb)

        study.optimize(
            objective,
            n_trials          = self.n_trials,
            timeout           = self.hpo_timeout,
            show_progress_bar = False,
            callbacks         = callbacks,
        )

        self.hpo_metadata = {"best_params": study.best_params, "best_value": study.best_value, "n_trials": len(study.trials), "trial_states": [t.state.name for t in study.trials]}
        best              = study.best_params
        self.embed_dim    = best["embed_dim"]
        self.hidden_dim   = best["hidden_dim"]
        self.n_layers     = best["n_layers"]
        self.dropout      = best["dropout"]
        self.lr           = best["lr"]
        self.weight_decay = best["weight_decay"]
        self.batch_size   = best["batch_size"]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    # Fit the model to the training data
    def fit(

        self,
        X_train: list[str],
        y_train: np.ndarray,
        X_val  : list[str]  | None = None,
        y_val  : np.ndarray | None = None,
        log_fn                     = None,

    ) -> None:

        if self.n_trials > 0:
            if X_val is None or y_val is None:
                raise ValueError("n_trials > 0 requires X_val")
            self._tune(X_train, y_train, X_val, y_val, log_fn=log_fn)

        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        np.random.seed(self.seed)

        self.tokenizer.build_vocab(list(X_train))

        self._eff_bs = self._effective_bs(self.batch_size, len(X_train))

        self._net = _BiLSTMNet(
            self.tokenizer.vocab_size, self.embed_dim, self.hidden_dim,
            self.n_layers, self.dropout, self.tokenizer.pad_idx,
        ).to(self.device)

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        optimizer    = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion    = nn.BCEWithLogitsLoss() if self.task_type == "classification" else nn.MSELoss()

        best_val_loss = float("inf")
        patience_cnt  = 0
        best_state    = None

        for epoch in range(self.max_epochs):

            self._net.train()

            train_total      = 0.0
            train_n          = 0
            train_grad_total = 0.0
            train_batches    = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                grad_norm         = nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()
                train_total      += loss.item() * xb.size(0)
                train_n          += xb.size(0)
                train_grad_total += float(grad_norm)
                train_batches    += 1

            train_loss      = train_total      / max(train_n,       1)
            train_grad_norm = train_grad_total / max(train_batches, 1)

            if X_val is not None and y_val is not None:

                val_loss = self._val_loss(X_val, y_val, criterion)

                if log_fn is not None:

                    log = {
                        "train_epoch": epoch,
                        "train/loss": train_loss, "val/loss": val_loss,
                        "train/lr":        optimizer.param_groups[0]["lr"],
                        "train/grad_norm": train_grad_norm,
                    }

                    if self.task_type == "regression":
                        log["train/rmse"] = train_loss ** 0.5
                        log["val/rmse"]   = val_loss   ** 0.5

                    log_fn(log)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_cnt  = 0
                    best_state    = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}

                else:
                    patience_cnt += 1
                    if patience_cnt >= self.patience:
                        break

            elif log_fn is not None:

                log = {
                    "train_epoch": epoch,
                    "train/loss": train_loss,
                    "train/lr":        optimizer.param_groups[0]["lr"],
                    "train/grad_norm": train_grad_norm,
                }

                if self.task_type == "regression":
                    log["train/rmse"] = train_loss ** 0.5

                log_fn(log)

        if best_state is not None:
            self._net.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    # Return positive class probabilities (classification only)
    def predict_proba(self, X: list[str]) -> np.ndarray | None:
        if self.task_type != "classification":
            return None
        return 1.0 / (1.0 + np.exp(-self.predict(X)))

    # Predict on a list of SMILES strings
    def predict(self, X: list[str]) -> np.ndarray:

        assert self._net is not None, "Call fit() before predict()."

        self._net.eval()

        loader = self._make_loader(X, np.zeros(len(X)), shuffle=False)
        preds  = []

        with torch.no_grad():
            for xb, _ in loader:
                preds.append(self._net(xb.to(self.device)).cpu().numpy())

        return np.concatenate(preds)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Create DataLoader from SMILES and target arrays
    def _make_loader(self, smiles: list[str], y: np.ndarray, shuffle: bool) -> DataLoader:

        X_enc     = torch.tensor(self.tokenizer.encode_batch(smiles), dtype=torch.long)
        y_t       = torch.tensor(y, dtype=torch.float32)
        generator = torch.Generator().manual_seed(self.seed) if shuffle else None
        bs        = self._eff_bs if self._eff_bs is not None else self.batch_size

        return DataLoader(TensorDataset(X_enc, y_t), batch_size=bs, shuffle=shuffle, generator=generator)

    # Compute average loss on validation set weighted by number of samples
    def _val_loss(self, smiles: list[str], y: np.ndarray, criterion) -> float:

        self._net.eval()

        loader = self._make_loader(smiles, y, shuffle=False)
        total  = 0.0
        n      = 0

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb  = xb.to(self.device), yb.to(self.device)
                total   += criterion(self._net(xb), yb).item() * xb.size(0)
                n       += xb.size(0)

        return total / max(n, 1)
