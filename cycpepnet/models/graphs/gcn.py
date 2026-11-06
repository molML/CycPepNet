import time
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from ..base import BaseModel

# Prune any HPO trial that exceeds this wallclock budget
_TRIAL_TIMEOUT_SEC = 1800


# =============================================
#              NETWORK DEFINITION
# =============================================

# Graph Convolutional Network module
class _GCNNet(nn.Module):

    def __init__(self, atom_dim: int, hidden_dim: int, n_layers: int, dropout: float):

        super().__init__()

        self.input_proj = nn.Linear(atom_dim, hidden_dim)
        self.convs      = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms      = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        self.drop       = nn.Dropout(dropout)
        self.head       = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:

        x = self.input_proj(x).relu()

        for conv, norm in zip(self.convs, self.norms):
            x = self.drop(norm(conv(x, edge_index) + x).relu())

        x = global_mean_pool(x, batch)

        return self.head(x).squeeze(-1)


# GCN model for molecular graphs (regression and classification)
class GCNModel(BaseModel):

    def __init__(
        self,
        atom_dim:     int | None  = None,         # Input atom feature dimension (inferred from data if None)
        hidden_dim:   int         = 128,          # Hidden layer width
        n_layers:     int         = 3,            # Number of GCNConv layers
        dropout:      float       = 0.2,          # Dropout probability
        lr:           float       = 1e-3,         # Adam learning rate
        weight_decay: float       = 0.0,          # L2 regularization strength
        batch_size:   int         = 32,           # Number of graphs per mini-batch
        max_epochs:   int         = 200,          # Maximum training epochs
        patience:     int         = 15,           # Early stopping patience (epochs without val improvement)
        device:       str         = "auto",       # Device to use ("auto" | "cpu" | "cuda" | "mps")
        seed:         int         = 42,           # Random seed for weight initialization and data shuffling
        n_trials:     int         = 0,            # Optuna tuning trials
        hpo_timeout:  int | None  = None,         # Max tuning time in seconds
        task_type:    str         = "regression", # Task type (regression/classification)
    ):

        self._atom_dim            = atom_dim
        self.hidden_dim           = hidden_dim
        self.n_layers             = n_layers
        self.dropout              = dropout
        self.lr                   = lr
        self.weight_decay         = weight_decay
        self.batch_size           = batch_size
        self.max_epochs           = max_epochs
        self.patience             = patience
        self.seed                 = seed
        self.n_trials             = n_trials
        self.hpo_timeout          = hpo_timeout
        self.task_type            = task_type
        self.device               = self._resolve_device(device)
        self._net: _GCNNet | None = None

    @property
    def name(self) -> str:
        return "GCN"

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    # Attach target labels to PyG Data objects, filtering out None graphs
    def _build_dataset(self, graphs: list[Data | None], y: np.ndarray) -> tuple[list[Data], np.ndarray]:

        data_list  = []
        valid_mask = []

        for g, yi in zip(graphs, y):

            if g is not None:
                data      = g.clone()
                data.y    = torch.tensor([float(yi)], dtype=torch.float)
                data_list.append(data)
                valid_mask.append(True)

            else:
                valid_mask.append(False)

        return data_list, np.array(valid_mask)

    # Effective batch size adapted to dataset size
    @staticmethod
    def _effective_bs(batch_size: int, n_samples: int) -> int:
        return min(batch_size, max(8, n_samples // 4))

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    # Run Optuna TPE search
    def _tune(

        self,
        X_train:  list[Data | None],
        y_train:  np.ndarray,
        X_val:    list[Data | None],
        y_val:    np.ndarray,
        atom_dim: int,
        log_fn=None,

    ) -> None:

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        seed   = self.seed
        device = self.device

        train_data, _ = self._build_dataset(X_train, y_train)
        val_data,   _ = self._build_dataset(X_val,   y_val)

        n_train = len(train_data)

        if n_train < 100:
            budget = max(10, self.max_epochs // 10)
        elif n_train < 500:
            budget = max(15, self.max_epochs // 5)
        else:
            budget = max(20, self.max_epochs // 3)

        def objective(trial: optuna.Trial) -> float:

            hidden_dim   = trial.suggest_categorical("hidden_dim", [64, 128, 256])
            n_layers     = trial.suggest_int("n_layers", 2, 4)
            dropout      = trial.suggest_float("dropout", 0.0, 0.4)
            lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])

            torch.manual_seed(seed)

            net = _GCNNet(atom_dim, hidden_dim, n_layers, dropout).to(device)

            eff_bs       = self._effective_bs(batch_size, n_train)
            gen          = torch.Generator().manual_seed(seed)
            train_loader = PyGDataLoader(train_data, batch_size=eff_bs, shuffle=True, generator=gen)
            val_loader   = PyGDataLoader(val_data,   batch_size=eff_bs, shuffle=False)
            optimizer    = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
            criterion    = nn.BCEWithLogitsLoss() if self.task_type == "classification" else nn.MSELoss()

            best_val_loss = float("inf")
            trial_start   = time.time()

            for step in range(budget):
                net.train()
                for batch in train_loader:

                    if time.time() - trial_start > _TRIAL_TIMEOUT_SEC:
                        raise optuna.TrialPruned()

                    batch = batch.to(device)
                    optimizer.zero_grad()
                    criterion(net(batch.x, batch.edge_index, batch.batch), batch.y.view(-1)).backward()
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()

                net.eval()
                total, n = 0.0, 0
                with torch.no_grad():
                    for batch in val_loader:

                        if time.time() - trial_start > _TRIAL_TIMEOUT_SEC:
                            raise optuna.TrialPruned()

                        batch   = batch.to(device)
                        total  += criterion(net(batch.x, batch.edge_index, batch.batch), batch.y.view(-1)).item() * batch.num_graphs
                        n      += batch.num_graphs

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
        X_train: list[Data | None],
        y_train: np.ndarray,
        X_val:   list[Data | None] | None = None,
        y_val:   np.ndarray        | None = None,
        log_fn                            = None,

    ) -> None:

        train_data, _ = self._build_dataset(X_train, y_train)

        if not train_data:
            raise ValueError("No valid graph samples found in X_train after graph construction/filtering.")

        if self._atom_dim is None:
            self._atom_dim = train_data[0].x.shape[1]

        if self.n_trials > 0:
            if X_val is None or y_val is None:
                raise ValueError("n_trials > 0 requires X_val")
            val_data, _ = self._build_dataset(X_val, y_val)
            if not val_data:
                raise ValueError("No valid graph samples found in X_val after graph construction/filtering.")
            self._tune(X_train, y_train, X_val, y_val, self._atom_dim, log_fn=log_fn)

        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        np.random.seed(self.seed)

        self._net = _GCNNet(self._atom_dim, self.hidden_dim, self.n_layers, self.dropout).to(self.device)

        eff_bs       = self._effective_bs(self.batch_size, len(train_data))
        generator    = torch.Generator().manual_seed(self.seed)
        train_loader = PyGDataLoader(train_data, batch_size=eff_bs, shuffle=True, generator=generator)
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

            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._net(batch.x, batch.edge_index, batch.batch), batch.y.view(-1))
                loss.backward()
                grad_norm         = nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()
                train_total      += loss.item() * batch.num_graphs
                train_n          += batch.num_graphs
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
    def predict_proba(self, X: list[Data | None]) -> np.ndarray | None:
        if self.task_type != "classification":
            return None
        return 1.0 / (1.0 + np.exp(-self.predict(X)))

    # Predict property values for a list of PyG Data objects (None entries yield NaN)
    def predict(self, X: list[Data | None]) -> np.ndarray:

        assert self._net is not None, "Call fit() before predict()."

        dummy_y               = np.zeros(len(X))
        data_list, valid_mask = self._build_dataset(X, dummy_y)
        preds_valid           = self._predict_data(data_list)
        preds                 = np.full(len(X), np.nan)
        preds[valid_mask]     = preds_valid

        return preds

    # Predict on a list of PyG Data objects and return concatenated predictions
    def _predict_data(self, data_list: list[Data]) -> np.ndarray:

        self._net.eval()

        loader = PyGDataLoader(data_list, batch_size=self.batch_size, shuffle=False)
        preds  = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds.append(self._net(batch.x, batch.edge_index, batch.batch).cpu().numpy())

        return np.concatenate(preds) if preds else np.array([], dtype=np.float32)

    # Compute average loss on validation set (graphs with None are skipped)
    def _val_loss(self, X_val: list[Data | None], y_val: np.ndarray, criterion) -> float:

        data_list, _ = self._build_dataset(X_val, y_val)

        self._net.eval()

        loader = PyGDataLoader(data_list, batch_size=self.batch_size, shuffle=False)
        total  = 0.0
        n      = 0

        with torch.no_grad():
            for batch in loader:
                batch   = batch.to(self.device)
                total  += criterion(self._net(batch.x, batch.edge_index, batch.batch), batch.y.view(-1)).item() * batch.num_graphs
                n      += batch.num_graphs

        return total / max(n, 1)
