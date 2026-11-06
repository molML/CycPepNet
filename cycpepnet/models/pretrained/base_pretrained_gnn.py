import io
import logging
import time
import urllib.request
from pathlib import Path
import numpy as np
import optuna
import torch
import torch.nn as nn
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import MessagePassing

from ..base import BaseModel

logger = logging.getLogger(__name__)

# Prune any HPO trial that exceeds this wallclock budget
_TRIAL_TIMEOUT_SEC = 1800

# Local cache for downloaded pretrained checkpoints
_CACHE_DIR = Path.home() / ".cache" / "cycpepnet"

_CHIRALITY_MAP = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED:     0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:  1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
}

_BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE:   0,
    Chem.rdchem.BondType.DOUBLE:   1,
    Chem.rdchem.BondType.TRIPLE:   2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

_BOND_DIR_MAP = {
    Chem.rdchem.BondDir.NONE:         0,
    Chem.rdchem.BondDir.ENDUPRIGHT:   1,
    Chem.rdchem.BondDir.ENDDOWNRIGHT: 2,
}

# =============================================
#              SMILES → GRAPH
# =============================================

# Convert SMILES to integer-encoded PyG Data matching pretrained-GNN checkpoint formats
def _smiles_to_indexed_graph(smiles: str, atom_idx_max: int, add_hs: bool) -> Data | None:

    if not smiles or not isinstance(smiles, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if add_hs:
            mol = Chem.AddHs(mol)

        atom_features = []
        for atom in mol.GetAtoms():
            anum       = atom.GetAtomicNum()
            atom_idx   = min(anum - 1, atom_idx_max) if 1 <= anum <= 118 else 0
            chiral_idx = _CHIRALITY_MAP.get(atom.GetChiralTag(), 0)
            atom_features.append([atom_idx, chiral_idx])

        x = torch.tensor(atom_features, dtype=torch.long)

        if mol.GetNumBonds() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, 2), dtype=torch.long)
        else:
            rows, cols, edge_feats = [], [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bt   = _BOND_TYPE_MAP.get(bond.GetBondType(), 0)
                bd   = _BOND_DIR_MAP.get(bond.GetBondDir(), 0)
                rows       += [i, j]
                cols       += [j, i]
                edge_feats += [[bt, bd], [bt, bd]]
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_attr  = torch.tensor(edge_feats, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    except Exception as e:
        logger.debug(f"Failed to convert SMILES to indexed graph '{smiles}': {e}")
        return None


# =============================================
#              WEIGHT LOADING
# =============================================

# Download and cache a pretrained checkpoint from a remote URL
def _download_pretrained_weights(url: str, cache_file: str, device: torch.device) -> dict:

    local_path = _CACHE_DIR / cache_file

    if not local_path.exists():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading pretrained weights → {local_path}")
        urllib.request.urlretrieve(url, str(local_path))

    return torch.load(str(local_path), map_location=device, weights_only=True)


# =============================================
#              NETWORK DEFINITION
# =============================================

# Single GIN convolution layer matching the pretrained-GNN checkpoint layout
class _GINConvLayer(MessagePassing):

    def __init__(self, hidden_dim: int, num_bond_types: int, num_bond_dirs: int):

        super().__init__(aggr="add")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        self.edge_embedding1 = nn.Embedding(num_bond_types, hidden_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_dirs,  hidden_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight)
        nn.init.xavier_uniform_(self.edge_embedding2.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:

        edge_emb = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.mlp(x + self.propagate(edge_index, x=x, edge_attr=edge_emb))

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return x_j + edge_attr


# Pretrained GNN encoder + regression/classification head
class _FineTunedGNN(nn.Module):

    def __init__(self, encoder: nn.Module, encoder_out_dim: int, head_dropout: float):

        super().__init__()

        self.encoder = encoder
        self.head    = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(encoder_out_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:

        graph_rep = self.encoder(x, edge_index, edge_attr, batch)

        return self.head(graph_rep).squeeze(-1)


# =============================================
#              BASE MODEL
# =============================================

# Shared base class for pretrained GNN models (Hu et al. GIN, MolCLR, ...)
class _PretrainedGNN(BaseModel):

    # Checkpoint
    _CHECKPOINT_URL:       str
    _CACHE_FILE:           str
    _ARCHITECTURE:         str

    # Display name used in result tables
    _NAME:                 str

    # Feature space
    _ATOM_IDX_MAX:         int
    _ADD_HS:               bool                   = False

    # Encoder
    _ENCODER_CLS:          type[nn.Module]
    _ENCODER_OUT_DIM:      int                    = 300

    # Checkpoint keys to drop before loading (e.g. contrastive projection heads)
    _STRIP_KEY_PREFIXES:   tuple[str, ...]        = ()

    def __init__(
        self,
        encoder_dropout: float       = 0.0,          # Dropout inside GNN encoder layers (pretrained at 0.0)
        head_dropout:    float       = 0.1,          # Dropout in prediction head
        lr_encoder:      float       = 2e-5,         # AdamW learning rate for pretrained encoder
        lr_head:         float       = 1e-4,         # AdamW learning rate for prediction head
        weight_decay:    float       = 1e-2,         # L2 regularization strength
        batch_size:      int         = 32,           # Number of graphs per mini-batch
        max_epochs:      int         = 200,          # Maximum training epochs
        patience:        int         = 15,           # Early stopping patience (epochs without val improvement)
        device:          str         = "auto",       # Device to use ("auto" | "cpu" | "cuda" | "mps")
        seed:            int         = 42,           # Random seed for weight initialization and data shuffling
        n_trials:        int         = 0,            # Optuna tuning trials (0 = no HPO)
        hpo_timeout:     int | None  = None,         # Max tuning time in seconds
        task_type:       str         = "regression", # Task type (regression/classification)
    ):

        self.encoder_dropout = encoder_dropout
        self.head_dropout    = head_dropout
        self.lr_encoder      = lr_encoder
        self.lr_head         = lr_head
        self.weight_decay    = weight_decay
        self.batch_size      = batch_size
        self.max_epochs      = max_epochs
        self.patience        = patience
        self.seed            = seed
        self.n_trials        = n_trials
        self.hpo_timeout     = hpo_timeout
        self.task_type       = task_type
        self.device          = self._resolve_device(device)

        self._net: _FineTunedGNN | None = None

    @property
    def name(self) -> str:
        return self._NAME

    # Persist model source details alongside metrics for reproducibility
    def benchmark_metadata(self) -> dict[str, object]:
        return {
            "pretrained_checkpoint_url":  self._CHECKPOINT_URL,
            "pretrained_checkpoint_file": self._CACHE_FILE,
            "pretrained_architecture":    self._ARCHITECTURE,
        }

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    # Effective batch size adapted to dataset size
    @staticmethod
    def _effective_bs(batch_size: int, n_samples: int) -> int:
        return min(batch_size, max(8, n_samples // 4))

    # Convert one SMILES to a Data object using the subclass-specific featurization
    def _smiles_to_graph(self, smiles: str) -> Data | None:
        return _smiles_to_indexed_graph(smiles, atom_idx_max=self._ATOM_IDX_MAX, add_hs=self._ADD_HS)

    # Convert SMILES to graphs, attach targets, filter invalid entries
    def _build_dataset(self, smiles_list: list[str], y: np.ndarray) -> tuple[list[Data], np.ndarray]:

        data_list  = []
        valid_mask = []

        for smi, yi in zip(smiles_list, y):

            g = self._smiles_to_graph(smi)

            if g is not None:
                g.y = torch.tensor([float(yi)], dtype=torch.float)
                data_list.append(g)
                valid_mask.append(True)

            else:
                valid_mask.append(False)

        return data_list, np.array(valid_mask)

    # Create encoder, load pretrained weights (optionally stripping projection keys), return on device
    def _load_encoder(self, drop_ratio: float) -> nn.Module:

        encoder    = self._ENCODER_CLS(drop_ratio=drop_ratio)
        pretrained = _download_pretrained_weights(self._CHECKPOINT_URL, self._CACHE_FILE, self.device)

        if self._STRIP_KEY_PREFIXES:
            pretrained = {
                k: v for k, v in pretrained.items()
                if not any(k.startswith(p) for p in self._STRIP_KEY_PREFIXES)
            }

        missing, _ = encoder.load_state_dict(pretrained, strict=False)

        if missing:
            logger.warning(f"Missing keys when loading pretrained {self.name}: {missing}")

        return encoder.to(self.device)

    # AdamW with separate lr groups: encoder (small) vs head (large)
    @staticmethod
    def _build_optimizer(net: _FineTunedGNN, lr_encoder: float, lr_head: float, weight_decay: float):

        return torch.optim.AdamW(
            [
                {"params": net.encoder.parameters(), "lr": lr_encoder},
                {"params": net.head.parameters(),    "lr": lr_head},
            ],
            weight_decay=weight_decay,
        )

    # Compute average val loss weighted by number of samples (uses pre-built loader)
    def _val_loss(self, loader: PyGDataLoader, criterion) -> float:

        self._net.eval()

        total, n = 0.0, 0

        with torch.no_grad():
            for batch in loader:
                batch   = batch.to(self.device)
                pred    = self._net(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                total  += criterion(pred, batch.y.view(-1)).item() * batch.num_graphs
                n      += batch.num_graphs

        return total / max(n, 1)

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    # Run Optuna TPE search using the validation set with a reduced epoch budget
    def _tune(self, X_train: list[str], y_train: np.ndarray, X_val: list[str], y_val: np.ndarray, log_fn=None) -> None:

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

        encoder = self._load_encoder(drop_ratio=0.0)

        buf = io.BytesIO()
        torch.save(encoder.state_dict(), buf)

        criterion = nn.BCEWithLogitsLoss() if self.task_type == "classification" else nn.MSELoss()

        def objective(trial: optuna.Trial) -> float:

            lr_encoder      = trial.suggest_float("lr_encoder",      1e-6, 1e-4, log=True)
            lr_head         = trial.suggest_float("lr_head",         1e-4, 1e-2, log=True)
            encoder_dropout = trial.suggest_float("encoder_dropout", 0.0,  0.5)
            head_dropout    = trial.suggest_float("head_dropout",    0.0,  0.3)
            batch_size      = trial.suggest_categorical("batch_size", [8, 16, 32])
            weight_decay    = trial.suggest_float("weight_decay",    1e-5, 1e-2, log=True)

            torch.manual_seed(seed + trial.number)

            buf.seek(0)
            encoder.load_state_dict(torch.load(buf, weights_only=True, map_location=device))
            encoder.drop_ratio = encoder_dropout

            net    = _FineTunedGNN(encoder, self._ENCODER_OUT_DIM, head_dropout).to(device)
            eff_bs = self._effective_bs(batch_size, n_train)
            gen    = torch.Generator().manual_seed(seed + trial.number)

            train_loader = PyGDataLoader(train_data, batch_size=eff_bs, shuffle=True,  generator=gen)
            val_loader   = PyGDataLoader(val_data,   batch_size=eff_bs, shuffle=False)

            optimizer     = self._build_optimizer(net, lr_encoder, lr_head, weight_decay)
            scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget)
            best_val_loss = float("inf")
            trial_start   = time.time()

            try:
                for step in range(budget):

                    net.train()
                    for batch in train_loader:

                        if time.time() - trial_start > _TRIAL_TIMEOUT_SEC:
                            raise optuna.TrialPruned()

                        batch = batch.to(device)
                        optimizer.zero_grad()
                        try:
                            criterion(net(batch.x, batch.edge_index, batch.edge_attr, batch.batch), batch.y.view(-1)).backward()
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            raise optuna.TrialPruned()
                        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                        optimizer.step()

                    scheduler.step()

                    net.eval()
                    total, n = 0.0, 0
                    with torch.no_grad():
                        for batch in val_loader:

                            if time.time() - trial_start > _TRIAL_TIMEOUT_SEC:
                                raise optuna.TrialPruned()

                            batch   = batch.to(device)
                            total  += criterion(net(batch.x, batch.edge_index, batch.edge_attr, batch.batch), batch.y.view(-1)).item() * batch.num_graphs
                            n      += batch.num_graphs

                    val_loss      = total / max(n, 1)
                    best_val_loss = min(best_val_loss, val_loss)

                    trial.report(val_loss, step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            finally:
                del net
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
        best                 = study.best_params
        self.lr_encoder      = best["lr_encoder"]
        self.lr_head         = best["lr_head"]
        self.encoder_dropout = best["encoder_dropout"]
        self.head_dropout    = best["head_dropout"]
        self.batch_size      = best["batch_size"]
        self.weight_decay    = best["weight_decay"]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    # Fit the model to the training data
    def fit(

        self,
        X_train: list[str],
        y_train: np.ndarray,
        X_val:   list[str] | None  = None,
        y_val:   np.ndarray | None = None,
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

        train_data, _ = self._build_dataset(X_train, y_train)

        if not train_data:
            raise ValueError("No valid graph samples found in X_train after SMILES conversion.")

        encoder   = self._load_encoder(self.encoder_dropout)
        self._net = _FineTunedGNN(encoder, self._ENCODER_OUT_DIM, self.head_dropout).to(self.device)

        eff_bs       = self._effective_bs(self.batch_size, len(train_data))
        generator    = torch.Generator().manual_seed(self.seed)
        train_loader = PyGDataLoader(train_data, batch_size=eff_bs, shuffle=True, generator=generator)
        optimizer    = self._build_optimizer(self._net, self.lr_encoder, self.lr_head, self.weight_decay)
        scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        criterion    = nn.BCEWithLogitsLoss() if self.task_type == "classification" else nn.MSELoss()

        val_loader = None
        if X_val is not None and y_val is not None:
            val_data, _ = self._build_dataset(X_val, y_val)
            if val_data:
                val_loader = PyGDataLoader(val_data, batch_size=eff_bs, shuffle=False)

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
                loss      = criterion(self._net(batch.x, batch.edge_index, batch.edge_attr, batch.batch), batch.y.view(-1))
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()

                train_total      += loss.item() * batch.num_graphs
                train_n          += batch.num_graphs
                train_grad_total += float(grad_norm)
                train_batches    += 1

            current_lr_encoder = optimizer.param_groups[0]["lr"]
            current_lr_head    = optimizer.param_groups[1]["lr"]
            scheduler.step()

            train_loss      = train_total      / max(train_n,       1)
            train_grad_norm = train_grad_total / max(train_batches, 1)

            if val_loader is not None:

                val_loss = self._val_loss(val_loader, criterion)

                if log_fn is not None:
                    log = {
                        "train_epoch":     epoch,
                        "train/loss":      train_loss,
                        "val/loss":        val_loss,
                        "train/lr":        current_lr_encoder,
                        "train/lr_head":   current_lr_head,
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
                    "train_epoch":     epoch,
                    "train/loss":      train_loss,
                    "train/lr":        current_lr_encoder,
                    "train/lr_head":   current_lr_head,
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

    # Predict property values for a list of SMILES (invalid SMILES yield NaN)
    def predict(self, X: list[str]) -> np.ndarray:

        assert self._net is not None, "Call fit() before predict()."

        dummy_y               = np.zeros(len(X))
        data_list, valid_mask = self._build_dataset(X, dummy_y)

        self._net.eval()

        loader = PyGDataLoader(data_list, batch_size=self.batch_size, shuffle=False)
        preds  = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds.append(self._net(batch.x, batch.edge_index, batch.edge_attr, batch.batch).cpu().numpy())

        preds_valid        = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
        result             = np.full(len(X), np.nan)
        result[valid_mask] = preds_valid

        return result
