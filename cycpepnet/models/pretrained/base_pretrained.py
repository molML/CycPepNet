import io
import time
import numpy as np
import torch
import optuna
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from ..base import BaseModel

# Prune any HPO trial that exceeds this wallclock budget
_TRIAL_TIMEOUT_SEC = 1800


# =============================================
#              NETWORK DEFINITION
# =============================================

# Pretrained encoder + regression/classification head
class _FineTunedEncoder(nn.Module):

    def __init__(self, encoder, hidden_size: int, dropout: float, pool_fn):

        super().__init__()

        self.encoder = encoder
        self.pool_fn = pool_fn
        self.head    = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool_fn(out, attention_mask)

        return self.head(pooled).squeeze(-1)


# =============================================
#              BASE MODEL
# =============================================

class _PretrainedModel(BaseModel):

    _MODEL_ID         : str
    _MODEL_REVISION   : str | None = None
    _TRUST_REMOTE_CODE: bool = False

    def __init__(
        self,
        dropout:      float       = 0.1,
        lr_encoder:   float       = 2e-5,
        lr_head:      float       = 1e-4,
        weight_decay: float       = 1e-2,
        batch_size:   int         = 16,
        max_epochs:   int         = 200,
        patience:     int         = 15,
        max_length:   int         = 512,
        device:       str         = "auto",
        seed:         int         = 42,
        n_trials:     int         = 0,
        hpo_timeout:  int | None  = None,
        task_type:    str         = "regression",
    ):

        self.dropout      = dropout
        self.lr_encoder   = lr_encoder
        self.lr_head      = lr_head
        self.weight_decay = weight_decay
        self.batch_size   = batch_size
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.max_length   = max_length
        self.seed         = seed
        self.n_trials     = n_trials
        self.hpo_timeout  = hpo_timeout
        self.task_type    = task_type
        self.device       = self._resolve_device(device)

        self._net:        _FineTunedEncoder | None = None
        self._eff_bs:     int | None               = None
        self._tokenizer                            = None

    # Shared Hugging Face load kwargs so encoder/tokenizer stay pinned consistently
    def _hf_load_kwargs(self) -> dict[str, object]:

        kwargs: dict[str, object] = {
            "trust_remote_code": self._TRUST_REMOTE_CODE,
        }

        if self._MODEL_REVISION is not None:
            kwargs["revision"] = self._MODEL_REVISION

        return kwargs

    # Load encoder and return
    def _load_encoder(self) -> tuple:
        encoder = AutoModel.from_pretrained(self._MODEL_ID, **self._hf_load_kwargs())
        return encoder, encoder.config.hidden_size

    # Persist model source details alongside metrics for reproducibility
    def benchmark_metadata(self) -> dict[str, object]:
        return {
            "pretrained_model_id"   : self._MODEL_ID,
            "pretrained_revision"   : self._MODEL_REVISION,
            "pretrained_remote_code": self._TRUST_REMOTE_CODE,
        }

    # Extract pooled representation from encoder output (default: CLS token)
    def _pool_representation(self, out, attention_mask: torch.Tensor) -> torch.Tensor:
        return out.last_hidden_state[:, 0, :]

    # Effective batch size adapted to dataset size
    @staticmethod
    def _effective_bs(batch_size: int, n_samples: int) -> int:
        return min(batch_size, max(8, n_samples // 4))

    # Tokenize a batch of SMILES strings using the model's own HuggingFace tokenizer
    def _encode_batch(self, smiles: list[str]) -> tuple[torch.Tensor, torch.Tensor]:

        enc = self._tokenizer(
            smiles,
            padding        = True,
            truncation     = True,
            max_length     = self.max_length,
            return_tensors = "pt",
        )

        return enc["input_ids"], enc["attention_mask"]

    # Build DataLoader from SMILES list and targets
    def _make_loader(self, smiles: list[str], y: np.ndarray, shuffle: bool) -> DataLoader:

        input_ids, attn_mask = self._encode_batch(smiles)
        y_t                  = torch.tensor(y, dtype=torch.float32)
        generator            = torch.Generator().manual_seed(self.seed) if shuffle else None
        bs                   = self._eff_bs if self._eff_bs is not None else self.batch_size

        return DataLoader(
            TensorDataset(input_ids, attn_mask, y_t),
            batch_size = bs,
            shuffle    = shuffle,
            generator  = generator,
        )

    # Compute average val loss weighted by number of samples (uses pre-built loader)
    def _val_loss(self, loader: DataLoader, criterion) -> float:

        self._net.eval()

        total, n = 0.0, 0

        with torch.no_grad():
            for xb_ids, xb_mask, yb in loader:
                xb_ids, xb_mask, yb  = xb_ids.to(self.device), xb_mask.to(self.device), yb.to(self.device)
                total               += criterion(self._net(xb_ids, xb_mask), yb).item() * xb_ids.size(0)
                n                   += xb_ids.size(0)

        return total / max(n, 1)

    # AdamW with separate lr groups: encoder (small) vs head (large)
    def _build_optimizer(self, net: _FineTunedEncoder, lr_encoder: float, lr_head: float, weight_decay: float):

        return torch.optim.AdamW(
            [
                {"params": net.encoder.parameters(), "lr": lr_encoder},
                {"params": net.head.parameters(),    "lr": lr_head},
            ],
            weight_decay = weight_decay,
        )

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

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._MODEL_ID, **self._hf_load_kwargs())

        encoder, hidden_size = self._load_encoder()
        encoder              = encoder.to(device)

        buf = io.BytesIO()
        torch.save(encoder.state_dict(), buf)

        def _tok(smiles: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
            enc = self._tokenizer(
                smiles, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            return enc["input_ids"], enc["attention_mask"]

        ids_train, mask_train = _tok(X_train)
        ids_val,   mask_val   = _tok(X_val)
        y_train_t             = torch.tensor(y_train, dtype=torch.float32)
        y_val_t               = torch.tensor(y_val,   dtype=torch.float32)

        criterion = nn.BCEWithLogitsLoss() if self.task_type == "classification" else nn.MSELoss()
        pool_fn   = self._pool_representation

        def objective(trial: optuna.Trial) -> float:

            lr_encoder   = trial.suggest_float("lr_encoder",   1e-6, 1e-4, log=True)
            lr_head      = trial.suggest_float("lr_head",      1e-4, 1e-2, log=True)
            dropout      = trial.suggest_float("dropout",      0.0,  0.3)
            batch_size   = trial.suggest_categorical("batch_size", [8, 16, 32])
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

            torch.manual_seed(seed + trial.number)

            buf.seek(0)
            encoder.load_state_dict(torch.load(buf, weights_only=True, map_location=device))

            net    = _FineTunedEncoder(encoder, hidden_size, dropout, pool_fn=pool_fn).to(device)
            eff_bs = self._effective_bs(batch_size, n_train)
            gen    = torch.Generator().manual_seed(seed + trial.number)

            train_loader = DataLoader(
                TensorDataset(ids_train, mask_train, y_train_t),
                batch_size=eff_bs, shuffle=True, generator=gen,
            )
            val_loader = DataLoader(
                TensorDataset(ids_val, mask_val, y_val_t),
                batch_size=eff_bs, shuffle=False,
            )

            optimizer     = self._build_optimizer(net, lr_encoder, lr_head, weight_decay)
            scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget)
            best_val_loss = float("inf")
            trial_start   = time.time()

            try:
                for step in range(budget):

                    net.train()
                    for xb_ids, xb_mask, yb in train_loader:

                        if time.time() - trial_start > _TRIAL_TIMEOUT_SEC:
                            raise optuna.TrialPruned()

                        xb_ids, xb_mask, yb = xb_ids.to(device), xb_mask.to(device), yb.to(device)
                        optimizer.zero_grad()
                        try:
                            criterion(net(xb_ids, xb_mask), yb).backward()
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            raise optuna.TrialPruned()
                        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                        optimizer.step()

                    scheduler.step()

                    net.eval()
                    total, n = 0.0, 0
                    with torch.no_grad():
                        for xb_ids, xb_mask, yb in val_loader:

                            if time.time() - trial_start > _TRIAL_TIMEOUT_SEC:
                                raise optuna.TrialPruned()

                            xb_ids, xb_mask, yb  = xb_ids.to(device), xb_mask.to(device), yb.to(device)
                            total               += criterion(net(xb_ids, xb_mask), yb).item() * xb_ids.size(0)
                            n                   += xb_ids.size(0)

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
        best              = study.best_params
        self.lr_encoder   = best["lr_encoder"]
        self.lr_head      = best["lr_head"]
        self.dropout      = best["dropout"]
        self.batch_size   = best["batch_size"]
        self.weight_decay = best["weight_decay"]

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

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._MODEL_ID, **self._hf_load_kwargs())

        encoder, hidden_size = self._load_encoder()
        encoder              = encoder.to(self.device)

        self._eff_bs = self._effective_bs(self.batch_size, len(X_train))
        self._net    = _FineTunedEncoder(
            encoder, hidden_size, self.dropout, pool_fn=self._pool_representation,
        ).to(self.device)

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        optimizer    = self._build_optimizer(self._net, self.lr_encoder, self.lr_head, self.weight_decay)
        scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        criterion    = nn.BCEWithLogitsLoss() if self.task_type == "classification" else nn.MSELoss()

        val_loader = self._make_loader(X_val, y_val, shuffle=False) if X_val is not None and y_val is not None else None

        best_val_loss = float("inf")
        patience_cnt  = 0
        best_state    = None

        for epoch in range(self.max_epochs):

            self._net.train()

            train_total      = 0.0
            train_n          = 0
            train_grad_total = 0.0
            train_batches    = 0

            for xb_ids, xb_mask, yb in train_loader:

                xb_ids, xb_mask, yb = xb_ids.to(self.device), xb_mask.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss      = criterion(self._net(xb_ids, xb_mask), yb)
                loss.backward()
                grad_norm  = nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()

                train_total      += loss.item() * xb_ids.size(0)
                train_n          += xb_ids.size(0)
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

    # Predict property values
    def predict(self, X: list[str]) -> np.ndarray:

        assert self._net is not None, "Call fit() before predict()."

        self._net.eval()

        loader = self._make_loader(X, np.zeros(len(X)), shuffle=False)
        preds  = []

        with torch.no_grad():
            for xb_ids, xb_mask, _ in loader:
                preds.append(self._net(xb_ids.to(self.device), xb_mask.to(self.device)).cpu().numpy())

        return np.concatenate(preds)
