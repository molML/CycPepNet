import argparse
from pathlib import Path
from cycpepnet.benchmark import BENCHMARK_TASKS
from cycpepnet.models.baseline import MeanPredictor
from cycpepnet.models.fps import RandomForestModel, XGBoostModel
from cycpepnet.models.graphs import GCNModel, MPNNModel
from cycpepnet.models.pretrained import ChemBERTa2Model, MolCLRModel, PretrainedGINModel, PeptideCLM2Model
from cycpepnet.models.strings import LSTMModel, TransformerModel
from cycpepnet.evaluation import RepType, evaluate_model

_TASK_MAP  = {t.name: t for t in BENCHMARK_TASKS}
_MODEL_MAP = {
    "mean":           MeanPredictor,
    "rf":             RandomForestModel,
    "xgboost":        XGBoostModel,
    "lstm":           LSTMModel,
    "transformer":    TransformerModel,
    "gcn":            GCNModel,
    "mpnn":           MPNNModel,
    "chemberta2":     ChemBERTa2Model,
    "peptide_clm2":   PeptideCLM2Model,
    "pretrained_gin": PretrainedGINModel,
    "molclr":         MolCLRModel,
}
_REP_MAP = {r.value: r for r in RepType}

# Build argument parser for CLI
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and evaluate a model on a framework task")
    p.add_argument("--task",      required=True, choices=list(_TASK_MAP),  help="Framework task name")
    p.add_argument("--model",     required=True, choices=list(_MODEL_MAP), help="Model identifier")
    p.add_argument("--rep",       required=True, choices=list(_REP_MAP),   help="Representation type")
    p.add_argument("--split",     required=True, type=Path,                help="Path to split JSON file")
    p.add_argument("--seed",      default=42,    type=int,                 help="Random seed")
    p.add_argument("--n-trials",  default=0,     type=int,                 help="Optuna HPO trials (0 = no tuning, the default; use 50 to match the framework)")
    p.add_argument("--data-root", default="data/processed", type=Path)
    return p


if __name__ == "__main__":
    args         = _build_parser().parse_args()
    task         = _TASK_MAP[args.task]
    model_kwargs = {"n_trials": args.n_trials} if args.n_trials > 0 else {}
    model        = _MODEL_MAP[args.model](seed=args.seed, task_type=task.task_type, **model_kwargs)
    rep          = _REP_MAP[args.rep]

    evaluate_model(task, model, rep, args.split, data_root=args.data_root)
