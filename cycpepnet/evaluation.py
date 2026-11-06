from enum import Enum
from pathlib import Path
import numpy as np
from cycpepnet.benchmark import BenchmarkTask, DATA_ROOT, load_task_data
from cycpepnet.utils import (
    PHYSICOCHEMICAL_COLUMNS,
    get_processed_dataset_dir,
    load_hdf_fingerprints,
    load_map4c_fingerprints,
    load_map6c_fingerprints,
    load_morgan_r2_binary_fingerprints,
    load_morgan_r2_count_fingerprints,
    load_mxfp_fingerprints,
    load_pyg_graphs,
)
from cycpepnet.utils.metrics import evaluate, evaluate_classification
from cycpepnet.utils.provenance import table_identity
from cycpepnet.utils.splitting import assert_valid_split, load_split


# Representation types
class RepType(str, Enum):

    # Baseline
    PHYSCHEM_ONLY             = "physchem"                     # Physicochemical descriptors only (7)

    # ECFP4b (radius=2)
    MORGAN_BINARY_R2_PHYSCHEM = "morgan_binary_r2+physchem"    # ECFP4b + physicochemical (4096+7)
    MORGAN_BINARY_R2          = "morgan_binary_r2"             # ECFP4b only (4096)

    # ECFP4c
    MORGAN_COUNT_R2_PHYSCHEM  = "morgan_count_r2+physchem"     # ECFP4c + physicochemical (4096+7)
    MORGAN_COUNT_R2           = "morgan_count_r2"              # ECFP4c only (4096)

    # ECFP6b
    MORGAN_BINARY_PHYSCHEM    = "morgan_binary+physchem"       # ECFP6b + physicochemical (4096+7)
    MORGAN_BINARY             = "morgan_binary"                # ECFP6b only (4096)

    # ECFP6c
    MORGAN_COUNT_PHYSCHEM     = "morgan_count+physchem"        # ECFP6c + physicochemical (4096+7)
    MORGAN_COUNT              = "morgan_count"                 # ECFP6c only (4096)

    # MAP4C
    MAP4C_PHYSCHEM            = "map4c+physchem"               # MAP4C + physicochemical (2048+7)
    MAP4C                     = "map4c"                        # MAP4C only (2048)

    # MAP6C
    MAP6C_PHYSCHEM            = "map6c+physchem"               # MAP6C + physicochemical (2048+7)
    MAP6C                     = "map6c"                        # MAP6C only (2048)

    # MXFP
    MXFP_PHYSCHEM             = "mxfp+physchem"                # MXFP + physicochemical (217+7)
    MXFP                      = "mxfp"                         # MXFP only (217)

    # HDF
    HDF_PHYSCHEM              = "hdf+physchem"                 # HDF + physicochemical (2048+7)
    HDF                       = "hdf"                          # HDF only (2048)

    # Sequence and graph
    SMILES                    = "smiles"                       # SMILES
    FRAGSMILES                = "fragsmiles"                   # fragSMILES
    GRAPHS                    = "graphs"                       # PyG graph dicts

# Train and evaluate a model on a single framework task
def evaluate_model(

    task:         BenchmarkTask,
    model,
    rep_type:     RepType,
    split_file:   Path,
    data_root:    Path = DATA_ROOT,
    verbose:      bool = True,
    log_fn             = None,
    return_preds: bool = False,
    task_data          = None,

) -> dict[str, float | int] | tuple[dict[str, float | int], dict]:

    if task_data is None:
        task_data = load_task_data(task, data_root)

    df, fps_morgan, fps_morgan_count, orig_idx, _ = task_data
    dataset_dir = get_processed_dataset_dir(task.dataset, data_root=data_root)

    split, metadata = load_split(split_file)

    if metadata.get("task_smiles_sha256") not in (None, table_identity(df)):
        raise ValueError(f"Task row identity disagrees with split: {split_file}")

    assert_valid_split(split, len(df), split_file)
    train_idx = split["train"]
    val_idx   = split["val"]
    test_idx  = split["test"]

    # Physicochemical descriptor matrix of the task rows
    def _physchem():
        return df[list(PHYSICOCHEMICAL_COLUMNS)].values.astype(np.float32)

    if rep_type == RepType.MORGAN_BINARY_PHYSCHEM:

        physchem = _physchem()
        X_train = np.concatenate([fps_morgan[train_idx].astype(np.float32), physchem[train_idx]], axis=1)
        X_val   = np.concatenate([fps_morgan[val_idx].astype(np.float32),   physchem[val_idx]],   axis=1)
        X_test  = np.concatenate([fps_morgan[test_idx].astype(np.float32),  physchem[test_idx]],  axis=1)

    elif rep_type == RepType.MORGAN_COUNT_PHYSCHEM:

        assert fps_morgan_count is not None, f"Count FP not found for '{task.dataset}'"
        physchem = _physchem()
        X_train  = np.concatenate([fps_morgan_count[train_idx].astype(np.float32), physchem[train_idx]], axis=1)
        X_val    = np.concatenate([fps_morgan_count[val_idx].astype(np.float32),   physchem[val_idx]],   axis=1)
        X_test   = np.concatenate([fps_morgan_count[test_idx].astype(np.float32),  physchem[test_idx]],  axis=1)

    elif rep_type == RepType.MAP6C_PHYSCHEM:
        map6c_path = dataset_dir / "map6c_fp.npy"
        assert map6c_path.exists(), f"MAP6C FP not found for '{task.dataset}'"
        fps_map6c = load_map6c_fingerprints(map6c_path)[orig_idx]
        physchem  = _physchem()
        X_train   = np.concatenate([fps_map6c[train_idx].astype(np.float32), physchem[train_idx]], axis=1)
        X_val     = np.concatenate([fps_map6c[val_idx].astype(np.float32),   physchem[val_idx]],   axis=1)
        X_test    = np.concatenate([fps_map6c[test_idx].astype(np.float32),  physchem[test_idx]],  axis=1)

    elif rep_type == RepType.MAP4C_PHYSCHEM:
        map4c_path = dataset_dir / "map4c_fp.npy"
        assert map4c_path.exists(), f"MAP4C FP not found for '{task.dataset}'"
        fps_map4c = load_map4c_fingerprints(map4c_path)[orig_idx]
        physchem  = _physchem()
        X_train   = np.concatenate([fps_map4c[train_idx].astype(np.float32), physchem[train_idx]], axis=1)
        X_val     = np.concatenate([fps_map4c[val_idx].astype(np.float32),   physchem[val_idx]],   axis=1)
        X_test    = np.concatenate([fps_map4c[test_idx].astype(np.float32),  physchem[test_idx]],  axis=1)

    elif rep_type == RepType.MORGAN_BINARY:
        X_train = fps_morgan[train_idx].astype(np.float32)
        X_val   = fps_morgan[val_idx].astype(np.float32)
        X_test  = fps_morgan[test_idx].astype(np.float32)

    elif rep_type == RepType.MORGAN_COUNT:
        assert fps_morgan_count is not None, f"Count FP not found for '{task.dataset}'"
        X_train = fps_morgan_count[train_idx].astype(np.float32)
        X_val   = fps_morgan_count[val_idx].astype(np.float32)
        X_test  = fps_morgan_count[test_idx].astype(np.float32)

    elif rep_type == RepType.MAP6C:
        map6c_path = dataset_dir / "map6c_fp.npy"
        assert map6c_path.exists(), f"MAP6C FP not found for '{task.dataset}'"
        fps_map6c = load_map6c_fingerprints(map6c_path)[orig_idx]
        X_train   = fps_map6c[train_idx].astype(np.float32)
        X_val     = fps_map6c[val_idx].astype(np.float32)
        X_test    = fps_map6c[test_idx].astype(np.float32)

    elif rep_type == RepType.MAP4C:
        map4c_path = dataset_dir / "map4c_fp.npy"
        assert map4c_path.exists(), f"MAP4C FP not found for '{task.dataset}'"
        fps_map4c = load_map4c_fingerprints(map4c_path)[orig_idx]
        X_train   = fps_map4c[train_idx].astype(np.float32)
        X_val     = fps_map4c[val_idx].astype(np.float32)
        X_test    = fps_map4c[test_idx].astype(np.float32)

    elif rep_type == RepType.MORGAN_BINARY_R2_PHYSCHEM:
        r2b_path = dataset_dir / "morgan_r2_fp.npy"
        assert r2b_path.exists(), f"ECFP4b FP not found for '{task.dataset}'"
        physchem  = _physchem()
        fps_r2b   = load_morgan_r2_binary_fingerprints(r2b_path)[orig_idx]
        X_train   = np.concatenate([fps_r2b[train_idx].astype(np.float32), physchem[train_idx]], axis=1)
        X_val     = np.concatenate([fps_r2b[val_idx].astype(np.float32),   physchem[val_idx]],   axis=1)
        X_test    = np.concatenate([fps_r2b[test_idx].astype(np.float32),  physchem[test_idx]],  axis=1)

    elif rep_type == RepType.MORGAN_BINARY_R2:
        r2b_path = dataset_dir / "morgan_r2_fp.npy"
        assert r2b_path.exists(), f"ECFP4b FP not found for '{task.dataset}'"
        fps_r2b  = load_morgan_r2_binary_fingerprints(r2b_path)[orig_idx]
        X_train  = fps_r2b[train_idx].astype(np.float32)
        X_val    = fps_r2b[val_idx].astype(np.float32)
        X_test   = fps_r2b[test_idx].astype(np.float32)

    elif rep_type == RepType.MORGAN_COUNT_R2_PHYSCHEM:
        r2_path  = dataset_dir / "morgan_r2_count_fp.npy"
        assert r2_path.exists(), f"ECFP4c FP not found for '{task.dataset}'"
        physchem = _physchem()
        fps_r2   = load_morgan_r2_count_fingerprints(r2_path)[orig_idx]
        X_train  = np.concatenate([fps_r2[train_idx].astype(np.float32), physchem[train_idx]], axis=1)
        X_val    = np.concatenate([fps_r2[val_idx].astype(np.float32),   physchem[val_idx]],   axis=1)
        X_test   = np.concatenate([fps_r2[test_idx].astype(np.float32),  physchem[test_idx]],  axis=1)

    elif rep_type == RepType.MORGAN_COUNT_R2:
        r2_path = dataset_dir / "morgan_r2_count_fp.npy"
        assert r2_path.exists(), f"ECFP4c FP not found for '{task.dataset}'"
        fps_r2  = load_morgan_r2_count_fingerprints(r2_path)[orig_idx]
        X_train = fps_r2[train_idx].astype(np.float32)
        X_val   = fps_r2[val_idx].astype(np.float32)
        X_test  = fps_r2[test_idx].astype(np.float32)

    elif rep_type == RepType.PHYSCHEM_ONLY:
        physchem = _physchem()
        X_train  = physchem[train_idx]
        X_val    = physchem[val_idx]
        X_test   = physchem[test_idx]

    elif rep_type == RepType.MXFP_PHYSCHEM:
        mxfp_path = dataset_dir / "mxfp_fp.npy"
        assert mxfp_path.exists(), f"MXFP FP not found for '{task.dataset}'"
        physchem  = _physchem()
        fps_mxfp  = load_mxfp_fingerprints(mxfp_path)[orig_idx]
        X_train   = np.concatenate([fps_mxfp[train_idx], physchem[train_idx]], axis=1)
        X_val     = np.concatenate([fps_mxfp[val_idx],   physchem[val_idx]],   axis=1)
        X_test    = np.concatenate([fps_mxfp[test_idx],  physchem[test_idx]],  axis=1)

    elif rep_type == RepType.MXFP:
        mxfp_path = dataset_dir / "mxfp_fp.npy"
        assert mxfp_path.exists(), f"MXFP FP not found for '{task.dataset}'"
        fps_mxfp  = load_mxfp_fingerprints(mxfp_path)[orig_idx]
        X_train   = fps_mxfp[train_idx]
        X_val     = fps_mxfp[val_idx]
        X_test    = fps_mxfp[test_idx]

    elif rep_type == RepType.HDF_PHYSCHEM:
        hdf_path = dataset_dir / "hdf_fp.npy"
        assert hdf_path.exists(), f"HDF FP not found for '{task.dataset}': run process/combine step first"
        physchem  = _physchem()
        fps_hdf   = load_hdf_fingerprints(hdf_path)[orig_idx]
        X_train   = np.concatenate([fps_hdf[train_idx], physchem[train_idx]], axis=1)
        X_val     = np.concatenate([fps_hdf[val_idx],   physchem[val_idx]],   axis=1)
        X_test    = np.concatenate([fps_hdf[test_idx],  physchem[test_idx]],  axis=1)

    elif rep_type == RepType.HDF:
        hdf_path = dataset_dir / "hdf_fp.npy"
        assert hdf_path.exists(), f"HDF FP not found for '{task.dataset}': run process/combine step first"
        fps_hdf  = load_hdf_fingerprints(hdf_path)[orig_idx]
        X_train  = fps_hdf[train_idx]
        X_val    = fps_hdf[val_idx]
        X_test   = fps_hdf[test_idx]

    elif rep_type == RepType.SMILES:
        smiles  = df["SMILES"].tolist()
        X_train = [smiles[i] for i in train_idx]
        X_val   = [smiles[i] for i in val_idx]
        X_test  = [smiles[i] for i in test_idx]

    elif rep_type == RepType.FRAGSMILES:
        frag_path = dataset_dir / "fragsmiles.txt"
        assert frag_path.exists(), f"fragsmiles.txt not found for '{task.dataset}': re-run the process script for this dataset"
        all_frags  = frag_path.read_text().splitlines()
        frags      = [all_frags[i] for i in orig_idx]
        X_train    = [frags[i] for i in train_idx]
        X_val      = [frags[i] for i in val_idx]
        X_test     = [frags[i] for i in test_idx]

    elif rep_type == RepType.GRAPHS:
        all_graphs = load_pyg_graphs(dataset_dir / "graphs.pt")
        graphs     = [all_graphs[i] for i in orig_idx]
        X_train    = [graphs[i] for i in train_idx]
        X_val      = [graphs[i] for i in val_idx]
        X_test     = [graphs[i] for i in test_idx]

    else:
        raise ValueError(f"Unknown rep_type: {rep_type}")

    y       = df[task.target_column].values.astype(np.float32)
    y_train = y[train_idx]
    y_val   = y[val_idx]
    y_test  = y[test_idx]

    is_cls = task.task_type == "classification"

    if verbose:
        print(f"  [{model.name} | {rep_type.value}] task={task.name}")
        print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
        print("  Fitting...")

    model.fit(X_train, y_train, X_val, y_val, log_fn=log_fn)

    if is_cls:

        y_score_val    = model.predict_proba(X_val)
        valid_mask_val = ~np.isnan(y_score_val)
        metrics_val    = evaluate_classification(y_val[valid_mask_val], y_score_val[valid_mask_val])

        y_score    = model.predict_proba(X_test)
        valid_mask = ~np.isnan(y_score)
        metrics    = evaluate_classification(y_test[valid_mask], y_score[valid_mask])

        result = {
            "val_auroc"       : metrics_val["auroc"],
            "val_tpr"         : metrics_val["tpr"],
            "val_tnr"         : metrics_val["tnr"],
            "val_precision"   : metrics_val["precision"],
            "val_balanced_acc": metrics_val["balanced_acc"],
            **metrics,
            "n_train":      int(len(train_idx)),
            "n_val":        int(len(val_idx)),
            "n_test":       int(len(test_idx)),
            "n_valid_test": int(valid_mask.sum()),
        }

        if verbose:
            print(f"  val  AUROC = {metrics_val['auroc']:.4f}  BalAcc = {metrics_val['balanced_acc']:.4f}  TPR = {metrics_val['tpr']:.4f}  TNR = {metrics_val['tnr']:.4f}  Prec = {metrics_val['precision']:.4f}")
            print(f"  test AUROC = {metrics['auroc']:.4f}  BalAcc = {metrics['balanced_acc']:.4f}  TPR = {metrics['tpr']:.4f}  TNR = {metrics['tnr']:.4f}  Prec = {metrics['precision']:.4f}")

        if return_preds:
            rows = np.asarray(test_idx)[valid_mask]
            return result, {
                "y_true"      : y_test[valid_mask].astype(int),
                "task_row"    : rows,
                "original_row": orig_idx[rows],
                "smiles"      : df.iloc[rows]["SMILES"].to_numpy(dtype=str),
                "y_score"     : y_score[valid_mask],
            }

    else:

        y_pred_val      = model.predict(X_val)
        valid_mask_val  = ~np.isnan(y_pred_val)
        metrics_val     = evaluate(y_val[valid_mask_val], y_pred_val[valid_mask_val], y_train)

        y_pred     = model.predict(X_test)
        valid_mask = ~np.isnan(y_pred)
        metrics    = evaluate(y_test[valid_mask], y_pred[valid_mask], y_train)

        result = {
            "val_rmse":     metrics_val["rmse"],
            "val_qf3":      metrics_val["qf3"],
            **metrics,
            "n_train":      int(len(train_idx)),
            "n_val":        int(len(val_idx)),
            "n_test":       int(len(test_idx)),
            "n_valid_test": int(valid_mask.sum()),
        }

        if verbose:
            print(f"  val  RMSE = {metrics_val['rmse']:.4f}  QF3² = {metrics_val['qf3']:.4f}")
            print(f"  test RMSE = {metrics['rmse']:.4f}  QF3² = {metrics['qf3']:.4f}")

        if return_preds:
            rows = np.asarray(test_idx)[valid_mask]
            return result, {
                "y_true"      : y_test[valid_mask],
                "task_row"    : rows,
                "original_row": orig_idx[rows],
                "smiles"      : df.iloc[rows]["SMILES"].to_numpy(dtype=str),
                "y_pred"      : y_pred[valid_mask],
            }

    return result
