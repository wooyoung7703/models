import argparse
import json
import os
import csv
import math
from typing import Dict, List, Tuple

import numpy as np

# Minimal sigmoid/logit helpers (avoid extra imports)
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def logit(p: np.ndarray) -> np.ndarray:
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _read_prob_file(path: str) -> Dict[int, float]:
    """Read a CSV with header id,prob and return mapping."""
    data: Dict[int, float] = {}
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if 'id' not in reader.fieldnames or 'prob' not in reader.fieldnames:
            raise ValueError(f"File {path} must contain headers: id,prob")
        for row in reader:
            try:
                i = int(row['id'])
                p = float(row['prob'])
            except Exception:
                continue
            data[i] = p
    return data


def _align_probs(files: Dict[str, str]) -> Tuple[List[int], np.ndarray, List[str]]:
    """Align probability files by common id intersection.
    Returns ids list, matrix shape (N, M) ordered by model key order, and model order list.
    """
    maps = {k: _read_prob_file(v) for k, v in files.items()}
    # intersection of ids
    common_ids = None
    for m in maps.values():
        ids = set(m.keys())
        common_ids = ids if common_ids is None else (common_ids & ids)
    if not common_ids:
        raise ValueError("No common ids across provided probability files")
    ordered_ids = sorted(common_ids)
    model_order = sorted(files.keys())
    mat = []
    for mid in model_order:
        m = maps[mid]
        mat.append([m[i] for i in ordered_ids])
    return ordered_ids, np.stack(mat, axis=1), model_order


def infer(args):
    # Load meta config
    with open(args.meta_config, 'r') as f:
        meta = json.load(f)
    ensemble = meta.get('ensemble', 'logistic')
    model_order = meta.get('model_order') or meta.get('models') or []
    if not model_order:
        raise ValueError("Meta config missing model_order/models")

    # Build file mapping
    prob_files: Dict[str, str] = {}
    for spec in args.base_probs:
        # expect format model=path
        if '=' not in spec:
            raise ValueError(f"Base prob spec must be model=path, got {spec}")
        name, path = spec.split('=', 1)
        if name not in model_order:
            raise ValueError(f"Model {name} not in meta model_order {model_order}")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        prob_files[name] = path

    ids, base_mat, order_read = _align_probs(prob_files)
    # reorder columns to match meta model_order
    if order_read != sorted(order_read):
        # base_mat constructed with sorted order; ensure we map correctly
        pass
    # permute if needed
    if order_read != model_order:
        col_index = [order_read.index(m) for m in model_order]
        base_mat = base_mat[:, col_index]

    # Combine
    if ensemble == 'logistic' and 'coef' in meta and 'intercept' in meta:
        coef = np.asarray(meta['coef']).reshape(-1)
        inter = float(meta['intercept'])
        if coef.shape[0] != base_mat.shape[1]:
            raise ValueError("Coefficient length mismatch")
        z = inter + np.sum(coef.reshape(1, -1) * logit(base_mat), axis=1)
        final_prob = sigmoid(z)
    elif ensemble == 'dynamic' and 'dynamic_weights' in meta:
        weights = meta['dynamic_weights']
        w = np.array([float(weights[m]) for m in model_order], dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(w) / len(w)
        z = np.sum(w.reshape(1, -1) * logit(base_mat), axis=1)
        final_prob = sigmoid(z)
    else:
        # fallback: simple mean
        final_prob = base_mat.mean(axis=1)

    # Threshold selection
    threshold = args.threshold
    if threshold is None:
        # try sidecar best threshold
        sidecar = os.path.splitext(args.meta_config)[0] + '.metrics.json'
        if os.path.exists(sidecar):
            try:
                with open(sidecar, 'r') as f:
                    metrics = json.load(f)
                threshold = float(metrics.get('best_threshold_precision', 0.5))
            except Exception:
                threshold = 0.5
        else:
            threshold = 0.5

    decisions = (final_prob >= threshold).astype(int)

    # Write output
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        wri = csv.writer(f)
        wri.writerow(['id','prob','decision'])
        for i, p, d in zip(ids, final_prob, decisions):
            wri.writerow([i, f"{p:.6f}", int(d)])

    # Optionally write summary json
    summary = {
        'meta_config': args.meta_config,
        'models_used': model_order,
        'threshold_used': threshold,
        'num_samples': len(ids),
        'positives_selected': int(decisions.sum()),
    }
    with open(os.path.splitext(args.output)[0] + '.summary.json', 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description='Stacking inference CLI')
    p.add_argument('--meta-config', type=str, required=True, help='Path to stacking_meta.json or stacking_oof_meta.json')
    p.add_argument('--base-probs', nargs='+', required=True, help='List like model=path/to/file.csv for each base model probability file')
    p.add_argument('--output', type=str, required=True, help='Output CSV path with id,prob,decision')
    p.add_argument('--threshold', type=float, default=None, help='Custom threshold override; if omitted uses best_threshold_precision sidecar or 0.5')
    args = p.parse_args()
    infer(args)

if __name__ == '__main__':
    main()
