import argparse
import os
import json
from datetime import datetime, timezone

import numpy as np

from .ae import train_timestep_ae, save_ae

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-source', type=str, default='real', choices=['real','synthetic'])
    p.add_argument('--days', type=int, default=30)
    p.add_argument('--interval', type=str, default='1m')
    p.add_argument('--feature-set', type=int, default=16, choices=[16])
    p.add_argument('--latent-dim', type=int, default=8)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', type=str, default='backend/app/training/models/ae_timestep.pt')
    args = p.parse_args()

    # Build tabular features to train AE on per-timestep vectors
    if args.data_source == 'real':
        try:
            from .dataset_real import build_tabular_dataset_real
            from .sequence_features import SEQUENCE_FEATURES_16
            X, _, feats = build_tabular_dataset_real(days=args.days, interval=args.interval, feature_subset=SEQUENCE_FEATURES_16)
        except Exception:
            from .dataset import build_tabular_dataset
            X, _, feats = build_tabular_dataset(days=args.days, interval=args.interval)
    else:
        from .dataset import build_tabular_dataset
        X, _, feats = build_tabular_dataset(days=args.days, interval=args.interval)

    pkg = train_timestep_ae(X.astype(np.float32), latent_dim=args.latent_dim, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    save_ae(pkg, args.out)
    # Simple sidecar
    sidecar = os.path.splitext(args.out)[0] + '.json'
    with open(sidecar, 'w') as f:
        json.dump({
            'timestamp_utc': datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'input_dim': int(pkg.model.input_dim),
            'latent_dim': int(pkg.model.latent_dim),
            'feature_list': feats,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved AE -> {args.out}")

if __name__ == '__main__':
    main()
