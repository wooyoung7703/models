import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TimestepAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        h = max(16, latent_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h), nn.ReLU(),
            nn.Linear(h, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h), nn.ReLU(),
            nn.Linear(h, input_dim)
        )
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


@dataclass
class AEPackage:
    model: TimestepAE
    mean: torch.Tensor
    std: torch.Tensor


def _standardize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    xz = (x - mean) / std
    return xz.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def train_timestep_ae(
    X: np.ndarray,
    latent_dim: int = 8,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "auto",
) -> AEPackage:
    assert X.ndim == 2, "Expected (N, F) input for timestep AE"
    xz, mean, std = _standardize(X)
    N, F = xz.shape
    ds = TensorDataset(torch.from_numpy(xz))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    dev = torch.device('cuda' if (device == 'auto' and torch.cuda.is_available()) else (device if device != 'auto' else 'cpu'))
    model = TimestepAE(F, latent_dim=latent_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    for _ in range(max(1, epochs)):
        for (xb,) in dl:
            xb = xb.to(dev).float()
            opt.zero_grad(set_to_none=True)
            xh, _ = model(xb)
            loss = crit(xh, xb)
            loss.backward(); opt.step()
    return AEPackage(model=model, mean=torch.from_numpy(mean).to(dev), std=torch.from_numpy(std).to(dev))


def save_ae(pkg: AEPackage, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'state_dict': pkg.model.state_dict(),
        'input_dim': pkg.model.input_dim,
        'latent_dim': pkg.model.latent_dim,
        'mean': pkg.mean.cpu(),
        'std': pkg.std.cpu(),
    }, path)


def load_ae(path: str, map_location: str | None = None) -> Optional[AEPackage]:
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location=map_location or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = TimestepAE(int(ck['input_dim']), latent_dim=int(ck.get('latent_dim', 8)))
    model.load_state_dict(ck['state_dict'])
    dev = next(model.parameters()).device
    mean = ck['mean'].to(dev)
    std = ck['std'].to(dev)
    return AEPackage(model=model.to(dev), mean=mean, std=std)


def transform_latent(pkg: AEPackage, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    assert X.ndim == 2, "Expected (N, F) input"
    model = pkg.model
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            chunk = torch.from_numpy(X[i:i+batch_size]).to(dev).float()
            chunk = (chunk - pkg.mean) / pkg.std
            _, z = model(chunk)
            out.append(z.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)
