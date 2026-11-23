# features.py - turns each image into a few numbers that the simple detector can learn from.
# Both kinds use the DETECTOR's own models (d-vae-cifar and d-ddpm-cifar). They were
# trained only on the 'ddpm_train' part, so they have never seen the classifier's images.
#
#   vae_features   2 numbers per image: how badly the VAE rebuilds it (mean squared
#                  error per pixel) and the KL term (how odd its latent numbers are).
#   ddpm_features  4 numbers per image: add a little noise (t_start = 200), let the
#                  diffusion model clean it up again, then look at the per-pixel
#                  error map: its mean and its 50th, 90th and 99th percentile.
#
# Images are tensors (N,3,32,32) with pixels in [-1,1]. The features come back as
# numpy arrays (N,2) or (N,4). cached_features() saves them to fakes/feat_<name>.npz
# so the slow part runs only once (the folder is ignored by git).
# All numbers are "proposed, to tune".

import hashlib
import json

import numpy as np
import torch

from model.ddpm import TinyUNet, reconstruct
from model.utils import CONFIGS_DIR, FAKES_DIR, WEIGHTS_DIR, get_device
from model.vae import ConvVAE, per_image_terms


def load_vae_detector():
    """The detector's VAE (d-vae-cifar), ready to use."""
    config = json.loads((CONFIGS_DIR / "d-vae-cifar.json").read_text(encoding="utf-8"))
    vae = ConvVAE(3, config["latent_size"])
    vae.load_state_dict(torch.load(WEIGHTS_DIR / "d-vae-cifar.pt", map_location="cpu", weights_only=True))
    return vae.to(get_device()).eval()


def load_ddpm_detector():
    """The detector's diffusion model (d-ddpm-cifar, EMA weights), ready to use."""
    ddpm = TinyUNet(3, 0)
    ddpm.load_state_dict(torch.load(WEIGHTS_DIR / "d-ddpm-cifar.pt", map_location="cpu", weights_only=True))
    return ddpm.to(get_device()).eval()


@torch.no_grad()
def vae_features(vae, images, batch=1000):
    """Two numbers per image: (reconstruction error per pixel, KL). Returns (N,2)."""
    vae.eval()
    device = next(vae.parameters()).device
    rows = []
    for i in range(0, len(images), batch):
        x = images[i:i + batch].to(device)
        mu, logvar = vae.encode(x)
        # The same two terms that ConvVAE.anomaly_score adds up, but kept apart here.
        recon_err, kl = per_image_terms(vae.decode(mu), x, mu, logvar)
        rows.append(torch.stack([recon_err / x[0].numel(), kl], dim=1).cpu())
    return torch.cat(rows).numpy()


@torch.no_grad()
def ddpm_features(ddpm, images, t_start=200, steps=20, batch=256):
    """Four numbers per image from the reconstruction error map. Returns (N,4):
    mean, 50th, 90th and 99th percentile of the per-pixel error."""
    ddpm.eval()
    device = next(ddpm.parameters()).device
    percents = torch.tensor([0.5, 0.9, 0.99], device=device)
    rows = []
    for i in range(0, len(images), batch):
        x = images[i:i + batch].to(device)
        error = reconstruct(ddpm, x, t_start, steps)[1].flatten(1)      # (n, 3072)
        spread = torch.quantile(error, percents, dim=1).T               # (n, 3)
        rows.append(torch.cat([error.mean(1, keepdim=True), spread], dim=1).cpu())
    return torch.cat(rows).numpy()


def cached_features(name, images, weights_name, compute, model):
    """Return compute(model, images), but load it from fakes/feat_<name>.npz if it is already there.
    The saved file remembers a fingerprint of the images and of the weights file; if either
    changed (for example the model was trained again), the features are made again."""
    weights = WEIGHTS_DIR / f"{weights_name}.pt"
    fingerprint = (hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()
                   + f"|{weights.stat().st_size}|{weights.stat().st_mtime_ns}")
    path = FAKES_DIR / f"feat_{name}.npz"
    if path.exists():
        with np.load(path) as saved:
            if str(saved["fingerprint"]) == fingerprint:
                return saved["features"]
    features = compute(model, images)
    FAKES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features, fingerprint=np.array(fingerprint))
    return features
