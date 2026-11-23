# train_vae.py - trains one VAE and saves everything about the run.
#
# Run it from the project folder, for example:
#   python -m model.train_vae --run g-vae-cifar
#   python -m model.train_vae --run d-vae-cifar
#   python -m model.train_vae --run g-vae-fmnist
#   python -m model.train_vae --run g-vae-cifar --epochs 1     (quick test)
#
# The three runs:
#   g-vae-cifar   the GENERATOR for CIFAR-10, trained on all 50,000 CIFAR-10 train images (seed 42)
#   d-vae-cifar   the DETECTOR's anomaly-score VAE, trained ONLY on the 'ddpm_train' part
#                 (30,000 images) with a different seed (43), so the classifier's
#                 real images (the 'clf_train' part) stay unseen by it
#   g-vae-fmnist  the GENERATOR for Fashion-MNIST, trained on its 60,000 train images (seed 42)
#
# What it writes (nothing is overwritten except files of the same run):
#   configs/<run>.json        settings, written BEFORE training starts
#   weights/<run>.pt          the trained weights (ignored by git)
#   fakes/<run>_samples.png   a grid of 16 sample images (ignored by git)
#   results/train_<run>.json  numbers MEASURED during this run

import argparse
import time

import torch
from torchvision.utils import save_image

from model.data import load_cifar10, load_fmnist, load_part
from model.utils import (CONFIGS_DIR, FAKES_DIR, RESULTS_DIR, SEED, WEIGHTS_DIR,
                         count_params, get_device, save_json, set_seed)
from model.vae import ConvVAE, batch_loss

# These values are all "proposed, to tune".
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
LATENT_SIZE = 128

# For each run: which images, how many channels, which seed.
RUNS = {
    "g-vae-cifar": dict(dataset="cifar10", part="train (all 50,000)", channels=3, seed=SEED,
                        load=lambda: load_cifar10("train")),
    "d-vae-cifar": dict(dataset="cifar10", part="ddpm_train", channels=3, seed=SEED + 1,
                        load=lambda: load_part("ddpm_train")),
    "g-vae-fmnist": dict(dataset="fmnist", part="train (all 60,000)", channels=1, seed=SEED,
                         load=lambda: load_fmnist("train")),
}


def train_one_epoch(model, optimizer, images):
    """One pass over all images in random order. Returns the average loss."""
    model.train()
    order = torch.randperm(len(images), device=images.device)
    total = 0.0
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[order[i:i + BATCH_SIZE]]
        loss = batch_loss(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
    return total / len(images)


def save_sample_grid(model, run, channels):
    """Save 16 generated images as a 4x4 PNG."""
    FAKES_DIR.mkdir(parents=True, exist_ok=True)
    x = (model.sample(16, seed=SEED) + 1) / 2          # [-1,1] -> [0,1]
    if channels == 1:
        x = x[:, :, 2:30, 2:30]                        # cut the padding off: 32x32 -> 28x28
    save_image(x.clamp(0, 1), FAKES_DIR / f"{run}_samples.png", nrow=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=list(RUNS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    run, info = args.run, RUNS[args.run]

    images = info["load"]()
    save_json(CONFIGS_DIR / f"{run}.json", {           # written BEFORE training
        "run_id": run, "dataset": info["dataset"], "part": info["part"],
        "num_images": len(images), "seed": info["seed"], "epochs": args.epochs,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "latent_size": LATENT_SIZE, "note": "proposed, to tune",
    })

    set_seed(info["seed"])
    device = get_device()
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    images = images.to(device)
    model = ConvVAE(info["channels"], LATENT_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"run {run} | {len(images)} images | device: {gpu}", flush=True)

    seconds, loss = [], None
    for epoch in range(args.epochs):
        start = time.time()
        loss = train_one_epoch(model, optimizer, images)
        if device.type == "cuda":
            torch.cuda.synchronize()                   # wait for the GPU so the timing is honest
        seconds.append(time.time() - start)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {loss:.2f}  {seconds[-1]:.1f} s", flush=True)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weights_file = WEIGHTS_DIR / f"{run}.pt"
    torch.save(model.state_dict(), weights_file)
    save_sample_grid(model, run, info["channels"])
    save_json(RESULTS_DIR / f"train_{run}.json", {     # only numbers measured in this run
        "run_id": run, "epochs_run": args.epochs,
        "seconds_per_epoch": round(sum(seconds) / len(seconds), 2),
        "final_loss": round(loss, 2),                  # average training loss of the last epoch
        "parameter_count": count_params(model),
        "weights_size_mb": round(weights_file.stat().st_size / 1e6, 2),
        "device": gpu,
    })
    print("done", flush=True)


if __name__ == "__main__":
    main()
