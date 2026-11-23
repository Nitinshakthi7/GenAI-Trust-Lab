# train_ddpm.py - trains one diffusion model (DDPM) and saves everything about the run.
#
# Run it from the project folder, for example:
#   python -m model.train_ddpm --run g-cddpm-fmnist
#   python -m model.train_ddpm --run g-ddpm-cifar
#   python -m model.train_ddpm --run d-ddpm-cifar
#   python -m model.train_ddpm --run g-ddpm-cifar --epochs 1      (quick test)
#   python -m model.train_ddpm --run g-ddpm-cifar --max-minutes 35 (stop cleanly after ~35 min)
#
# The three runs:
#   g-cddpm-fmnist  the GENERATOR for Fashion-MNIST: class-conditional (10 labels) with
#                   classifier-free guidance, trained on the 60,000 train images (seed 42)
#   g-ddpm-cifar    the GENERATOR for CIFAR-10: no labels, trained on all 50,000
#                   CIFAR-10 train images (seed 42)
#   d-ddpm-cifar    the DETECTOR's reconstruction model: no labels, trained ONLY on the
#                   'ddpm_train' part (30,000 images) with a different seed (43), so the
#                   classifier's real images (the 'clf_train' part) stay unseen by it
#
# What it writes (nothing is overwritten except files of the same run):
#   configs/<run>.json        settings, written BEFORE training starts
#   weights/<run>.pt          the EMA (averaged) weights, saved every 5 epochs and at the end
#   fakes/<run>_samples.png   sample images (ignored by git)
#   results/train_<run>.json  numbers MEASURED during this run

import argparse
import time

import torch
from torchvision.utils import save_image

from model.data import load_cifar10, load_fmnist_labeled, load_part
from model.ddpm import EMA, TinyUNet, ddim_sample, ddpm_loss, make_schedule
from model.utils import (CONFIGS_DIR, FAKES_DIR, RESULTS_DIR, SEED, WEIGHTS_DIR,
                         count_params, get_device, save_json, set_seed)

# These values are all "proposed, to tune".
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
EMA_DECAY = 0.999
TIMESTEPS = 1000
SCHEDULE = "cosine"
LABEL_DROPOUT = 0.1     # how often a label is hidden while training (for guidance)
GUIDANCE = 3.0          # guidance strength for the sample picture (conditional model only)
SAMPLE_STEPS = 50       # DDIM steps for the sample picture
SAVE_EVERY = 5          # save the weights every this many epochs

# For each run: which images (and labels, if any), channels, number of classes, seed.
RUNS = {
    "g-cddpm-fmnist": dict(dataset="fmnist", part="train (all 60,000)", channels=1, classes=10,
                           seed=SEED, load=lambda: load_fmnist_labeled("train")),
    "g-ddpm-cifar": dict(dataset="cifar10", part="train (all 50,000)", channels=3, classes=0,
                         seed=SEED, load=lambda: (load_cifar10("train"), None)),
    "d-ddpm-cifar": dict(dataset="cifar10", part="ddpm_train", channels=3, classes=0,
                         seed=SEED + 1, load=lambda: (load_part("ddpm_train"), None)),
}


def train_one_epoch(model, ema, optimizer, schedule, images, labels):
    """One pass over all images in random order. Returns the average loss."""
    model.train()
    order = torch.randperm(len(images), device=images.device)
    total = torch.zeros((), device=images.device)
    for i in range(0, len(images), BATCH_SIZE):
        idx = order[i:i + BATCH_SIZE]
        ids = None if labels is None else labels[idx]
        loss = ddpm_loss(model, images[idx], schedule, ids, LABEL_DROPOUT)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)
        total += loss.detach() * len(idx)
    return total.item() / len(images)


def save_weights(ema, run):
    """Save the averaged (EMA) weights. Returns the file path."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHTS_DIR / f"{run}.pt"
    torch.save(ema.model.state_dict(), path)
    return path


def save_sample_grid(model, run, info):
    """Save a picture of generated images (DDIM, 50 steps).
    Labelled model: one row per class (8 images each) with guidance. Otherwise a 4x4 grid."""
    FAKES_DIR.mkdir(parents=True, exist_ok=True)
    if info["classes"]:
        ids = torch.arange(info["classes"]).repeat_interleave(8)   # 0,0,..,1,1,..
        x = ddim_sample(model, (len(ids), info["channels"], 32, 32), SAMPLE_STEPS, SEED,
                        class_ids=ids, guidance=GUIDANCE)
        nrow = 8
    else:
        x = ddim_sample(model, (16, info["channels"], 32, 32), SAMPLE_STEPS, SEED)
        nrow = 4
    x = (x + 1) / 2                                    # [-1,1] -> [0,1]
    if info["channels"] == 1:
        x = x[:, :, 2:30, 2:30]                        # cut the padding off: 32x32 -> 28x28
    save_image(x.clamp(0, 1), FAKES_DIR / f"{run}_samples.png", nrow=nrow)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=list(RUNS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-minutes", type=float, default=0,
                        help="stop cleanly after about this many minutes (0 = no limit)")
    parser.add_argument("--epochs-reason", default="set by hand on the command line",
                        help="one sentence on why this number of epochs, saved in the config")
    args = parser.parse_args()
    run, info = args.run, RUNS[args.run]

    images, labels = info["load"]()
    save_json(CONFIGS_DIR / f"{run}.json", {           # written BEFORE training
        "run_id": run, "dataset": info["dataset"], "part": info["part"],
        "num_images": len(images), "seed": info["seed"],
        "epochs": args.epochs, "epochs_reason": args.epochs_reason,
        "max_minutes": args.max_minutes, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "optimizer": "Adam",
        "ema_decay": EMA_DECAY, "timesteps": TIMESTEPS, "noise_schedule": SCHEDULE,
        "num_classes": info["classes"],
        "label_dropout": LABEL_DROPOUT if info["classes"] else None,
        "sample_guidance": GUIDANCE if info["classes"] else None,
        "sample_steps": SAMPLE_STEPS, "note": "proposed, to tune",
    })

    set_seed(info["seed"])
    device = get_device()
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    images = images.to(device)
    labels = None if labels is None else labels.to(device)
    schedule = make_schedule(TIMESTEPS, SCHEDULE, device)
    model = TinyUNet(info["channels"], info["classes"]).to(device)
    ema = EMA(model, EMA_DECAY)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"run {run} | {len(images)} images | device: {gpu} | parameters: {count_params(model):,}", flush=True)

    run_start = time.time()
    seconds, loss, stopped_early = [], None, False
    for epoch in range(args.epochs):
        start = time.time()
        loss = train_one_epoch(model, ema, optimizer, schedule, images, labels)
        if device.type == "cuda":
            torch.cuda.synchronize()                   # wait for the GPU so the timing is honest
        seconds.append(time.time() - start)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {loss:.4f}  {seconds[-1]:.1f} s", flush=True)
        if (epoch + 1) % SAVE_EVERY == 0:
            save_weights(ema, run)                     # so a crash does not lose everything
        elapsed = time.time() - run_start
        if args.max_minutes and elapsed + seconds[-1] > args.max_minutes * 60 and epoch + 1 < args.epochs:
            print(f"time limit of {args.max_minutes} min reached, stopping cleanly", flush=True)
            stopped_early = True
            break

    weights_file = save_weights(ema, run)
    save_sample_grid(ema.model, run, info)
    save_json(RESULTS_DIR / f"train_{run}.json", {     # only numbers measured in this run
        "run_id": run, "epochs_run": len(seconds),
        "stopped_by_time_limit": stopped_early,
        "seconds_per_epoch": round(sum(seconds) / len(seconds), 2),
        "total_minutes": round((time.time() - run_start) / 60, 2),
        "final_loss": round(loss, 4),                  # average training loss of the last epoch
        "parameter_count": count_params(model),
        "weights_size_mb": round(weights_file.stat().st_size / 1e6, 2),
        "device": gpu,
    })
    print("done", flush=True)


if __name__ == "__main__":
    main()
