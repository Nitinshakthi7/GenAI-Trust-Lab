# train_auditor_run.py - one run of the Auditor: train a small diffusion model on a SMALL part of
# CIFAR-10, make images with it, and measure how many of them look like a training image.
#
# Run it from the project folder, for example:
#   python -m model.train_auditor_run --run a-500
#   python -m model.train_auditor_run --run a-10000
#   python -m model.train_auditor_run --run a-500-dup
#   python -m model.train_auditor_run --run a-500 --seed 2      (the same run with another training seed: run id a-500-s2)
#   python -m model.train_auditor_run --run a-500 --steps 20 --samples 16 --out-dir SOME_FOLDER   (quick test)
#
# Seeds (for error bars): the default --seed 42 is today's run, unchanged (run id a-500, sample seed 4242).
# Any other seed changes the training seed AND the sample seed (4242 + 100 x (seed - 42)) and adds "-s<seed>"
# to the run id (a-500-s2: configs/a-500-s2.json, weights/a-500-s2.pt, results/audit_a-500-s2.json).
# The training IMAGES stay the same (the subset is always drawn with seed 42); only the training randomness changes.
# EXTRA_SEEDS lists the seeds model/audit_summary.py looks for. POST /audit/check only uses the base runs (RUNS).
#
# The three runs (the training images are drawn from the 50,000 CIFAR-10 TRAIN images, seed 42):
#   a-500       500 different images
#   a-10000     10,000 different images
#   a-500-dup   500 training rows made of 350 images that appear once and 75 further images that
#               appear twice (so 30% of the rows are repeats). 425 different images in all.
# All three runs use the SAME number of optimiser steps, so only the size and the repeats differ.
#
# The model is the same as g-ddpm-cifar (model.ddpm.TinyUNet, cosine schedule, EMA, DDIM sampling).
# The measuring is in model/auditor.py. Held-out real images = the 'calibration' part only.
# Everything here is "proposed, to tune".
#
# What it writes (by default; with --out-dir FOLDER all three go straight into FOLDER):
#   configs/<run>.json          the settings, written BEFORE training starts
#   weights/<run>.pt            the EMA (averaged) weights
#   results/audit_<run>.json    only numbers MEASURED in this run

import argparse
import time

import numpy as np
import torch

from model.auditor import calibrate, closest_pairs, load_heldout, memorization_rate, nearest_distances
from model.data import load_cifar10
from model.ddpm import EMA, TinyUNet, ddim_sample, ddpm_loss, make_schedule
from model.utils import CONFIGS_DIR, RESULTS_DIR, SEED, WEIGHTS_DIR, get_device, save_json, set_seed

# These values are all "proposed, to tune".
STEPS = 6000              # optimiser steps (the conductor sets this from the measured speed)
MAX_BATCH = 128
LEARNING_RATE = 2e-4
EMA_DECAY = 0.999
TIMESTEPS = 1000
SCHEDULE = "cosine"
SAMPLES = 1000            # images generated and audited
SAMPLE_STEPS = 50         # DDIM steps
SAMPLE_SEED = 4242
SAMPLE_CHUNK = 250        # images made at once (keeps the GPU memory small)
PERCENTILE = 1.0          # the threshold is the 1st percentile of the held-out distances
PROGRESS_EVERY = 500

# For each run: how many images appear once ("once") and how many appear twice ("twice").
RUNS = {
    "a-500": dict(once=500, twice=0),
    "a-10000": dict(once=10000, twice=0),
    "a-500-dup": dict(once=350, twice=75),
}
EXTRA_SEEDS = (2, 3)      # the extra training seeds for error bars (run ids a-500-s2, a-500-s3, ...)


def run_id(run, seed=SEED):
    """The name a run's files get: the base name for the default seed (today's runs), else e.g. a-500-s2."""
    return run if seed == SEED else f"{run}-s{seed}"


def subset_rows(run):
    """Which CIFAR-10 train images (their numbers) make up the training rows of this run.
    The same every time: a fixed shuffle of 0..49999 (seed 42), and the first ones are taken."""
    shuffled = np.random.RandomState(SEED).permutation(50000)
    once, twice = RUNS[run]["once"], RUNS[run]["twice"]
    chosen = shuffled[:once + twice]
    return np.concatenate([chosen, chosen[once:]])      # the last `twice` images are listed twice


def duplicate_share(rows):
    """Percent of the rows that belong to an image appearing more than once."""
    counts = np.unique(rows, return_counts=True)[1]
    return 100.0 * counts[counts > 1].sum() / len(rows)


def make_samples(model, schedule, count, sample_seed=SAMPLE_SEED):
    """DDIM images from the averaged model, in chunks. The first chunk uses sample_seed (4242), the next 4243, ..."""
    pieces = []
    for i, start in enumerate(range(0, count, SAMPLE_CHUNK)):
        shape = (min(SAMPLE_CHUNK, count - start), 3, 32, 32)
        pieces.append(ddim_sample(model, shape, SAMPLE_STEPS, sample_seed + i, schedule=schedule))
    return torch.cat(pieces)


def summary(values, percentile):
    """min, median and one percentile of a list of distances (plain numbers)."""
    values = np.asarray(values)
    return {"min": round(float(values.min()), 3), "median": round(float(np.median(values)), 3),
            f"p{int(percentile)}": round(float(np.percentile(values, percentile)), 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=list(RUNS))
    parser.add_argument("--seed", type=int, default=SEED, help="training seed (default 42 = today's runs); another seed writes <run>-s<seed>")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--max-minutes", type=float, default=0,
                        help="stop training cleanly after about this many minutes (0 = no limit)")
    parser.add_argument("--out-dir", default=None,
                        help="put the config, weights and result all in this folder (default: configs/, weights/, results/)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (keeps the computer usable)")
    args = parser.parse_args()
    run, seed = run_id(args.run, args.seed), args.seed
    sample_seed = SAMPLE_SEED + 100 * (seed - SEED)     # seed 42 -> 4242 (today), seed 2 -> 242, seed 3 -> 342
    torch.set_num_threads(args.threads)

    if args.out_dir:
        config_file, weights_file, result_file = (f"{args.out_dir}/{name}" for name in (f"{run}.json", f"{run}.pt", f"audit_{run}.json"))
    else:
        config_file, weights_file, result_file = CONFIGS_DIR / f"{run}.json", WEIGHTS_DIR / f"{run}.pt", RESULTS_DIR / f"audit_{run}.json"

    # The training rows come from the CIFAR-10 TRAIN images only.
    rows = subset_rows(args.run)
    unique = np.unique(rows)
    train_all = load_cifar10("train")
    batch_size = min(MAX_BATCH, len(rows))
    save_json(config_file, {                           # written BEFORE training
        "run_id": run, "dataset": "cifar10", "part": "random subset of the 50,000 train images (seed 42)",
        "training_rows": len(rows), "unique_images": len(unique),
        "duplicate_share_percent": round(duplicate_share(rows), 2), "seed": seed,
        "steps": args.steps, "max_minutes": args.max_minutes, "batch_size": batch_size,
        "learning_rate": LEARNING_RATE, "optimizer": "Adam", "ema_decay": EMA_DECAY,
        "timesteps": TIMESTEPS, "noise_schedule": SCHEDULE, "samples": args.samples,
        "sample_steps": SAMPLE_STEPS, "sample_seed": sample_seed,
        "calibration_percentile": PERCENTILE, "heldout_part": "calibration", "note": "proposed, to tune",
    })

    set_seed(seed)
    device = get_device()
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    images = train_all[torch.from_numpy(rows)].to(device)         # the training rows (repeats included)
    schedule = make_schedule(TIMESTEPS, SCHEDULE, device)
    model = TinyUNet(3, 0).to(device)
    ema = EMA(model, EMA_DECAY)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"run {run} | {len(rows)} rows, {len(unique)} different images | steps {args.steps} | device: {gpu}", flush=True)

    # Training: every step takes a random batch of rows, so a repeated image is picked twice as often.
    losses, stopped_early, start = [], False, time.time()
    for step in range(1, args.steps + 1):
        batch = images[torch.randperm(len(images), device=device)[:batch_size]]
        loss = ddpm_loss(model, batch, schedule)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)
        losses.append(loss.item())          # a plain number: keeping a tiny GPU tensor per step pinned GPU memory (it grew until training crawled)
        if step % PROGRESS_EVERY == 0:
            average = sum(losses[-PROGRESS_EVERY:]) / PROGRESS_EVERY
            print(f"step {step}/{args.steps}  loss {average:.4f}  {(time.time() - start) / 60:.1f} min", flush=True)
        if args.max_minutes and time.time() - start > args.max_minutes * 60 and step < args.steps:
            print(f"time limit of {args.max_minutes} min reached, stopping cleanly", flush=True)
            stopped_early = True
            break
    if device.type == "cuda":
        torch.cuda.synchronize()                       # wait for the GPU so the timing is honest
    seconds, steps_run = time.time() - start, len(losses)
    final_loss = sum(losses[-100:]) / len(losses[-100:])          # the average of the last 100 steps
    torch.save(ema.model.state_dict(), weights_file)

    # Audit: make images, then measure the distance to the nearest training image.
    generated = make_samples(ema.model, schedule, args.samples, sample_seed)
    train_set = train_all[torch.from_numpy(unique)].to(device)    # each training image once
    heldout = load_heldout().to(device)                           # the calibration part only
    threshold, held_distances = calibrate(heldout, train_set, PERCENTILE)
    gen_distances, _ = nearest_distances(generated, train_set)
    flagged, rate = memorization_rate(gen_distances, threshold)

    save_json(result_file, {                           # only numbers measured in this run
        "run": run, "training_rows": len(rows), "unique_images": len(unique),
        "duplicate_share_percent": round(duplicate_share(rows), 2),
        "steps": steps_run, "steps_requested": args.steps, "stopped_by_time_limit": stopped_early,
        "seconds": round(seconds, 1), "seconds_per_step": round(seconds / steps_run, 4),
        "final_loss": round(final_loss, 4), "samples": len(gen_distances),
        "calibration_threshold": round(threshold, 3), "flagged": flagged, "memorization_rate": round(rate, 4),
        "distance_summary": {"generated": summary(gen_distances, 5), "heldout": summary(held_distances, 1)},
        "device": gpu,
        "examples": closest_pairs(generated, train_set, k=3),      # the 3 closest pairs (not chosen for looks)
    })
    print(f"done | threshold {threshold:.2f} | flagged {flagged} of {len(gen_distances)} | memorization_rate {rate:.4f}", flush=True)


if __name__ == "__main__":
    main()
