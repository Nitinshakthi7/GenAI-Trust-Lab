# make_fakes.py - makes the FAKE images that the Detector learns from.
# A trained generator (the VAE or the diffusion model) makes brand-new images from
# random numbers. No CIFAR-10 picture is used to make them. They are saved as
# ordinary 0..255 pixel pictures in one file, in three parts:
#   train        20,000 fakes  (for training the classifier)
#   calibration   5,000 fakes  (for setting the thresholds)
#   test          5,000 fakes  (made now, but NOT used until the final run)
# Each part has its own seed (1000, 2000, 3000 + the generator's own seed 42),
# so the three parts never contain the same image. All numbers are "proposed, to tune".
#
# Run it from the project folder, for example:
#   python -m model.make_fakes --generator g-vae-cifar
#   python -m model.make_fakes --generator g-ddpm-cifar      (only after its training has finished)
#
# It writes fakes/<generator>_fakes.npz (ignored by git). Running it twice gives the same file.
# Other files read the fakes with load_fakes(generator, part) below.

import argparse
import hashlib
import json
import time

import numpy as np
import torch

from model.data import _to_range   # the same "0..255 -> -1..1" step the real images use
from model.ddpm import TinyUNet, ddim_sample, make_schedule
from model.utils import CONFIGS_DIR, FAKES_DIR, WEIGHTS_DIR, get_device
from model.vae import ConvVAE

PART_SIZES = {"train": 20000, "calibration": 5000, "test": 5000}
PART_SEEDS = {"train": 1000, "calibration": 2000, "test": 3000}   # the generator's own seed is added
DDIM_STEPS = 50
BATCH_SIZE = {"vae": 1000, "ddpm": 500}     # how many images are made at once
FAMILY = {"g-vae-cifar": "vae", "g-ddpm-cifar": "ddpm"}


def load_generator(name):
    """Build the generator and load its trained weights, the same way api/generators.py does.
    Returns (family, network, schedule, config). The schedule is None for the VAE."""
    config = json.loads((CONFIGS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    weights = torch.load(WEIGHTS_DIR / f"{name}.pt", map_location="cpu", weights_only=True)
    device = get_device()
    if FAMILY[name] == "vae":
        network, schedule = ConvVAE(3, config["latent_size"]), None
    else:
        network = TinyUNet(3, config["num_classes"])
        schedule = make_schedule(config["timesteps"], config["noise_schedule"], device)
    network.load_state_dict(weights)
    return FAMILY[name], network.to(device).eval(), schedule, config


def make_batch(family, network, schedule, n, seed):
    """n new images, shape (n,3,32,32), pixels in [-1,1]. Same seed = same images."""
    if family == "vae":
        return network.sample(n, seed)
    return ddim_sample(network, (n, 3, 32, 32), DDIM_STEPS, seed, schedule=schedule)


def to_uint8(images):
    """Floats in [-1,1], shape (N,3,32,32) -> pixels 0..255, shape (N,32,32,3)."""
    pixels = ((images.clamp(-1, 1) + 1) * 127.5).round().byte()
    return pixels.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def make_part(family, network, schedule, n, part_seed):
    """Make n images in batches. Every batch gets its own seed, so no two batches repeat."""
    batch = BATCH_SIZE[family]
    pieces = []
    for i, start in enumerate(range(0, n, batch)):
        images = make_batch(family, network, schedule, min(batch, n - start), part_seed * 1000 + i)
        pieces.append(to_uint8(images))
    return np.concatenate(pieces)


def to_float(pixels):
    """The way back: pixels 0..255, shape (N,32,32,3) -> floats in [-1,1], shape (N,3,32,32)."""
    return _to_range(pixels).permute(0, 3, 1, 2).contiguous()


def load_fakes(generator, part):
    """Read one part ('train', 'calibration' or 'test') of a fakes file.
    Returns floats in [-1,1], shape (N,3,32,32), like the real images."""
    with np.load(FAKES_DIR / f"{generator}_fakes.npz") as data:
        pixels = data[part]
    return to_float(pixels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True, choices=list(FAMILY))
    name = parser.parse_args().generator

    torch.set_num_threads(4)                      # keep the computer usable while other things run
    family, network, schedule, config = load_generator(name)
    print(f"generator {name} | device: {get_device()}", flush=True)

    seeds = {part: PART_SEEDS[part] + config["seed"] for part in PART_SIZES}
    arrays = {}
    for part, size in PART_SIZES.items():
        start = time.time()
        arrays[part] = make_part(family, network, schedule, size, seeds[part])
        print(f"{part}: shape {arrays[part].shape}  seed {seeds[part]}  {time.time() - start:.1f} s", flush=True)

    FAKES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(FAKES_DIR / f"{name}_fakes.npz", **arrays,
             **{f"seed_{part}": np.array(seed) for part, seed in seeds.items()})
    print("sha256 of the train array:", hashlib.sha256(arrays["train"].tobytes()).hexdigest())


if __name__ == "__main__":
    main()
