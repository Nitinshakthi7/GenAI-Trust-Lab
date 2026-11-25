# generators.py - the list of image generators and the code that makes images with them.
#
# What is in here:
#   CATALOGUE       the list of generators (name, dataset, which settings each one accepts)
#   list_generators the answer for GET /generators (status is checked live from the files)
#   generate        the answer for POST /generate (checks the request, makes the images)
#
# How models are loaded: only when first needed ("lazily"), then kept in memory.
# The trained files in weights/ can be re-written while the API runs (training does that),
# so we remember the file's size and time, and load it again if either one changes.
# If the file cannot be read right now, we answer 503 ("try again soon") instead of crashing.

import base64
import io
import json
import threading
from typing import Optional

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pydantic import BaseModel, ConfigDict

from api.common import ApiError
from model.data import FMNIST_CLASSES
from model.ddpm import TinyUNet, ddim_sample, make_schedule
from model.utils import CONFIGS_DIR, WEIGHTS_DIR, count_params, get_device
from model.vae import ConvVAE

# Licence notes shown to the user (CIFAR-10 has no licence on its official page: to verify).
FMNIST_LICENCE = "Fashion-MNIST: MIT licence"
MNIST_LICENCE = "MNIST: CC BY-SA 3.0 licence"
CIFAR_LICENCE = "CIFAR-10: the official page states no licence (to verify)"

# The generators. "controls" lists the settings each one accepts (the website reads this).
# g-dcgan-cifar is documented only: it has no config and no weights, so its status is 'not_available'.
CATALOGUE = {
    "g-vae-cifar": dict(name="Conv VAE", dataset="CIFAR-10", family="vae", channels=3,
                        controls=["seed", "n", "interpolate"], licence=CIFAR_LICENCE),
    "g-vae-fmnist": dict(name="Conv VAE", dataset="Fashion-MNIST", family="vae", channels=1,
                         controls=["seed", "n", "interpolate"], licence=FMNIST_LICENCE),
    "g-cddpm-fmnist": dict(name="Conditional diffusion (DDIM)", dataset="Fashion-MNIST", family="cddpm",
                           channels=1, controls=["seed", "n", "steps", "class_id", "guidance", "show_steps"],
                           classes=FMNIST_CLASSES, licence=FMNIST_LICENCE),
    "g-vae-mnist": dict(name="Conv VAE", dataset="MNIST", family="vae", channels=1,
                        controls=["seed", "n", "interpolate"], licence=MNIST_LICENCE),
    "g-cddpm-mnist": dict(name="Conditional diffusion (DDIM)", dataset="MNIST", family="cddpm",
                          channels=1, controls=["seed", "n", "steps", "class_id", "guidance", "show_steps"],
                          classes=[str(digit) for digit in range(10)], licence=MNIST_LICENCE),
    "g-ddpm-cifar": dict(name="Diffusion (DDIM)", dataset="CIFAR-10", family="ddpm", channels=3,
                         controls=["seed", "n", "steps", "show_steps"], licence=CIFAR_LICENCE),
    "g-dcgan-cifar": dict(name="DCGAN", dataset="CIFAR-10", family="dcgan", channels=3,
                          controls=[], licence=CIFAR_LICENCE),
}

# Limits and defaults (proposed, to tune).
STEPS_ALLOWED = (20, 50, 100)
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 3.0          # the same strength the training script uses for its sample picture
MAX_SEED = 2_000_000_000
LABEL = "small model, expect low quality"

_cache = {}                     # generator id -> the loaded model and what we know about it
_load_lock = threading.Lock()   # one load at a time
# ponytail: no lock around generation, so two requests at once just share the CPU; add a queue if that hurts.


# ------------------------------------------------------------ the request shape

class Interpolate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed_b: int
    frames: int


class GenerateRequest(BaseModel):
    """The JSON body of POST /generate. A field left out (or null) means "not set"."""
    model_config = ConfigDict(extra="forbid")      # unknown field names are refused (catches typos)
    generator_id: str
    seed: int = 42
    n: int = 1
    steps: Optional[int] = None
    guidance: Optional[float] = None
    class_id: Optional[int] = None
    interpolate: Optional[Interpolate] = None
    show_steps: Optional[bool] = None             # diffusion only: also send 8 frames of the first image's denoising


# ------------------------------------------------------- loading models lazily

def _weights_path(gid):
    return WEIGHTS_DIR / f"{gid}.pt"


def _config_path(gid):
    return CONFIGS_DIR / f"{gid}.json"


def _build(gid):
    """Build the empty network for this generator from its config file. Returns (network, schedule)."""
    info = CATALOGUE[gid]
    config = json.loads(_config_path(gid).read_text(encoding="utf-8"))
    if info["family"] == "vae":
        return ConvVAE(info["channels"], config["latent_size"]), None
    if info["family"] in ("ddpm", "cddpm"):
        schedule = make_schedule(config["timesteps"], config["noise_schedule"], get_device())
        return TinyUNet(info["channels"], config["num_classes"]), schedule
    raise ValueError("this generator cannot be built")


def _load(gid):
    """Return the loaded model entry for gid. Raises ApiError(503) if it is not ready."""
    weights = _weights_path(gid)
    if not _config_path(gid).exists():
        raise ApiError(503, f"{gid} is not available (no config file)")
    if not weights.exists():
        raise ApiError(503, f"{gid} is still being trained (no weights file yet). Try again later.")
    stamp = (weights.stat().st_size, weights.stat().st_mtime_ns)
    with _load_lock:
        entry = _cache.get(gid)
        if entry and entry["stamp"] == stamp:
            return entry
        try:
            network, schedule = _build(gid)
            network.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            network = network.to(get_device()).eval()
        except Exception:
            _cache.pop(gid, None)
            raise ApiError(503, f"the weights of {gid} cannot be read right now "
                                "(they may be being written by training). Try again in a minute.")
        entry = dict(stamp=stamp, model=network, schedule=schedule, params=count_params(network))
        _cache[gid] = entry
        return entry


# ------------------------------------------------------------ GET /generators

def _describe(gid):
    info = CATALOGUE[gid]
    has_config = _config_path(gid).exists()
    path = _weights_path(gid)
    status, params = "not_available", None
    if has_config:
        try:
            params = _load(gid)["params"]
            status = "ready"
        except ApiError:
            status = "training"                  # no weights yet, or they are being written
            try:
                params = count_params(_build(gid)[0])
            except Exception:
                params = None
    return {
        "id": gid, "name": info["name"], "dataset": info["dataset"], "family": info["family"],
        "params": params,
        "size_mb": round(path.stat().st_size / 1e6, 2) if path.exists() else None,
        "status": status,
        "controls": info["controls"],
        "classes": info.get("classes", []),
        "licence_note": info["licence"],
    }


def list_generators():
    return {"generators": [_describe(gid) for gid in CATALOGUE],
            "note": "Small models trained for this project. Expect low quality."}


# -------------------------------------------------------------- POST /generate

def _check_request(req, info):
    """Check the request. Returns (steps, guidance) after filling in the defaults."""
    for control in ("steps", "guidance", "class_id", "interpolate", "show_steps"):
        if getattr(req, control) is not None and control not in info["controls"]:
            raise ApiError(400, f"{control} is not supported by {req.generator_id}")
    if not 0 <= req.seed <= MAX_SEED:
        raise ApiError(400, f"seed must be between 0 and {MAX_SEED}")
    if not 1 <= req.n <= 16:
        raise ApiError(400, "n must be between 1 and 16")
    if req.interpolate:
        if not 2 <= req.interpolate.frames <= 16:
            raise ApiError(400, "interpolate.frames must be between 2 and 16")
        if not 0 <= req.interpolate.seed_b <= MAX_SEED:
            raise ApiError(400, f"interpolate.seed_b must be between 0 and {MAX_SEED}")
    if req.class_id is not None and not 0 <= req.class_id <= 9:
        raise ApiError(400, "class_id must be between 0 and 9")
    steps = guidance = None
    if info["family"] in ("ddpm", "cddpm"):
        steps = DEFAULT_STEPS if req.steps is None else req.steps
        if steps not in STEPS_ALLOWED:
            raise ApiError(400, f"steps must be one of {list(STEPS_ALLOWED)}")
    if info["family"] == "cddpm":
        guidance = req.guidance
        if guidance is None:
            guidance = DEFAULT_GUIDANCE if req.class_id is not None else 0.0
        if not 0 <= guidance <= 10:
            raise ApiError(400, "guidance must be between 0 and 10")
        if guidance > 0 and req.class_id is None:
            raise ApiError(400, "guidance needs a class_id (guidance pushes the image towards that class)")
    return steps, guidance


def _make_vae(model, req):
    """Returns (images, seeds). With interpolate: a slide from seed to seed_b. Otherwise n images."""
    if req.interpolate:
        frames = req.interpolate.frames
        seeds = [req.seed] + [None] * (frames - 2) + [req.interpolate.seed_b]   # in-between frames are blends
        return model.interpolate(req.seed, req.interpolate.seed_b, frames), seeds
    seeds = [req.seed + i for i in range(req.n)]      # image i uses seed + i, so each one can be remade alone
    return torch.cat([model.sample(1, s) for s in seeds]), seeds


def _make_diffusion(entry, info, req, steps, guidance):
    """Returns (images, seeds, frames). One DDIM run per image, image i uses seed + i.
    frames: None, or with show_steps the 8 (step, image) frames of the first image."""
    seeds = [req.seed + i for i in range(req.n)]
    ids = None if req.class_id is None else torch.tensor([req.class_id])
    shape = (1, info["channels"], 32, 32)
    run = lambda s, keep: ddim_sample(entry["model"], shape, steps, s, class_ids=ids, guidance=guidance or 0.0,
                                      schedule=entry["schedule"], return_steps=keep)
    first, frames = run(seeds[0], True) if req.show_steps else (run(seeds[0], False), None)
    return torch.cat([first] + [run(s, False) for s in seeds[1:]]), seeds, frames


def _to_png_base64(image, gid, info, seed):
    """One image tensor (C,32,32) in [-1,1] -> PNG as plain base64 text, with a provenance note inside."""
    pixels = ((image.clamp(-1, 1) + 1) / 2 * 255).round().byte().cpu()
    if info["channels"] == 1:
        pixels = pixels[0, 2:30, 2:30]                # Fashion-MNIST: cut the padding off, 32x32 -> 28x28
    else:
        pixels = pixels.permute(1, 2, 0)              # (C,H,W) -> (H,W,C) for Pillow
    picture = Image.fromarray(pixels.contiguous().numpy())
    seed_text = "none (blend of two seeds)" if seed is None else str(seed)
    note = (f"model: {gid}; dataset: {info['dataset']}; seed: {seed_text}; "
            "made by GenAI Trust Lab; small model")
    meta = PngInfo()
    meta.add_text("provenance", note)
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG", pnginfo=meta)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def generate(req):
    """The answer for POST /generate. `req` is a GenerateRequest."""
    gid = req.generator_id
    if gid not in CATALOGUE:
        raise ApiError(404, f"unknown generator: {gid}")
    info = CATALOGUE[gid]
    steps, guidance = _check_request(req, info)       # bad requests are answered before any model is loaded
    entry = _load(gid)
    try:
        frames = None
        if info["family"] == "vae":
            batch, seeds = _make_vae(entry["model"], req)
        else:
            batch, seeds, frames = _make_diffusion(entry, info, req, steps, guidance)
        images = [{"png_base64": _to_png_base64(img, gid, info, s), "seed": s, "class_id": req.class_id}
                  for img, s in zip(batch, seeds)]
        if frames is not None:
            frames = [{"step": step, "png_base64": _to_png_base64(img[0], gid, info, seeds[0])} for step, img in frames]
    except ApiError:
        raise
    except Exception:
        raise ApiError(500, "making the images failed on the server")
    note = "Image number i uses seed + i, so any single image can be made again with n=1."
    if req.interpolate:
        note = "The first and last images come from seed and seed_b; the ones between are blends. n is ignored."
    answer = {
        "generator_id": gid,
        "images": images,
        "metadata": {"model": info["name"], "dataset": info["dataset"], "steps": steps,
                     "guidance": guidance, "label": LABEL},
        "note": note,
    }
    if frames is not None:              # only with show_steps, so the answer without it is unchanged
        answer["denoise_frames"] = frames
    return answer
