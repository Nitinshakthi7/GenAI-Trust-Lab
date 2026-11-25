# audit.py - the real GET /audit/results: the results of the Auditor's memorization experiments.
#
# The numbers are measured by model/train_auditor_run.py and gathered into results/audit_results.json
# by model/audit_summary.py. This file only hands that file to the website:
#   - the file exists  -> answer with its content, unchanged (after checking it has the contract's field names;
#                         if it does not, or it cannot be read, the answer is 500 {"error": ...})
#   - the file is missing -> the placeholder: the two planned experiments, every memorization_rate null,
#                         no examples, and a note that starts with "[PLACEHOLDER - not measured yet]"
# For tests, the environment variable AUDIT_RESULTS_FILE can name another file.
# The page text must say the Auditor measures similarity between images; it does not decide legal copying.
#
# Also here: the real POST /audit/check. The user uploads one picture; we find the 3 training images of ONE Auditor run
# (form field `run`, default a-500) that are nearest to it, by the same pixel L2 distance the Auditor uses
# (model/auditor.py: 32x32x3 numbers scaled to -1..1). The flag is "possible_copy" only if the nearest distance is below
# that run's threshold (calibration_threshold in results/audit_<run>.json), else "not_flagged".
#   - the run's training images are rebuilt with model/train_auditor_run.py's own subset_rows (same seed, same pictures)
#     and kept in memory after the first request
#   - a run whose results file does not exist yet (still training) answers 503; an unknown run answers 400
#   - a picture that is not 32x32 is resized to 32x32 first (bilinear, like the detector) and the note says so
#   - CLIP similarity (model/auditor_clip.py) next to pixel L2 and pHash: each of the 3 nearest gets clip_similarity, plus
#     clip_nearest (the training image with the highest CLIP similarity), clip_flag ("possible_copy" above the run's CLIP
#     threshold, the 99th percentile of held-out calibration images, else "not_flagged") and clip_threshold. CLIP finds
#     copies moved by a few pixels or mirrored, which pixel L2 misses. The CLIP numbers of a run's training images are
#     cached under logs/; while a run has no cache yet, the first request starts building it in the background and the
#     CLIP fields are null (the note says so). flag stays the pixel flag, as before.
#   - only the three base runs (RUNS); the extra-seed runs (a-500-s2, ...) are for the results summary only

import base64
import io
import json
import os
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.fft import dct

from api.common import ApiError
from api.detect import open_picture
from model import auditor_clip
from model.audit_summary import POINT_FIELDS, build_summary
from model.data import load_cifar10
from model.make_fakes import to_float, to_uint8
from model.train_auditor_run import RUNS, subset_rows
from model.utils import RESULTS_DIR

RESULTS_FILE = RESULTS_DIR / "audit_results.json"
PLACEHOLDER_NOTE = ("[PLACEHOLDER - not measured yet] The Auditor runs have not been done yet. "
                    "Similarity between images only: it does not decide legal copying.")


def has_names(item, names):
    """True if item is a dict with exactly these field names."""
    return isinstance(item, dict) and set(item) == set(names)


def has_contract_shape(data):
    """True if data has the field names of GET /audit/results in api/contract.json (values are not checked)."""
    if not has_names(data, ["experiments", "examples", "note"]):
        return False
    if not (isinstance(data["experiments"], list) and isinstance(data["examples"], list) and isinstance(data["note"], str)):
        return False
    for experiment in data["experiments"]:
        if not (has_names(experiment, ["name", "x_label", "points"]) and isinstance(experiment["points"], list)):
            return False
        if not all(has_names(point, POINT_FIELDS) for point in experiment["points"]):
            return False
    return all(has_names(pair, ["generated_png_base64", "nearest_png_base64", "distance"]) for pair in data["examples"])


def audit_results():
    """The answer of GET /audit/results. Raises ApiError(500) if the results file is broken."""
    path = Path(os.environ.get("AUDIT_RESULTS_FILE") or RESULTS_FILE)
    if not path.exists():
        return build_summary({}, note=PLACEHOLDER_NOTE)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ApiError(500, "the audit results file cannot be read")
    if not has_contract_shape(data):
        raise ApiError(500, "the audit results file does not have the expected fields")
    return data


# ------------------------------------------------------------------ POST /audit/check

DEFAULT_RUN = "a-500"
NEAREST = 3                       # how many nearest training images come back
SIZE = 32                         # the Auditor works on 32 x 32 pictures

_train = {}                       # run -> its different training images, floats in -1..1, shape (K,3,32,32)
_load_lock = threading.Lock()     # only one request loads them
CLIP_AUTO_BUILD = True            # a run without a CLIP cache starts building it in the background (api/check_api.py turns it off)
_clip = {}                        # run -> (CLIP embeddings of its training images, CLIP threshold), once its cache exists
_clip_jobs = {}                   # run -> the background thread building its CLIP cache
_clip_lock = threading.Lock()


def run_settings(run):
    """(threshold, number of different training images) of this run, from results/audit_<run>.json.
    Raises ApiError: 400 unknown run, 503 no results file yet, 500 file broken."""
    if run not in RUNS:
        raise ApiError(400, f"unknown run '{run}': use one of {', '.join(RUNS)}")
    path = RESULTS_DIR / f"audit_{run}.json"
    if not path.exists():
        raise ApiError(503, f"audit run not ready: {run} has no results file yet (it may still be training)")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data["calibration_threshold"]), int(data["unique_images"])
    except (OSError, ValueError, KeyError, TypeError):
        raise ApiError(500, f"the results file of run {run} cannot be read")


def training_images(run):
    """The images the Auditor compared against in this run: each different training image once, chosen by
    model/train_auditor_run.py's own subset_rows (same seed, same split), as it did. Loaded once, then kept."""
    with _load_lock:
        if run not in _train:
            try:
                unique = np.unique(subset_rows(run))
                _train[run] = load_cifar10("train")[torch.from_numpy(unique)]
            except Exception:
                raise ApiError(503, "audit run not ready: the CIFAR-10 training images could not be loaded")
        return _train[run]


def clip_index(run):
    """(training embeddings, CLIP threshold) of this run, or None while its CLIP cache does not exist yet. The first
    request for such a run starts building the cache in the background (about 30 s per 1,000 images on the GPU)."""
    with _clip_lock:
        if run not in _clip:
            found = auditor_clip.run_index(run, build=False)
            if found is not None:
                _clip[run] = found
            elif CLIP_AUTO_BUILD and not (run in _clip_jobs and _clip_jobs[run].is_alive()):
                _clip_jobs[run] = threading.Thread(target=auditor_clip.run_index, args=(run,), daemon=True)
                _clip_jobs[run].start()
        return _clip.get(run)


def clip_part(run, pixels, train, distances):
    """(CLIP similarity to every training image or None, the CLIP fields of the answer, the CLIP sentence of the note)."""
    empty = {"clip_nearest": None, "clip_flag": None, "clip_threshold": None}
    try:
        index = clip_index(run)
        if index is None:
            return None, empty, (f" CLIP similarity is not ready yet for run {run} (it is prepared once per run and can take "
                                 "a few minutes); the pixel answer above does not need it.")
        embeddings, threshold = index
        if len(embeddings) != len(train):
            raise ValueError("the CLIP cache does not match the training images")
        sims = embeddings @ auditor_clip.embed(pixels[None])[0]          # cosine similarity to every training image
    except Exception:
        return None, empty, " CLIP similarity is not available right now; the pixel answer above does not need it."
    best = int(sims.argmax())
    top, threshold = round(float(sims[best]), 4), round(threshold, 4)       # the flag follows the numbers the answer shows
    fields = {"clip_nearest": {"clip_similarity": top, "distance": round(float(distances[best]), 4),
                               "image_png_base64": png_base64(to_uint8(train[best:best + 1])[0])},
              "clip_flag": "possible_copy" if top > threshold else "not_flagged",
              "clip_threshold": threshold}
    return sims, fields, (" CLIP similarity (0 to 1; higher means more similar in content, and it also finds images moved by a "
                          f"few pixels or mirrored) makes it a possible copy above {threshold:.3f}, the 99th percentile of real "
                          "images the model never saw.")


def phash(pixels):
    """64-bit DCT perceptual hash of a 32x32 RGB picture (uint8, 32 x 32 x 3): 64 True/False values."""
    grey = np.asarray(Image.fromarray(pixels).convert("L"), dtype=float)
    low = dct(dct(grey, axis=0), axis=1)[:8, :8]                # the 8 x 8 lowest frequencies
    return (low > np.median(low)).ravel()


def png_base64(pixels):
    """One 32 x 32 RGB picture (uint8) -> plain base64 PNG (no data: prefix)."""
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def audit_check(data, run):
    """The answer of POST /audit/check (see api/contract.json). The upload is already checked (type, 5 MB)."""
    picture = open_picture(data)                                # a bad picture is answered before anything is loaded
    threshold, expected = run_settings(run)
    train = training_images(run)
    if len(train) != expected:
        raise ApiError(500, f"run {run}: {len(train)} training images were rebuilt but its results file says {expected}")
    size = picture.size
    if size != (SIZE, SIZE):
        picture = picture.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    pixels = np.array(picture)                                  # 32 x 32 x 3, numbers 0..255
    query = to_float(pixels[None])                              # the Auditor's scale: -1..1
    distances = (train.flatten(1) - query.flatten(1)).norm(dim=1)       # L2 distance to every training image
    nearest = torch.argsort(distances)[:NEAREST].tolist()
    mine = phash(pixels)
    sims, clip_fields, clip_note = clip_part(run, pixels, train, distances)
    found = []
    for rank, (i, theirs) in enumerate(zip(nearest, to_uint8(train[nearest])), start=1):
        found.append({"rank": rank, "distance": round(float(distances[i]), 4),
                      "phash_distance": int((mine != phash(theirs)).sum()),
                      "clip_similarity": None if sims is None else round(float(sims[i]), 4), "image_png_base64": png_base64(theirs)})
    note = (f"Run {run}: your image was compared with {len(train)} different training images by pixel distance "
            "(a lower distance means more similar). It is a possible copy only if the nearest distance is below "
            f"{threshold:.2f}, the 1st percentile of the distances of real images the model never saw."
            f"{clip_note} This measures similarity only; it is not legal copying.")
    if size != (SIZE, SIZE):
        note += f" Your image was {size[0]}x{size[1]}, so it was resized to {SIZE}x{SIZE} first."
    flag = "possible_copy" if float(distances[nearest[0]]) < threshold else "not_flagged"
    return {"nearest": found, "flag": flag, **clip_fields, "note": note}
