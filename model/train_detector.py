# train_detector.py - trains the Detector (real image or AI-made image?) in DEV MODE.
#
# Run it from the project folder, for example:
#   python -m model.train_detector --dev --generators g-vae-cifar
#   python -m model.train_detector --dev --generators g-vae-cifar --cnn-epochs 2   (quick test)
#   python -m model.train_detector --dev --generators g-vae-cifar --use-ddpm-features
#
# What it does, in order:
#   1. Loads REAL images (the 'clf_train' and 'calibration' parts of CIFAR-10) and FAKE
#      images (made earlier by model.make_fakes). The fakes are samples from a generator,
#      so no CIFAR-10 picture is used to make them. Reals and fakes are balanced.
#   2. Trains detector 1: logistic regression on a few numbers per image (model.features).
#   3. Trains detector 2: a small CNN on the raw pixels (model.cnn.SmallCNN), the baseline.
#   4. CALIBRATION: sets two thresholds on the 'calibration' part only.
#        high = the 95th percentile of the AI-scores of the REAL images (so at most 5% of
#               real images get the label 'ai'),
#        low  = the 5th percentile of the AI-scores of the FAKE images.
#        If low >= high, both are set to their midpoint (then there is no 'unclear' band).
#        Label: 'ai' if score >= high, 'real' if score <= low, otherwise 'unclear'.
#   5. DEV report: AUROC and accuracy per generator, label shares, leave-one-generator-out
#      and robustness (JPEG, blur, noise). All of it is on the CALIBRATION data, so it is
#      NOT a result. It only shows that the code works and what to tune.
#
# THE FINAL TEST SPLIT IS NEVER LOADED HERE (load_real below refuses it). The plan says:
# the test set is scored once per finished run and never used to choose anything.
# The --final flag exists but is refused in this batch.
#
# What it writes (ignored by git: weights/, fakes/):
#   configs/detector_dev.json     the settings, written BEFORE training
#   weights/detector_lr.joblib    the trained logistic regression
#   weights/cnn-baseline.pt       the trained CNN
#   results/detector_dev.json     numbers MEASURED in this run (and the dev warning)
# All numbers are "proposed, to tune".

import argparse
import io
import time

import joblib
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model.cnn import SmallCNN
from model.data import load_part
from model.features import (cached_features, ddpm_features, load_ddpm_detector,
                            load_vae_detector, vae_features)
from model.make_fakes import FAMILY, load_fakes, to_float, to_uint8
from model.utils import (CONFIGS_DIR, FAKES_DIR, RESULTS_DIR, SEED, WEIGHTS_DIR,
                         get_device, save_json, set_seed)

# These values are all "proposed, to tune".
CNN_EPOCHS = 10
CNN_BATCH_SIZE = 128
CNN_LEARNING_RATE = 1e-3
HIGH_PERCENTILE = 95        # of the REAL calibration scores
LOW_PERCENTILE = 5          # of the FAKE calibration scores
JPEG_QUALITY = 50
BLUR_SIGMA = 1.0
NOISE_STD = 0.05            # on the 0..1 pixel scale (that is 0.1 on the -1..1 scale)
DEV_NOTE = "dev: calibration data, not a result; final_test never touched"
PERTURBATIONS = ["jpeg_q50", "blur_sigma1.0", "noise_std0.05"]


# ------------------------------------------------------------- loading (safe parts only)

def load_real(part):
    """Real CIFAR-10 images of one part. Only the two dev parts are allowed."""
    assert part != "final_test", "dev mode must NEVER load the final test split"
    assert part in ("clf_train", "calibration"), f"part {part!r} is not allowed in dev mode"
    return load_part(part)


def load_fake(generator, part):
    """Fake images of one part. The fakes' 'test' part is not allowed in dev mode either."""
    assert part in ("train", "calibration"), f"fake part {part!r} is not allowed in dev mode"
    return load_fakes(generator, part)


def cut(images, limit):
    """Keep only the first `limit` images (0 = keep all). Used for quick code-path tests."""
    return images[:limit] if limit else images


def load_data(generators, limit):
    """All images the dev run needs, in a dict. The fakes for training are shared out evenly
    between the generators, so the fake total equals the number of real images."""
    real_train = cut(load_real("clf_train"), limit)
    share = len(real_train) // len(generators)
    return {
        "real_train": real_train,
        "real_cal": cut(load_real("calibration"), limit),
        "fake_train": {g: cut(load_fake(g, "train"), limit)[:share] for g in generators},
        "fake_cal": {g: cut(load_fake(g, "calibration"), limit) for g in generators},
    }


def load_models(use_ddpm):
    """The detector's own models that make the features."""
    models = {"vae": load_vae_detector()}
    if use_ddpm:
        try:
            models["ddpm"] = load_ddpm_detector()
        except Exception as error:
            print("diffusion features skipped (weights not available):", error)
    return models


# ------------------------------------------------------------------------- features

def get_features(images, models, name=None):
    """A table with one row per image (2 columns, or 6 with the diffusion features).
    If name is given, the numbers are cached in fakes/feat_<name>_*.npz."""
    tables = []
    for kind, fn, weights in [("vae", vae_features, "d-vae-cifar"),
                              ("ddpm", ddpm_features, "d-ddpm-cifar")]:
        if kind not in models:
            continue
        if name is None:
            tables.append(fn(models[kind], images))
        else:    # "t200_s20" = the noise level and steps of ddpm_features; part of the cache name
            tag = kind if kind == "vae" else "ddpm_t200_s20"
            tables.append(cached_features(f"{name}_{tag}", images, weights, fn, models[kind]))
    return np.concatenate(tables, axis=1)


def make_all_features(data, models):
    """Feature tables for every image set (same layout as `data`)."""
    return {
        "real_train": get_features(data["real_train"], models, "real_clf_train"),
        "real_cal": get_features(data["real_cal"], models, "real_calibration"),
        "fake_train": {g: get_features(x, models, f"{g}_train") for g, x in data["fake_train"].items()},
        "fake_cal": {g: get_features(x, models, f"{g}_calibration") for g, x in data["fake_cal"].items()},
    }


# ---------------------------------------------------------------- detector 1: logistic regression

def train_lr(real_features, fake_features):
    """Standardise the features, then fit a logistic regression (real = 0, fake = 1)."""
    x = np.concatenate([real_features] + list(fake_features))
    y = np.concatenate([np.zeros(len(real_features))] + [np.ones(len(f)) for f in fake_features])
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED, max_iter=1000))
    return model.fit(x, y)


def lr_scores(model, features):
    """AI-score between 0 and 1 for each row (higher = more likely AI-made)."""
    return model.predict_proba(features)[:, 1]


# ------------------------------------------------------------------- detector 2: the CNN

def train_cnn(images, labels, epochs):
    """Train SmallCNN on raw pixels. Returns (model, seconds per epoch, last epoch's loss)."""
    set_seed()
    device = get_device()
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    seconds, average = [], None
    for epoch in range(epochs):
        start = time.time()
        model.train()
        order = torch.randperm(len(images))
        total = 0.0
        for i in range(0, len(images), CNN_BATCH_SIZE):
            idx = order[i:i + CNN_BATCH_SIZE]
            loss = loss_fn(model(images[idx].to(device)).squeeze(1), labels[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        average = total / len(images)
        seconds.append(time.time() - start)
        print(f"cnn epoch {epoch + 1}/{epochs}  loss {average:.4f}  {seconds[-1]:.1f} s", flush=True)
    return model, seconds, average


@torch.no_grad()
def cnn_scores(model, images, batch=1000):
    """AI-score between 0 and 1 for each image."""
    model.eval()
    device = next(model.parameters()).device
    out = [torch.sigmoid(model(images[i:i + batch].to(device)).squeeze(1)).cpu()
           for i in range(0, len(images), batch)]
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- calibration and labels

def calibrate(real_scores, fake_scores):
    """Pick the two thresholds from the calibration scores. Returns (low, high)."""
    high = float(np.percentile(real_scores, HIGH_PERCENTILE))
    low = float(np.percentile(fake_scores, LOW_PERCENTILE))
    if low >= high:                      # the groups do not overlap: no 'unclear' band
        low = high = (low + high) / 2
    return low, high


def label_scores(scores, low, high):
    """'ai' if score >= high, 'real' if score <= low, otherwise 'unclear'."""
    labels = np.full(len(scores), "unclear", dtype=object)
    labels[scores <= low] = "real"
    labels[scores >= high] = "ai"        # set last, so 'ai' wins if low == high
    return labels


def label_shares(scores, low, high):
    """What fraction of the images got each label."""
    labels = label_scores(scores, low, high)
    return {name: round(float(np.mean(labels == name)), 4) for name in ("real", "ai", "unclear")}


def evaluate(real_scores, fake_scores):
    """AUROC and accuracy (a score of 0.5 or more counts as 'ai'); reals are 0, fakes are 1."""
    truth = np.concatenate([np.zeros(len(real_scores)), np.ones(len(fake_scores))])
    scores = np.concatenate([real_scores, fake_scores])
    return {"auroc": round(float(roc_auc_score(truth, scores)), 4),
            "accuracy_at_0.5": round(float(((scores >= 0.5) == truth).mean()), 4)}


def report_detector(real_scores, fake_scores):
    """The dev report of one detector. fake_scores is a dict: generator -> scores."""
    low, high = calibrate(real_scores, np.concatenate(list(fake_scores.values())))
    return {
        "low_threshold": round(low, 4), "high_threshold": round(high, 4),
        "real_images_labels": label_shares(real_scores, low, high),
        "per_generator": {g: {**evaluate(real_scores, s), "fake_images_labels": label_shares(s, low, high)}
                          for g, s in fake_scores.items()},
    }


# ------------------------------------------------- leave-one-generator-out and robustness

def leave_one_generator_out(feats):
    """Train the logistic regression WITHOUT one generator's fakes, then test it on the real
    images plus that generator's fakes (calibration part). `feats` is make_all_features' dict."""
    if len(feats["fake_train"]) < 2:
        return "not possible yet: needs two generators"
    out = {}
    for left_out in feats["fake_train"]:
        others = [f for g, f in feats["fake_train"].items() if g != left_out]
        model = train_lr(feats["real_train"], others)
        out[left_out] = evaluate(lr_scores(model, feats["real_cal"]),
                                 lr_scores(model, feats["fake_cal"][left_out]))
    return out


def perturb(images, kind):
    """A damaged copy of the images (floats in, floats out), like a re-saved or edited picture."""
    pixels = to_uint8(images)                                      # (N,32,32,3), 0..255
    if kind == "jpeg_q50":
        for i, p in enumerate(pixels):
            buffer = io.BytesIO()
            Image.fromarray(p).save(buffer, format="JPEG", quality=JPEG_QUALITY)
            pixels[i] = np.array(Image.open(buffer))
    elif kind == "blur_sigma1.0":
        blurred = gaussian_filter(pixels.astype(np.float32), sigma=(0, BLUR_SIGMA, BLUR_SIGMA, 0))
        pixels = blurred.round().clip(0, 255).astype(np.uint8)
    elif kind == "noise_std0.05":
        noise = np.random.RandomState(SEED).normal(0, NOISE_STD, pixels.shape)
        pixels = ((pixels / 255 + noise).clip(0, 1) * 255).round().astype(np.uint8)
    else:
        raise ValueError(f"unknown perturbation {kind}")
    return to_float(pixels)


def robustness(data, models, lr, cnn):
    """AUROC and accuracy of both detectors on damaged copies of the calibration images."""
    out = {}
    for kind in PERTURBATIONS:
        real = perturb(data["real_cal"], kind)
        real_scores = (lr_scores(lr, get_features(real, models)), cnn_scores(cnn, real))
        fake_scores = {}
        for g, images in data["fake_cal"].items():
            fakes = perturb(images, kind)
            fake_scores[g] = (lr_scores(lr, get_features(fakes, models)), cnn_scores(cnn, fakes))
        out[kind] = {name: {g: evaluate(real_scores[k], fake_scores[g][k]) for g in fake_scores}
                     for k, name in enumerate(["logistic_regression", "cnn_baseline"])}
    return out


# ------------------------------------------------------------------------------- main

def fake_seeds(generator):
    """The seeds that were used to make the fakes (saved inside the fakes file)."""
    with np.load(FAKES_DIR / f"{generator}_fakes.npz") as saved:
        return {part: int(saved[f"seed_{part}"]) for part in ("train", "calibration", "test")}


def write_config(args, data, models):
    """Save the settings BEFORE any training starts."""
    save_json(CONFIGS_DIR / "detector_dev.json", {
        "run_id": "detector_dev", "mode": "dev", "note": "proposed, to tune",
        "generators": args.generators,
        "fake_seeds": {g: fake_seeds(g) for g in args.generators},
        "real_parts_used": ["clf_train", "calibration"], "final_test_used": False,
        "images": {"real_train": len(data["real_train"]), "real_calibration": len(data["real_cal"]),
                   "fake_train_per_generator": {g: len(x) for g, x in data["fake_train"].items()},
                   "fake_calibration_per_generator": {g: len(x) for g, x in data["fake_cal"].items()}},
        "limit_images_per_set": args.limit,
        "features": {"vae": "d-vae-cifar: reconstruction error per pixel, KL",
                     "ddpm": "d-ddpm-cifar: error map mean, p50, p90, p99 (t_start 200, 20 steps)"
                             if "ddpm" in models else None},
        "logistic_regression": {"standardised": True, "random_state": SEED, "max_iter": 1000},
        "cnn": {"epochs": args.cnn_epochs, "batch_size": CNN_BATCH_SIZE, "optimizer": "Adam",
                "learning_rate": CNN_LEARNING_RATE, "loss": "BCEWithLogits", "seed": SEED},
        "threshold_rule": f"high = {HIGH_PERCENTILE}th percentile of the real calibration AI-scores; "
                          f"low = {LOW_PERCENTILE}th percentile of the fake calibration AI-scores; "
                          "if low >= high both become their midpoint",
        "perturbations": {"jpeg_quality": JPEG_QUALITY, "blur_sigma": BLUR_SIGMA,
                          "noise_std_on_0_to_1_scale": NOISE_STD},
        "dev_note": DEV_NOTE,
    })


def column_means(table):
    """The average of each feature column, rounded for the JSON file."""
    return [round(float(v), 4) for v in table.mean(0)]


def print_report(title, report):
    print(f"{title}: low {report['low_threshold']}  high {report['high_threshold']}  "
          f"| real images -> {report['real_images_labels']}")
    for g, r in report["per_generator"].items():
        print(f"   {g}: AUROC {r['auroc']}  accuracy {r['accuracy_at_0.5']}  | fake images -> {r['fake_images_labels']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="dev mode (the only mode allowed now)")
    parser.add_argument("--final", action="store_true", help="refused in this batch")
    parser.add_argument("--generators", nargs="+", required=True, choices=list(FAMILY))
    parser.add_argument("--use-ddpm-features", action="store_true")
    parser.add_argument("--cnn-epochs", type=int, default=CNN_EPOCHS)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N images of each set (quick test)")
    args = parser.parse_args()
    if args.final:
        parser.error("--final is refused in this batch: the final test split is scored only once, "
                     "after the dev results are agreed. Use --dev.")
    if not args.dev:
        parser.error("please add --dev (the only mode available now)")

    torch.set_num_threads(4)                 # keep the computer usable while other things run
    set_seed()
    run_start = time.time()
    data = load_data(args.generators, args.limit)
    models = load_models(args.use_ddpm_features)
    write_config(args, data, models)         # written BEFORE training
    seconds = {}

    start = time.time()
    feats = make_all_features(data, models)
    seconds["features"] = round(time.time() - start, 1)

    start = time.time()
    lr = train_lr(feats["real_train"], list(feats["fake_train"].values()))
    seconds["logistic_regression_fit"] = round(time.time() - start, 2)

    images = torch.cat([data["real_train"]] + list(data["fake_train"].values()))
    labels = torch.cat([torch.zeros(len(data["real_train"]))] +
                       [torch.ones(len(x)) for x in data["fake_train"].values()])
    cnn, cnn_seconds, cnn_loss = train_cnn(images, labels, args.cnn_epochs)
    seconds["cnn_per_epoch"] = [round(s, 1) for s in cnn_seconds]

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, WEIGHTS_DIR / "detector_lr.joblib")
    torch.save(cnn.state_dict(), WEIGHTS_DIR / "cnn-baseline.pt")

    start = time.time()
    scores = {
        "logistic_regression": (lr_scores(lr, feats["real_cal"]),
                                {g: lr_scores(lr, f) for g, f in feats["fake_cal"].items()}),
        "cnn_baseline": (cnn_scores(cnn, data["real_cal"]),
                         {g: cnn_scores(cnn, x) for g, x in data["fake_cal"].items()}),
    }
    reports = {name: report_detector(real, fakes) for name, (real, fakes) in scores.items()}
    seconds["scoring_and_calibration"] = round(time.time() - start, 1)

    start = time.time()
    loo = leave_one_generator_out(feats)
    seconds["leave_one_generator_out"] = round(time.time() - start, 2)
    start = time.time()
    robust = robustness(data, models, lr, cnn)
    seconds["robustness"] = round(time.time() - start, 1)
    seconds["total"] = round(time.time() - run_start, 1)

    print(f"\n=== {DEV_NOTE} ===")
    for name, report in reports.items():
        print_report(name, report)
    print("leave-one-generator-out:", loo)
    for kind, per_detector in robust.items():
        for name, per_generator in per_detector.items():
            print(f"robustness {kind} {name}:", per_generator)
    print("seconds:", seconds)

    save_json(RESULTS_DIR / "detector_dev.json", {      # only numbers measured in this run
        "run_id": "detector_dev", "warning": DEV_NOTE, "generators": args.generators,
        "limit_images_per_set": args.limit, "feature_columns": feats["real_cal"].shape[1],
        "feature_means": {"real_calibration": column_means(feats["real_cal"]),
                          "fake_calibration": {g: column_means(f) for g, f in feats["fake_cal"].items()}},
        "cnn_epochs_run": args.cnn_epochs, "cnn_last_epoch_loss": round(cnn_loss, 4),
        "dev_report": reports, "leave_one_generator_out": loo, "robustness": robust,
        "seconds": seconds, "device": str(get_device()),
    })
    print("done")


if __name__ == "__main__":
    main()
