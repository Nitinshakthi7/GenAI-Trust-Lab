# run_final_detector.py - builds the FINAL Detector and scores the final test split exactly ONCE.
#
# In plain words. Two steps, run one after the other (from the project folder):
#
#   python -m model.run_final_detector --run-id detector_final --generators g-vae-cifar g-ddpm-cifar --phase train
#   python -m model.run_final_detector --run-id detector_final --generators g-vae-cifar g-ddpm-cifar --phase score-final
#   (add --activate to the score-final command to make the API use this detector afterwards)
#
# PHASE train  uses ONLY the real 'clf_train' and 'calibration' parts and the fakes' 'train' and 'calibration'
#   parts. It never opens the final test split. It does, in order:
#     1. makes the numbers (features) for every image: 2 from the Detector's VAE, 4 from its diffusion model
#     2. trains THREE detectors on real images against the fakes of ALL given generators:
#          lr_vae       logistic regression on the 2 VAE numbers
#          lr_vae_ddpm  logistic regression on all 6 numbers (VAE + diffusion)  <- the main one
#          cnn          the small CNN on raw pixels (the baseline)
#     3. sets two thresholds (low, high) per detector on the calibration part only (the dev rule of train_detector.py)
#     4. leave-one-generator-out: trains again WITHOUT one generator and tests on that generator (calibration
#        data only). Results on an unseen generator are findings, not promises.
#     5. robustness: JPEG, blur and noise on the calibration images
#     6. FREEZE: saves the three weight files, then configs/<run-id>.json (settings + SHA-256 of each weight
#        file) and results/<run-id>_train.json (everything measured, on calibration data: NOT the final result)
#
# PHASE score-final is the ONLY place that opens the final test split. It refuses to run unless the frozen
#   weight files still have exactly the SHA-256 written in the config (nothing changed after freezing) and
#   results/<run-id>_final.json does not exist yet. It never trains and never changes a threshold. It writes
#   results/<run-id>_final.json. One run id = one look at the final test.
#
# --dry-run is for tests only. It needs --out-dir (everything is written there, nothing in the project's own
#   configs/results/weights/fakes folders), allows --limit, allows the pseudo-generator 'g-vae-cifar-b' (the
#   same VAE as g-vae-cifar, sampled with other seeds, so there are two generators to test leave-one-out)
#   and, in score-final, it reads the CALIBRATION data instead of the final test data. Its files say
#   'DRY RUN, not a result'.
#
# All numbers are "proposed, to tune".

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from model import features as feature_module
from model.cnn import SmallCNN
from model.data import load_part
from model.features import load_ddpm_detector, load_vae_detector
from model.make_fakes import FAMILY, load_fakes, load_generator, make_part, to_float
from model.train_detector import (BLUR_SIGMA, CNN_BATCH_SIZE, CNN_EPOCHS, CNN_LEARNING_RATE, HIGH_PERCENTILE,
                                  JPEG_QUALITY, LOW_PERCENTILE, NOISE_STD, PERTURBATIONS, calibrate, cnn_scores,
                                  cut, evaluate, get_features, label_shares, load_real, lr_scores,
                                  perturb, train_cnn, train_lr)
from model.utils import FAKES_DIR, ROOT, SEED, WEIGHTS_DIR, get_device, save_json, set_seed

CPU_THREADS = 2                                    # keep the computer usable while other things run
DETECTORS = ("lr_vae", "lr_vae_ddpm", "cnn")       # the three detectors, in this order
WEIGHT_FILES = {"lr_vae": "{run}_lr_vae.joblib", "lr_vae_ddpm": "{run}_lr_vae_ddpm.joblib", "cnn": "{run}_cnn.pt"}
FEATURE_MODEL_FILES = ("d-vae-cifar.pt", "d-ddpm-cifar.pt")    # the Detector's own models that make the features
PSEUDO = {"g-vae-cifar-b": "g-vae-cifar"}          # dry run only: a second 'generator' made from the first
PSEUDO_SEEDS = {"train": 4000, "calibration": 5000, "test": 6000}
DRY_STAMP = "DRY RUN, not a result"
TRAIN_STAMP = "calibration data: not the final result"
LOGO_NOTE = "unseen generator results are findings, not promises"
FAIRNESS_CAVEAT = ("the Detector's diffusion model and the fake-making diffusion model share an architecture; "
                   "detection of that generator may look better than on a truly unseen generator")
VAE_CAVEAT = "VAE fakes are blurry and easy."
REFUSED_ALREADY_SCORED = ("the final test was already scored for this run id; "
                          "a new run id (new frozen config) is needed")


# ------------------------------------------------------------------ small helpers

def refuse(message, code=2):
    """Stop with a plain message and a non-zero exit code."""
    print("REFUSED:", message)
    sys.exit(code)


def folders(out_dir):
    """Where the files go: the project's own folders, or four sub-folders inside --out-dir (tests)."""
    base = Path(out_dir) if out_dir else ROOT
    return {name: base / name for name in ("configs", "results", "weights", "fakes")}


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_with_retry(load, what):
    """Run load(). A weight file that is being written right now can fail once: wait a bit and try again once."""
    for attempt in (1, 2):
        try:
            return load()
        except Exception as error:
            if attempt == 2:
                refuse(f"{what} is missing or cannot be read right now ({type(error).__name__}); try again later", 3)
            time.sleep(3)


def snapshot_feature_models(paths):
    """Dry run only: copy the Detector's two model files into --out-dir and use the copies, so a test does not
    depend on a file that another job (the diffusion training) may be rewriting right now."""
    paths["weights"].mkdir(parents=True, exist_ok=True)
    for name in FEATURE_MODEL_FILES:
        for attempt in range(3):
            shutil.copy2(WEIGHTS_DIR / name, paths["weights"] / name)
            try:
                torch.load(paths["weights"] / name, map_location="cpu", weights_only=True)
                break
            except Exception:                           # the original was half-written: wait and copy again
                time.sleep(3)
        else:
            refuse(f"weights/{name} cannot be read right now; try again later", 3)


def load_feature_models(paths):
    """The Detector's own VAE and diffusion model (they make the features), read from paths['weights']
    (the project's weights folder, or the snapshot in --out-dir for a dry run)."""
    feature_module.WEIGHTS_DIR = paths["weights"]
    return {"vae": load_with_retry(load_vae_detector, "weights/d-vae-cifar.pt"),
            "ddpm": load_with_retry(load_ddpm_detector, "weights/d-ddpm-cifar.pt")}


# ------------------------------------------------------------------ loading images

def make_pseudo_fakes(name, part, n):
    """Dry run only: n fakes from the real generator behind a pseudo-generator, with its own seeds."""
    family, network, schedule, config = load_generator(PSEUDO[name])
    return to_float(make_part(family, network, schedule, n, PSEUDO_SEEDS[part] + config["seed"]))


def load_fakes_of(generator, part, limit):
    """One part of a generator's fakes (a pseudo-generator is made on the spot)."""
    if generator in PSEUDO:
        return make_pseudo_fakes(generator, part, limit)
    return load_fakes(generator, part)


def seeds_of(generator):
    """The seeds the training and calibration fakes were made with, for the config file.
    (The test part's seed is not read in the train phase.)"""
    if generator in PSEUDO:
        return {part: PSEUDO_SEEDS[part] + 42 for part in ("train", "calibration")}
    with np.load(FAKES_DIR / f"{generator}_fakes.npz") as saved:
        return {part: int(saved[f"seed_{part}"]) for part in ("train", "calibration")}


def train_fakes(generator, part, limit):
    """Training-phase fakes: only the 'train' and 'calibration' parts are allowed here."""
    assert part in ("train", "calibration"), f"fake part {part!r} is not allowed in the train phase"
    return load_fakes_of(generator, part, limit)


def load_train_data(generators, limit):
    """All images the train phase needs. Each generator's training fakes are loaded up to the number of
    real training images (so leave-one-out can use a balanced set too)."""
    real_train = cut(load_real("clf_train"), limit)
    return {
        "real_train": real_train,
        "real_cal": cut(load_real("calibration"), limit),
        "fake_train": {g: cut(train_fakes(g, "train", limit), limit)[:len(real_train)] for g in generators},
        "fake_cal": {g: cut(train_fakes(g, "calibration", limit), limit) for g in generators},
    }


def balanced(per_generator, names, n_real):
    """The first (n_real // number of generators) items of each named generator: fakes and reals are equal in number."""
    share = n_real // len(names)
    return [per_generator[g][:share] for g in names]


# ------------------------------------------------------------------ features

def all_features(data, feature_models, cache_folder):
    """Feature tables (6 columns: 2 VAE + 4 diffusion) for every image set. They are cached in cache_folder,
    so a second run does not repeat the slow diffusion part."""
    feature_module.FAKES_DIR = cache_folder            # cached_features() writes its files here
    def table(images, name):
        return get_features(images, feature_models, name)
    return {
        "real_train": table(data["real_train"], "real_clf_train"),
        "real_cal": table(data["real_cal"], "real_calibration"),
        "fake_train": {g: table(x, f"{g}_train") for g, x in data["fake_train"].items()},
        "fake_cal": {g: table(x, f"{g}_calibration") for g, x in data["fake_cal"].items()},
    }


# ------------------------------------------------------------------ training and scoring

def train_detectors(real_images, fake_images, real_feats, fake_feats, epochs):
    """Train the three detectors on the real images against the fakes. fake_images and fake_feats are
    lists with one entry per generator. Columns 0-1 of the features are the VAE ones, 0-5 are all six."""
    lr_vae = train_lr(real_feats[:, :2], [f[:, :2] for f in fake_feats])
    lr_both = train_lr(real_feats, fake_feats)
    images = torch.cat([real_images] + list(fake_images))
    labels = torch.cat([torch.zeros(len(real_images))] + [torch.ones(len(x)) for x in fake_images])
    cnn = train_cnn(images, labels, epochs)[0]
    return {"lr_vae": lr_vae, "lr_vae_ddpm": lr_both, "cnn": cnn}


def score_images(models, images, features):
    """The AI-score (0 to 1) of each detector for these images. `features` is the 6-column table of the same images."""
    return {"lr_vae": lr_scores(models["lr_vae"], features[:, :2]),
            "lr_vae_ddpm": lr_scores(models["lr_vae_ddpm"], features),
            "cnn": cnn_scores(models["cnn"], images)}


def split_scores(models, real_images, real_feats, fake_images, fake_feats):
    """{detector: (real scores, {generator: fake scores})} for one set of real images and fakes."""
    real = score_images(models, real_images, real_feats)
    fakes = {g: score_images(models, x, fake_feats[g]) for g, x in fake_images.items()}
    return {d: (real[d], {g: fakes[g][d] for g in fakes}) for d in DETECTORS}


def thresholds_of(real_scores, fake_scores):
    """(low, high) by the dev rule, from the calibration scores. fake_scores is {generator: scores}."""
    return calibrate(real_scores, np.concatenate(list(fake_scores.values())))


def detector_report(real_scores, fake_scores, low, high):
    """AUROC, accuracy and the label shares (real / ai / unclear) per generator, for GIVEN thresholds."""
    return {
        "low_threshold": low, "high_threshold": high,
        "real_images_labels": label_shares(real_scores, low, high),
        "per_generator": {g: {**evaluate(real_scores, s), "fake_images_labels": label_shares(s, low, high)}
                          for g, s in fake_scores.items()},
    }


def calibration_reports(scores):
    """Thresholds from the calibration scores of ALL generators, and the report that goes with them."""
    reports = {}
    for d, (real, fakes) in scores.items():
        low, high = thresholds_of(real, fakes)
        reports[d] = detector_report(real, fakes, low, high)
    return reports


# ------------------------------------------------------------------ leave-one-generator-out

def unseen_result(real_scores, fake_scores, left_out, same):
    """One detector, one left-out generator. The thresholds come from the generators the detector HAS seen
    (like a real deployment). `same` = this detector's report from the model trained on ALL generators."""
    seen_fakes = {g: s for g, s in fake_scores.items() if g != left_out}
    low, high = thresholds_of(real_scores, seen_fakes)
    report = detector_report(real_scores, {left_out: fake_scores[left_out]}, low, high)
    unseen = report["per_generator"][left_out]
    same_generator = same["per_generator"][left_out]
    return {
        "unseen_generator": {**unseen, "low_threshold": low, "high_threshold": high,
                             "real_images_labels": report["real_images_labels"]},
        "same_generator": same_generator,
        "drop_same_minus_unseen": {k: round(same_generator[k] - unseen[k], 4) for k in ("auroc", "accuracy_at_0.5")},
    }


def leave_one_generator_out(data, feats, reports, epochs):
    """For each generator g: train all three detectors on the reals and the fakes of the OTHER generators only,
    then test on the real calibration images against g's calibration fakes. Calibration data only."""
    names = list(data["fake_train"])
    if len(names) < 2:
        return "not possible: needs two generators"
    n_real = len(data["real_train"])
    out = {}
    for left_out in names:
        others = [g for g in names if g != left_out]
        print(f"leave-one-generator-out: training without {left_out}", flush=True)
        models = train_detectors(data["real_train"], balanced(data["fake_train"], others, n_real),
                                 feats["real_train"], balanced(feats["fake_train"], others, n_real), epochs)
        scores = split_scores(models, data["real_cal"], feats["real_cal"], data["fake_cal"], feats["fake_cal"])
        out[left_out] = {d: unseen_result(*scores[d], left_out, reports[d]) for d in DETECTORS}
    return out


# ------------------------------------------------------------------ robustness

def damaged_scores(models, images, kind, feature_models):
    """Damage the images (JPEG, blur or noise), then score them with all three detectors."""
    damaged = perturb(images, kind)
    return score_images(models, damaged, get_features(damaged, feature_models))


def robustness(models, data, feature_models):
    """AUROC and accuracy of the three detectors on damaged copies of the calibration images, per generator."""
    out = {}
    for kind in PERTURBATIONS:
        real = damaged_scores(models, data["real_cal"], kind, feature_models)
        fakes = {g: damaged_scores(models, x, kind, feature_models) for g, x in data["fake_cal"].items()}
        out[kind] = {d: {g: evaluate(real[d], fakes[g][d]) for g in fakes} for d in DETECTORS}
    return out


# ------------------------------------------------------------------ freezing (weights + config + results)

def save_weights(models, run_id, paths):
    """Save the three weight files. Returns {detector: {'file': name, 'sha256': ...}} (the hash of the saved file)."""
    paths["weights"].mkdir(parents=True, exist_ok=True)
    info = {}
    for d in DETECTORS:
        name = WEIGHT_FILES[d].format(run=run_id)
        if d == "cnn":
            torch.save(models[d].state_dict(), paths["weights"] / name)
        else:
            joblib.dump(models[d], paths["weights"] / name)
        info[d] = {"file": name, "sha256": sha256_of(paths["weights"] / name)}
    return info


def feature_model_hashes(paths):
    """The SHA-256 of the Detector's own VAE and diffusion model files (they make the features)."""
    return {name: sha256_of(paths["weights"] / name) for name in FEATURE_MODEL_FILES}


def write_config(args, paths, data, reports, weight_info):
    """configs/<run-id>.json: the settings, the thresholds and the SHA-256 of every weight file (the 'freeze')."""
    save_json(paths["configs"] / f"{args.run_id}.json", {
        "run_id": args.run_id, "mode": "final", "note": "proposed, to tune",
        "stamp": DRY_STAMP if args.dry_run else None, "dry_run": args.dry_run,
        "generators": args.generators, "fake_seeds": {g: seeds_of(g) for g in args.generators}, "seed": SEED,
        "real_parts_used": ["clf_train", "calibration"], "final_test_used": False,
        "images": {"real_train": len(data["real_train"]), "real_calibration": len(data["real_cal"]),
                   "fake_train_per_generator": {g: len(x) for g, x in data["fake_train"].items()},
                   "fake_calibration_per_generator": {g: len(x) for g, x in data["fake_cal"].items()}},
        "limit_images_per_set": args.limit,
        "features": {"vae": "d-vae-cifar: reconstruction error per pixel, KL (2 numbers)",
                     "ddpm": "d-ddpm-cifar: t_start 200, 20 DDIM steps; mean, p50, p90, p99 of the per-pixel error (4 numbers)"},
        "detectors": {"lr_vae": "logistic regression on the 2 VAE numbers",
                      "lr_vae_ddpm": "logistic regression on all 6 numbers (the main detector)",
                      "cnn": "SmallCNN on raw pixels (the baseline)"},
        "logistic_regression": {"standardised": True, "random_state": SEED, "max_iter": 1000},
        "cnn": {"epochs": args.cnn_epochs, "batch_size": CNN_BATCH_SIZE, "optimizer": "Adam",
                "learning_rate": CNN_LEARNING_RATE, "loss": "BCEWithLogits", "seed": SEED},
        "threshold_rule": f"high = {HIGH_PERCENTILE}th percentile of the real calibration AI-scores; "
                          f"low = {LOW_PERCENTILE}th percentile of the fake calibration AI-scores (all generators); "
                          "if low >= high both become their midpoint",
        "thresholds": {d: {"low": reports[d]["low_threshold"], "high": reports[d]["high_threshold"]} for d in DETECTORS},
        "perturbations": {"jpeg_quality": JPEG_QUALITY, "blur_sigma": BLUR_SIGMA, "noise_std_on_0_to_1_scale": NOISE_STD},
        "weight_files": weight_info,
        "feature_model_files_sha256": feature_model_hashes(paths),
    })


def write_train_results(args, paths, feats, reports, logo, robust, seconds):
    """results/<run-id>_train.json: numbers MEASURED on the calibration data (not the final result)."""
    save_json(paths["results"] / f"{args.run_id}_train.json", {
        "run_id": args.run_id, "warning": DRY_STAMP if args.dry_run else TRAIN_STAMP,
        "also": TRAIN_STAMP, "dry_run": args.dry_run, "final_test_used": False,
        "generators": args.generators, "limit_images_per_set": args.limit,
        "feature_columns": int(feats["real_cal"].shape[1]), "cnn_epochs_run": args.cnn_epochs,
        "calibration_report": reports,
        "leave_one_generator_out": logo, "leave_one_generator_out_note": LOGO_NOTE,
        "robustness": robust, "seconds": seconds, "device": str(get_device()),
    })


def print_summary(reports, logo, robust):
    print(f"\n=== {TRAIN_STAMP} ===")
    for d, r in reports.items():
        print(f"{d}: low {r['low_threshold']:.4f}  high {r['high_threshold']:.4f}  | real -> {r['real_images_labels']}")
        for g, x in r["per_generator"].items():
            print(f"   {g}: AUROC {x['auroc']}  accuracy {x['accuracy_at_0.5']}  | fakes -> {x['fake_images_labels']}")
    print(f"leave-one-generator-out ({LOGO_NOTE}):")
    if isinstance(logo, str):
        print("  ", logo)
    else:
        for g, per_detector in logo.items():
            for d, x in per_detector.items():
                print(f"   without {g} | {d}: unseen AUROC {x['unseen_generator']['auroc']}, "
                      f"same AUROC {x['same_generator']['auroc']}, drop {x['drop_same_minus_unseen']}")
    for kind, per_detector in robust.items():
        for d, per_generator in per_detector.items():
            print(f"robustness {kind} {d}:", {g: r["auroc"] for g, r in per_generator.items()}, "(AUROC)")


# ------------------------------------------------------------------ PHASE train

def run_train(args, paths):
    if (paths["results"] / f"{args.run_id}_final.json").exists():
        refuse(REFUSED_ALREADY_SCORED)
    set_seed()
    seconds, start = {}, time.time()
    if args.dry_run:
        snapshot_feature_models(paths)
    data = load_train_data(args.generators, args.limit)
    feature_models = load_feature_models(paths)

    step = time.time()
    feats = all_features(data, feature_models, paths["fakes"])
    seconds["features"] = round(time.time() - step, 1)

    step = time.time()
    n_real = len(data["real_train"])
    models = train_detectors(data["real_train"], balanced(data["fake_train"], args.generators, n_real),
                             feats["real_train"], balanced(feats["fake_train"], args.generators, n_real), args.cnn_epochs)
    seconds["training_three_detectors"] = round(time.time() - step, 1)
    weight_info = save_weights(models, args.run_id, paths)       # saved now, so a crash later loses nothing

    scores = split_scores(models, data["real_cal"], feats["real_cal"], data["fake_cal"], feats["fake_cal"])
    reports = calibration_reports(scores)

    step = time.time()
    logo = leave_one_generator_out(data, feats, reports, args.cnn_epochs)
    seconds["leave_one_generator_out"] = round(time.time() - step, 1)
    step = time.time()
    robust = robustness(models, data, feature_models)
    seconds["robustness"] = round(time.time() - step, 1)
    seconds["total"] = round(time.time() - start, 1)

    write_config(args, paths, data, reports, weight_info)        # the freeze: written last, with the weight hashes
    write_train_results(args, paths, feats, reports, logo, robust, seconds)
    print_summary(reports, logo, robust)
    print("seconds:", seconds)
    print(f"frozen: {paths['configs'] / (args.run_id + '.json')}")
    print("train phase done" + (f"  [{DRY_STAMP}]" if args.dry_run else ""))


# ------------------------------------------------------------------ PHASE score-final

def check_frozen(args, paths):
    """Refuse unless the frozen config exists, and every weight file is EXACTLY as it was frozen
    (same SHA-256). Also checks the Detector's own two model files and that the thresholds in the config
    equal the ones in the train results. Returns the config."""
    config_path = paths["configs"] / f"{args.run_id}.json"
    if not config_path.exists():
        refuse(f"no frozen config {config_path.name}: run the train phase first", 3)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["dry_run"] != args.dry_run:
        refuse("this config is a dry run (or a real run) but this command is the other kind", 3)
    if config["generators"] != args.generators:
        refuse(f"--generators must be exactly the frozen ones: {config['generators']}", 3)
    for info in config["weight_files"].values():
        path = paths["weights"] / info["file"]
        if not path.exists():
            refuse(f"the frozen weight file {info['file']} is missing", 3)
        if sha256_of(path) != info["sha256"]:
            refuse(f"{info['file']} changed after freezing (its SHA-256 is different)", 3)
    for name, expected in config["feature_model_files_sha256"].items():
        if not (paths["weights"] / name).exists() or sha256_of(paths["weights"] / name) != expected:
            refuse(f"{name} (the Detector's own model) changed after freezing", 3)
    results_path = paths["results"] / f"{args.run_id}_train.json"
    if not results_path.exists():
        refuse(f"no {results_path.name}: run the train phase first", 3)
    reports = json.loads(results_path.read_text(encoding="utf-8"))["calibration_report"]
    for d in DETECTORS:
        if (config["thresholds"][d]["low"], config["thresholds"][d]["high"]) != (reports[d]["low_threshold"], reports[d]["high_threshold"]):
            refuse(f"the thresholds of {d} in the config and in {results_path.name} are different", 3)
    return config


def load_frozen(config, paths):
    """The three frozen detectors (call this only AFTER check_frozen has checked the hashes)."""
    files = {d: paths["weights"] / info["file"] for d, info in config["weight_files"].items()}
    cnn = SmallCNN()
    cnn.load_state_dict(torch.load(files["cnn"], map_location="cpu", weights_only=True))
    return {"lr_vae": joblib.load(files["lr_vae"]), "lr_vae_ddpm": joblib.load(files["lr_vae_ddpm"]),
            "cnn": cnn.to(get_device()).eval()}


def load_final_data(generators, dry_run, limit):
    """THE ONLY PLACE THAT OPENS THE FINAL TEST SPLIT: the real 'final_test' images and the fakes' 'test' arrays.
    In a dry run the CALIBRATION parts stand in for them, so the real final test is never read."""
    if dry_run:
        return (cut(load_real("calibration"), limit),
                {g: cut(load_fakes_of(g, "calibration", limit), limit) for g in generators})
    reals = load_part("final_test")
    fakes = {g: load_fakes_of(g, "test", 0) for g in generators}
    assert len(reals) == 5000 and all(len(x) == 5000 for x in fakes.values()), "the final test sets should have 5,000 images each"
    return reals, fakes


def write_new_json(path, data):
    """Write a JSON file that must NOT exist yet (mode 'x'): even two runs at the same moment cannot both write it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as file:
            file.write(json.dumps(data, indent=2) + "\n")
    except FileExistsError:
        refuse(REFUSED_ALREADY_SCORED)


def activate(args, paths):
    """Make the API use this run: write configs/detector_active.json (same keys as before, plus 'features')."""
    note = (f"{DRY_STAMP}" if args.dry_run else
            "final detector: trained on VAE and diffusion fakes with diffusion features; the final test split was "
            f"scored once (results/{args.run_id}_final.json); all numbers proposed, to tune")
    save_json(paths["configs"] / "detector_active.json", {
        "active_run": args.run_id, "thresholds_from": f"results/{args.run_id}_train.json",
        "features": ["vae", "ddpm"], "note": note})
    print(f"activated: {paths['configs'] / 'detector_active.json'}")


def run_score_final(args, paths):
    final_path = paths["results"] / f"{args.run_id}_final.json"
    if final_path.exists():
        refuse(REFUSED_ALREADY_SCORED)
    config = check_frozen(args, paths)                     # nothing is loaded or scored before these checks pass
    models = load_frozen(config, paths)
    feature_models = load_feature_models(paths)

    reals, fakes = load_final_data(args.generators, args.dry_run, args.limit)
    real_feats = get_features(reals, feature_models)
    fake_feats = {g: get_features(x, feature_models) for g, x in fakes.items()}
    scores = split_scores(models, reals, real_feats, fakes, fake_feats)
    detectors = {d: detector_report(real, fake_scores, config["thresholds"][d]["low"], config["thresholds"][d]["high"])
                 for d, (real, fake_scores) in scores.items()}       # the FROZEN thresholds: no change, no retraining

    write_new_json(final_path, {
        "run_id": args.run_id,
        "stamp": DRY_STAMP if args.dry_run else "final test split, scored ONCE with the frozen detectors and thresholds",
        "dry_run": args.dry_run, "final_test_used": not args.dry_run,
        "data_used": "CALIBRATION data standing in for the test data (dry run)" if args.dry_run
                     else "the real final_test images and the fakes' 'test' arrays",
        "generators": args.generators,
        "images": {"real": len(reals), "fake_per_generator": {g: len(x) for g, x in fakes.items()}},
        "frozen_thresholds": config["thresholds"],
        "weights_sha256_checked": {i["file"]: i["sha256"] for i in config["weight_files"].values()},
        "detectors": detectors,
        "caveats": [FAIRNESS_CAVEAT, VAE_CAVEAT, LOGO_NOTE],
    })
    print(f"=== {DRY_STAMP if args.dry_run else 'FINAL TEST, scored once'} ===")
    for d, report in detectors.items():
        print(f"{d}: real -> {report['real_images_labels']}")
        for g, x in report["per_generator"].items():
            print(f"   {g}: AUROC {x['auroc']}  accuracy {x['accuracy_at_0.5']}  | fakes -> {x['fake_images_labels']}")
    print(f"written: {final_path}")
    if args.activate:
        activate(args, paths)

# ------------------------------------------------------------------ the command line

def read_args():
    parser = argparse.ArgumentParser(description="Final Detector: train phase, then score-final (once).")
    parser.add_argument("--run-id", default="detector_final")
    parser.add_argument("--generators", nargs="+", required=True)
    parser.add_argument("--phase", required=True, choices=["train", "score-final"])
    parser.add_argument("--activate", action="store_true", help="score-final only: afterwards write configs/detector_active.json")
    parser.add_argument("--dry-run", action="store_true", help="tests only: needs --out-dir; the final test split is NOT read")
    parser.add_argument("--out-dir", help="tests only (with --dry-run): write everything here instead of the project folders")
    parser.add_argument("--limit", type=int, default=0, help="tests only (with --dry-run): use only the first N images of each set")
    parser.add_argument("--cnn-epochs", type=int, default=CNN_EPOCHS)
    args = parser.parse_args()
    allowed = list(FAMILY) + (list(PSEUDO) if args.dry_run else [])
    if any(g not in allowed for g in args.generators) or len(set(args.generators)) != len(args.generators):
        parser.error(f"--generators must be different names from {allowed}")
    if args.dry_run and not args.out_dir:
        parser.error("--dry-run needs --out-dir (so nothing is written into the project folders)")
    if args.dry_run and args.limit <= 0:
        parser.error("--dry-run needs --limit N (a tiny test: use 200 or less)")
    if not args.dry_run and (args.out_dir or args.limit):
        parser.error("--out-dir and --limit are only for --dry-run tests")
    if args.activate and args.phase != "score-final":
        parser.error("--activate only goes with --phase score-final")
    return args


def main():
    args = read_args()
    torch.set_num_threads(CPU_THREADS)
    paths = folders(args.out_dir)
    for g in args.generators:
        if g not in PSEUDO and not (FAKES_DIR / f"{g}_fakes.npz").exists():
            refuse(f"fakes/{g}_fakes.npz is missing: run  python -m model.make_fakes --generator {g}  first", 3)
    if args.phase == "train":
        run_train(args, paths)
    else:
        run_score_final(args, paths)


if __name__ == "__main__":
    main()
