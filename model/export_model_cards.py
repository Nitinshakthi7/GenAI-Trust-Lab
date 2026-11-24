# export_model_cards.py - writes the numbers for the website's "Model cards" (Limits page).
#
# What it does, in plain words:
#   It copies measured numbers out of the files in results/ into one small file,
#   web/public/results/model-cards.json, which the Limits page reads.
#   It never calculates, rounds or invents a number. A number that is not in a results
#   file is simply left out, and the page then shows "[PLACEHOLDER - not measured yet]".
#   A card whose results files do not exist yet (for example a-10000) is left out completely.
#
# Files it READS (all in results/, read only):
#   train_<card id>.json          training numbers of one model (epochs, loss, parameters, ...)
#   ddim_timing.json              how long DDIM sampling takes (for g-ddpm-cifar)
#   detector_final_final.json     final test split, scored once (the detectors)
#   detector_final_train.json     calibration split: unseen-generator test and robustness (the detectors)
#   detector_intervals.json       95% intervals; the final-test ones are formula-based (the detectors)
#   detector_blur_control.json    sharpness-only baseline and AUROC on blur-matched real images (calibration split)
#   audit_<card id>.json          one Auditor run (a-500, a-10000, a-500-dup)
# File it WRITES:
#   web/public/results/model-cards.json
#
# The page reads:  { "models": { "<card id>": { "script": "...", "config": "...",
#                  "measurements": { "<label>": <number>, ... } } } }
#   (it shows measurements only when script and config are both present; "source" is extra
#    and is ignored by the page)
#
# Run it from the project folder:  .venv/Scripts/python model/export_model_cards.py
# Only the Python standard library is used.

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "web" / "public" / "results" / "model-cards.json"

# (results key, label shown on the page) for a training file
TRAIN_NUMBERS = [
    ("epochs_run", "Epochs run"),
    ("seconds_per_epoch", "Seconds per epoch"),
    ("total_minutes", "Total training minutes"),
    ("final_loss", "Final training loss"),
    ("parameter_count", "Parameters"),
    ("weights_size_mb", "Weights file size (MB)"),
]

# the three detectors: id of the card -> (name inside the results files, short name used in labels)
DETECTORS = {
    "d-vae-cifar": ("lr_vae", "VAE-number detector"),
    "d-ddpm-cifar": ("lr_vae_ddpm", "VAE+diffusion detector"),
    "cnn-baseline": ("cnn", "CNN baseline"),
}
GENERATORS = ["g-vae-cifar", "g-ddpm-cifar"]
PERTURBATIONS = [("jpeg_q50", "JPEG quality 50"), ("blur_sigma1.0", "blur sigma 1.0"), ("noise_std0.05", "noise std 0.05")]

# (path inside the audit file, label shown on the page)
AUDIT_NUMBERS = [
    (("training_rows",), "Training rows"),
    (("unique_images",), "Different images in the training rows"),
    (("duplicate_share_percent",), "Repeated share of training rows (%)"),
    (("steps",), "Training steps"),
    (("seconds",), "Training seconds"),
    (("final_loss",), "Final training loss"),
    (("samples",), "Generated images checked"),
    (("calibration_threshold",), "Similarity cut-off (pixel distance, from held-out real images)"),
    (("flagged",), "Generated images flagged as similar (possible copy)"),
    (("memorization_rate",), "Share of generated images flagged (possible copy)"),
    (("distance_summary", "generated", "min"), "Generated: smallest distance to a training image"),
    (("distance_summary", "generated", "p5"), "Generated: 5th percentile distance"),
    (("distance_summary", "generated", "median"), "Generated: median distance"),
    (("distance_summary", "heldout", "min"), "Held-out real: smallest distance"),
    (("distance_summary", "heldout", "p1"), "Held-out real: 1st percentile distance"),
    (("distance_summary", "heldout", "median"), "Held-out real: median distance"),
]


@lru_cache(maxsize=None)
def load(name):
    """The contents of results/<name>, or None if the file is missing or unreadable."""
    try:
        return json.loads((RESULTS / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def new_card(script, config):
    return {"script": script, "config": config, "measurements": {}, "_files": []}


def value_at(filename, *path):
    """The number at `path` inside results/<filename>, or None if it is not there."""
    value = load(filename)
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return None if isinstance(value, bool) or not isinstance(value, (int, float)) else value


def put(card, label, filename, *path):
    """Copy the number at `path` inside results/<filename> into the card. Skip it if it is not there."""
    value = value_at(filename, *path)
    if value is None:
        return
    card["measurements"][label] = value
    if filename not in card["_files"]:
        card["_files"].append(filename)


def put_training(card, run_id):
    for key, label in TRAIN_NUMBERS:
        put(card, label, f"train_{run_id}.json", key)


def put_detector(card, detector, name):
    final, train = "detector_final_final.json", "detector_final_train.json"
    for g in GENERATORS:
        put(card, f"{name}, final test AUROC, {g} fakes", final, "detectors", detector, "per_generator", g, "auroc")
        put(card, f"{name}, final test accuracy at 0.5, {g} fakes", final, "detectors", detector, "per_generator", g, "accuracy_at_0.5")
    for g in GENERATORS:  # the detector was trained WITHOUT generator g, then tested on g (calibration split)
        loo = ("leave_one_generator_out", g, detector)
        put(card, f"{name}, AUROC on unseen {g} (trained without it, calibration split)", train, *loo, "unseen_generator", "auroc")
        put(card, f"{name}, AUROC drop, seen minus unseen {g} (calibration split)", train, *loo, "drop_same_minus_unseen", "auroc")
    for key, text in PERTURBATIONS:
        for g in GENERATORS:
            put(card, f"{name}, AUROC after {text}, {g} fakes (calibration split)", train, "robustness", key, detector, g, "auroc")
    intervals, blur = "detector_intervals.json", "detector_blur_control.json"
    for g in GENERATORS:
        for end in ("low", "high"):
            put(card, f"{name}, final test AUROC 95% interval, {end} end (formula-based interval), {g} fakes", intervals,
                "final_test_closed_form_only", "detectors", detector, g, "auroc", f"hanley_mcneil_{end}")
    for g in GENERATORS:  # real images blurred to the sharpness of g's fakes (sigma*), then scored against g's fakes
        sigma = value_at(blur, "blur_matched", "sigma_star", g)
        if sigma is not None:
            put(card, f"{name}, AUROC on blur-matched real images (blur sigma {sigma}) vs {g} fakes (calibration split)",
                blur, "blur_matched", "auroc_at_sigma_star", detector, g)


def build_cards():
    cards = {}

    for run_id, script in [("g-vae-cifar", "model/train_vae.py"), ("g-vae-fmnist", "model/train_vae.py"),
                           ("g-cddpm-fmnist", "model/train_ddpm.py"), ("g-ddpm-cifar", "model/train_ddpm.py")]:
        cards[run_id] = new_card(script, f"configs/{run_id}.json")
        put_training(cards[run_id], run_id)

    # speed of DDIM sampling: timed with an untrained network of the same size as g-ddpm-cifar
    for where, text in [("gpu", "GPU"), ("cpu", "CPU")]:
        put(cards["g-ddpm-cifar"], f"DDIM sampling, 50 steps, 16 images, {text} (seconds; untrained network of the same size)",
            "ddim_timing.json", "seconds_by_steps", where, "50")

    for run_id, script in [("d-vae-cifar", "model/train_vae.py"), ("d-ddpm-cifar", "model/train_ddpm.py")]:
        cards[run_id] = new_card(f"{script}, model/run_final_detector.py", f"configs/{run_id}.json, configs/detector_final.json")
        put_training(cards[run_id], run_id)
    cards["cnn-baseline"] = new_card("model/run_final_detector.py", "configs/detector_final.json")
    put(cards["cnn-baseline"], "CNN epochs run", "detector_final_train.json", "cnn_epochs_run")
    for run_id, (detector, name) in DETECTORS.items():
        put_detector(cards[run_id], detector, name)

    # shortcut check: how well does sharpness alone separate real from fake? (calibration split)
    blur = "detector_blur_control.json"
    sharp = cards["sharpness-baseline"] = new_card("model/blur_control.py", "no config file: the settings are at the top of model/blur_control.py")
    for g in GENERATORS:
        put(sharp, f"AUROC of minus sharpness, real images vs {g} fakes (calibration split)", blur, "sharpness_only", "auroc_real_vs_fakes", g)
    for key, text in [("real_calibration", "real images"), ("g-vae-cifar_fakes", "g-vae-cifar fakes"), ("g-ddpm-cifar_fakes", "g-ddpm-cifar fakes")]:
        put(sharp, f"Median sharpness, {text} (calibration split)", blur, "sharpness_only", "median_sharpness", key)
    for g in GENERATORS:
        put(sharp, f"Blur sigma matched to the sharpness of {g} fakes (sigma*; 0 = fakes not blurrier than real)", blur, "blur_matched", "sigma_star", g)

    for run_id in ["a-500", "a-10000", "a-500-dup"]:
        cards[run_id] = new_card("model/train_auditor_run.py", f"configs/{run_id}.json")
        for path, label in AUDIT_NUMBERS:
            put(cards[run_id], label, f"audit_{run_id}.json", *path)

    return cards


def main():
    filled, placeholders, models = [], [], {}
    for run_id, card in build_cards().items():
        if not card["measurements"]:
            placeholders.append(run_id)  # nothing measured: leave it out, the page shows the placeholder
            continue
        card["source"] = ", ".join("results/" + f for f in card.pop("_files"))
        models[run_id] = card
        filled.append(f"{run_id} ({len(card['measurements'])} numbers from {card['source']})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"models": models}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT)
    print("filled:", *filled, sep="\n  ")
    print("still placeholders:", ", ".join(placeholders) or "none")


if __name__ == "__main__":
    main()
