# blur_control.py - two checks on the FROZEN final Detector, plus error bars on its numbers.
#
# Run it from the project folder:
#   .venv/Scripts/python -m model.blur_control --quick          smoke test: 300 images per set, files go to logs/
#   .venv/Scripts/python -m model.blur_control                  the full run (5-15 minutes on the GPU)
#   .venv/Scripts/python -m model.blur_control --reuse-scores   skip the scoring, reuse logs/blur_control_scores.npz
#
# PART 1  sharpness only. sharpness(image) = variance of the Laplacian of the grayscale image (pixels on a
#         0..1 scale, scipy.ndimage.laplace). The "AI-ness" score is minus sharpness. We report its AUROC for
#         the real calibration images against each generator's calibration fakes, and the median sharpness.
#         (An AUROC above 0.5 means the fakes are blurrier than the real images on this measure.)
# PART 2  blur-matched real images. All real calibration images are blurred with a Gaussian (each colour
#         channel on its own), rounded back to whole 0..255 pixels (the Detector sees PNGs), and scored by the
#         three frozen detectors. sigma* of a generator = the grid sigma whose median real sharpness is closest
#         to the median sharpness of that generator's fakes (sigma* = 0 if the fakes are NOT blurrier than the
#         unblurred real images). AUROC = blurred real images vs the original fakes, for every sigma.
# PART 3  95% intervals on the calibration split: percentile bootstrap (2000 resamples, real and fake images
#         resampled separately) next to the closed-form ones (Hanley-McNeil for AUROC, Wilson for accuracy).
# PART 4  95% intervals for the FINAL TEST numbers, from the saved summary only (closed form, NOT bootstrap:
#         the final test was scored once and the per-image scores were not saved).
#
# THE FINAL TEST SPLIT IS NEVER LOADED HERE. Only the real 'calibration' part and the fakes' 'calibration'
# array are used. The only thing read about the final test is results/detector_final_final.json (read only).
# The frozen detector is never trained or changed: its SHA-256 hashes are checked first.
#
# Files it READS:   configs/detector_final.json, weights/detector_final_*.joblib / detector_final_cnn.pt,
#                   weights/d-vae-cifar.pt, weights/d-ddpm-cifar.pt, the real CIFAR-10 'calibration' part,
#                   fakes/<generator>_fakes.npz ('calibration' array only),
#                   results/detector_final_train.json (calibration numbers), results/detector_final_final.json
# Files it WRITES:  results/detector_blur_control.json   (parts 1 and 2)
#                   results/detector_intervals.json      (parts 3 and 4)
#                   logs/blur_control_scores.npz         (cache of the per-image scores, ignored by git)
#                   (with --quick the three files go to logs/ with '_quick' in the name instead)
# All numbers are "proposed, to tune".

import argparse
import json
import math
import sys
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, laplace
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from model.data import load_part
from model.make_fakes import to_float, to_uint8
from model.run_final_detector import (CPU_THREADS, DETECTORS, folders, load_feature_models, load_frozen,
                                      score_images, sha256_of)
from model.train_detector import evaluate, get_features
from model.utils import CONFIGS_DIR, FAKES_DIR, LOGS_DIR, RESULTS_DIR, SEED, WEIGHTS_DIR, get_device, save_json, set_seed

RUN_ID = "detector_final"
REAL_PART = "calibration"        # the ONLY real part used here
FAKE_PART = "calibration"        # the ONLY fakes part used here
SIGMAS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
N_BOOT = 2000
Z95 = 1.96
QUICK_N = 300
GRAY = np.array([0.299, 0.587, 0.114])      # the usual grayscale weights for R, G, B
FLAG_DIFFERENCE = 0.005


# ------------------------------------------------------------------ small numbers helpers

def r4(x):
    return round(float(x), 4)


def sig6(x):
    """6 significant digits (sharpness values are small numbers)."""
    return float(f"{float(x):.6g}")


def fast_auroc(real_scores, fake_scores):
    """AUROC from ranks (ties get the average rank): the chance a random fake scores higher than a random real."""
    ranks = rankdata(np.concatenate([real_scores, fake_scores]))
    n0, n1 = len(real_scores), len(fake_scores)
    return (ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def hanley_mcneil(a, n_fake, n_real):
    """Hanley and McNeil (1982): (standard error, low, high) of an AUROC `a`. Fakes are the positives (n1)."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    variance = (a * (1 - a) + (n_fake - 1) * (q1 - a * a) + (n_real - 1) * (q2 - a * a)) / (n_fake * n_real)
    se = math.sqrt(max(variance, 0.0))
    return se, max(0.0, a - Z95 * se), min(1.0, a + Z95 * se)


def wilson(p, n):
    """Wilson 95% interval (low, high) of a share p measured on n images."""
    denominator = 1 + Z95 ** 2 / n
    centre = (p + Z95 ** 2 / (2 * n)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / denominator
    return centre - half, centre + half


def self_check():
    """A small test that must pass before anything is scored."""
    rng = np.random.default_rng(0)
    real, fake = rng.integers(0, 20, 300) / 20, rng.integers(5, 25, 400) / 20           # many ties on purpose
    truth = np.concatenate([np.zeros(300), np.ones(400)])
    assert abs(fast_auroc(real, fake) - roc_auc_score(truth, np.concatenate([real, fake]))) < 1e-12
    low, high = wilson(0.5, 100)
    assert abs((low + high) / 2 - 0.5) < 1e-9 and 0.403 < low < 0.404
    se, low, high = hanley_mcneil(0.5, 100, 100)
    assert abs(se - 0.040927) < 1e-5 and abs(low - (0.5 - Z95 * se)) < 1e-12
    print("self-check passed", flush=True)


# ------------------------------------------------------------------ the frozen detector

def check_frozen_hashes():
    """Compare the SHA-256 of every frozen file with configs/detector_final.json. Stop if any differs."""
    config = json.loads((CONFIGS_DIR / f"{RUN_ID}.json").read_text(encoding="utf-8"))
    expected = {info["file"]: info["sha256"] for info in config["weight_files"].values()}
    expected.update(config["feature_model_files_sha256"])
    result = {}
    for name, sha in expected.items():
        path = WEIGHTS_DIR / name
        actual = sha256_of(path) if path.exists() else None
        result[name] = {"expected": sha, "actual": actual, "match": actual == sha}
    bad = [name for name, r in result.items() if not r["match"]]
    if bad:
        sys.exit("STOPPED: these frozen files are missing or changed after freezing (SHA-256 differs from "
                 f"configs/{RUN_ID}.json): {', '.join(bad)}")
    return config, result


# ------------------------------------------------------------------ images, sharpness, blur

def sharpness(pixels):
    """(N,32,32,3) uint8 -> (N,) variance of the Laplacian of the grayscale image, pixels on a 0..1 scale."""
    gray = (pixels.astype(np.float64) / 255) @ GRAY                  # (N,32,32)
    return np.array([laplace(g).var() for g in gray])               # one image at a time (laplace works on all axes)


def blur(pixels, sigma):
    """Gaussian blur of each colour channel on its own (never across channels), rounded back to uint8."""
    if sigma == 0:
        return pixels
    blurred = gaussian_filter(pixels.astype(np.float32), sigma=(0, sigma, sigma, 0))
    return blurred.round().clip(0, 255).astype(np.uint8)


def load_fake_pixels(generator, limit):
    with np.load(FAKES_DIR / f"{generator}_fakes.npz") as saved:
        pixels = saved[FAKE_PART]                                     # only the calibration array is opened
    return pixels[:limit] if limit else pixels


def score_pixels(pixels, models, feature_models):
    """{detector: scores (N,)} for uint8 images (N,32,32,3). Features and scores come back as CPU numpy, batch by batch."""
    images = to_float(pixels)
    return score_images(models, images, get_features(images, feature_models))


def compute_scores(limit, generators, models, feature_models):
    """Sharpness and the three detector scores of: each generator's fakes, and the real images at every sigma."""
    assert REAL_PART == "calibration" and FAKE_PART == "calibration"
    real = to_uint8(load_part(REAL_PART))
    real = real[:limit] if limit else real
    out = {}
    for g in generators:
        start = time.time()
        pixels = load_fake_pixels(g, limit)
        out[f"sharp_fake_{g}"] = sharpness(pixels)
        for d, s in score_pixels(pixels, models, feature_models).items():
            out[f"fake_{g}_{d}"] = s
        print(f"scored {len(pixels)} fakes of {g}  ({time.time() - start:.0f} s)", flush=True)
    for sigma in SIGMAS:
        start = time.time()
        pixels = blur(real, sigma)
        out[f"sharp_real_{sigma}"] = sharpness(pixels)
        for d, s in score_pixels(pixels, models, feature_models).items():
            out[f"real_{sigma}_{d}"] = s
        print(f"scored {len(pixels)} real images blurred with sigma {sigma}  ({time.time() - start:.0f} s)", flush=True)
    return out


# ------------------------------------------------------------------ parts 1 and 2

def sharpness_and_blur(scores, generators, train_report):
    """Parts 1 and 2 from the scores. Returns (sharpness_only dict, blur_matched dict)."""
    real_sharp = {s: scores[f"sharp_real_{s}"] for s in SIGMAS}
    real_median = {s: float(np.median(real_sharp[s])) for s in SIGMAS}
    n_real = len(real_sharp[0.0])

    only = {"definition": "variance of the Laplacian (scipy.ndimage.laplace) of the grayscale image, pixels on a 0..1 scale",
            "score": "AI-ness = minus sharpness; AUROC above 0.5 means the fakes are blurrier than the real images",
            "median_sharpness": {"real_calibration": sig6(real_median[0.0])}, "auroc_real_vs_fakes": {}}
    star = {}
    print("\nPART 1  sharpness only (calibration split)")
    print(f"  median sharpness, real images: {real_median[0.0]:.6g}")
    for g in generators:
        fake_sharp = scores[f"sharp_fake_{g}"]
        fake_median = float(np.median(fake_sharp))
        auroc = fast_auroc(-real_sharp[0.0], -fake_sharp)
        only["median_sharpness"][f"{g}_fakes"] = sig6(fake_median)
        only["auroc_real_vs_fakes"][g] = r4(auroc)
        print(f"  {g}: median sharpness of fakes {fake_median:.6g} | AUROC (minus sharpness) {auroc:.4f}")
        blurrier = fake_median < real_median[0.0]
        chosen = min(SIGMAS, key=lambda s: abs(real_median[s] - fake_median)) if blurrier else 0.0
        star[g] = {"fake_median_sharpness": sig6(fake_median), "fakes_blurrier_than_unblurred_real": bool(blurrier),
                   "sigma_star": chosen,
                   "fakes_blurrier_than_most_blurred_real": bool(fake_median < real_median[SIGMAS[-1]])}
        if not blurrier:
            print(f"  {g}: the fakes are NOT blurrier than the unblurred real images -> sigma* = 0")
        else:
            print(f"  {g}: sigma* = {chosen}" + ("  (fakes are blurrier than even the most blurred real set)"
                                                if star[g]["fakes_blurrier_than_most_blurred_real"] else ""))

    table, at_star, at_zero = {}, {}, {}
    for d in DETECTORS:
        table[d], at_star[d], at_zero[d] = {}, {}, {}
        for g in generators:
            fake = scores[f"fake_{g}_{d}"]
            table[d][g] = {str(s): r4(fast_auroc(scores[f"real_{s}_{d}"], fake)) for s in SIGMAS}
            at_star[d][g] = table[d][g][str(star[g]["sigma_star"])]
            earlier = train_report[d]["per_generator"][g]["auroc"]
            at_zero[d][g] = {"this_run": table[d][g]["0.0"], "detector_final_train.json": earlier,
                             "abs_difference": r4(abs(table[d][g]["0.0"] - earlier)),
                             f"flag_difference_above_{FLAG_DIFFERENCE}": abs(table[d][g]["0.0"] - earlier) > FLAG_DIFFERENCE}

    print("\nPART 2  AUROC of blurred real images vs the original fakes (* = sigma* of that generator)")
    print("  " + f"{'detector':12s} {'generator':13s}" + "".join(f"{s:>9}" for s in SIGMAS))
    for d in DETECTORS:
        for g in generators:
            cells = "".join(f"{table[d][g][str(s)]:>8.4f}" + ("*" if s == star[g]["sigma_star"] else " ") for s in SIGMAS)
            print("  " + f"{d:12s} {g:13s}" + cells)
    print(f"  sigma 0 against results/detector_final_train.json (flag if the difference is above {FLAG_DIFFERENCE}):")
    for d in DETECTORS:
        for g in generators:
            x = at_zero[d][g]
            print(f"    {d:12s} {g:13s} this run {x['this_run']:.4f}  earlier {x['detector_final_train.json']:.4f}  "
                  f"difference {x['abs_difference']:.4f}" + ("  <-- FLAG" if x[f"flag_difference_above_{FLAG_DIFFERENCE}"] else ""))

    only["images"] = {"real_calibration": n_real}
    blur_matched = {
        "how_blurred": "scipy.ndimage.gaussian_filter on each colour channel separately (sigma 0 across channels), "
                       "rounded to whole 0..255 pixels; fakes are NOT blurred",
        "sigmas": SIGMAS,
        "median_sharpness_real_by_sigma": {str(s): sig6(real_median[s]) for s in SIGMAS},
        "sigma_star_rule": "the grid sigma whose median real sharpness is closest (absolute difference) to the median "
                           "sharpness of the generator's fakes; 0 if the fakes are not blurrier than the unblurred real images",
        "sigma_star": {g: star[g]["sigma_star"] for g in generators},
        "sigma_star_details": star,
        "auroc_by_sigma": table, "auroc_at_sigma_star": at_star,
        "sigma_0_against_detector_final_train_json": at_zero,
    }
    return only, blur_matched


# ------------------------------------------------------------------ part 3

def bootstrap_and_closed_form(scores, generators):
    """AUROC and accuracy at 0.5 of every detector x generator on the calibration split, with the percentile
    bootstrap interval and the closed-form one. The same resamples are used for the three detectors."""
    rng = np.random.default_rng(SEED)
    out = {d: {} for d in DETECTORS}
    n_real = len(scores[f"real_0.0_{DETECTORS[0]}"])
    for g in generators:
        real = {d: scores[f"real_0.0_{d}"] for d in DETECTORS}
        fake = {d: scores[f"fake_{g}_{d}"] for d in DETECTORS}
        n_fake = len(fake[DETECTORS[0]])
        draws = {d: ([], []) for d in DETECTORS}                      # (AUROCs, accuracies) of the resamples
        print(f"bootstrap for {g}: {N_BOOT} resamples x {len(DETECTORS)} detectors", flush=True)
        for b in range(N_BOOT):
            real_idx, fake_idx = rng.integers(0, n_real, n_real), rng.integers(0, n_fake, n_fake)
            for d in DETECTORS:
                r, f = real[d][real_idx], fake[d][fake_idx]
                draws[d][0].append(fast_auroc(r, f))
                draws[d][1].append(((r < 0.5).sum() + (f >= 0.5).sum()) / (n_real + n_fake))
        for d in DETECTORS:
            point = evaluate(real[d], fake[d])                         # the same definition as run_final_detector.py
            auroc = fast_auroc(real[d], fake[d])
            accuracy = ((real[d] < 0.5).sum() + (fake[d] >= 0.5).sum()) / (n_real + n_fake)
            assert abs(auroc - point["auroc"]) < 1e-4 and abs(accuracy - point["accuracy_at_0.5"]) < 1e-4
            auroc_ci = np.percentile(draws[d][0], [2.5, 97.5])
            accuracy_ci = np.percentile(draws[d][1], [2.5, 97.5])
            se, hm_low, hm_high = hanley_mcneil(auroc, n_fake, n_real)
            w_low, w_high = wilson(accuracy, n_real + n_fake)
            out[d][g] = {
                "auroc": {"value": r4(auroc), "bootstrap_low": r4(auroc_ci[0]), "bootstrap_high": r4(auroc_ci[1]),
                          "hanley_mcneil_se": round(se, 5), "hanley_mcneil_low": r4(hm_low), "hanley_mcneil_high": r4(hm_high)},
                "accuracy_at_0.5": {"value": r4(accuracy), "bootstrap_low": r4(accuracy_ci[0]), "bootstrap_high": r4(accuracy_ci[1]),
                                    "wilson_low": r4(w_low), "wilson_high": r4(w_high)},
            }
    return out, n_real


def max_difference(by_detector):
    """The largest |bootstrap - closed form| over every detector x generator, for each end of each interval."""
    worst = {"auroc_low": 0.0, "auroc_high": 0.0, "accuracy_low": 0.0, "accuracy_high": 0.0}
    for per_generator in by_detector.values():
        for x in per_generator.values():
            a, c = x["auroc"], x["accuracy_at_0.5"]
            worst["auroc_low"] = max(worst["auroc_low"], abs(a["bootstrap_low"] - a["hanley_mcneil_low"]))
            worst["auroc_high"] = max(worst["auroc_high"], abs(a["bootstrap_high"] - a["hanley_mcneil_high"]))
            worst["accuracy_low"] = max(worst["accuracy_low"], abs(c["bootstrap_low"] - c["wilson_low"]))
            worst["accuracy_high"] = max(worst["accuracy_high"], abs(c["bootstrap_high"] - c["wilson_high"]))
    worst = {k: r4(v) for k, v in worst.items()}
    worst["overall"] = max(worst.values())
    return worst


def print_part3(by_detector):
    print("\nPART 3  calibration split, 95% intervals: bootstrap vs closed form")
    for d, per_generator in by_detector.items():
        for g, x in per_generator.items():
            a, c = x["auroc"], x["accuracy_at_0.5"]
            print(f"  {d:12s} {g:13s} AUROC {a['value']:.4f}  bootstrap [{a['bootstrap_low']:.4f}, {a['bootstrap_high']:.4f}]  "
                  f"Hanley-McNeil [{a['hanley_mcneil_low']:.4f}, {a['hanley_mcneil_high']:.4f}]")
            print(f"  {'':12s} {'':13s} acc   {c['value']:.4f}  bootstrap [{c['bootstrap_low']:.4f}, {c['bootstrap_high']:.4f}]  "
                  f"Wilson [{c['wilson_low']:.4f}, {c['wilson_high']:.4f}]")


# ------------------------------------------------------------------ part 4

def final_test_intervals(generators):
    """Closed-form intervals for the numbers in results/detector_final_final.json (read only; nothing is scored)."""
    final = json.loads((RESULTS_DIR / f"{RUN_ID}_final.json").read_text(encoding="utf-8"))
    n_real = final["images"]["real"]
    out = {d: {} for d in DETECTORS}
    for d in DETECTORS:
        for g in generators:
            n_fake = final["images"]["fake_per_generator"][g]
            saved = final["detectors"][d]["per_generator"][g]
            se, hm_low, hm_high = hanley_mcneil(saved["auroc"], n_fake, n_real)
            w_low, w_high = wilson(saved["accuracy_at_0.5"], n_real + n_fake)
            out[d][g] = {
                "auroc": {"value": saved["auroc"], "hanley_mcneil_se": round(se, 5),
                          "hanley_mcneil_low": r4(hm_low), "hanley_mcneil_high": r4(hm_high)},
                "accuracy_at_0.5": {"value": saved["accuracy_at_0.5"], "wilson_low": r4(w_low), "wilson_high": r4(w_high)},
                "n_real": n_real, "n_fake": n_fake,
            }
    print("\nPART 4  final test, formula-based 95% intervals (Hanley-McNeil / Wilson), nothing re-scored")
    print(f"  set sizes from results/{RUN_ID}_final.json: images.real = {n_real}, images.fake_per_generator")
    for d in DETECTORS:
        for g in generators:
            a, c = out[d][g]["auroc"], out[d][g]["accuracy_at_0.5"]
            print(f"  {d:12s} {g:13s} AUROC {a['value']:.4f} [{a['hanley_mcneil_low']:.4f}, {a['hanley_mcneil_high']:.4f}]  "
                  f"acc {c['value']:.4f} [{c['wilson_low']:.4f}, {c['wilson_high']:.4f}]")
    return out


# ------------------------------------------------------------------ main

def read_args():
    parser = argparse.ArgumentParser(description="Sharpness baseline, blur-matched reals and 95% intervals for the frozen Detector.")
    parser.add_argument("--quick", action="store_true", help=f"smoke test: only {QUICK_N} images per set; files go to logs/")
    parser.add_argument("--reuse-scores", action="store_true", help="skip the scoring: read the cached scores from logs/")
    return parser.parse_args()


def get_scores(args, config, hash_digest, limit):
    """The per-image scores: from the cache (--reuse-scores) or freshly computed (and then cached)."""
    cache = LOGS_DIR / ("blur_control_scores_quick.npz" if args.quick else "blur_control_scores.npz")
    if args.reuse_scores:
        if not cache.exists():
            sys.exit(f"STOPPED: --reuse-scores but {cache} does not exist")
        with np.load(cache) as saved:
            if str(saved["frozen_digest"]) != hash_digest:
                sys.exit("STOPPED: the cached scores were made with other frozen files")
            print(f"reusing the scores in {cache}", flush=True)
            return {k: saved[k] for k in saved.files if k != "frozen_digest"}
    paths = folders(None)
    models = load_frozen(config, paths)                                # only after the hash check has passed
    feature_models = load_feature_models(paths)
    scores = compute_scores(limit, config["generators"], models, feature_models)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **scores, frozen_digest=np.array(hash_digest))
    print(f"cached the scores in {cache}", flush=True)
    return scores


def main():
    args = read_args()
    start = time.time()
    set_seed()
    torch.set_num_threads(CPU_THREADS)
    self_check()
    config, hash_check = check_frozen_hashes()
    print("frozen files: all", len(hash_check), "SHA-256 hashes match configs/detector_final.json", flush=True)
    hash_digest = "".join(r["actual"] for r in hash_check.values())
    generators = config["generators"]
    limit = QUICK_N if args.quick else 0
    train_report = json.loads((RESULTS_DIR / f"{RUN_ID}_train.json").read_text(encoding="utf-8"))["calibration_report"]

    scores = get_scores(args, config, hash_digest, limit)
    scoring_seconds = round(time.time() - start, 1)

    only, blur_matched = sharpness_and_blur(scores, generators, train_report)
    intervals, n_real = bootstrap_and_closed_form(scores, generators)
    print_part3(intervals)
    print("  max |bootstrap - closed form|:", max_difference(intervals))
    final = final_test_intervals(generators)

    seconds = {"scoring_or_loading": scoring_seconds, "total": round(time.time() - start, 1)}
    command = "python -m model.blur_control" + "".join(" " + a for a in sys.argv[1:])
    common = {"run_id": RUN_ID, "quick_smoke_test": args.quick, "command": command, "seed": SEED,
              "device": str(get_device()), "seconds": seconds, "scores_reused_from_cache": args.reuse_scores}
    folder, suffix = (LOGS_DIR, "_quick") if args.quick else (RESULTS_DIR, "")
    n_fake = {g: len(scores[f"fake_{g}_{DETECTORS[0]}"]) for g in generators}
    save_json(folder / f"detector_blur_control{suffix}.json", {
        **common,
        "stamp": "calibration split only (real 'calibration' part and the fakes' 'calibration' arrays); final test not used",
        "final_test_used": False,
        "images": {"real_calibration": n_real, "fake_calibration_per_generator": n_fake},
        "frozen_hash_check": {"config": f"configs/{RUN_ID}.json", "all_match": True, "files": hash_check},
        "sharpness_only": only, "blur_matched": blur_matched,
    })
    save_json(folder / f"detector_intervals{suffix}.json", {
        **common,
        "images_calibration": {"real": n_real, "fake_per_generator": n_fake},
        "frozen_hash_check_all_match": True,
        "calibration_bootstrap_and_closed_form": {
            "note": "calibration split, scored with the frozen detectors (see scores_reused_from_cache); accuracy_at_0.5 counts a score of "
                    "0.5 or more as 'ai' (as in run_final_detector.py); bootstrap = percentile interval, "
                    f"{N_BOOT} resamples, real and fake images resampled separately; closed form = Hanley-McNeil (AUROC, "
                    "fakes are the positives) and Wilson (accuracy over all real and fake images)",
            "confidence": 0.95, "bootstrap_resamples": N_BOOT, "bootstrap_seed": SEED,
            "detectors": intervals,
            "max_abs_difference_bootstrap_vs_closed_form": max_difference(intervals),
        },
        "final_test_closed_form_only": {
            "note": "FORMULA-BASED intervals (Hanley-McNeil for AUROC, Wilson for accuracy), NOT bootstrap: the final "
                    "test was scored once and the per-image scores were not saved. Nothing was re-scored here.",
            "confidence": 0.95,
            "values_from": f"results/{RUN_ID}_final.json (AUROC and accuracy are rounded to 4 decimals there)",
            "set_sizes_from": f"results/{RUN_ID}_final.json: images.real and images.fake_per_generator",
            "detectors": final,
        },
    })
    print(f"\nwrote {folder / ('detector_blur_control' + suffix + '.json')} and {folder / ('detector_intervals' + suffix + '.json')}")
    print("seconds:", seconds)
    print("blur_control done" + ("  [quick smoke test]" if args.quick else ""))


if __name__ == "__main__":
    main()
