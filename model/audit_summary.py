# audit_summary.py - gathers the Auditor runs into the ONE file that GET /audit/results returns.
#
# Each run (model/train_auditor_run.py) wrote its own results/audit_<run>.json. This file reads
# the ones that exist and writes results/audit_results.json in the shape of GET /audit/results
# (see api/contract.json): experiments (points for the two graphs), examples (closest pairs) and a note.
# A run that has not been done yet gives memorization_rate null. Nothing is invented here.
#
# Error bars: each run may also have been trained with the extra seeds (results/audit_a-500-s2.json, ...; see
# EXTRA_SEEDS in model/train_auditor_run.py). Every point keeps memorization_rate = the seed-42 run (as before) and
# adds, over the seeds found: memorization_rate_mean, memorization_rate_sd, flagged_mean, flagged_sd and n_seeds.
# sd is the sample standard deviation (n - 1), null when fewer than 2 seeds have run; a missing run has n_seeds 0.
#
# Run it from the project folder:
#   python -m model.audit_summary
#   python -m model.audit_summary --in-dir SOME_FOLDER --out SOME_FILE.json     (for tests)
#
# build_summary() is also used by api/audit.py to make the placeholder answer (no runs at all).

import argparse
import json
from pathlib import Path

import numpy as np

from model.train_auditor_run import EXTRA_SEEDS, run_id
from model.utils import RESULTS_DIR, SEED, save_json

NOTE = ("Each run audits its own small diffusion model. Similarity between images only: it does not decide "
        "legal copying. Measured by model/train_auditor_run.py; settings are proposed. Error bars: mean and standard "
        "deviation over the training seeds that have run (n_seeds).")

RUN_ORDER = ["a-500", "a-10000", "a-500-dup"]
# Each graph: its name, the label under the x axis, and (x value, run that gives the point).
EXPERIMENTS = [
    ("memorization vs training-set size", "training images", [(500, "a-500"), (10000, "a-10000")]),
    ("effect of duplicates (500 training images)", "duplicate share of the training set (%)", [(0, "a-500"), (30, "a-500-dup")]),
]
MAX_EXAMPLES = 9
POINT_FIELDS = ["x", "memorization_rate", "memorization_rate_mean", "memorization_rate_sd", "flagged_mean", "flagged_sd", "n_seeds"]


def mean_sd(values, digits):
    """(mean, sample standard deviation) rounded; None for what cannot be measured (no values / one value)."""
    mean = round(float(np.mean(values)), digits) if values else None
    sd = round(float(np.std(values, ddof=1)), digits) if len(values) > 1 else None
    return mean, sd


def point(x, run, results):
    """One graph point: the seed-42 rate (as before) plus mean, sd and n over every seed of this run found."""
    found = [results[rid] for rid in [run_id(run, seed) for seed in (SEED, *EXTRA_SEEDS)] if rid in results]
    rate_mean, rate_sd = mean_sd([r["memorization_rate"] for r in found], 4)
    flagged_mean, flagged_sd = mean_sd([r["flagged"] for r in found], 2)
    return {"x": x, "memorization_rate": results[run]["memorization_rate"] if run in results else None,
            "memorization_rate_mean": rate_mean, "memorization_rate_sd": rate_sd,
            "flagged_mean": flagged_mean, "flagged_sd": flagged_sd, "n_seeds": len(found)}


def build_summary(results, note=NOTE):
    """results = {run id: the dict from audit_<run id>.json} (base runs and seed runs such as a-500-s2).
    A run that is missing is simply not in it. Examples come from the seed-42 runs only, as before."""
    experiments = []
    for name, x_label, points in EXPERIMENTS:
        experiments.append({"name": name, "x_label": x_label, "points": [point(x, run, results) for x, run in points]})
    examples = [pair for run in RUN_ORDER if run in results for pair in results[run]["examples"]]
    return {"experiments": experiments, "examples": examples[:MAX_EXAMPLES], "note": note}


def load_runs(in_dir):
    """Read the audit_<run id>.json files that exist in in_dir, every seed of every run. Returns {run id: dict}."""
    found = {}
    for rid in [run_id(run, seed) for run in RUN_ORDER for seed in (SEED, *EXTRA_SEEDS)]:
        path = Path(in_dir) / f"audit_{rid}.json"
        if path.exists():
            found[rid] = json.loads(path.read_text(encoding="utf-8"))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default=RESULTS_DIR, help="where the audit_<run>.json files are (default: results/)")
    parser.add_argument("--out", default=RESULTS_DIR / "audit_results.json", help="the file to write")
    args = parser.parse_args()

    results = load_runs(args.in_dir)
    summary = build_summary(results)
    save_json(args.out, summary)
    print(f"run files found: {list(results)}  base runs missing: {[run for run in RUN_ORDER if run not in results]}")
    for experiment in summary["experiments"]:
        print(experiment["name"], [(p["x"], p["memorization_rate"], f"mean {p['memorization_rate_mean']} sd {p['memorization_rate_sd']} n {p['n_seeds']}")
                                   for p in experiment["points"]])
    print(f"examples: {len(summary['examples'])} | written to {args.out}")


if __name__ == "__main__":
    main()
