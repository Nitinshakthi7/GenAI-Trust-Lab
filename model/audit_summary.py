# audit_summary.py - gathers the Auditor runs into the ONE file that GET /audit/results returns.
#
# Each run (model/train_auditor_run.py) wrote its own results/audit_<run>.json. This file reads
# the ones that exist and writes results/audit_results.json in the shape of GET /audit/results
# (see api/contract.json): experiments (points for the two graphs), examples (closest pairs) and a note.
# A run that has not been done yet gives memorization_rate null. Nothing is invented here.
#
# Run it from the project folder:
#   python -m model.audit_summary
#   python -m model.audit_summary --in-dir SOME_FOLDER --out SOME_FILE.json     (for tests)
#
# build_summary() is also used by api/audit.py to make the placeholder answer (no runs at all).

import argparse
import json
from pathlib import Path

from model.utils import RESULTS_DIR, save_json

NOTE = ("Each run audits its own small diffusion model. Similarity between images only: it does not decide "
        "legal copying. Measured by model/train_auditor_run.py; settings are proposed.")

RUN_ORDER = ["a-500", "a-10000", "a-500-dup"]
# Each graph: its name, the label under the x axis, and (x value, run that gives the point).
EXPERIMENTS = [
    ("memorization vs training-set size", "training images", [(500, "a-500"), (10000, "a-10000")]),
    ("effect of duplicates (500 training images)", "duplicate share of the training set (%)", [(0, "a-500"), (30, "a-500-dup")]),
]
MAX_EXAMPLES = 9


def build_summary(results, note=NOTE):
    """results = {run name: the dict from audit_<run>.json}. A run that is missing is simply not in it."""
    experiments = []
    for name, x_label, points in EXPERIMENTS:
        experiments.append({
            "name": name, "x_label": x_label,
            "points": [{"x": x, "memorization_rate": results[run]["memorization_rate"] if run in results else None}
                       for x, run in points],
        })
    examples = [pair for run in RUN_ORDER if run in results for pair in results[run]["examples"]]
    return {"experiments": experiments, "examples": examples[:MAX_EXAMPLES], "note": note}


def load_runs(in_dir):
    """Read the audit_<run>.json files that exist in in_dir. Returns {run: dict}."""
    found = {}
    for run in RUN_ORDER:
        path = Path(in_dir) / f"audit_{run}.json"
        if path.exists():
            found[run] = json.loads(path.read_text(encoding="utf-8"))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default=RESULTS_DIR, help="where the audit_<run>.json files are (default: results/)")
    parser.add_argument("--out", default=RESULTS_DIR / "audit_results.json", help="the file to write")
    args = parser.parse_args()

    results = load_runs(args.in_dir)
    summary = build_summary(results)
    save_json(args.out, summary)
    print(f"runs found: {[run for run in RUN_ORDER if run in results]}  missing: {[run for run in RUN_ORDER if run not in results]}")
    for experiment in summary["experiments"]:
        print(experiment["name"], [(p["x"], p["memorization_rate"]) for p in experiment["points"]])
    print(f"examples: {len(summary['examples'])} | written to {args.out}")


if __name__ == "__main__":
    main()
