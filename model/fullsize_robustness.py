# fullsize_robustness.py - how the full-size check (model/fullsize.py) holds up when a picture is changed, on a
# held-out folder no run trained on (data/openfake/heldout, made by model/prepare_openfake.py).
#
# Run from the project folder:
#   python -m model.fullsize_robustness --run-id v2 --folder data/openfake/heldout --cache-only
#       only computes and stores the CLIP / measured features and Smaug scores of every change (the slow part);
#       no number is computed or shown. CLIP and Smaug are frozen, so one cache run serves every run-id.
#   python -m model.fullsize_robustness --run-id pilot --folder data/openfake/heldout
#   python -m model.fullsize_robustness --run-id v2 --folder data/openfake/heldout
#       scores the folder ONCE per run-id (kept in logs/heldout_scored.json; a second time needs --force) and
#       writes results/fullsize_robustness_<run-id>.json
#
# Changes (model/fullsize.py augment(), applied before normalize()): clean; JPEG quality 70 and 50; half size and
# back; screenshot-style (0.7x, 4% of each side cut off, PNG). For each: accuracy at 0.5 and AUROC of our CLIP
# model, our feature model, Smaug and the combiner (from that run's models file), the combiner's real / unclear / ai
# labels (fullsize.label_of), per AI tool (meta.csv), and AI tools seen vs unseen in that run's training folders.
# A row of one AI tool (or of seen / unseen tools) counts only its AI images; its AUROC is those AI images against
# all real images of the folder. All numbers are "proposed, to tune".

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from model import fullsize
from model.utils import LOGS_DIR, RESULTS_DIR, ROOT, save_json

CONDITIONS = {"clean": "", "jpeg70": "jpeg:70", "jpeg50": "jpeg:50", "resize0.5": "resize:0.5",
              "screenshot": "screenshot:0.7:0.04"}
GUARD = LOGS_DIR / "heldout_scored.json"
FEATURE_CACHE = LOGS_DIR / "fullsize_features"


def cached_features(where, rows, condition):
    """(measured features, CLIP embeddings) of the changed pictures, kept in logs/fullsize_features/."""
    keys = np.array([f"{p.parent.name}/{p.name}" for p, _, _ in rows])
    file = FEATURE_CACHE / f"{where.replace('/', '_')}_{condition}.npz"
    if file.exists():
        stored = np.load(file)
        if np.array_equal(stored["keys"], keys):
            return stored["feats"], stored["clip"]
    feats, _, clip = fullsize.describe(rows)
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(file, keys=keys, feats=feats, clip=clip)
    return feats, clip


def row(condition, group, name, idx, scores, y, cut, real, seen=None):
    both = np.union1d(idx, real)                       # AUROC: these images against all real images
    return {"condition": condition, "group": group, "name": name, "seen": seen,
            "real_images": int((y[idx] == 0).sum()), "ai_images": int((y[idx] == 1).sum()),
            "models": {k: dict(fullsize.accuracy(v[idx], y[idx]), auroc=fullsize.auroc(v[both], y[both]))
                       for k, v in scores.items()},
            "combiner_labels": fullsize.three_way(scores["combiner"][idx], y[idx], cut,
                                                  scores["ours-clip"][idx], scores["smaug"][idx])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="pilot, v2, ... (the models of model/fullsize.py --run-id)")
    parser.add_argument("--folder", required=True, help="held-out folder with real/, ai/ and meta.csv")
    parser.add_argument("--cache-only", action="store_true", help="only compute and store features and Smaug scores")
    parser.add_argument("--force", action="store_true", help="score a folder again for the same run-id")
    args = parser.parse_args()
    folder = Path(args.folder)
    where = folder.resolve().relative_to(ROOT).as_posix()
    files = fullsize.list_images(folder)
    y = np.array([label for _, label in files], dtype=int)
    guard = json.loads(GUARD.read_text(encoding="utf-8")) if GUARD.exists() else {}
    key = f"{where}|{args.run_id}"
    if not args.cache_only:
        if guard.get(key) and not args.force:
            raise SystemExit(f"{where} was already scored for run {args.run_id} at {guard[key]}; "
                             "it is scored once (use --force only if that run crashed)")
        bundle_file, _, config_file = fullsize.run_files(args.run_id)
        bundle = joblib.load(bundle_file)
        config = json.loads(config_file.read_text(encoding="utf-8"))

    inputs = {}
    for condition, spec in CONDITIONS.items():
        print(f"{condition}: {len(files)} images", flush=True)
        rows = [(p, label, spec) for p, label in files]
        feats, clip = cached_features(where, rows, condition)
        inputs[condition] = feats, clip, fullsize.smaug_cached(rows)
    if args.cache_only:
        print("features and Smaug scores cached; nothing was scored")
        return

    now = datetime.now().isoformat(timespec="seconds")
    guard[key] = guard.get(key, []) + [now]            # written before any number is computed
    save_json(GUARD, guard)
    train = config["train_folder"]
    seen = {g for d in ([train] if isinstance(train, str) else train)
            for p, g in fullsize.generator_names(ROOT / d).items() if p.parent.name == "ai"}
    names = fullsize.generator_names(folder)
    gen = np.array([names.get(p, "unknown") if label else "real photos" for p, label in files])
    real, cut = np.flatnonzero(y == 0), bundle["band"]
    out = []
    for condition, (feats, clip, smaug) in inputs.items():
        scores = {"ours-clip": bundle["clip"].predict_proba(clip)[:, 1],
                  "ours-features": bundle["features"].predict_proba(feats)[:, 1], "smaug": smaug}
        scores["combiner"] = bundle["combiner"].predict_proba(
            np.column_stack([fullsize.logit(scores[k]) for k in bundle["combiner_inputs"]]))[:, 1]
        out.append(row(condition, "all", "all images", np.arange(len(y)), scores, y, cut, real))
        for group, keep in (("seen", np.isin(gen, list(seen))), ("unseen", (y == 1) & ~np.isin(gen, list(seen)))):
            if keep.any():
                out.append(row(condition, group, f"AI tools {group} in training", np.flatnonzero(keep), scores, y, cut, real))
        for g in sorted(set(gen)):
            out.append(row(condition, "generator", g, np.flatnonzero(gen == g), scores, y, cut, real,
                           seen=None if g == "real photos" else g in seen))
    result_file = RESULTS_DIR / f"fullsize_robustness_{args.run_id}.json"
    save_json(result_file, {
        "run_id": args.run_id, "folder": where, "timestamp": now, "times_scored": len(guard[key]),
        "models_file": bundle_file.relative_to(ROOT).as_posix(),
        "models_sha256": hashlib.sha256(bundle_file.read_bytes()).hexdigest(),
        "conditions": CONDITIONS, "images": {"real": int((y == 0).sum()), "ai": int((y == 1).sum())},
        "band": cut, "train_generators": sorted(seen), "unclear_rule": fullsize.UNCLEAR_RULE,
        "row_rule": "group all = every image; seen / unseen / one AI tool = only those AI images (accuracy = share "
                    "called AI at 0.5), AUROC = those AI images against all real images; real photos = only real "
                    "images (no AUROC)",
        "rows": out,
        "note": "Our models never trained on this folder. Smaug is pretrained and its training may include some of "
                "these AI tools, so its numbers can be too good. Proposed, to tune."})
    for r in out:
        if r["group"] == "all":
            m = r["models"]
            print(f"{r['condition']:11s} " + "  ".join(f"{k} {v['accuracy']}/{v['auroc']}" for k, v in m.items())
                  + f"  | combiner labels {r['combiner_labels']}")
    print("written:", result_file)


if __name__ == "__main__":
    main()
