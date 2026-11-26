# fullcheck.py - the "full_check" part of POST /detect, for pictures that are not 32x32.
#
# Three detectors score the picture (see model/fullsize.py): our CLIP model, our measured-feature model, and
# Smaug (pretrained by Abacus.AI, not ours); OUR combiner (logistic regression) turns the three scores into
# real / unclear / ai. The answer also lists how much each measured feature pushed our
# feature model, Smaug's score per crop, and simple clues from the file itself (camera EXIF, C2PA, IPTC).
# Our two models exist only after `python -m model.fullsize` has been run on the pilot images; until then
# their probabilities are null. The trained file is re-read when it changes, so retraining needs no restart.

import io
import json

import joblib
import numpy as np
from PIL import Image

from model import fullsize

MAKE, MODEL = 271, 272                       # EXIF tag numbers of the camera make and model
NOTE = ("A pilot: our models were trained on about 600 labelled images (OpenFakeTiny), and Smaug is a "
        "published detector we did not train. Detecting images from AI tools none of them has seen is an open "
        "problem, so this answer can be wrong. 'Push' explains our measured-feature model only.")

_bundle, _bundle_time = None, None


ACTIVE_FILE = fullsize.CONFIGS_DIR / "fullsize_active.json"     # {"run_id": "pilot" or "v2"}: which run the site uses


def active_run():
    try:
        return json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))["run_id"]
    except (OSError, ValueError, KeyError):
        return "pilot"


def bundle():
    """The active run's trained models (re-read when the file or the active run changes), or None before training."""
    global _bundle, _bundle_time
    path = fullsize.run_files(active_run())[0]
    if not path.exists():
        return None
    key = (str(path), path.stat().st_mtime)
    if key != _bundle_time:
        _bundle, _bundle_time = joblib.load(path), key
    return _bundle


def heldout_tested(run_id):
    """'Tested' texts from the run's once-scored held-out report (clean pictures, all images), or {} if not scored yet."""
    path = fullsize.RESULTS_DIR / f"fullsize_robustness_{run_id}.json"
    try:
        row = next(r for r in json.loads(path.read_text(encoding="utf-8"))["rows"]
                   if r["condition"] == "clean" and r["group"] == "all")
    except (OSError, ValueError, KeyError, StopIteration):
        return {}
    how = "held-out images, scored once"
    texts = {k: f"{m['correct']} of {m['total']} correct ({how})" for k, m in row["models"].items() if m.get("total")}
    lab = row["combiner_labels"]
    texts["combiner"] = (f"{lab['correct_among_decided']} of {lab['decided']} decided answers correct, "
                         f"{lab['unclear']} of {lab['real_images'] + lab['ai_images']} unclear ({how})")
    return texts


def file_clues(data, picture):
    """What the file says about itself. Byte checks only: they find a label, they do not verify it."""
    try:
        exif = picture.getexif()
        camera = " ".join(str(exif.get(tag, "")).strip("\x00 ") for tag in (MAKE, MODEL)).strip()
    except Exception:
        camera = ""
    c2pa = b"c2pa" in data and (b"caBX" in data or b"\xff\xeb" in data)     # PNG chunk caBX / JPEG APP11 with a c2pa box
    iptc = b"trainedAlgorithmicMedia" in data
    return [
        {"clue": "Camera details (EXIF)", "found": bool(camera), "detail": camera or None},
        {"clue": "C2PA content label (simple byte check, not verified)", "found": c2pa,
         "detail": "a C2PA manifest is in the file; AI tools such as ChatGPT add one" if c2pa else None},
        {"clue": "IPTC 'made by AI' tag", "found": iptc,
         "detail": "the file says trainedAlgorithmicMedia" if iptc else None},
    ]


def full_check(data):
    """The full_check answer for one uploaded file (bytes). The upload was already checked."""
    picture = Image.open(io.BytesIO(data))
    img = fullsize.open_image(io.BytesIO(data))
    feats = np.array([fullsize.measured_features(img)])
    smaug, crops = fullsize.smaug_score(img)
    trained = bundle()
    if trained and "combiner" not in trained:            # an older trained file (before the combiner): ignore it
        trained = None
    scores = {"ours-clip": None, "ours-features": None, "smaug": smaug}
    push = [None] * len(fullsize.FEATURE_NAMES)
    medians, tested, cut, counted = {}, {}, None, {}
    if trained:
        scores["ours-clip"] = float(trained["clip"].predict_proba(np.array([fullsize.clip_embedding(img)]))[0, 1])
        scores["ours-features"] = float(trained["features"].predict_proba(feats)[0, 1])
        scaler, lr = trained["features"][0], trained["features"][-1]
        push = (lr.coef_[0] * scaler.transform(feats)[0]).tolist()   # each feature's share of the decision (log-odds)
        medians, cut = trained["medians"], trained["band"]
        tested = heldout_tested(active_run()) or trained["tested"]   # held-out report once scored, else training text
    known = [p for p in scores.values() if p is not None]
    if trained and smaug is not None:                    # OUR combiner decides (logistic regression on the 3 scores)
        inputs = trained["combiner_inputs"]
        x = fullsize.logit([[scores[k] for k in inputs]])
        combiner = trained["combiner"]
        final = float(combiner.predict_proba(x)[0, 1])
        counted = dict(zip(inputs, (combiner[-1].coef_[0] * combiner[0].transform(x)[0]).round(3).tolist()))
        label = fullsize.label_of(final, cut, scores["ours-clip"], smaug)
        # "disagreement": the combiner leaned real, but our CLIP model or Smaug says AI, so it is not cleared as real
        decided_by = "disagreement" if label == "unclear" else "combiner"
    else:                                                # fallback: plain average, cut at 0.5
        final = float(np.mean(known)) if known else None
        label = None if final is None else ("ai" if final >= fullsize.THRESHOLD else "real")
        decided_by = "average"
    smaug_name = "Smaug AI Image Detector by Abacus.AI (pretrained, not trained by us)"
    if smaug is None:
        smaug_name += " - not available right now (its weights are missing or still downloading)"
    return {
        "label": label,
        "probability_ai": final,
        "decided_by": decided_by,
        "band": cut,
        "agree": len({p >= fullsize.THRESHOLD for p in known}) <= 1,
        "models": [
            {"id": "ours-clip", "name": "Our CLIP model (frozen CLIP ViT-L/14 features + logistic regression, method of Ojha et al. 2023)",
             "ours": True, "probability_ai": scores["ours-clip"], "counted": counted.get("ours-clip"), "tested": tested.get("ours-clip")},
            {"id": "ours-features", "name": "Our measured-feature model (logistic regression)", "ours": True,
             "probability_ai": scores["ours-features"], "counted": counted.get("ours-features"), "tested": tested.get("ours-features")},
            {"id": "smaug", "name": smaug_name, "ours": False, "probability_ai": smaug,
             "counted": counted.get("smaug"), "tested": tested.get("smaug")},
        ],
        "features": [
            {"name": name, "value": round(value, 4),
             "typical_real": medians.get(name, {}).get("real"), "typical_ai": medians.get(name, {}).get("ai"),
             "push": None if p is None else round(p, 3), "meaning": meaning}
            for (name, meaning), value, p in zip(fullsize.FEATURES, feats[0], push)
        ],
        "crops": [{"part": name, "probability_ai": round(p, 4)} for name, p in crops],
        "file_clues": file_clues(data, picture),
        "vote_tested": tested.get("combiner"),
        "note": NOTE,
    }
