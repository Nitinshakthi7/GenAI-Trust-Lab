# fullsize.py - the full-size image check: is a normal photo (any size) real or AI-made?
#
# Three detectors score the same picture, and OUR learned combiner (a logistic regression) turns their three
# scores into one answer: real, ai or unclear.
#   1. ours-clip      OUR model, the method of Ojha et al. 2023 ("Towards Universal Fake Image Detectors"):
#                     a frozen pretrained CLIP ViT-L/14 image encoder (OpenAI, not trained by us) turns the picture
#                     into 768 numbers; a logistic regression trained on OUR training images reads them.
#   2. ours-features  OUR model: a logistic regression on 5 measured features (sharpness, noise, fine detail,
#                     colour saturation, edges). Logistic regression can be explained exactly, so the website
#                     shows how much each feature pushed it ("push").
#   3. smaug          Smaug AI Image Detector by Abacus.AI (Apache-2.0), pretrained, NOT trained by us.
#                     The crop code below follows its smaug_image_detect.py (Apache-2.0, Abacus.AI).
#   combiner          OUR logistic regression on the three scores (stacking). It is trained on out-of-fold scores
#                     of the training images, and the "unclear" band is chosen on those too (never on the test set).
# A ResNet-18 (ImageNet) + logistic regression model is also trained and tested, as a baseline only.
#
# Train, then test once, from the project folder:
#   python -m model.fullsize --train data/openfake/train --test data/openfake/test
# Without --test it uses leave-one-out on the training folder (for a small pilot set).
# It writes configs/fullsize_pilot.json (before training), weights/fullsize_models.joblib and
# results/fullsize_pilot.json (only measured numbers). Everything here is "proposed, to tune".

import argparse
import csv
import io
import json
import math
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import compare_cnn
from model.utils import CONFIGS_DIR, DATA_DIR, LOGS_DIR, RESULTS_DIR, SEED, WEIGHTS_DIR, save_json

PILOT_DIR = DATA_DIR / "pilot"
BUNDLE_FILE = WEIGHTS_DIR / "fullsize_models.joblib"
RESULT_FILE = RESULTS_DIR / "fullsize_pilot.json"
SMAUG_ID = "abacusai/Smaug-AI-Image-Detector"
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SIDE = 512                     # every picture is resized (shorter side) and centre-cropped to 512x512 first,
                               # so size and file format cannot become a shortcut (proposed)
THRESHOLD = 0.5                # AI if the vote is at least this (proposed; Smaug's card also uses 0.5)
CNN_C, CLIP_C, FEATURE_C, COMBINER_C = 0.1, 1.0, 1.0, 1.0   # logistic-regression strength (proposed, to tune)
COMBINER_INPUTS = ["ours-clip", "ours-features", "smaug"]

# name, plain-English meaning (what it measures and what it does not prove)
FEATURES = [
    ("Sharpness", "How crisp the edges are (variance of the Laplacian). Camera photos and AI images can both be sharp or blurry."),
    ("Noise level", "The fine random grain a camera sensor leaves. AI images are often smoother, but phones also smooth noise away."),
    ("Fine-detail energy (%)", "Share of the image's energy in very fine patterns, from the frequency spectrum (FFT). Generators can leave too little or unusual fine detail."),
    ("Colour saturation", "How strong the colours are (0 to 1). AI images are often more saturated, but so are edited photos."),
    ("Edge amount (%)", "Share of pixels on a strong edge. It describes the scene as much as how the picture was made."),
]
FEATURE_NAMES = [name for name, _ in FEATURES]


def open_image(path_or_file):
    """A picture file -> RGB Pillow image, turned upright using its EXIF orientation."""
    return ImageOps.exif_transpose(Image.open(path_or_file)).convert("RGB")


def square(img, side=SIDE):
    """Shorter side -> `side` pixels, then the centre side x side square."""
    w, h = img.size
    s = side / min(w, h)
    img = img.resize((max(side, round(w * s)), max(side, round(h * s))), Image.LANCZOS)
    w, h = img.size
    x, y = (w - side) // 2, (h - side) // 2
    return img.crop((x, y, x + side, y + side))


def normalize(img):
    """512x512 square, then saved and re-read as JPEG quality 90: every picture goes through the same steps, so
    'phone JPEG vs generator PNG' cannot become a shortcut for our two models (Smaug gets the picture as it is)."""
    buffer = io.BytesIO()
    square(img).save(buffer, "JPEG", quality=90)
    return Image.open(buffer).convert("RGB")


def measured_features(img):
    """The 5 measured features of one picture, as a list in FEATURE_NAMES order."""
    sq = normalize(img)
    rgb = np.asarray(sq, dtype=np.float64)
    gray = rgb @ np.array([0.299, 0.587, 0.114])                      # 0..255
    sharpness = ndimage.laplace(gray).var()
    noise = (gray - ndimage.median_filter(gray, size=3)).std()
    window = np.outer(np.hanning(SIDE), np.hanning(SIDE))              # softens the borders before the FFT
    power = np.abs(np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * window))) ** 2
    yy, xx = np.indices(power.shape) - SIDE // 2
    radius = np.hypot(xx, yy) / (SIDE // 2)                            # 0 = centre, 1 = edge of the spectrum
    fine = 100.0 * power[radius > 0.5].sum() / power[radius > 0].sum()
    saturation = np.asarray(sq.convert("HSV"), dtype=np.float64)[..., 1].mean() / 255.0
    edges = np.hypot(ndimage.sobel(gray, 0), ndimage.sobel(gray, 1))
    edge_share = 100.0 * (edges > 100).mean()                          # 100 = a strong edge on the 0..255 scale (proposed)
    return [float(sharpness), float(noise), float(fine), float(saturation), float(edge_share)]


@torch.no_grad()
def cnn_embedding(img):
    """512 numbers from the pretrained ResNet-18 (the layer just before its ImageNet answer)."""
    net = compare_cnn.load()
    x = compare_cnn.WEIGHTS.transforms()(normalize(img)).unsqueeze(0).to(next(net.parameters()).device)
    for layer in (net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4, net.avgpool):
        x = layer(x)
    return x.flatten(1)[0].cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------- CLIP image features (pretrained, frozen)
CLIP_ID = "openai/clip-vit-large-patch14"
_clip = None
_clip_lock = threading.Lock()


def load_clip():
    """(processor, CLIP image encoder, device) from the local Hugging Face cache, loaded once (no download here)."""
    global _clip
    with _clip_lock:
        if _clip is None:
            from huggingface_hub import hf_hub_download
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
            folder = Path(hf_hub_download(CLIP_ID, "config.json", local_files_only=True)).parent
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPVisionModelWithProjection.from_pretrained(
                folder, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
            _clip = (CLIPImageProcessor.from_pretrained(folder), model, device)
        return _clip


@torch.no_grad()
def clip_embedding(img):
    """768 numbers: CLIP's image embedding of the normalized picture, scaled to length 1."""
    processor, model, device = load_clip()
    pixels = processor(images=normalize(img), return_tensors="pt")["pixel_values"].to(device, model.dtype)
    emb = model(pixel_values=pixels).image_embeds[0].float()
    return (emb / emb.norm()).cpu().numpy().astype(np.float64)


def logit(p):
    p = np.clip(np.nan_to_num(np.asarray(p, dtype=float), nan=0.5), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- Smaug (pretrained, not ours)
CROP, SMAUG_SIDE, BUDGET = 528, 576, 140      # must match Smaug's training (from smaug_image_detect.py)
_smaug, _smaug_error = None, None
_smaug_lock = threading.Lock()


def smaug_views(img):
    """Shorter side to 576 (or up to 528), then three 528x528 crops along the long side, with their names."""
    w, h = img.size
    if min(w, h) > SMAUG_SIDE:
        s = SMAUG_SIDE / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    elif min(w, h) < CROP:
        s = CROP / min(w, h)
        img = img.resize((math.ceil(w * s), math.ceil(h * s)), Image.BICUBIC)
    w, h = img.size
    cx, cy = (w - CROP) // 2, (h - CROP) // 2
    if w >= h:
        return [img.crop((x, cy, x + CROP, cy + CROP)) for x in (0, cx, w - CROP)], ["left", "centre", "right"]
    return [img.crop((cx, y, cx + CROP, y + CROP)) for y in (0, cy, h - CROP)], ["top", "centre", "bottom"]


def load_smaug():
    """Smaug from the local Hugging Face cache (no download here). None if it cannot be loaded; see smaug_error()."""
    global _smaug, _smaug_error
    with _smaug_lock:
        if _smaug is None:                                   # tried again on every call until it loads
            try:
                from huggingface_hub import snapshot_download
                from sentence_transformers import SentenceTransformer
                path = snapshot_download(SMAUG_ID, local_files_only=True)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                st = SentenceTransformer(path, device=device, config_kwargs={"audio_config": None},
                                         model_kwargs={"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32})
                processor = st[0].processor.image_processor
                processor.max_soft_tokens = processor.image_seq_length = BUDGET
                _smaug, _smaug_error = st, None
            except Exception as error:                       # missing package or weights: the other two models still work
                _smaug_error = f"{type(error).__name__}: {error}"[:300]
        return _smaug


def smaug_error():
    return _smaug_error


SMAUG_CACHE = LOGS_DIR / "smaug_cache.json"


def smaug_cached(paths):
    """Smaug scores for image files, kept in logs/smaug_cache.json (key: path inside data/) so a rerun is quick."""
    cache = json.loads(SMAUG_CACHE.read_text(encoding="utf-8")) if SMAUG_CACHE.exists() else {}
    out = []
    for i, path in enumerate(paths, 1):
        key = Path(path).resolve().relative_to(DATA_DIR.resolve()).as_posix()
        if cache.get(key) is None:
            score = smaug_score(open_image(path))[0]
            if score is not None:
                cache[key] = score
            if i % 25 == 0:
                SMAUG_CACHE.write_text(json.dumps(cache), encoding="utf-8")
                print(f"  smaug {i}/{len(paths)}", flush=True)
        out.append(cache.get(key, np.nan))
    SMAUG_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return np.array(out, dtype=float)


@torch.no_grad()
def smaug_score(img):
    """(AI probability, [(crop name, AI probability), ...]) from Smaug, or (None, []) if Smaug is not available."""
    st = load_smaug()
    if st is None:
        return None, []
    crops, names = smaug_views(img)
    logits = st.encode([{"image": c} for c in crops], batch_size=3, convert_to_tensor=True).float().view(-1)
    overall = torch.sigmoid(logits.mean()).item()                      # Smaug averages the crops' logits
    return overall, [(n, torch.sigmoid(v).item()) for n, v in zip(names, logits)]


# ---------------------------------------------------------------- training on the pilot images
def wilson(correct, total, z=1.96):
    """95% Wilson interval of a share, as [low, high]."""
    if total == 0:
        return [None, None]
    p = correct / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [round(centre - half, 4), round(centre + half, 4)]


def accuracy(probs, y):
    probs = np.asarray(probs, dtype=float)
    keep = ~np.isnan(probs)
    correct = int(((probs[keep] >= THRESHOLD).astype(int) == y[keep]).sum())
    total = int(keep.sum())
    return {"correct": correct, "total": total, "accuracy": round(correct / total, 4) if total else None,
            "interval_95": wilson(correct, total)}


def tested_text(acc, how):
    return f"{acc['correct']} of {acc['total']} correct ({how})" if acc["total"] else None


def list_images(folder):
    """[(path, 0 real / 1 ai)] from folder/real and folder/ai."""
    return [(p, label) for label, sub in ((0, "real"), (1, "ai"))
            for p in sorted((folder / sub).glob("*")) if p.suffix.lower() in SUFFIXES]


def generator_names(folder):
    """{file name: AI model name} from folder/meta.csv if it exists (made by model/prepare_openfake.py)."""
    meta = folder / "meta.csv" if folder else None
    if not meta or not meta.exists():
        return {}
    with open(meta, newline="", encoding="utf-8") as f:
        return {row["file"]: row["model"] for row in csv.DictReader(f)}


def describe(files):
    """Measured features, ResNet-18 and CLIP embeddings for a list of images."""
    feats, resnet, clip = [], [], []
    for i, (path, _) in enumerate(files, 1):
        img = open_image(path)
        feats.append(measured_features(img))
        resnet.append(cnn_embedding(img))
        clip.append(clip_embedding(img))
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} images", flush=True)
    return np.array(feats), np.array(resnet), np.array(clip)


def pipe(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=SEED))


def band(scores, y):
    """The unclear band, the same rule as the CIFAR Detector: high = 95th percentile of the REAL scores (at most
    5 percent of real images called AI), low = 5th percentile of the AI scores; if low >= high both become their middle."""
    high, low = float(np.percentile(scores[y == 0], 95)), float(np.percentile(scores[y == 1], 5))
    if low >= high:
        high = low = (high + low) / 2
    return {"low": round(low, 4), "high": round(high, 4)}


def label_of(score, cut, clip=None, smaug=None):
    """real / unclear / ai from the combiner's score, with one rule added after two live tests on 25 Nov:
    the answer may only be 'real' if neither of the two strong detectors (our CLIP model, Smaug) says AI (>= 0.5);
    otherwise it is 'unclear'. An 'ai' answer from the combiner is kept as it is."""
    label = "ai" if score >= cut["high"] else "real" if score <= cut["low"] else "unclear"
    if label == "real" and any(p is not None and not np.isnan(p) and p >= THRESHOLD for p in (clip, smaug)):
        return "unclear"
    return label


def auroc(scores, y):
    from sklearn.metrics import roc_auc_score
    keep = ~np.isnan(scores)
    return round(float(roc_auc_score(y[keep], scores[keep])), 4) if len(set(y[keep])) == 2 else None


def three_way(scores, y, cut, clip, smaug):
    """Counts of the real / unclear / ai labels, and how often a decided label is right."""
    labels = np.array([label_of(s, cut, c, m) for s, c, m in zip(scores, clip, smaug)])
    decided = labels != "unclear"
    right = int(((labels == "ai") & (y == 1)).sum() + ((labels == "real") & (y == 0)).sum())
    return {"ai": int((labels == "ai").sum()), "real": int((labels == "real").sum()), "unclear": int((~decided).sum()),
            "correct_among_decided": right, "decided": int(decided.sum()),
            "accuracy_among_decided": round(right / decided.sum(), 4) if decided.sum() else None,
            "real_called_ai": int(((labels == "ai") & (y == 0)).sum()), "real_images": int((y == 0).sum()),
            "ai_called_real": int(((labels == "real") & (y == 1)).sum()), "ai_images": int((y == 1).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(PILOT_DIR), help="folder with real/ and ai/ sub-folders to train on")
    parser.add_argument("--test", default=None, help="folder with real/ and ai/ to test on ONCE (never trained on); "
                                                      "without it, leave-one-out on the training folder")
    args = parser.parse_args()
    train_dir, test_dir = Path(args.train), Path(args.test) if args.test else None
    files = list_images(train_dir)
    y = np.array([label for _, label in files], dtype=int)
    if (y == 0).sum() < 5 or (y == 1).sum() < 5:
        raise SystemExit(f"need at least 5 real and 5 AI images in {train_dir} (found {(y == 0).sum()} real, {(y == 1).sum()} AI)")
    test_files = list_images(test_dir) if test_dir else []
    how = "test images from AI tools it never saw" if test_dir else "leave-one-out"
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED) if test_dir else LeaveOneOut()
    save_json(CONFIGS_DIR / "fullsize_pilot.json", {
        "run_id": "fullsize-pilot", "train_folder": str(train_dir), "test_folder": str(test_dir) if test_dir else None,
        "train_real": int((y == 0).sum()), "train_ai": int((y == 1).sum()), "test_images": len(test_files),
        "side": SIDE, "jpeg_quality": 90, "cnn_C": CNN_C, "clip_C": CLIP_C, "feature_C": FEATURE_C,
        "combiner_C": COMBINER_C, "combiner_inputs": COMBINER_INPUTS, "features": FEATURE_NAMES,
        "clip": CLIP_ID + " (frozen, image_embeds 768, normalized)",
        "cnn_baseline": "torchvision resnet18 IMAGENET1K_V1 (frozen, avgpool 512)", "smaug": SMAUG_ID,
        "folds": "stratified 5-fold" if test_dir else "leave-one-out", "seed": SEED,
        "unclear_band": "95th percentile of real out-of-fold combiner scores / 5th percentile of AI ones",
        "evaluation": how, "note": "proposed, to tune"})

    start = time.time()
    print(f"training images: {len(files)}", flush=True)
    feats, resnet, clip = describe(files)
    smaug = smaug_cached([p for p, _ in files])
    # out-of-fold scores: each training image is scored by models that did not see it
    oof = {"ours-clip": cross_val_predict(pipe(CLIP_C), clip, y, cv=folds, method="predict_proba")[:, 1],
           "ours-features": cross_val_predict(pipe(FEATURE_C), feats, y, cv=folds, method="predict_proba")[:, 1],
           "smaug": smaug}
    stack = np.column_stack([logit(oof[k]) for k in COMBINER_INPUTS])
    oof_comb = cross_val_predict(pipe(COMBINER_C), stack, y, cv=folds, method="predict_proba")[:, 1]
    cut = band(oof_comb, y)
    train_check = three_way(oof_comb, y, cut, oof["ours-clip"], oof["smaug"])   # the unclear rule, measured on training images
    print("training images, out-of-fold, with the unclear rule:", train_check, flush=True)
    clip_model, feature_model = pipe(CLIP_C).fit(clip, y), pipe(FEATURE_C).fit(feats, y)
    combiner = pipe(COMBINER_C).fit(stack, y)
    weights = dict(zip(COMBINER_INPUTS, (combiner[-1].coef_[0] / combiner[0].scale_).round(4).tolist()))
    print("combiner weight per detector (per unit of log-odds):", weights, "| unclear band:", cut, flush=True)

    if test_dir:
        print(f"test images: {len(test_files)}", flush=True)
        y_eval, eval_files = np.array([label for _, label in test_files], dtype=int), test_files
        t_feats, t_resnet, t_clip = describe(test_files)
        resnet_model = pipe(CNN_C).fit(resnet, y)
        scores = {"ours-clip": clip_model.predict_proba(t_clip)[:, 1],
                  "ours-features": feature_model.predict_proba(t_feats)[:, 1],
                  "smaug": smaug_cached([p for p, _ in test_files]),
                  "baseline-resnet18": resnet_model.predict_proba(t_resnet)[:, 1]}
        scores["combiner"] = combiner.predict_proba(np.column_stack([logit(scores[k]) for k in COMBINER_INPUTS]))[:, 1]
    else:
        y_eval, eval_files = y, files
        scores = dict(oof, combiner=oof_comb)
    acc = {name: dict(accuracy(p, y_eval), auroc=auroc(np.asarray(p, dtype=float), y_eval)) for name, p in scores.items()}
    final = three_way(scores["combiner"], y_eval, cut, scores["ours-clip"], scores["smaug"])

    names = generator_names(test_dir)
    groups = {}
    for k, (path, label) in enumerate(eval_files):
        groups.setdefault("real photos" if label == 0 else names.get(path.name, "unknown"), []).append(k)
    by_generator = {g: dict({name: accuracy(np.asarray(p, dtype=float)[idx], y_eval[idx]) for name, p in scores.items()},
                            combiner_labels=three_way(scores["combiner"][idx], y_eval[idx], cut,
                                                     np.asarray(scores["ours-clip"])[idx], np.asarray(scores["smaug"])[idx]))
                    for g, idx in sorted(groups.items())}

    medians = {name: {"real": float(np.median(feats[y == 0, k])), "ai": float(np.median(feats[y == 1, k]))}
               for k, name in enumerate(FEATURE_NAMES)}
    final_text = (f"{final['correct_among_decided']} of {final['decided']} decided answers correct, "
                  f"{final['unclear']} of {len(y_eval)} unclear ({how})")
    joblib.dump({"clip": clip_model, "features": feature_model, "combiner": combiner, "combiner_inputs": COMBINER_INPUTS,
                 "band": cut, "feature_names": FEATURE_NAMES, "medians": medians,
                 "tested": dict({k: tested_text(v, how) for k, v in acc.items()}, combiner=final_text),
                 "images": {"real": int((y == 0).sum()), "ai": int((y == 1).sum())}}, BUNDLE_FILE)
    save_json(RESULT_FILE, {
        "run": "fullsize-pilot", "train_folder": str(train_dir), "test_folder": str(test_dir) if test_dir else None,
        "train_real": int((y == 0).sum()), "train_ai": int((y == 1).sum()),
        "evaluation": how + " (scored once; Smaug is pretrained, so it is simply scored)",
        "accuracy_at_0_5": acc, "combiner": dict(final, band=cut, weights=weights),
        "combiner_training_out_of_fold": train_check,
        "unclear_rule": "the answer may only be real if neither our CLIP model nor Smaug says AI (>= 0.5); otherwise "
                        "unclear. Chosen on 25 Nov from two live test images (an AI doodle the combiner called real, and "
                        "an AI image our CLIP model called real), not from the test folder. The test folder has now been "
                        "scored three times: no rule (results/fullsize_pilot_before_unclear_rule.json), unclear whenever "
                        "CLIP and Smaug disagree (results/fullsize_pilot_disagree_rule.json), and this rule (this file).",
        "accuracy_by_generator": by_generator, "feature_medians": medians,
        "smaug_available": bool(not np.isnan(np.asarray(scores["smaug"], dtype=float)).all()),
        "seconds": round(time.time() - start, 1),
        "per_image": [{"image": f"{p.parent.name}/{p.name}", "truth": "ai" if t else "real",
                       "generator": names.get(p.name) if t else None,
                       **{k: (None if np.isnan(float(v[i])) else round(float(v[i]), 4)) for k, v in scores.items()},
                       "label": label_of(scores["combiner"][i], cut, scores["ours-clip"][i], scores["smaug"][i])}
                      for i, (p, t) in enumerate(eval_files)],
        "note": "Our models were trained on the training folder only; every AI tool in the test folder is missing from "
                "the training folder. Smaug's card lists several of these tools among its training generators, so its "
                "score here is not an unseen-tool result and can be too good.",
    })
    for name, a in acc.items():
        print(f"{name:18s} {a['correct']}/{a['total']}  accuracy {a['accuracy']}  95% {a['interval_95']}  AUROC {a['auroc']}")
    print("combiner three-way:", final)


if __name__ == "__main__":
    main()
