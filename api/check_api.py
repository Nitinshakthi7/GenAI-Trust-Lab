# check_api.py - a quick self-check of the whole API. It does not need a running server:
# FastAPI's TestClient calls the app directly.
#
# Run it from the project folder:   .venv\Scripts\python -m api.check_api
# It stops at the first check that fails and says which one. At the end it prints how many checks passed.
#
# What it checks:
#   - all 8 endpoints answer, with exactly the field names written in api/contract.json
#   - POST /generate works for every generator whose weights are ready (n=2, steps 20); the MNIST generators are
#     'training' until their weights exist; show_steps true adds 8 denoise frames (the last = the first image),
#     and without it the answer keeps exactly its old keys
#   - every PNG carries its provenance text note
#   - bad requests give 400 with {"error": ...}, an unknown generator gives 404,
#     a model that is not ready gives 503
#   - no endpoint is a stub any more: no answer has the X-Stub header
#   - POST /compare is real (no X-Stub header) and gives the right answers for pictures where we
#     know the answer in advance (same image twice, orange vs black, shifted copy, a cat-like pair,
#     two real CIFAR-10 calibration images), and 400 for bad uploads. If the pretrained ResNet-18 weights are in the
#     torch cache, regions_png_base64 must be a valid PNG and the 3 CNN rows must be in the table (else regions is null);
#     the CLIP row is always last (1.00 for the same picture twice, or 'not available' if CLIP cannot be loaded)
#   - POST /detect is real (no X-Stub header): a real CIFAR-10 calibration image, a VAE
#     fake and a 64x64 photo all answer 200 with the contract's field names (the 64x64 photo is
#     out_of_scope with the exact warning), bad uploads give 400, missing detector files give 503.
#     These checks read configs/detector_active.json to know WHICH detector is active:
#       dev detector   -> heatmap_source vae and a note starting 'Dev detector'
#       final detector -> heatmap_source ddpm (vae only if the diffusion weights cannot be loaded; a WARNING is
#                         printed then), a note starting 'Final detector', any label for a real image (real, ai or
#                         unclear are all honest), and the CNN's baseline_score above 0.5 on average for 10 VAE fakes
#                         and 10 diffusion fakes (calibration pictures only)
#   - POST /detect with a FINAL detector (a tiny made-up run in a temp folder, random weights; only the code path is
#     tested): the pointer's features vae + ddpm select it, the note is the final note, the heatmap is the diffusion one
#     (heatmap_source ddpm; skipped with a message if weights/d-ddpm-cifar.pt cannot be read right now), and a missing or
#     broken diffusion file falls back to the VAE heatmap (heatmap_source vae) and comes back by itself
#   - Grad-CAM (model/gradcam.py): the map is 32 x 32, its values are 0 to 1, it is not one flat colour,
#     and /detect now returns a real gradcam_png_base64 (a 128 x 128 PNG)
#   - POST /evidence is real (EXIF only, no X-Stub header): a JPG with EXIF and GPS gives the camera make and
#     model, says "GPSInfo: present (coordinates hidden)" and never returns the GPS numbers; a PNG without EXIF gives
#     found false; long values are cut, binary tags are skipped, at most 30 tags; bad uploads give 400
#   - GET /audit/results is real (no X-Stub header): with no results file it answers the placeholder (two planned
#     experiments, every rate null, no examples, a note starting with '[PLACEHOLDER - not measured yet]'); with a
#     results file (made here from tiny random pictures) it answers the file unchanged with the contract's field
#     names, and each closest-pair example has its three fields; a broken file gives 500 with {"error": ...}
#   - POST /audit/check is real (no X-Stub header, a note that does not start with 'stub:'): a training image of run a-500
#     (read straight from the CIFAR-10 files) gives distance 0, flag possible_copy, and its own picture back; the training-image
#     count equals the results file; the 3 nearest come back ranked, smallest first; random noise is not_flagged; a 64x64 picture
#     and a 30x20 JPG are resized (the note says so); a bad `run` or a missing image is 400; bad uploads are 400; a run whose
#     results file is missing (a temporary folder is used, nothing is renamed) is 503, a broken one 500; every other run that
#     already has a results file is checked the same way (count, distance 0)
#   - Auditor seeds and CLIP: /audit/results points carry mean, sd and n_seeds (one-seed and three-seed files both pass; a
#     point without n_seeds is 500); /audit/check's CLIP fields (clip_similarity per match, clip_nearest, clip_flag,
#     clip_threshold) are consistent, a training image and its mirror image find the original by CLIP, a seed run
#     (a-500-s2) is not a /audit/check run (400), and a run without a CLIP cache answers with null CLIP fields
#     (the background cache build is switched off here; the a-500 cache is built directly if it is missing)
#   - the CORS header is present for the Vite website address
# The final test split of CIFAR-10 is never opened here: real pictures come from the 'calibration' part only.

import base64
import io
import json
import os
import pickle
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image
from PIL.ExifTags import TAGS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from api import audit, compare, detect, generators
from api.common import MAX_UPLOAD_BYTES, PNG_START
from api.main import app
from model import compare_cnn, detector, features
from model.audit_summary import build_summary
from model.auditor import closest_pairs, nearest_distances
from model.data import get_part
from model.cnn import SmallCNN
from model.gradcam import gradcam
from model.make_fakes import to_float
from model.train_auditor_run import RUNS, subset_rows

torch.set_num_threads(2)         # keep the computer usable while other things run
ROOT = Path(__file__).resolve().parent.parent
CONTRACT = json.loads((Path(__file__).parent / "contract.json").read_text(encoding="utf-8"))["endpoints"]
VITE = "http://localhost:5173"
client = TestClient(app, raise_server_exceptions=False)
count = 0


def check(condition, message):
    """One check. Stops with the message if the condition is false."""
    global count
    assert condition, f"CHECK FAILED: {message}"
    count += 1


def picture_bytes(kind="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 50, 50)).save(buffer, format=kind)
    return buffer.getvalue()


OPTIONAL = {"full_check", "full_check_error",     # /detect adds these only for pictures that are not 32x32
            "denoise_frames"}                     # /generate adds this only with show_steps true


def same_names(real, example):
    """True if `real` has exactly the same field names as `example` (values are ignored).
    Exceptions: "fields" (the EXIF tags) is a free list, so only its being a dict is checked;
    the OPTIONAL fields may be missing, but when they are there their insides are checked too."""
    if isinstance(example, dict):
        return (isinstance(real, dict) and real.keys() - OPTIONAL == example.keys() - OPTIONAL
                and all(same_names(real[k], example[k]) for k in example if k != "fields" and k in real))
    if isinstance(example, list) and example:
        return isinstance(real, list) and all(same_names(item, example[0]) for item in real)
    return True


def call(endpoint):
    """Call one of the 8 endpoints with the example request from the contract."""
    method, path = endpoint.split(" ")
    png = picture_bytes()
    if endpoint == "POST /generate":
        return client.post(path, json=CONTRACT[endpoint]["request"])
    if endpoint == "POST /compare":
        files = {"image_a": ("a.png", png, "image/png"), "image_b": ("b.png", png, "image/png")}
        return client.post(path, files=files, data={"use_words": "false", "focus": "sleeves"})
    if method == "POST":
        return client.post(path, files={"image": ("x.png", png, "image/png")})
    return client.get(path)


def decode(image):
    return Image.open(io.BytesIO(base64.b64decode(image["png_base64"])))


# ------------------------------------------------------------ 1. the 8 endpoints

def check_endpoints():
    for endpoint in CONTRACT:
        reply = call(endpoint)
        check(reply.status_code == 200, f"{endpoint} answers 200 (got {reply.status_code}: {reply.text[:200]})")
        check(same_names(reply.json(), CONTRACT[endpoint]["response"]), f"{endpoint} has the contract's field names")
        check("X-Stub" not in reply.headers, f"{endpoint} is real, so no X-Stub header")
    check(client.get("/health").json() == {"status": "ok"}, "/health says ok")
    print(f"endpoints: all {len(CONTRACT)} answer with the contract's field names")


# ------------------------------------------------------------ 2. POST /generate

def check_generate(statuses):
    ids = [g["id"] for g in client.get("/generators").json()["generators"]]
    check(ids == ["g-vae-cifar", "g-vae-fmnist", "g-cddpm-fmnist", "g-vae-mnist", "g-cddpm-mnist", "g-ddpm-cifar",
                  "g-dcgan-cifar"], "catalogue ids")
    check(statuses["g-dcgan-cifar"] == "not_available", "g-dcgan-cifar is not_available")
    for gid in ("g-vae-mnist", "g-cddpm-mnist"):
        check(statuses[gid] in ("training", "ready"), f"{gid} has a config, so it is training (no weights yet) or ready")
    catalogue = {g["id"]: g for g in client.get("/generators").json()["generators"]}
    check(catalogue["g-cddpm-mnist"]["classes"] == [str(d) for d in range(10)] and catalogue["g-cddpm-mnist"]["dataset"] == "MNIST",
          "g-cddpm-mnist: dataset MNIST, classes are the digits 0-9")
    for gid in ("g-vae-cifar", "g-vae-fmnist"):
        check(statuses[gid] == "ready", f"{gid} must be ready")

    requests = [
        ("g-vae-cifar", dict(n=2), 2), ("g-vae-cifar", dict(interpolate={"seed_b": 7, "frames": 3}), 3),
        ("g-vae-fmnist", dict(n=2), 2), ("g-vae-fmnist", dict(interpolate={"seed_b": 7, "frames": 2}), 2),
        ("g-ddpm-cifar", dict(n=2, steps=20), 2),
        ("g-cddpm-fmnist", dict(n=2, steps=20, class_id=3, guidance=3.0), 2),
        ("g-cddpm-fmnist", dict(n=2, steps=20), 2),
        ("g-vae-mnist", dict(n=2), 2),
        ("g-cddpm-mnist", dict(n=2, steps=20, class_id=7, guidance=3.0), 2),
    ]
    for gid, extra, how_many in requests:
        if statuses[gid] != "ready":
            print(f"generate {gid}: skipped (status {statuses[gid]})")
            continue
        start = time.time()
        reply = client.post("/generate", json={"generator_id": gid, "seed": 42, **extra})
        seconds = time.time() - start
        check(reply.status_code == 200, f"generate {gid} {extra} answers 200 (got {reply.status_code}: {reply.text[:200]})")
        body = reply.json()
        check(len(body["images"]) == how_many, f"generate {gid} {extra} gives {how_many} images")
        check(body["metadata"]["label"] == "small model, expect low quality", "metadata label")
        check(not body["images"][0]["png_base64"].startswith("data:"), "plain base64, no data: prefix")
        for image in body["images"]:
            picture = decode(image)
            want = (28, 28) if "mnist" in gid else (32, 32)          # Fashion-MNIST and MNIST
            check(picture.size == want, f"{gid} image size is {want}")
            text = picture.text.get("provenance", "")
            check(f"model: {gid};" in text and "made by GenAI Trust Lab" in text, f"{gid} PNG carries the provenance note")
            if image["seed"] is not None:
                check(f"seed: {image['seed']};" in text, f"{gid} PNG note has its seed")
        print(f"generate {gid} {extra}: ok, {seconds:.2f} s on this CPU. PNG note: {decode(body['images'][0]).text['provenance']}")

    # Image number i uses seed + i: the second image of (seed 42, n=2) equals (seed 43, n=1).
    for gid, extra in (("g-vae-cifar", {}), ("g-ddpm-cifar", {"steps": 20})):
        if statuses[gid] == "ready":
            two = client.post("/generate", json={"generator_id": gid, "seed": 42, "n": 2, **extra}).json()
            one = client.post("/generate", json={"generator_id": gid, "seed": 43, "n": 1, **extra}).json()
            check(two["images"][1]["png_base64"] == one["images"][0]["png_base64"], f"{gid}: image i is seed + i")

    # "Watch it denoise": show_steps adds 8 frames of the first image; without it the answer has exactly the old keys.
    for gid in ("g-cddpm-fmnist", "g-ddpm-cifar", "g-cddpm-mnist"):
        if statuses[gid] != "ready":
            continue
        ask = {"generator_id": gid, "seed": 42, "n": 2, "steps": 20}
        plain = client.post("/generate", json=ask).json()
        check(list(plain) == ["generator_id", "images", "metadata", "note"], f"{gid}: without show_steps the answer has the old keys")
        reply = client.post("/generate", json={**ask, "show_steps": True})
        body, frames = reply.json(), reply.json().get("denoise_frames", [])
        check(reply.status_code == 200 and same_names(body, CONTRACT["POST /generate"]["response"]), f"{gid} show_steps: 200 with the contract's names")
        check([f["step"] for f in frames] == [1, 4, 6, 9, 12, 15, 17, 20], f"{gid} show_steps: 8 frames at evenly spaced steps (got {[f['step'] for f in frames]})")
        check(frames[-1]["png_base64"] == body["images"][0]["png_base64"], f"{gid} show_steps: the last frame is the first image")
        check(body["images"] == plain["images"], f"{gid} show_steps: the images are the same as without it")
        check(all(decode(f).size == decode(body["images"][0]).size for f in frames), f"{gid} show_steps: frames are PNGs of the image's size")
        print(f"generate {gid} show_steps: 8 frames at steps {[f['step'] for f in frames]}")


# ------------------------------------------------------- 3. errors (400, 404, 503)

def check_error(reply, status, what):
    check(reply.status_code == status, f"{what}: expected {status}, got {reply.status_code} ({reply.text[:150]})")
    check(list(reply.json()) == ["error"] and isinstance(reply.json()["error"], str), f"{what}: body is {{'error': text}}")


def check_bad_requests():
    bad = [
        ("n = 0", dict(generator_id="g-vae-cifar", n=0)),
        ("n = 17", dict(generator_id="g-vae-cifar", n=17)),
        ("seed below 0", dict(generator_id="g-vae-cifar", seed=-1)),
        ("steps on a VAE", dict(generator_id="g-vae-cifar", steps=20)),
        ("steps = 30", dict(generator_id="g-ddpm-cifar", steps=30)),
        ("guidance on g-ddpm-cifar", dict(generator_id="g-ddpm-cifar", guidance=3.0)),
        ("class_id on g-ddpm-cifar", dict(generator_id="g-ddpm-cifar", class_id=1)),
        ("class_id = 10", dict(generator_id="g-cddpm-fmnist", class_id=10)),
        ("guidance = 11", dict(generator_id="g-cddpm-fmnist", class_id=1, guidance=11)),
        ("guidance without class_id", dict(generator_id="g-cddpm-fmnist", guidance=3.0)),
        ("interpolate on a diffusion model", dict(generator_id="g-ddpm-cifar", interpolate={"seed_b": 1, "frames": 4})),
        ("show_steps on a VAE", dict(generator_id="g-vae-cifar", show_steps=True)),
        ("frames = 1", dict(generator_id="g-vae-cifar", interpolate={"seed_b": 1, "frames": 1})),
        ("frames = 17", dict(generator_id="g-vae-cifar", interpolate={"seed_b": 1, "frames": 17})),
        ("n is text", dict(generator_id="g-vae-cifar", n="many")),
        ("unknown field", dict(generator_id="g-vae-cifar", colour="red")),
        ("no generator_id", dict(n=1)),
    ]
    for what, body in bad:
        check_error(client.post("/generate", json=body), 400, what)
    check_error(client.post("/generate", content="{not json", headers={"Content-Type": "application/json"}), 400, "broken JSON")
    check_error(client.post("/generate", json={"generator_id": "g-nope"}), 404, "unknown generator")
    check_error(client.get("/no-such-address"), 404, "unknown address")
    check_error(client.post("/generate", json={"generator_id": "g-dcgan-cifar"}), 503, "g-dcgan-cifar (documented only)")
    print(f"errors: {len(bad) + 4} bad requests answered with the right status and {{'error': ...}}")


def check_bad_uploads():
    png = picture_bytes()
    check(client.post("/detect", files={"image": ("x.jpg", picture_bytes("JPEG"), "image/jpeg")}).status_code == 200, "a JPG is accepted")
    big = PNG_START + b"0" * MAX_UPLOAD_BYTES
    for path in ("/detect", "/audit/check", "/evidence"):
        check_error(client.post(path, files={"image": ("x.txt", b"hello", "text/plain")}), 400, f"{path} text file")
        check_error(client.post(path, files={"image": ("x.png", b"", "image/png")}), 400, f"{path} empty file")
        check_error(client.post(path, files={"image": ("x.png", big, "image/png")}), 400, f"{path} bigger than 5 MB")
        check_error(client.post(path), 400, f"{path} no file")
    files = {"image_a": ("a.png", png, "image/png"), "image_b": ("b.txt", b"hello", "text/plain")}
    check_error(client.post("/compare", files=files), 400, "/compare second file is text")
    files = {"image_a": ("a.png", png, "image/png"), "image_b": ("b.png", png, "image/png")}
    check_error(client.post("/compare", files=files, data={"focus": "x" * 101}), 400, "/compare focus too long")
    print("uploads: wrong type, empty, over 5 MB and missing files all give 400")


# ------------------------------------------------- 5. POST /compare with known answers

def image_png(pixels):
    """numpy picture (height x width x 3, numbers 0 to 255) -> PNG bytes."""
    buffer = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def solid(colour, width=224, height=224):
    return np.full((height, width, 3), colour)


def blob_scene(colour, centre_x):
    """A grey 224 x 224 background with one round blob of the given colour (a 'cat' made of one circle)."""
    scene = solid((128, 128, 128))
    y, x = np.mgrid[0:224, 0:224]
    scene[(y - 112) ** 2 + (x - centre_x) ** 2 < 50 ** 2] = colour
    return scene


def compare_pictures(pixels_a, pixels_b, **form):
    """Send two numpy pictures to POST /compare. Returns the reply."""
    files = {"image_a": ("a.png", image_png(pixels_a), "image/png"), "image_b": ("b.png", image_png(pixels_b), "image/png")}
    return client.post("/compare", files=files, data=form)


def table_of(reply):
    """The feature table as {feature: (text for image A, text for image B)}."""
    return {r["feature"]: (r["image_a"], r["image_b"]) for r in reply.json()["feature_table"]}


def cifar_test_pictures():
    """Two real CIFAR-10 images from the CALIBRATION part (read from data/cifar10). Two noise images if the data is missing."""
    batch = ROOT / "data" / "cifar10" / "cifar-10-batches-py" / "test_batch"
    if batch.exists():
        with open(batch, "rb") as file:
            data = pickle.load(file, encoding="bytes")[b"data"]
        first_two = [int(i) for i in get_part("calibration")[:2]]       # never a final_test image
        return [data[i].reshape(3, 32, 32).transpose(1, 2, 0) for i in first_two], f"CIFAR-10 calibration images {first_two}"
    noise = np.random.default_rng(0).integers(0, 256, (2, 32, 32, 3))
    return list(noise), "random noise (CIFAR-10 data not found)"


def check_compare():
    rows = ["mean brightness", "average colour (R,G,B)", "colour spread", "sharpness", "edge amount", "texture",
            "original size", "dominant colours", "colour histogram distance"]
    example = CONTRACT["POST /compare"]["response"]
    cnn_rows = ["CNN feature similarity (ResNet-18, whole image)", "CNN best-match similarity (A to B, B to A)",
                "CNN top guesses (ImageNet, may be wrong)"]
    try:
        compare_cnn.load()                  # the ResNet-18 weights come from the torch cache; no download
        cnn_ready = True
    except Exception as error:
        cnn_ready = False
        print(f"WARNING: ResNet-18 weights cannot be loaded ({error}); checking /compare without the CNN part")

    # (a) the same picture twice
    noise = np.random.default_rng(1).integers(0, 256, (224, 224, 3))
    reply = compare_pictures(noise, noise)
    table = table_of(reply)
    check(reply.status_code == 200 and reply.json()["similarity"]["ssim"] == 1.0, "same image twice: ssim is 1.0")
    check(table["dominant colours"][0] == table["dominant colours"][1], "same image twice: same dominant colours")
    check(table["colour histogram distance"] == ("0.00", "0.00"), "same image twice: histogram distance is 0")
    clip = table[compare.CLIP_FEATURE]
    if clip[0] == "not available":
        print("WARNING: CLIP cannot be loaded; the /compare CLIP row says 'not available'")
    check(clip in (("1.00", "1.00"), ("not available", "not available")), f"same image twice: CLIP similarity is 1.00 (got {clip})")
    check(decode({"png_base64": reply.json()["diff_heatmap_png_base64"]}).convert("RGB").getextrema()[1][0] == 255,
          "same image twice: the difference map is all white (no difference)")

    # (b) solid orange (300 x 200) against solid black (224 x 224)
    start = time.time()
    reply = compare_pictures(solid((255, 140, 0), 300, 200), solid((0, 0, 0)))
    seconds = time.time() - start
    body, table = reply.json(), table_of(reply)
    check(table["dominant colours"][0].startswith("orange") and table["dominant colours"][1].startswith("black"),
          f"orange vs black: dominant colours are named orange and black (got {table['dominant colours']})")
    check(float(table["mean brightness"][0]) > float(table["mean brightness"][1]), "orange vs black: orange is brighter")
    check(body["similarity"]["ssim"] < 0.1, f"orange vs black: ssim is low (got {body['similarity']['ssim']})")
    check(decode({"png_base64": body["diff_heatmap_png_base64"]}).convert("RGB").getextrema()[1][0] < 255,
          "orange vs black: the difference map is not all zero")
    check(table["original size"] == ("300x200", "224x224"), "orange vs black: original sizes are reported")
    print(f"compare orange vs black: ssim {body['similarity']['ssim']}, dominant colours {table['dominant colours']}, "
          f"brightness {table['mean brightness']}, {seconds:.2f} s")

    # (c) a picture against a copy shifted by 20 pixels
    scene = blob_scene((240, 140, 40), 80)
    ssim = compare_pictures(scene, np.roll(scene, 20, axis=1)).json()["similarity"]["ssim"]
    check(ssim < 1, f"shifted by 20 pixels: ssim is below 1 (got {ssim})")

    # (d) a cat-like pair: an orange blob and a black blob on the same grey background
    reply = compare_pictures(blob_scene((240, 140, 40), 80), blob_scene((20, 20, 20), 140), use_words="true", focus="fur colour")
    body, table = reply.json(), table_of(reply)
    check("orange" in table["dominant colours"][0], f"cat-like pair: A has orange (got {table['dominant colours'][0]})")
    check("black" in table["dominant colours"][1], f"cat-like pair: B has black (got {table['dominant colours'][1]})")
    check(body["source"] == "heatmap_only" and body["differences"] == [], "use_words true: still heatmap_only, no differences")
    check(body["summary"].startswith("Words are not enabled on this server (no vision API configured)."), "use_words true: summary starts with the notice")
    check(body["note"].startswith(example["note"]), "the note starts with the contract's note")
    print(f"compare cat-like pair: dominant colours {table['dominant colours']}")

    # (e) two real CIFAR-10 test images: every field, as the contract says
    (first, second), where = cifar_test_pictures()
    reply = compare_pictures(first, second)
    body = reply.json()
    check(reply.status_code == 200 and same_names(body, example), f"{where}: 200 with the contract's field names")
    check([r["feature"] for r in body["feature_table"]] == rows + (cnn_rows if cnn_ready else []) + [compare.CLIP_FEATURE],
          f"the feature table has the 9 rows{' and the 3 CNN rows' if cnn_ready else ''} and the CLIP row")
    clip = table_of(reply)[compare.CLIP_FEATURE]
    check(clip[0] == clip[1] and (clip[0] == "not available" or -1 <= float(clip[0]) <= 1), f"{where}: CLIP similarity is -1 to 1 (got {clip})")
    print(f"compare {where}: CLIP similarity {clip[0]}")
    check(all(isinstance(v, str) and v for r in body["feature_table"] for v in r.values()), "every table cell is a text")
    check(body["similarity"]["lpips"] is None, "lpips is null (not built)")
    if cnn_ready:
        regions = decode({"png_base64": body["regions_png_base64"]})
        check(regions.format == "PNG" and regions.size == (448, 224), f"CNN loaded: regions is a valid 448 x 224 PNG (got {regions.format} {regions.size})")
        check("pretrained ResNet-18" in body["note"], "CNN loaded: the note says the CNN is pretrained")
    else:
        check(body["regions_png_base64"] is None and "CNN part is not available" in body["note"], "no CNN: regions null and the note says so")
    check(body["source"] == "heatmap_only" and 2 <= body["summary"].count(". ") + 1 <= 4, "source heatmap_only, summary is 2 to 4 sentences")
    check(not body["diff_heatmap_png_base64"].startswith("data:"), "heatmap is plain base64")
    check(decode({"png_base64": body["diff_heatmap_png_base64"]}).size == (224, 224), "heatmap is 224 x 224")
    check(table_of(reply)["original size"] == ("32x32", "32x32"), "CIFAR original size is 32x32")
    check("X-Stub" not in reply.headers, "/compare is real: no X-Stub header")
    print(f"compare {where}: ssim {body['similarity']['ssim']}")
    print(f"compare summary: {body['summary']}")

    # (f) bad uploads
    good = image_png(solid((1, 2, 3), 8, 8))
    send = lambda a, b, **form: client.post("/compare", files={"image_a": ("a.png", a, "image/png"), "image_b": ("b.png", b, "image/png")}, data=form)
    check_error(send(b"hello, I am a text file", good), 400, "/compare text file renamed .png")
    check_error(send(good, PNG_START + b"0" * (6 * 1024 * 1024)), 400, "/compare 6 MB file")
    check_error(send(good, good, focus="x" * 101), 400, "/compare focus of 101 characters")
    check(send(good, good, focus="x" * 100).status_code == 200, "/compare focus of exactly 100 characters is fine")
    check_error(send(good, PNG_START + b"this is not a real picture"), 400, "/compare PNG start but broken picture")
    check_error(send(good, image_png(np.zeros((1, 1, 3)))[:20]), 400, "/compare cut-off PNG")
    buffer = io.BytesIO()
    Image.new("L", (7000, 7000)).save(buffer, format="PNG")                  # 49 million pixels, but a small file
    check_error(send(good, buffer.getvalue()), 400, "/compare more than 40 million pixels")
    buffer = io.BytesIO()
    Image.new("RGB", (30, 20), (200, 50, 50)).save(buffer, format="JPEG")
    check(send(good, buffer.getvalue()).status_code == 200, "/compare accepts a JPG")
    print("compare: known answers, CIFAR-10 pair and bad uploads all as expected")


def check_not_ready():
    """Pretend the weights folder is empty / holds a broken file: the API must answer 503, not crash."""
    real_dir = generators.WEIGHTS_DIR
    generators._cache.clear()
    with tempfile.TemporaryDirectory() as fake_dir:
        generators.WEIGHTS_DIR = Path(fake_dir)
        try:
            check_error(client.post("/generate", json={"generator_id": "g-vae-cifar"}), 503, "no weights file yet")
            statuses = {g["id"]: g["status"] for g in client.get("/generators").json()["generators"]}
            check(statuses["g-vae-cifar"] == "training" and statuses["g-dcgan-cifar"] == "not_available", "no weights: status 'training'")
            (Path(fake_dir) / "g-vae-cifar.pt").write_bytes(b"this is not a real weights file")
            check_error(client.post("/generate", json={"generator_id": "g-vae-cifar"}), 503, "broken weights file")
        finally:
            generators.WEIGHTS_DIR = real_dir
            generators._cache.clear()
    check(client.post("/generate", json={"generator_id": "g-vae-cifar"}).status_code == 200, "works again with the real folder")
    print("not ready: missing and broken weights files give 503")


# ------------------------------------------------------- 6. POST /detect (the dev OR the final detector)

WARNING_TEXT = "This detector was trained only on 32x32 CIFAR-10-style images. Results on other photos are not meaningful."
REAL_POINTER = detector.ACTIVE_FILE      # configs/detector_active.json (some checks below swap detector.ACTIVE_FILE for a moment)


def final_is_active():
    """True if configs/detector_active.json says the FINAL detector is in use (its features include 'ddpm'),
    False for the dev detector. Read fresh each time, so the checks always match what /detect really does."""
    return "ddpm" in json.loads(REAL_POINTER.read_text(encoding="utf-8")).get("features", ["vae"])


def send_detect(picture, kind="PNG"):
    """Send a PIL picture to POST /detect. Returns the reply."""
    buffer = io.BytesIO()
    picture.save(buffer, format=kind)
    return client.post("/detect", files={"image": ("x." + kind.lower(), buffer.getvalue(), "image/" + kind.lower())})


def detect_pictures():
    """(a real CIFAR-10 picture from the CALIBRATION part, a VAE fake from the calibration fakes).
    The index comes from data/splits.csv; the final_test part is never opened."""
    with open(ROOT / "data" / "cifar10" / "cifar-10-batches-py" / "test_batch", "rb") as file:
        data = pickle.load(file, encoding="bytes")[b"data"]
    index = int(get_part("calibration")[0])
    real = Image.fromarray(data[index].reshape(3, 32, 32).transpose(1, 2, 0))
    with np.load(ROOT / "fakes" / "g-vae-cifar_fakes.npz") as saved:
        fake = Image.fromarray(saved["calibration"][0])               # only the calibration array is read
    return real, fake, index


def check_answer(reply, what):
    """The checks every good /detect answer must pass. Returns the answer."""
    check(reply.status_code == 200, f"{what}: /detect answers 200 (got {reply.status_code}: {reply.text[:200]})")
    body = reply.json()
    check(same_names(body, CONTRACT["POST /detect"]["response"]), f"{what}: the contract's field names")
    check("X-Stub" not in reply.headers, f"{what}: /detect is real, so no X-Stub header")
    check(body["label"] in ("real", "ai", "unclear"), f"{what}: label is real, ai or unclear")
    check(0 <= body["score"] <= 1 and 0 <= body["baseline_score"] <= 1, f"{what}: score and baseline_score are in [0,1]")
    check(isinstance(body["vae_score"], float) and body["vae_score"] >= 0, f"{what}: vae_score is a number, 0 or more")
    final = final_is_active()
    if not final:
        check(body["heatmap_source"] == "vae", f"{what}: heatmap_source is vae")
    elif body["heatmap_source"] == "vae":      # allowed only when the diffusion model could not be loaded
        check(detect._detector.ddpm is None, f"{what}: heatmap_source is vae although the diffusion weights are loaded")
        print(f"WARNING: {what}: the diffusion weights (weights/d-ddpm-cifar.pt) cannot be loaded, so the heatmap is the VAE one")
    else:
        check(body["heatmap_source"] == "ddpm", f"{what}: heatmap_source is ddpm (got {body['heatmap_source']})")
    heatmap = body["heatmap_png_base64"]
    check(not heatmap.startswith("data:") and decode({"png_base64": heatmap}).size == (128, 128), f"{what}: heatmap is a plain base64 128 x 128 PNG")
    cam = body["gradcam_png_base64"]
    check(isinstance(cam, str) and not cam.startswith("data:"), f"{what}: gradcam is plain base64 text (no data: prefix)")
    cam_picture = decode({"png_base64": cam})
    check(cam_picture.format == "PNG" and cam_picture.size == (128, 128), f"{what}: gradcam decodes to a 128 x 128 PNG")
    if final:
        check(body["note"].startswith("Final detector"), f"{what}: the note starts with 'Final detector'")
    else:
        check(body["note"].startswith("Dev detector") and "not a result" in body["note"], f"{what}: the honesty note is there")
    check("Grad-CAM shows where the CNN baseline looked; it does not prove why the image is real or AI." in body["note"],
          f"{what}: the Grad-CAM sentence is in the note")
    return body


def check_detect():
    real, fake, index = detect_pictures()

    start = time.time()
    reply = send_detect(real)
    seconds = time.time() - start
    body = check_answer(reply, "real calibration image")
    check(body["out_of_scope"] is False and body["warning"] is None and body["input_size"] == "32x32",
          "real calibration image: in scope, no warning, input_size 32x32")
    print(f"detect real calibration image {index}: label {body['label']}, score {body['score']}, baseline {body['baseline_score']}, "
          f"vae_score {body['vae_score']}, {seconds:.2f} s")
    check(send_detect(real).json() == body, "the same picture twice gives the same answer")

    body = check_answer(send_detect(fake), "VAE fake")
    check(body["out_of_scope"] is False and body["warning"] is None, "VAE fake: in scope, no warning")
    print(f"detect VAE fake (calibration 0): label {body['label']}, score {body['score']}, baseline {body['baseline_score']}, vae_score {body['vae_score']}")

    if final_is_active():
        # The CNN baseline separates fakes from real pictures well on calibration data, so the MEAN baseline_score of
        # 10 fakes must be above 0.5 (a mean, so one odd picture cannot fail the check). Two kinds of fakes.
        for name in ("g-vae-cifar", "g-ddpm-cifar"):
            file = ROOT / "fakes" / f"{name}_fakes.npz"
            if not file.exists():
                print(f"detect final: {file.name} not found, its fake-picture check is skipped")
                continue
            with np.load(file) as saved:
                pictures = saved["calibration"][:10]                  # only the calibration array is read, never 'test'
            start = time.time()
            scores = [send_detect(Image.fromarray(p)).json()["baseline_score"] for p in pictures]
            mean = sum(scores) / len(scores)
            check(len(scores) == 10 and mean > 0.5, f"final detector: mean baseline_score of 10 {name} calibration fakes is above 0.5 (got {mean:.3f})")
            print(f"detect final, 10 {name} calibration fakes: mean baseline_score {mean:.3f}, {(time.time() - start) / len(scores):.2f} s per picture")

    body = check_answer(send_detect(real.resize((64, 64), Image.Resampling.BILINEAR)), "64x64 photo")
    check(body["out_of_scope"] is True and body["input_size"] == "64x64", "64x64 photo: out_of_scope true, input_size 64x64")
    check(body["warning"] == WARNING_TEXT, "64x64 photo: the exact warning text")
    print(f"detect 64x64 photo: out_of_scope {body['out_of_scope']}, warning {body['warning']!r}")

    body = check_answer(send_detect(Image.new("RGB", (30, 20), (200, 50, 50)), "JPEG"), "30x20 JPG")
    check(body["out_of_scope"] is True and body["input_size"] == "30x20" and body["warning"] == WARNING_TEXT, "30x20 JPG: out of scope, original size reported")

    # bad uploads (the 5 MB and empty-file cases are already checked in check_bad_uploads)
    good = image_png(solid((1, 2, 3), 8, 8))
    send = lambda data: client.post("/detect", files={"image": ("x.png", data, "image/png")})
    check_error(send(b"hello, I am a text file"), 400, "/detect text file named .png")
    check_error(send(PNG_START + b"this is not a real picture"), 400, "/detect PNG start but broken picture")
    check_error(send(good[:20]), 400, "/detect cut-off PNG")
    buffer = io.BytesIO()
    Image.new("L", (7000, 7000)).save(buffer, format="PNG")                  # 49 million pixels, but a small file
    check_error(send(buffer.getvalue()), 400, "/detect more than 40 million pixels")
    print("detect: real image, VAE fake, 64x64 photo, 30x20 JPG and bad uploads all as expected")


def check_detect_not_ready():
    """Pretend the detector's files are missing or broken: /detect must answer 503, not crash."""
    png = picture_bytes()
    post = lambda: client.post("/detect", files={"image": ("x.png", png, "image/png")})
    real_pointer, real_weights, real_detector = detector.ACTIVE_FILE, detector.WEIGHTS_DIR, detect._detector
    detect._detector = None                                   # forget the loaded detector, so it loads again
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        try:
            detector.ACTIVE_FILE = folder / "no_such_file.json"
            check_error(post(), 503, "/detect: the active-detector file is missing")
            (folder / "results.json").write_text(json.dumps({"run_id": "detector_dev", "dev_report": {"logistic_regression": {}}}))
            (folder / "pointer.json").write_text(json.dumps({"active_run": "detector_dev", "thresholds_from": str(folder / "results.json")}))
            detector.ACTIVE_FILE = folder / "pointer.json"
            check_error(post(), 503, "/detect: a threshold is missing")
            detector.ACTIVE_FILE = real_pointer
            detector.WEIGHTS_DIR = folder                     # no detector_lr.joblib in here
            check_error(post(), 503, "/detect: the logistic regression file is missing")
            first_file = f"{json.loads(real_pointer.read_text(encoding='utf-8'))['active_run']}_lr_vae.joblib" if final_is_active() else "detector_lr.joblib"
            (folder / first_file).write_bytes(b"this is not a real weights file")      # the first file the active detector reads
            check_error(post(), 503, "/detect: the logistic regression file is broken")
        finally:
            detector.ACTIVE_FILE, detector.WEIGHTS_DIR, detect._detector = real_pointer, real_weights, real_detector
    check(post().status_code == 200, "/detect works again with the real files")
    print("detect not ready: missing threshold or weights files and a broken file give 503")


def make_made_up_final_run(folder):
    """A tiny made-up 'final run' in `folder` (random weights, made-up thresholds). It only tests the code path:
    the numbers it gives mean nothing. The real one is made by model/run_final_detector.py."""
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 20)
    for name, columns in (("lr_vae", 2), ("lr_vae_ddpm", 6)):
        model = make_pipeline(StandardScaler(), LogisticRegression()).fit(rng.normal(size=(40, columns)), labels)
        joblib.dump(model, folder / f"detector_final_{name}.joblib")
    torch.save(SmallCNN().state_dict(), folder / "detector_final_cnn.pt")
    limits = {"low_threshold": 0.3, "high_threshold": 0.7}
    report = {"run_id": "detector_final", "calibration_report": {name: limits for name in ("lr_vae", "lr_vae_ddpm", "cnn")}}
    (folder / "results.json").write_text(json.dumps(report))
    pointer = {"active_run": "detector_final", "thresholds_from": str(folder / "results.json"), "features": ["vae", "ddpm"]}
    (folder / "pointer.json").write_text(json.dumps(pointer))


def check_detect_final():
    """POST /detect when configs/detector_active.json points to a FINAL run (features vae + ddpm): the diffusion
    heatmap, the final note, and the fallback to the VAE heatmap when the diffusion file cannot be read."""
    real, fake, index = detect_pictures()
    saved = detector.ACTIVE_FILE, detector.WEIGHTS_DIR, features.WEIGHTS_DIR, detect._detector
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        make_made_up_final_run(folder)
        try:
            detector.ACTIVE_FILE, detector.WEIGHTS_DIR, detect._detector = folder / "pointer.json", folder, None
            start = time.time()
            reply = send_detect(real)
            seconds = time.time() - start
            body = reply.json()
            check(reply.status_code == 200 and same_names(body, CONTRACT["POST /detect"]["response"]),
                  f"final detector: 200 with the contract's field names (got {reply.status_code}: {reply.text[:200]})")
            check(detect._detector.final is True, "final detector: the pointer's features vae + ddpm select the final detector")
            check(body["note"].startswith("Final detector") and "Grad-CAM shows where the CNN baseline looked" in body["note"],
                  "final detector: the final note (not the dev note) with the Grad-CAM sentence")
            check(body["label"] in ("real", "ai", "unclear") and 0 <= body["score"] <= 1 and 0 <= body["baseline_score"] <= 1,
                  "final detector: label, score and baseline_score are valid")
            if detect._detector.ddpm is None:
                print("final detector: the diffusion model weights/d-ddpm-cifar.pt cannot be read right now; the 'ddpm' heatmap check is skipped")
            else:
                check(body["heatmap_source"] == "ddpm", "final detector: heatmap_source is ddpm")
                check(decode({"png_base64": body["heatmap_png_base64"]}).size == (128, 128), "final detector: the ddpm heatmap is a 128 x 128 PNG")
                check(send_detect(real).json() == body, "final detector: the same picture twice gives the same answer")
                print(f"detect final detector (made-up run): heatmap_source {body['heatmap_source']}, first request incl. loading {seconds:.2f} s")
                # the diffusion file cannot be read (missing, then half-written): VAE heatmap and VAE-only score
                detect._detector.ddpm = None
                features.WEIGHTS_DIR = folder
                for what in ("missing", "broken"):
                    if what == "broken":
                        (folder / "d-ddpm-cifar.pt").write_bytes(b"half written file")
                    body = send_detect(real).json()
                    check(body["heatmap_source"] == "vae" and body["label"] in ("real", "ai", "unclear") and 0 <= body["score"] <= 1,
                          f"final detector, diffusion file {what}: falls back to the VAE heatmap and score")
                features.WEIGHTS_DIR = saved[2]
                check(send_detect(real).json()["heatmap_source"] == "ddpm", "final detector: the diffusion model is picked up again by itself")
        finally:
            detector.ACTIVE_FILE, detector.WEIGHTS_DIR, features.WEIGHTS_DIR, detect._detector = saved
    print("detect final detector: final note, ddpm heatmap and the fallback as expected")


# ------------------------------------------------------- 7. Grad-CAM (model/gradcam.py)

def check_gradcam():
    """Grad-CAM on the trained CNN, for a real calibration image and a VAE fake."""
    real, fake, index = detect_pictures()
    cnn = detector.Detector._load_cnn()                       # the trained CNN, loaded on its own
    for what, picture in (("real calibration image", real), ("VAE fake", fake)):
        cam = gradcam(cnn, to_float(np.array(picture)[None]))
        check(cam.shape == (32, 32), f"Grad-CAM {what}: the map is 32 x 32 (got {cam.shape})")
        check(float(cam.min()) >= 0 and float(cam.max()) <= 1, f"Grad-CAM {what}: values are within 0..1")
        check(float(cam.max()) > float(cam.min()), f"Grad-CAM {what}: the map is not one flat colour")
        print(f"gradcam {what}: shape {cam.shape}, min {float(cam.min()):.2f}, max {float(cam.max()):.2f}, spread (std) {float(cam.std()):.3f}")
    check(all(p.grad is None for p in cnn.parameters()), "Grad-CAM leaves no gradients behind in the CNN")


# ------------------------------------------------------- 8. POST /evidence (EXIF only)

GPS_NUMBERS = (47.0, 36.0, 22.5, 122.0, 19.0, 59.25)         # the GPS numbers we hide in the test photo
EXIF_NOTE_TEXT = "Missing EXIF is weak evidence; present EXIF does not mean real."


def jpeg_with_exif(extra=None):
    """A small JPG in memory with Make 'TestCam', Model 'X1', a DateTime and a GPS block. `extra` adds more tags."""
    exif = Image.Exif()
    exif[0x010F], exif[0x0110], exif[0x0132] = "TestCam", "X1", "2024:05:06 07:08:09"
    exif[0x8825] = {1: "N", 2: GPS_NUMBERS[:3], 3: "W", 4: GPS_NUMBERS[3:]}
    for number, value in (extra or {}).items():
        exif[number] = value
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (10, 200, 90)).save(buffer, format="JPEG", exif=exif.tobytes())
    return buffer.getvalue()


def send_evidence(data, name="x.jpg", kind="image/jpeg"):
    return client.post("/evidence", files={"image": (name, data, kind)})


def check_evidence():
    example = CONTRACT["POST /evidence"]["response"]

    # (a) a JPG that has EXIF and GPS
    reply = send_evidence(jpeg_with_exif())
    check(reply.status_code == 200, f"/evidence JPG with EXIF: answers 200 (got {reply.status_code}: {reply.text[:200]})")
    body = reply.json()
    check(same_names(body, example), "/evidence JPG with EXIF: the contract's field names")
    check("X-Stub" not in reply.headers, "/evidence is real, so no X-Stub header")
    fields = body["exif"]["fields"]
    check(body["exif"]["found"] is True, "/evidence JPG with EXIF: found is true")
    check(fields.get("Make") == "TestCam" and fields.get("Model") == "X1", f"/evidence: Make and Model are in the fields (got {fields})")
    check(fields.get("DateTime") == "2024:05:06 07:08:09", "/evidence: DateTime is in the fields")
    check(fields.get("GPSInfo") == "present (coordinates hidden)", "/evidence: says 'GPSInfo: present (coordinates hidden)'")
    for number in GPS_NUMBERS:
        check(str(number) not in reply.text and str(int(number)) not in reply.text, f"/evidence: the GPS number {number} is NOT in the response")
    check("GPSLatitude" not in reply.text and "GPSLongitude" not in reply.text, "/evidence: no GPS tag names with coordinates")
    check(body["exif"]["does_not_prove"] == EXIF_NOTE_TEXT, "/evidence: the exif does_not_prove text")
    check(body["c2pa"] == {"status": "not_checked", "summary": None, "does_not_prove": "A missing manifest means nothing."}, "/evidence: c2pa is not_checked")
    check(body["ela_png_base64"] is None and body["fft_png_base64"] is None, "/evidence: ela and fft are null (not built)")
    check(body["note"] == "Evidence, not proof. Nothing is stored.", "/evidence: the note text")
    print(f"evidence JPG with EXIF: found {body['exif']['found']}, fields {fields}")

    # (b) a PNG with no EXIF
    body = send_evidence(picture_bytes(), "x.png", "image/png").json()
    check(body["exif"]["found"] is False and body["exif"]["fields"] == {}, "/evidence PNG without EXIF: found is false, no fields")
    check(body["exif"]["does_not_prove"] == EXIF_NOTE_TEXT, "/evidence PNG without EXIF: the does_not_prove text")
    check(body["c2pa"]["status"] == "not_checked", "/evidence PNG without EXIF: c2pa is not_checked")

    # (c) a busy JPG: a long text is cut to 100 characters, a binary tag is skipped, at most 30 tags come back
    extra = {number: 7 for number in range(0x0100, 0x0160) if number in TAGS}      # many numeric tags
    extra.update({0x010E: "x" * 300, 0x010F: "TestCam", 0x0110: "X1", 0x0132: "2024:05:06 07:08:09",
                  0x927C: b"\x00\x01binary-blob"})                                  # (set last, so they are not overwritten)
    fields = send_evidence(jpeg_with_exif(extra)).json()["exif"]["fields"]
    check(len(fields) <= 30, f"/evidence: at most 30 tags (got {len(fields)})")
    check(len(fields.get("ImageDescription", "x")) == 100, "/evidence: a 300-character value is cut to 100 characters")
    check("MakerNote" not in fields, "/evidence: the binary MakerNote tag is skipped")
    check(fields.get("GPSInfo") == "present (coordinates hidden)", "/evidence: the GPS line is kept even when there are many tags")
    check(all(isinstance(v, str) and 0 < len(v) <= 100 for v in fields.values()), "/evidence: every value is a short text")

    # (d) bad uploads (the empty, 5 MB and no-file cases are already checked in check_bad_uploads)
    check_error(send_evidence(b"hello, I am a text file", "x.png", "image/png"), 400, "/evidence text file named .png")
    check_error(send_evidence(PNG_START + b"this is not a real picture", "x.png", "image/png"), 400, "/evidence PNG start but broken picture")
    buffer = io.BytesIO()
    Image.new("L", (7000, 7000)).save(buffer, format="PNG")                  # 49 million pixels, but a small file
    check_error(send_evidence(buffer.getvalue(), "x.png", "image/png"), 400, "/evidence more than 40 million pixels")
    print("evidence: EXIF JPG, PNG without EXIF, many tags and bad uploads all as expected")


# ------------------------------------------------------- 9. GET /audit/results (real; placeholder until the runs exist)

PLACEHOLDER_START = "[PLACEHOLDER - not measured yet]"
PAIR_NAMES = {"generated_png_base64", "nearest_png_base64", "distance"}


def get_audit(path):
    """GET /audit/results while AUDIT_RESULTS_FILE points at `path` (the setting is removed again afterwards)."""
    os.environ["AUDIT_RESULTS_FILE"] = str(path)
    try:
        return client.get("/audit/results")
    finally:
        os.environ.pop("AUDIT_RESULTS_FILE", None)


def check_audit():
    example = CONTRACT["GET /audit/results"]["response"]
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)

        # (a) no results file yet: the placeholder
        reply = get_audit(folder / "no_such_file.json")
        body = reply.json()
        check(reply.status_code == 200 and same_names(body, example), f"audit placeholder: 200 with the contract's field names (got {reply.status_code})")
        check("X-Stub" not in reply.headers, "/audit/results is real, so no X-Stub header")
        rates = [point["memorization_rate"] for experiment in body["experiments"] for point in experiment["points"]]
        check(len(body["experiments"]) == 2 and rates == [None] * 4, f"audit placeholder: two experiments, every rate null (got {rates})")
        check(all(p["n_seeds"] == 0 and p["memorization_rate_mean"] is None and p["memorization_rate_sd"] is None and p["flagged_mean"] is None
                  for e in body["experiments"] for p in e["points"]), "audit placeholder: n_seeds 0 and every mean and sd null")
        check(body["examples"] == [], "audit placeholder: no examples")
        check(body["note"].startswith(PLACEHOLDER_START), "audit placeholder: the note starts with the PLACEHOLDER words")
        check("not decide legal copying" in body["note"], "audit placeholder: the note says similarity only, not legal copying")
        print(f"audit placeholder: {[e['name'] for e in body['experiments']]}, note: {body['note']}")

        # (b) a results file made from tiny random pictures (the first 'generated' picture is an exact copy of a training one)
        g = torch.Generator().manual_seed(0)
        train = torch.rand(20, 3, 32, 32, generator=g) * 2 - 1
        generated = torch.cat([train[:1], torch.rand(5, 3, 32, 32, generator=g) * 2 - 1])
        pairs = closest_pairs(generated, train, k=3)
        runs = {"a-500": {"memorization_rate": 0.25, "flagged": 4, "examples": pairs}, "a-10000": {"memorization_rate": 0.0625, "flagged": 1, "examples": pairs}}
        results_file = folder / "audit_results.json"
        results_file.write_text(json.dumps(build_summary(runs)), encoding="utf-8")        # a-500-dup is "not done yet"
        reply = get_audit(results_file)
        body = reply.json()
        check(reply.status_code == 200 and same_names(body, example), f"audit results file: 200 with the contract's field names (got {reply.status_code})")
        check(body == json.loads(results_file.read_text(encoding="utf-8")), "audit results file: the answer is the file, unchanged")
        check("X-Stub" not in reply.headers, "audit results file: no X-Stub header")
        points = [[p["memorization_rate"] for p in e["points"]] for e in body["experiments"]]
        check(points == [[0.25, 0.0625], [0.25, None]], f"audit results file: the rates are the file's (got {points})")
        check(len(body["examples"]) == 6 and all(set(pair) == PAIR_NAMES for pair in body["examples"]), "audit results file: 6 examples with the three field names")
        check(body["examples"][0]["distance"] == 0.0, "audit results file: the exact copy has distance 0")
        for key in ("generated_png_base64", "nearest_png_base64"):
            picture = decode({"png_base64": body["examples"][0][key]})
            check(picture.format == "PNG" and picture.size == (128, 128), f"audit results file: {key} is a 128 x 128 PNG")
            check(not body["examples"][0][key].startswith("data:"), f"audit results file: {key} is plain base64")
        print(f"audit results file: rates {points}, {len(body['examples'])} examples, distances {[p['distance'] for p in body['examples'][:3]]}")
        one = [[(p["n_seeds"], p["memorization_rate_mean"], p["memorization_rate_sd"], p["flagged_mean"], p["flagged_sd"]) for p in e["points"]] for e in body["experiments"]]
        check(one == [[(1, 0.25, None, 4, None), (1, 0.0625, None, 1, None)], [(1, 0.25, None, 4, None), (0, None, None, None, None)]],
              f"audit results file, one seed: n_seeds 1, mean = the rate, sd null (got {one})")

        # (b2) the same with two extra seeds of a-500 (a-500-s2, a-500-s3): mean and sd over the 3 seeds; examples still from seed 42 only
        three_file = folder / "audit_results_three_seeds.json"
        runs3 = {**runs, "a-500-s2": {"memorization_rate": 0.3, "flagged": 5, "examples": pairs}, "a-500-s3": {"memorization_rate": 0.35, "flagged": 6, "examples": pairs}}
        three_file.write_text(json.dumps(build_summary(runs3)), encoding="utf-8")
        reply = get_audit(three_file)
        body = reply.json()
        check(reply.status_code == 200 and same_names(body, example) and body == json.loads(three_file.read_text(encoding="utf-8")),
              f"audit results file, three seeds: 200, the contract's field names, the file unchanged (got {reply.status_code})")
        p = body["experiments"][0]["points"][0]
        check((p["memorization_rate"], p["n_seeds"], p["memorization_rate_mean"], p["memorization_rate_sd"], p["flagged_mean"], p["flagged_sd"]) == (0.25, 3, 0.3, 0.05, 5, 1),
              f"audit results file, three seeds: seed-42 rate kept, mean 0.3, sd 0.05, flagged 5 +/- 1 (got {p})")
        check(len(body["examples"]) == 6, "audit results file, three seeds: the seed runs add no examples")
        print(f"audit results file, three seeds: a-500 point {p}")

        # (c) a broken results file gives 500 with {"error": ...}
        good = json.loads(results_file.read_text(encoding="utf-8"))
        missing_field = json.loads(results_file.read_text(encoding="utf-8"))
        del missing_field["examples"][0]["distance"]
        bad_point = json.loads(results_file.read_text(encoding="utf-8"))
        bad_point["experiments"][0]["points"][0] = {"x": 500}
        no_seeds = json.loads(results_file.read_text(encoding="utf-8"))
        del no_seeds["experiments"][1]["points"][1]["n_seeds"]
        for what, content in (("text that is not JSON", "this is not json {"), ("an empty file", ""), ("a JSON list", "[]"),
                              ("other field names", json.dumps({"hello": 1})), ("a missing 'note'", json.dumps({k: v for k, v in good.items() if k != "note"})),
                              ("an example without 'distance'", json.dumps(missing_field)), ("a point without 'memorization_rate'", json.dumps(bad_point)),
                              ("a point without 'n_seeds'", json.dumps(no_seeds))):
            results_file.write_text(content, encoding="utf-8")
            check_error(get_audit(results_file), 500, f"audit results file with {what}")
    print("audit: placeholder, one-seed and three-seed results files and 8 kinds of broken file all as expected")


# ------------------------------------------------------- 10. POST /audit/check (real)

CHECK_NAMES = {"rank", "distance", "phash_distance", "clip_similarity", "image_png_base64"}
CHECK_TOP = {"nearest", "flag", "clip_nearest", "clip_flag", "clip_threshold", "note"}
BANNED_WORDS = ("copied", "stolen", "plagiarism", "infringement")        # never used as a conclusion


def train_picture(index):
    """CIFAR-10 TRAIN picture number `index` (32 x 32 x 3 numbers), read straight from the data_batch files,
    not through the API code (so the check does not use what it is checking)."""
    with open(ROOT / "data" / "cifar10" / "cifar-10-batches-py" / f"data_batch_{index // 10000 + 1}", "rb") as file:
        return pickle.load(file, encoding="bytes")[b"data"][index % 10000].reshape(3, 32, 32).transpose(1, 2, 0)


def send_check(data, **form):
    """Send picture bytes to POST /audit/check (form: optional run). Returns the reply."""
    return client.post("/audit/check", files={"image": ("x.png", data, "image/png")}, data=form)


def check_good_answer(reply, what):
    """The checks every good /audit/check answer must pass. Returns the answer."""
    check(reply.status_code == 200, f"{what}: /audit/check answers 200 (got {reply.status_code}: {reply.text[:200]})")
    body = reply.json()
    example = dict(CONTRACT["POST /audit/check"]["response"])
    if body.get("clip_nearest") is None:
        example["clip_nearest"] = None                  # null while the run's CLIP cache is not ready
    check(same_names(body, example), f"{what}: the contract's field names")
    check(set(body) == CHECK_TOP and all(set(n) == CHECK_NAMES for n in body["nearest"]), f"{what}: exactly the contract's keys")
    sims = [n["clip_similarity"] for n in body["nearest"]]
    if body["clip_flag"] is None:
        check(body["clip_nearest"] is None and body["clip_threshold"] is None and sims == [None] * 3
              and ("CLIP similarity is not ready yet" in body["note"] or "CLIP similarity is not available" in body["note"]),
              f"{what}: CLIP not ready: every CLIP field null and the note says so")
    else:
        best, cut = body["clip_nearest"], body["clip_threshold"]
        check(0 < cut <= 1 and all(isinstance(s, float) and -1 <= s <= 1.0001 for s in sims + [best["clip_similarity"]]),
              f"{what}: CLIP threshold and similarities are numbers up to 1 (got {cut}, {sims}, {best['clip_similarity']})")
        check(best["clip_similarity"] >= max(sims) and body["clip_flag"] == ("possible_copy" if best["clip_similarity"] > cut else "not_flagged"),
              f"{what}: clip_nearest is the most similar and clip_flag follows the CLIP threshold")
        check(best["distance"] >= body["nearest"][0]["distance"] and decode({"png_base64": best["image_png_base64"]}).size == (32, 32)
              and f"{cut:.3f}" in body["note"], f"{what}: clip_nearest has a pixel distance and a 32 x 32 PNG; the note gives the CLIP threshold")
    check("X-Stub" not in reply.headers and not body["note"].startswith("stub:"), f"{what}: real answer, no X-Stub header, no 'stub:' note")
    near = body["nearest"]
    check([n["rank"] for n in near] == [1, 2, 3], f"{what}: the 3 nearest, ranks 1 to 3")
    distances = [n["distance"] for n in near]
    check(distances == sorted(distances) and distances[0] >= 0, f"{what}: distances are 0 or more, smallest first (got {distances})")
    check(all(isinstance(n["phash_distance"], int) and 0 <= n["phash_distance"] <= 64 for n in near), f"{what}: phash_distance is a whole number 0 to 64")
    for n in near:
        picture = decode({"png_base64": n["image_png_base64"]})
        check(picture.format == "PNG" and picture.size == (32, 32) and not n["image_png_base64"].startswith("data:"),
              f"{what}: image of rank {n['rank']} is a plain base64 32 x 32 PNG")
    check(body["flag"] in ("possible_copy", "not_flagged"), f"{what}: flag is possible_copy or not_flagged")
    check("similar" in body["note"] and "possible copy" in body["note"] and "not legal copying" in body["note"], f"{what}: the note's plain words")
    check(not any(word in body["note"].lower() for word in BANNED_WORDS), f"{what}: the note never says copied / stolen / plagiarism / infringement")
    return body


def check_audit_check():
    threshold, unique = audit.run_settings("a-500")         # the threshold in results/audit_a-500.json
    first = train_picture(int(subset_rows("a-500")[0]))     # a picture that WAS in the training set of a-500
    audit.CLIP_AUTO_BUILD = False                           # never start a long CLIP cache build from the checks
    audit.auditor_clip.run_index("a-500")                   # the a-500 CLIP cache: read, or built now (about 1 minute)

    # (a) a training image goes back in: distance 0, possible_copy, itself as rank 1
    start = time.time()
    reply = send_check(image_png(first))
    seconds = time.time() - start
    body = check_good_answer(reply, "a-500 training image")
    near = body["nearest"]
    check(near[0]["distance"] == 0 and near[0]["phash_distance"] == 0 and body["flag"] == "possible_copy",
          f"a-500 training image: distance 0, flag possible_copy (got {near[0]['distance']}, {body['flag']})")
    check((np.array(decode({"png_base64": near[0]["image_png_base64"]})) == first).all(), "a-500 training image: rank 1 is the same picture")
    check(near[1]["distance"] > 0 and "a-500" in body["note"] and f"{unique} different training images" in body["note"] and f"{threshold:.2f}" in body["note"],
          "a-500 training image: the note names the run, the number of training images and the threshold")
    check("resized" not in body["note"], "a-500 training image: a 32x32 upload is not resized")
    check(send_check(image_png(first), run="a-500").json() == body, "run a-500 is the default, and the same picture twice gives the same answer")
    train = audit.training_images("a-500")
    check(len(train) == unique == 500, f"a-500: {len(train)} training images rebuilt, the results file says {unique}")
    exact, _ = nearest_distances(to_float(first[None]), train)
    check(abs(float(exact[0]) - near[0]["distance"]) < 1e-4, "rank 1 has the distance model/auditor.py's nearest_distances gives")
    print(f"audit check a-500 training image: distances {[n['distance'] for n in near]}, phash {[n['phash_distance'] for n in near]}, "
          f"flag {body['flag']}, threshold {threshold}, {len(train)} training images, {seconds:.2f} s")
    check(body["clip_flag"] == "possible_copy" and near[0]["clip_similarity"] >= 0.999 and body["clip_nearest"]["clip_similarity"] >= 0.999
          and (np.array(decode({"png_base64": body["clip_nearest"]["image_png_base64"]})) == first).all(),
          f"a-500 training image: CLIP similarity 1, clip_flag possible_copy, clip_nearest is the same picture (got {body['clip_flag']}, {near[0]['clip_similarity']})")

    # (a2) its mirror image: far in pixels, but CLIP still finds the original
    body = check_good_answer(send_check(image_png(first[:, ::-1].copy())), "a-500 mirrored training image")
    check(body["nearest"][0]["distance"] > 0 and (np.array(decode({"png_base64": body["clip_nearest"]["image_png_base64"]})) == first).all(),
          "a-500 mirrored training image: not distance 0, and clip_nearest is the original picture")
    print(f"audit check a-500 mirrored training image: pixel flag {body['flag']} (distance {body['nearest'][0]['distance']}), "
          f"clip_flag {body['clip_flag']} (similarity {body['clip_nearest']['clip_similarity']}, threshold {body['clip_threshold']})")

    # (b) a real picture the model never saw (CIFAR-10 calibration image): never at distance 0
    (held_out, _), where = cifar_test_pictures()
    body = check_good_answer(send_check(image_png(held_out)), "calibration image")
    check(body["nearest"][0]["distance"] > 0, f"{where}: not a training image, so the distance is above 0")
    check(body["flag"] == ("possible_copy" if body["nearest"][0]["distance"] < threshold else "not_flagged"), f"{where}: the flag follows the threshold")
    print(f"audit check {where}: nearest distance {body['nearest'][0]['distance']}, flag {body['flag']}")

    # (c) random noise is far from every training image
    noise = np.random.default_rng(2).integers(0, 256, (32, 32, 3))
    body = check_good_answer(send_check(image_png(noise)), "random noise")
    check(body["flag"] == "not_flagged" and body["nearest"][0]["distance"] >= threshold, "random noise: not_flagged")

    # (d) other sizes are resized to 32x32, and the note says so
    body = check_good_answer(send_check(image_png(np.array(Image.fromarray(first).resize((64, 64), Image.Resampling.NEAREST)))), "64x64 picture")
    check("Your image was 64x64, so it was resized to 32x32 first." in body["note"], "64x64 picture: the note says it was resized")
    buffer = io.BytesIO()
    Image.new("RGB", (30, 20), (200, 50, 50)).save(buffer, format="JPEG")
    body = check_good_answer(send_check(buffer.getvalue()), "30x20 JPG")
    check("Your image was 30x20, so it was resized to 32x32 first." in body["note"], "30x20 JPG: the note says it was resized")

    # (e) the form field `run`, and a missing image
    for run in ("a-999", "A-500", "a-500 ", "../results", "a-500-s2"):          # a-500-s2: seed runs are for the summary only
        check_error(send_check(image_png(first), run=run), 400, f"/audit/check run {run!r}")
    check_error(client.post("/audit/check", data={"run": "a-500"}), 400, "/audit/check no image field")

    # (f) bad uploads (the wrong type, empty, 5 MB and no-file cases are already checked in check_bad_uploads)
    good = image_png(solid((1, 2, 3), 8, 8))
    check_error(send_check(PNG_START + b"this is not a real picture"), 400, "/audit/check PNG start but broken picture")
    check_error(send_check(good[:20]), 400, "/audit/check cut-off PNG")
    buffer = io.BytesIO()
    Image.new("L", (7000, 7000)).save(buffer, format="PNG")                  # 49 million pixels, but a small file
    check_error(send_check(buffer.getvalue()), 400, "/audit/check more than 40 million pixels")

    # (g) a run without a results file is not ready (503), a broken one is 500. A temporary folder stands in for results/.
    real_dir = audit.RESULTS_DIR
    with tempfile.TemporaryDirectory() as folder:
        audit.RESULTS_DIR = Path(folder)
        try:
            for run in RUNS:
                check_error(send_check(good, run=run), 503, f"/audit/check run {run} without a results file")
            (Path(folder) / "audit_a-500.json").write_text("this is not json {")
            check_error(send_check(good), 500, "/audit/check a-500 results file that is not JSON")
            (Path(folder) / "audit_a-500.json").write_text(json.dumps({"calibration_threshold": 11.8, "unique_images": 499}))
            check_error(send_check(good), 500, "/audit/check results file with another number of training images")
        finally:
            audit.RESULTS_DIR = real_dir
    check(send_check(good).status_code == 200, "/audit/check works again with the real results folder")

    # (h) the other runs, as soon as their results files exist: the rebuilt training set has the size the file says, and a
    # training image gives distance 0 (a-500-dup has 500 training rows but only 425 different images)
    for run in RUNS:
        if run == "a-500":
            continue
        if not (audit.RESULTS_DIR / f"audit_{run}.json").exists():
            print(f"audit check {run}: no results file yet, its checks are skipped")
            continue
        count = audit.run_settings(run)[1]
        check(len(audit.training_images(run)) == count == len(set(subset_rows(run).tolist())), f"{run}: rebuilt training images equal the results file's {count}")
        start = time.time()
        body = check_good_answer(send_check(image_png(train_picture(int(subset_rows(run)[0]))), run=run), f"{run} training image")
        check(body["nearest"][0]["distance"] == 0 and body["flag"] == "possible_copy", f"{run}: a training image gives distance 0 and possible_copy")
        print(f"audit check {run}: {count} training images, training image distance 0, clip_flag {body['clip_flag']} "
              f"(null = no CLIP cache yet), {time.time() - start:.2f} s")
    print("audit check: training image, held-out image, noise, resizing, run field, bad uploads and not-ready runs all as expected")


# ------------------------------------------------------------------- 4. CORS

def check_cors():
    for origin in (VITE, "http://127.0.0.1:5173"):
        reply = client.get("/health", headers={"Origin": origin})
        check(reply.headers.get("access-control-allow-origin") == origin, f"CORS header for {origin}")
    check("access-control-allow-origin" not in client.get("/health", headers={"Origin": "http://evil.example"}).headers,
          "no CORS header for another website")
    ask = {"Origin": VITE, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    reply = client.options("/generate", headers=ask)
    check(reply.status_code == 200 and reply.headers.get("access-control-allow-origin") == VITE, "CORS preflight for POST /generate")
    reply = client.get("/audit/results", headers={"Origin": VITE})
    check("X-Stub" in reply.headers.get("access-control-expose-headers", ""), "the website may read X-Stub")
    print("cors: header present for the Vite addresses, absent for other websites")


if __name__ == "__main__":
    check_endpoints()
    check_generate({g["id"]: g["status"] for g in client.get("/generators").json()["generators"]})
    check_bad_requests()
    check_bad_uploads()
    check_compare()
    check_not_ready()
    check_detect()
    check_detect_not_ready()
    check_detect_final()
    check_gradcam()
    check_evidence()
    check_audit()
    check_audit_check()
    check_cors()
    print(f"\nALL {count} CHECKS PASSED")
