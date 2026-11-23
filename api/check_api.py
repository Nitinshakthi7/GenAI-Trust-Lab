# check_api.py - a quick self-check of the whole API. It does not need a running server:
# FastAPI's TestClient calls the app directly.
#
# Run it from the project folder:   .venv\Scripts\python -m api.check_api
# It stops at the first check that fails and says which one. At the end it prints how many checks passed.
#
# What it checks:
#   - all 8 endpoints answer, with exactly the field names written in api/contract.json
#   - POST /generate works for every generator whose weights are ready (n=2, steps 20)
#   - every PNG carries its provenance text note
#   - bad requests give 400 with {"error": ...}, an unknown generator gives 404,
#     a model that is not ready gives 503
#   - the three stub answers carry the X-Stub: true header and a "stub:" note
#   - POST /compare is real (no X-Stub header) and gives the right answers for pictures where we
#     know the answer in advance (same image twice, orange vs black, shifted copy, a cat-like pair,
#     two real CIFAR-10 calibration images), and 400 for bad uploads
#   - POST /detect is real (dev detector, no X-Stub header): a real CIFAR-10 calibration image, a VAE
#     fake and a 64x64 photo all answer 200 with the contract's field names (the 64x64 photo is
#     out_of_scope with the exact warning), bad uploads give 400, missing detector files give 503
#   - Grad-CAM (model/gradcam.py): the map is 32 x 32, its values are 0 to 1, it is not one flat colour,
#     and /detect now returns a real gradcam_png_base64 (a 128 x 128 PNG)
#   - POST /evidence is real (EXIF only, no X-Stub header): a JPG with EXIF and GPS gives the camera make and
#     model, says "GPSInfo: present (coordinates hidden)" and never returns the GPS numbers; a PNG without EXIF gives
#     found false; long values are cut, binary tags are skipped, at most 30 tags; bad uploads give 400
#   - the CORS header is present for the Vite website address
# The final test split of CIFAR-10 is never opened here: real pictures come from the 'calibration' part only.

import base64
import io
import json
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image
from PIL.ExifTags import TAGS

from api import detect, generators
from api.common import MAX_UPLOAD_BYTES, PNG_START
from api.main import app
from model import detector
from model.data import get_part
from model.gradcam import gradcam
from model.make_fakes import to_float

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


def same_names(real, example):
    """True if `real` has exactly the same field names as `example` (values are ignored).
    The one exception: "fields" (the EXIF tags) is a free list, so only its being a dict is checked."""
    if isinstance(example, dict):
        return (isinstance(real, dict) and real.keys() == example.keys()
                and all(same_names(real[k], example[k]) for k in example if k != "fields"))
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
        is_stub = endpoint not in ("GET /health", "GET /generators", "POST /generate", "POST /compare", "POST /detect", "POST /evidence")
        if is_stub:
            check(reply.headers.get("X-Stub") == "true", f"{endpoint} has the header X-Stub: true")
            check(reply.json()["note"].startswith("stub:"), f"{endpoint} note starts with 'stub:'")
        else:
            check("X-Stub" not in reply.headers, f"{endpoint} is real, so no X-Stub header")
    check(client.get("/health").json() == {"status": "ok"}, "/health says ok")
    print(f"endpoints: all {len(CONTRACT)} answer with the contract's field names")


# ------------------------------------------------------------ 2. POST /generate

def check_generate(statuses):
    ids = [g["id"] for g in client.get("/generators").json()["generators"]]
    check(ids == ["g-vae-cifar", "g-vae-fmnist", "g-cddpm-fmnist", "g-ddpm-cifar", "g-dcgan-cifar"], "catalogue ids")
    check(statuses["g-dcgan-cifar"] == "not_available", "g-dcgan-cifar is not_available")
    for gid in ("g-vae-cifar", "g-vae-fmnist"):
        check(statuses[gid] == "ready", f"{gid} must be ready")

    requests = [
        ("g-vae-cifar", dict(n=2), 2), ("g-vae-cifar", dict(interpolate={"seed_b": 7, "frames": 3}), 3),
        ("g-vae-fmnist", dict(n=2), 2), ("g-vae-fmnist", dict(interpolate={"seed_b": 7, "frames": 2}), 2),
        ("g-ddpm-cifar", dict(n=2, steps=20), 2),
        ("g-cddpm-fmnist", dict(n=2, steps=20, class_id=3, guidance=3.0), 2),
        ("g-cddpm-fmnist", dict(n=2, steps=20), 2),
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
            want = (28, 28) if "fmnist" in gid else (32, 32)
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

    # (a) the same picture twice
    noise = np.random.default_rng(1).integers(0, 256, (224, 224, 3))
    reply = compare_pictures(noise, noise)
    table = table_of(reply)
    check(reply.status_code == 200 and reply.json()["similarity"]["ssim"] == 1.0, "same image twice: ssim is 1.0")
    check(table["dominant colours"][0] == table["dominant colours"][1], "same image twice: same dominant colours")
    check(table["colour histogram distance"] == ("0.00", "0.00"), "same image twice: histogram distance is 0")
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
    check(body["note"] == example["note"], "the note is the contract's note")
    print(f"compare cat-like pair: dominant colours {table['dominant colours']}")

    # (e) two real CIFAR-10 test images: every field, as the contract says
    (first, second), where = cifar_test_pictures()
    reply = compare_pictures(first, second)
    body = reply.json()
    check(reply.status_code == 200 and same_names(body, example), f"{where}: 200 with the contract's field names")
    check([r["feature"] for r in body["feature_table"]] == rows, "the feature table has the 9 rows")
    check(all(isinstance(v, str) and v for r in body["feature_table"] for v in r.values()), "every table cell is a text")
    check(body["similarity"]["lpips"] is None and body["regions_png_base64"] is None, "lpips and regions are null (not built)")
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


# ------------------------------------------------------- 6. POST /detect (the dev detector)

WARNING_TEXT = "This detector was trained only on 32x32 CIFAR-10-style images. Results on other photos are not meaningful."


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
    check(body["heatmap_source"] == "vae", f"{what}: heatmap_source is vae")
    heatmap = body["heatmap_png_base64"]
    check(not heatmap.startswith("data:") and decode({"png_base64": heatmap}).size == (128, 128), f"{what}: heatmap is a plain base64 128 x 128 PNG")
    cam = body["gradcam_png_base64"]
    check(isinstance(cam, str) and not cam.startswith("data:"), f"{what}: gradcam is plain base64 text (no data: prefix)")
    cam_picture = decode({"png_base64": cam})
    check(cam_picture.format == "PNG" and cam_picture.size == (128, 128), f"{what}: gradcam decodes to a 128 x 128 PNG")
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
            (folder / "detector_lr.joblib").write_bytes(b"this is not a real weights file")
            check_error(post(), 503, "/detect: the logistic regression file is broken")
        finally:
            detector.ACTIVE_FILE, detector.WEIGHTS_DIR, detect._detector = real_pointer, real_weights, real_detector
    check(post().status_code == 200, "/detect works again with the real files")
    print("detect not ready: missing threshold or weights files and a broken file give 503")


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
    check_gradcam()
    check_evidence()
    check_cors()
    print(f"\nALL {count} CHECKS PASSED")
