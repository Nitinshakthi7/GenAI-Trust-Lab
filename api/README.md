# API (the part the website talks to)

A small web API built with FastAPI. The website (Tejash's part) sends requests to it and shows the answers.
The exact field names of every request and answer are written in `api/contract.json`
(proposed: Nitin and Tejash agree it at the first meeting).

## Start it

Run this from the project folder (the one that holds `api/`, `model/`, `weights/`):

```
.venv\Scripts\python -m uvicorn api.main:app --port 8000
```

Then open <http://localhost:8000/docs>. It is an interactive page where you can try every endpoint
with a button, without writing any code.

Stop the API with `Ctrl+C` in the same window.

## The 8 endpoints

| Endpoint | What it does | Status |
|----------|--------------|--------|
| `GET /health` | Answers `{"status": "ok"}` | real |
| `GET /generators` | Lists the image generators, their settings and whether each one is ready | real |
| `POST /generate` | Makes images with one generator (JSON body, see the contract) | real |
| `POST /detect` | Real or AI-made? (upload one image) | real (dev detector; Grad-CAM built) |
| `POST /compare` | Compares two images (upload `image_a`, `image_b`; optional `use_words`, `focus`) | real (no Words) |
| `GET /audit/results` | Memorization experiment results | stub |
| `POST /audit/check` | Finds the training images most like an uploaded image | stub |
| `POST /evidence` | EXIF / C2PA / ELA / FFT evidence for an uploaded image | real (EXIF only; C2PA, ELA, FFT not built) |

The two remaining **stubs** answer with the fixed example from `contract.json`, so the website can already be wired up.
Every stub answer has a `note` starting with `stub:` and the response header `X-Stub: true`.
Upload endpoints already check the upload (see below), so the website sees the real error behaviour.

## Detect (dev detector)

`POST /detect` says whether a picture looks real or AI-made. Send a multipart form with `image` (PNG or JPG,
at most 5 MB, at most 40 million pixels; a bad upload gives 400). The detector is the **dev detector**: it was
trained on **one weak generator only** (blurry VAE fakes), so its scores say nothing about diffusion fakes or
other unseen generators. It is a work-in-progress check, **not a result**. The `note` field says this in every answer.

The answer has exactly the contract's fields:

- `label`: `real`, `ai` or `unclear` (from the logistic regression's two thresholds, set on the calibration data).
- `score`: logistic regression, the probability that the image is AI-made, 0 to 1.
- `baseline_score`: the small CNN, the probability that the image is AI-made, 0 to 1.
- `vae_score`: the detector VAE's anomaly score (reconstruction error + KL). It is **not** limited to 0..1
  (real CIFAR-10 images give about 100 to 200). On the dev data the VAE fakes get a LOWER number than real
  images, because the VAE rebuilds its own kind of blurry picture well. Do not draw it as a 0..1 bar.
- `heatmap_png_base64`: where the VAE rebuilds the picture badly, a 128 x 128 PNG (white = no error, red = big
  error), plain base64, no `data:` prefix. `heatmap_source` is `vae`.
- `gradcam_png_base64`: **Grad-CAM** of the small CNN baseline (built in `model/gradcam.py`): the 32x32 picture
  shown at 128 x 128 and tinted red where the CNN looked for its "AI" output. Plain base64 PNG, no `data:` prefix.
  It explains the **CNN classifier, NOT the forgery**: it shows where the CNN baseline looked, and it does not
  prove why the image is real or AI. The `note` says so. (It is always a map for the "AI" output: for a picture
  the CNN calls real it still marks the places that pushed most towards "AI", even though the total is low.)
- `input_size`: `WIDTHxHEIGHT` of the picture you sent. `out_of_scope`: `true` if it is not exactly 32x32.
  The picture is still shrunk to 32x32 and scored, but `warning` then says: "This detector was trained only on
  32x32 CIFAR-10-style images. Results on other photos are not meaningful." (`warning` is `null` otherwise.)

The detector loads the first time it is needed and is kept in memory (about 2 seconds once). Scoring one picture
takes a few milliseconds on the CPU. If its files are missing or are being written, the answer is 503
(`detector not ready`). Which detector is used is set in `configs/detector_active.json` (one line to switch to
the final detector later); the thresholds are read from the results file named there.

## Evidence (EXIF only)

`POST /evidence` shows what the picture FILE says about itself. Send a multipart form with `image` (PNG or JPG,
at most 5 MB, at most 40 million pixels; a bad upload gives 400). Only **EXIF** is built so far. EXIF is the
small block of text a camera or phone writes into a photo. Nothing is stored. The answer has exactly the
contract's fields:

- `exif.found`: `true` only if at least one readable tag exists.
- `exif.fields`: tag name -> short text (for example `"Make": "TestCam"`), read with Pillow. At most 30 tags,
  a value longer than 100 characters is cut, binary tags (such as MakerNote) are skipped.
  **GPS privacy:** if the file has GPS data, the answer only says `"GPSInfo": "present (coordinates hidden)"`.
  The coordinates are never returned.
- `exif.does_not_prove`: "Missing EXIF is weak evidence; present EXIF does not mean real."
- `c2pa`: `{"status": "not_checked", "summary": null, "does_not_prove": "A missing manifest means nothing."}` (not built).
- `ela_png_base64`, `fft_png_base64`: `null` (not built).
- `note`: "Evidence, not proof. Nothing is stored."

## Compare

`POST /compare` compares any two photos (for example an orange cat and a black cat) with plain image
maths. It needs no GPU and no trained model. Send a multipart form with:

- `image_a`, `image_b`: PNG or JPG, at most 5 MB each, at most 40 million pixels.
- `use_words`: `true` or `false` (default `false`). The Words panel (a vision model that describes the
  differences) is **not built yet**. With `true` the answer is still `"source": "heatmap_only"` and the
  `summary` starts with `Words are not enabled on this server (no vision API configured).`
- `focus`: optional text, at most 100 characters (longer gives 400). For now only its length is checked;
  later it will tell the Words panel what to look at.

The answer has (see the contract for the exact shape):

- `similarity.ssim`: structural similarity of the two grey 224 x 224 pictures, -1 to 1 (1 = identical).
  Photos that are not aligned give a low value, which is normal. `similarity.lpips` is `null` (not built).
- `diff_heatmap_png_base64`: a 224 x 224 PNG, white where the images are the same and red where the
  colours differ. Plain base64, no `data:` prefix. `regions_png_base64` is `null`.
- `feature_table`: rows of `feature`, `image_a`, `image_b`, `does_not_prove`. The rows are mean brightness,
  average colour, colour spread, sharpness, edge amount, texture, original size, dominant colours
  (3 k-means colours with plain names and their share) and a colour histogram distance.
- `differences`: always `[]` for now. `source`: always `heatmap_only` for now.
- `summary`: 3 plain sentences made from the numbers by simple rules. `note`: the contract's note.

Both images are shrunk to 224 x 224 before measuring. The numbers (224, the edge level 0.1) are proposed
and can be tuned in `api/compare.py`.

## Generators

`GET /generators` gives, for each generator: `id`, `name`, `dataset`, `family`, `params`, `size_mb`,
`status`, `controls`, `classes`, `licence_note`.

| id | What | Controls |
|----|------|----------|
| `g-vae-cifar` | Conv VAE on CIFAR-10 | seed, n, interpolate |
| `g-vae-fmnist` | Conv VAE on Fashion-MNIST | seed, n, interpolate |
| `g-cddpm-fmnist` | Diffusion with class labels on Fashion-MNIST | seed, n, steps, class_id, guidance |
| `g-ddpm-cifar` | Diffusion on CIFAR-10 | seed, n, steps |
| `g-dcgan-cifar` | DCGAN on CIFAR-10: listed only, no model (`not_available`) | none |

Status: `ready` (weights file exists and loads), `training` (config exists but the weights are missing
or being written right now), `not_available` (no config).

Rules for `POST /generate` (a wrong value or an unsupported control gives 400):

- `n` 1 to 16. `seed` 0 to 2,000,000,000 (default 42).
- `steps` is 20, 50 or 100 (diffusion only, default 50).
- `class_id` 0 to 9 and `guidance` 0 to 10 are for `g-cddpm-fmnist` only. Without `class_id` the model
  makes images with no label (and `guidance` must be 0). With `class_id`, `guidance` defaults to 3.
- `interpolate` is `{"seed_b": ..., "frames": 2 to 16}` and is for the two VAEs only. It makes `frames`
  images sliding from `seed` to `seed_b` (`n` is ignored). The in-between frames have `seed: null`.
- Image number `i` uses seed `seed + i`, so any single image can be made again with `n = 1`.
- Images come back as plain base64 PNG text (no `data:image/png;base64,` prefix).
  Fashion-MNIST images are 28x28, CIFAR-10 images are 32x32.
- Every PNG has a text note called `provenance` inside it, for example
  `model: g-vae-cifar; dataset: CIFAR-10; seed: 42; made by GenAI Trust Lab; small model`.
  Read it with `Image.open(...).text["provenance"]` in Pillow.

Models are loaded the first time they are needed and kept in memory. If the file in `weights/` changes
(training saves new weights), the API loads the new file on the next request.

## Errors

Every error has the same shape: `{"error": "<plain text>"}`.

| Code | Meaning |
|------|---------|
| 400 | Bad request (wrong value, unsupported control, broken JSON, bad upload) |
| 404 | Unknown generator or unknown address |
| 503 | Model not ready: still training, or its weights file is being written. Try again later |
| 500 | Anything else |

Uploads: PNG or JPG only (the first bytes of the file are checked, not its name), at most 5 MB
(`MAX_UPLOAD_BYTES` in `api/common.py`).

## Environment variables

| Name | Meaning | Default |
|------|---------|---------|
| `ALLOWED_ORIGINS` | Website addresses allowed to call the API (comma-separated). Addresses only, never secrets | `http://localhost:5173,http://127.0.0.1:5173` (the Vite website, proposed) |

Example (PowerShell): `$env:ALLOWED_ORIGINS = "http://localhost:3000"`

## Check that it works

```
.venv\Scripts\python -m api.check_api
```

It calls all 8 endpoints, compares the field names with `contract.json`, makes images with every ready
generator, and tries bad requests. For `/compare` it also uses pictures with a known answer (the same
image twice, solid orange against solid black, a copy shifted by 20 pixels, a cat-like pair, two real
CIFAR-10 calibration images). For `/detect` it sends a real CIFAR-10 calibration image, a VAE fake and a 64x64
photo, and also pretends the detector's files are missing. It checks Grad-CAM (the map is 32x32, 0 to 1, not
flat, and `/detect` returns a real PNG) and `/evidence` (a JPG with EXIF and GPS: the make and model come back,
the GPS numbers never do). It prints how many checks passed. It needs `httpx`
(in `requirements.txt`), the detector's files in `weights/` and the VAE fakes in `fakes/`.

## Files

- `main.py`: the app, CORS, error answers and the real endpoints
- `generators.py`: the generator list and the image-making code
- `compare.py`: the image maths behind `POST /compare`
- `detect.py`: the web side of `POST /detect` (upload check, loads the detector once, warning and note)
- `evidence.py`: the web side of `POST /evidence` (upload check, EXIF reading, GPS hidden)
- `stubs.py`: the two stub endpoints
- `common.py`: the error class and the upload check
- `check_api.py`: the self-check
- `contract.json`: the proposed request/answer shapes
