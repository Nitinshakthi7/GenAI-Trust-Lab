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
| `POST /detect` | Real or AI-made? (upload one image) | real: final detector (VAE + diffusion features; DDPM heatmap; Grad-CAM; the final test split was scored once) |
| `POST /compare` | Compares two images (upload `image_a`, `image_b`; optional `use_words`, `focus`) | real (no Words) |
| `GET /audit/results` | Memorization experiment results | real (placeholders until the runs exist) |
| `POST /audit/check` | Finds the training images most like an uploaded image (upload `image`; optional `run`) | real (pixel distance to the training images of one Auditor run) |
| `POST /evidence` | EXIF / C2PA / ELA / FFT evidence for an uploaded image | real (EXIF only; C2PA, ELA, FFT not built) |

There is **no stub endpoint** any more: every endpoint is real, so no answer has an `X-Stub` header or a `stub:` note.
Upload endpoints check the upload (see "Errors" below).
The `flag` field of `POST /audit/check` has exactly two allowed values: `possible_copy` (the nearest training image is closer than the run's threshold) and `not_flagged`. It measures similarity only, not legal copying.

## Audit results

`GET /audit/results` returns the results of the Auditor's memorization experiments (the maths is in
`model/auditor.py`; each run is `model/train_auditor_run.py`; `model/audit_summary.py` gathers the runs into
`results/audit_results.json`). The Auditor measures **similarity between images** (pixel distance to the nearest training
image). It does not decide legal copying, and the words "similar" and "possible copy" are used on purpose.

- If `results/audit_results.json` exists, the answer is that file, unchanged. If it cannot be read or does not have the
  contract's field names (`experiments`, `examples`, `note`, and the fields inside them), the answer is 500 `{"error": ...}`.
- If the file does **not** exist yet, the answer is a **placeholder**: the two planned experiments, every `memorization_rate`
  `null`, `examples` `[]`, and a `note` starting with `[PLACEHOLDER - not measured yet]`. The website should show it as
  "not measured yet", not as a result. There is no `X-Stub` header, because the endpoint itself is real.
- `examples` are the closest pairs (generated image, its nearest training image, the distance), picked by smallest
  distance, not by looks. Images are 128 x 128 PNG, plain base64, no `data:` prefix.

## Audit check

`POST /audit/check` finds the training images of one Auditor run that are most similar to your picture. Send a multipart
form with `image` (PNG or JPG, at most 20 MB, at most 40 million pixels; a bad upload gives 400) and an optional form field
`run`: `a-500` (default), `a-10000` or `a-500-dup`. Any other `run` gives 400 `{"error": ...}`. If the run's
`results/audit_<run>.json` does not exist yet (the run is still training) the answer is 503 `{"error": ...}`.

- **Which training images.** The ones the run was trained on, rebuilt with `subset_rows` of `model/train_auditor_run.py`
  (same seed 42, same subset), each different image once (`a-500-dup` has 500 rows but 425 different images, so the answer
  never shows the same image twice). They are loaded on the first request (about 2 seconds) and kept in memory. If their number
  differs from `unique_images` in the results file the answer is 500.
- **Distance.** The Auditor's own L2 distance: the picture as 32x32x3 numbers scaled to -1..1, compared with every training
  image. `nearest` has the 3 closest, `rank` 1 to 3, smallest `distance` first (4 decimals). A picture that is not 32x32 is
  resized to 32x32 first (bilinear, like `/detect`) and the `note` says so.
- **`flag`.** `possible_copy` if the nearest `distance` is below the run's `calibration_threshold` (the 1st percentile of the
  distances of held-out real images, for `a-500` 11.822), otherwise `not_flagged`. It measures pixel similarity only, not legal copying.
- **`phash_distance`.** How many of the 64 bits of a simple perceptual hash (DCT of the 32x32 grey picture, lowest 8x8 values
  compared with their median) differ between your picture and that training image: 0 to 64, a second opinion beside the pixel distance.
- **`image_png_base64`.** The training image as a 32x32 PNG, plain base64, no `data:` prefix.
- **`note`.** Plain English: the run, how many training images were compared, the threshold, "similarity only, not legal copying",
  and the resize if there was one.

## Detect (final detector; the dev detector is still supported)

`POST /detect` says whether a picture looks real or AI-made. Send a multipart form with `image` (PNG or JPG,
at most 20 MB, at most 40 million pixels; a bad upload gives 400). `configs/detector_active.json` picks the detector.
Right now it is the **final detector** (VAE + diffusion features, a diffusion heatmap and Grad-CAM; the final test split
was scored once; see "The final detector" below). The older **dev detector** (trained on **one weak generator only**,
blurry VAE fakes; a work-in-progress check, **not a result**) is used when the file has no `features` key, and is what the
bullets below describe first. The `note` field says which detector answered.

**The three labels.** `real`, `ai` or `unclear`. The detector gives a score from 0 to 1, and two thresholds (set on the
calibration data, not on the test data) cut it into three: a score at or below the low threshold is `real`, at or above the
high threshold is `ai`, and anything in between is `unclear`. A real picture is often `unclear`; that is expected and honest,
not a bug. On a generator it has **never seen**, the detector does much worse; that is a finding of this project, not a promise.

The answer has exactly the contract's fields:

- `label`: `real`, `ai` or `unclear` (from the logistic regression's two thresholds, set on the calibration data).
- `score`: logistic regression, the probability that the image is AI-made, 0 to 1.
- `baseline_score`: the small CNN, the probability that the image is AI-made, 0 to 1.
- `vae_score`: the detector VAE's anomaly score (reconstruction error + KL). It is **not** limited to 0..1
  (real CIFAR-10 images give about 100 to 200). On the dev data the VAE fakes get a LOWER number than real
  images, because the VAE rebuilds its own kind of blurry picture well. Do not draw it as a 0..1 bar.
- `heatmap_png_base64`: where the VAE rebuilds the picture badly, a 128 x 128 PNG (white = no error, red = big
  error), plain base64, no `data:` prefix. `heatmap_source` is `vae` (for the final detector it is `ddpm`, see below).
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
(`detector not ready`). Which detector is used is set in `configs/detector_active.json`; the thresholds are read
from the results file named there.

**The final detector.** When `configs/detector_active.json` has `"features": ["vae", "ddpm"]` (written by
`python -m model.run_final_detector ... --phase score-final --activate`, see `model/README.md`), `/detect` uses the
final run instead of the dev one: the main `score` is the logistic regression on the 2 VAE numbers plus 4 numbers
from the detector's diffusion model, `baseline_score` is that run's CNN, and `heatmap_png_base64` is the **diffusion
model's** reconstruction error (`heatmap_source` is `ddpm`; white = no error, fully red = an error of 0.3 or more,
proposed). The `note` is then the "Final detector" note (it says unseen generators are findings, not promises) instead of
the dev note. If the diffusion model file is missing or cannot be read right now (for example while it is being
written), the answer still comes back, using the VAE-only logistic regression of the same run and the VAE heatmap
(`heatmap_source` is `vae`); the diffusion model is tried again on the next picture. The final detector is slower:
20 diffusion steps per picture, about half a second on a CPU (measured: see `model/README.md`). With the dev
pointer (no `features` key) the answers are exactly the same as before.

## Evidence (EXIF only)

`POST /evidence` shows what the picture FILE says about itself. Send a multipart form with `image` (PNG or JPG,
at most 20 MB, at most 40 million pixels; a bad upload gives 400). Only **EXIF** is built so far. EXIF is the
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
maths and a pretrained CNN. It needs no GPU and nothing trained by us. Send a multipart form with:

- `image_a`, `image_b`: PNG or JPG, at most 20 MB each, at most 40 million pixels.
- `use_words`: `true` or `false` (default `false`). The Words panel (a vision model that describes the
  differences) is **not built yet**. With `true` the answer is still `"source": "heatmap_only"` and the
  `summary` starts with `Words are not enabled on this server (no vision API configured).`
- `focus`: optional text, at most 100 characters (longer gives 400). For now only its length is checked;
  later it will tell the Words panel what to look at.

The answer has (see the contract for the exact shape):

- `similarity.ssim`: structural similarity of the two grey 224 x 224 pictures, -1 to 1 (1 = identical).
  Photos that are not aligned give a low value, which is normal. `similarity.lpips` is `null` (not built).
- `diff_heatmap_png_base64`: a 224 x 224 PNG, white where the images are the same and red where the
  colours differ. Plain base64, no `data:` prefix.
- `regions_png_base64`: a 448 x 224 PNG made with the CNN (see below): picture A on the left, picture B on the
  right, each with a red tint where the other picture has no similar part. `null` if the CNN cannot be loaded.
- `feature_table`: rows of `feature`, `image_a`, `image_b`, `does_not_prove`. The rows are mean brightness,
  average colour, colour spread, sharpness, edge amount, texture, original size, dominant colours
  (3 k-means colours with plain names and their share) and a colour histogram distance. When the CNN is
  available, three more rows follow: `CNN feature similarity (ResNet-18, whole image)` (one number, shown in both
  columns), `CNN best-match similarity (A to B, B to A)` and `CNN top guesses (ImageNet, may be wrong)`.
- `differences`: always `[]` for now. `source`: always `heatmap_only` for now.
- `summary`: 3 plain sentences made from the numbers by simple rules. `note`: the contract's note plus one
  sentence about the CNN (or, if it is missing, one sentence saying so).

Both images are shrunk to 224 x 224 before measuring. The numbers (224, the edge level 0.1) are proposed
and can be tuned in `api/compare.py`. The edge amount is measured at the picture's own size when it is smaller
than 224 (a 32 x 32 picture is not blown up first: that smeared every edge and gave 0%).

**The CNN part** (`model/compare_cnn.py`). It is a ResNet-18 from torchvision with its ImageNet weights: it is
**pretrained by others, not trained by us**. The weights file (`resnet18-f37072fd.pth`, 45 MB) is read from the
torch cache folder (`~/.cache/torch/hub/checkpoints`); the server never downloads it. It is loaded once, on the
first `/compare` call, and runs on the CPU (about 0.05 s per pair). Both pictures are stretched to 224 x 224 (their
shape is not kept) and go through the network. Each picture becomes a 14 x 14 grid of patches. For every patch of A
we look for the most similar patch anywhere in B (cosine similarity); "unmatched" = 1 minus that best match, so a
high value means nothing in the other picture looks like this patch. The same is done for B against A. The red
tint is stretched per picture (its strongest area is always the strongest red), so compare the table numbers, not
the colour strength between pictures. The table rows are: the cosine similarity of the two whole pictures, the
average best match from A to B and from B to A, and ResNet-18's three best ImageNet labels for each picture.
All of this is descriptive only: a matched patch can belong to a different object, an unmatched patch is not
necessarily a meaningful difference, and ImageNet's 1000 fixed labels can be wrong or too coarse. It does not
explain the differences in words.

If the weights file is missing or cannot be loaded (or the CNN fails for any other reason), `/compare` still
answers 200 as before: `regions_png_base64` is `null`, the three CNN rows are left out, and `note` says that the
CNN part is not available.

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

Uploads: PNG or JPG only (the first bytes of the file are checked, not its name), at most 20 MB
(`MAX_UPLOAD_BYTES` in `api/common.py`).

## Environment variables

| Name | Meaning | Default |
|------|---------|---------|
| `ALLOWED_ORIGINS` | Website addresses allowed to call the API (comma-separated). Addresses only, never secrets | `http://localhost:5173,http://127.0.0.1:5173` (the Vite website, proposed) |
| `AUDIT_RESULTS_FILE` | A file to use instead of `results/audit_results.json` for `GET /audit/results` (used by the tests) | not set |

Example (PowerShell): `$env:ALLOWED_ORIGINS = "http://localhost:3000"`

## Check that it works

```
.venv\Scripts\python -m api.check_api
```

It calls all 8 endpoints, compares the field names with `contract.json`, makes images with every ready
generator, and tries bad requests. For `/audit/check` it uploads a training image of `a-500` (distance 0, `possible_copy`),
random noise (`not_flagged`), a 64x64 picture (resize note), a bad `run` (400) and a run without a results file (503, using a
temporary folder), and, for every run whose results file exists, checks the training-image count against the file. For `/compare` it also uses pictures with a known answer (the same
image twice, solid orange against solid black, a copy shifted by 20 pixels, a cat-like pair, two real
CIFAR-10 calibration images). For `/detect` it sends a real CIFAR-10 calibration image, a VAE fake and a 64x64
photo, and also pretends the detector's files are missing. It reads `configs/detector_active.json` to know which detector
is active: for the final detector it expects `heatmap_source` `ddpm` (`vae` only, with a printed WARNING, if the diffusion
weights cannot be loaded), a note starting `Final detector`, any of the three labels for a real picture, and a mean
`baseline_score` above 0.5 for 10 VAE fakes and 10 diffusion fakes (calibration pictures only); for the dev detector it
expects `vae` and the `Dev detector` note. It also tries a tiny made-up final run (random weights,
in a temporary folder): the final note, the diffusion heatmap and the fallback to the VAE heatmap when the diffusion
file is missing or broken. It checks Grad-CAM (the map is 32x32, 0 to 1, not
flat, and `/detect` returns a real PNG) and `/evidence` (a JPG with EXIF and GPS: the make and model come back,
the GPS numbers never do). For `/audit/results` it checks the placeholder (no results file), a results file made from tiny
random pictures (unchanged answer, the three fields of each closest pair) and seven kinds of broken file (500). It prints how many checks passed. It needs `httpx`
(in `requirements.txt`), the detector's files in `weights/` and the VAE fakes in `fakes/`.

## Files

- `main.py`: the app, CORS, error answers and the real endpoints
- `generators.py`: the generator list and the image-making code
- `compare.py`: the image maths and the region map / table rows behind `POST /compare` (the pretrained ResNet-18 is in `model/compare_cnn.py`)
- `detect.py`: the web side of `POST /detect` (upload check, loads the detector once, warning and note)
- `evidence.py`: the web side of `POST /evidence` (upload check, EXIF reading, GPS hidden)
- `audit.py`: the web side of `GET /audit/results` (the results file, or the placeholder if it does not exist yet) and of `POST /audit/check` (nearest training images of one Auditor run)
- `common.py`: the error class and the upload check
- `check_api.py`: the self-check
- `contract.json`: the proposed request/answer shapes
