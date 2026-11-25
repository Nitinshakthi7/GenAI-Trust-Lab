# How to test the website (measured Tue 24 Nov 2026)

All "expected" results below were **measured on 24 Nov with the real API**, not guessed. They are for checking that
the demo still works, not results of the project. The 20 example images come from the *calibration* part of the data.
The final test split is never used here.

## 1. Start everything (do this fresh before every test or demo)

API, from the project folder:

```powershell
.venv\Scripts\python -m uvicorn api.main:app --port 8000
```

Website, in a second terminal, from the `web` folder:

```powershell
$env:VITE_USE_STUB='false'; $env:VITE_ENABLE_COMPARE_WORDS='false'; npm run dev
```

After a cold start the API takes about a minute before it answers (52 seconds measured on 25 Nov), so start it first.
Open http://localhost:5173. **Restart the website server after any `git pull`.** On 24 Nov an old server kept running
after a pull and every Detector click said "The lab service could not complete this request".
If you see that message, the website server is the first thing to restart.

## 2. Detector (page: Detector)

Click an example, then "Analyze image". Examples are named by what they are:

| Click this | Expected label | AI-made score | Use it to show |
|---|---|---|---|
| Made by our VAE 3 | AI-made | 0.87 | a fake it catches |
| Made by our VAE 1 | AI-made | 0.82 | a fake it catches |
| Made by our DDPM 5 | AI-made | 0.94 | a diffusion fake it catches |
| Made by our DDPM 4 | AI-made | 0.87 | a diffusion fake it catches |
| Real photo 6 | Real | 0.10 | a real photo it recognises |
| Real photo 8 | Real | 0.09 | a real photo it recognises |
| Real photo 1 | Unclear | 0.27 | the usual result for a real photo |
| Made by our DDPM 1 | **Real** (wrong) | 0.14 | an honest miss |
| Made by our VAE 2, VAE 5, DDPM 2, DDPM 3 | Unclear | 0.54 to 0.71 | in-between |

Totals over the 20 examples: real photos 2 Real + 8 Unclear + 0 AI-made. VAE fakes 3 AI-made + 2 Unclear. Diffusion fakes
2 AI-made + 2 Unclear + 1 Real (wrong). Nothing real was called AI-made. That is 20 pictures, so it shows the pattern and
proves nothing. The real numbers are on the Limits page.

Also try:
- **Your own photo (any size above 32x32):** you get a warning that results are not meaningful. This is on purpose.
- **A file that is not an image (renamed .txt):** a clear error, no crash.
- **Evidence tab:** shows EXIF, C2PA, ELA and FFT. Click it after Analyze.

## 3. Generator (page: Generator)

| Do this | Expect |
|---|---|
| CIFAR-10, Conv VAE, seed 42, 4 images | 4 blurry 32x32 images in under 1 second |
| Same again | the same 4 images (same seed gives the same images) |
| Seed 7 | different images |
| CIFAR-10, Diffusion (DDIM), seed 42, 4 images | 4 images in about 1 to 2 seconds, a bit sharper than the VAE |
| Fashion-MNIST, conditional diffusion, class Sneaker, guidance 2 | 4 grey clothing pictures in about 2 seconds |
| "Interpolate between two seeds" | a row of images that morph from one seed to the other |
| "Send to Detector" under any image | the Detector opens with that image loaded, ready to Analyze |

The model list shows only models that are ready (DCGAN was never built, so it is not listed).

## 4. Compare (page: Compare)

Upload two images (the files are in `web/public/examples/real` and `fake`). Words stays off.

| Image A | Image B | Expected |
|---|---|---|
| real_0.png | real_0.png | SSIM 1.0, CNN similarity 1.00 (the same picture) |
| real_0.png | real_1.png | SSIM about 0.40, CNN similarity 0.83 |
| real_0.png | fake_vae_0.png | SSIM about 0.47, CNN similarity 0.27 |
| real_0.png | fake_ddpm_0.png | SSIM about 0.25 |
| real_0.png | real_0.png with a red square painted on it | SSIM about 0.86, and the CNN map shows red over the square |
| two different-size pictures, for example `web/public/examples/demo/larger-frame.png` and `dusk.png` | works, both are resized |

The red-square test is the best one for the demo: it shows the map pointing at a real edit. Make it in Paint by drawing
a red box on a copy of `real_0.png`. The map is coarse (14 x 14 cells), so it marks an area, not an outline.

## 5. Auditor (page: Auditor)

The top of the page shows the measured runs: 500 training images 2.6% flagged, 10,000 images 0.4%, and with 30%
repeated images 3.9% (one run each, so these are noisy).

"Check similarity" compares an upload with the training images of the run you pick in "Compare with the training
images of" (default: 500 images, run a-500). I made a test kit in `data/test_kit/` (git ignores it, because it holds
CIFAR-10 pictures). Expected on the page with the default run a-500:

| Upload | Expected |
|---|---|
| `1_train_in_all_runs_a.png`, `..._b.png` | **Possible copy**, distance 0 (it is a training image) |
| `3_train_twice_in_dup_run.png` | Possible copy, distance 0 |
| `5_edit_brighter.png` | Possible copy, distance 10.8 (below the 11.82 threshold), pHash distance 0 |
| `2_train_in_a10000_only.png` | Not flagged, distance 23.1 (it is only in the 10,000-image run) |
| `4_train_in_no_run.png` | Not flagged, distance 24.9 |
| `5_edit_shifted_2px.png` and `5_edit_flipped.png` | **Not flagged** even though it is the same picture. Pixel distance cannot see a shift or flip. This is a limit to say out loud |
| `web/public/examples/real/real_0.png` | Not flagged, distance 26.6 |

**Best demo of the run picker:** upload `2_train_in_a10000_only.png`. With "500 images" it is not flagged (23.1).
Switch to "10,000 images" and press the button again: possible copy, distance 0, because that picture is in that run's
training set. With "10,000 images" the threshold is 9.74, so `5_edit_brighter.png` (10.8) is not flagged there.

## 6. Limits (page: Limits and Methods)

All 11 model cards should show measured numbers (no "PLACEHOLDER" text). The blur-shortcut paragraph is under "What the
Detector covers".

## 7. Not tested yet

- The phone-width layout.
- Wi-Fi switched off (the ResNet-18 weights are cached, so Compare should still work).
- A full restart of the laptop followed by a cold start of both servers.
- Fashion-MNIST VAE in the browser (the API call works).

## 8. Small things found on 24 Nov

- Fixed 25 Nov: "DCGAN" no longer appears in the Generator model list.
- Fixed 25 Nov: the Auditor intro no longer repeats "Each run audits its own small diffusion model".
- Fixed 25 Nov: the Auditor check has a run picker (a-500, a-10000, a-500-dup).
- Still there: the "Use Words" box is on the Compare page, with a note that Words is off.
