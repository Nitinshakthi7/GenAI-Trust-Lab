# model/ - what each file does

Run every command from the project folder (the one that contains `model/`), with the virtual environment on:

```
.venv\Scripts\activate
```

| File | What it does | How to run it |
|------|--------------|---------------|
| `__init__.py` | Empty. It lets Python treat `model/` as a package. | (nothing to run) |
| `utils.py` | Shared helpers: the one seed (42), folder paths, picking the GPU, counting parameters, saving JSON. | (imported by the other files) |
| `data.py` | Loads CIFAR-10 and Fashion-MNIST as tensors in [-1, 1] (Fashion-MNIST padded to 32x32) and cuts CIFAR-10 into four fixed parts, written to `data/splits.csv`. | `python -m model.data` (self-check) |
| `vae.py` | The small convolutional VAE: `sample`, `interpolate` and `anomaly_score`, plus the loss. | `python -m model.vae` (self-check) |
| `train_vae.py` | Trains one VAE run and saves its config, weights, 16 sample images and measured results. | `python -m model.train_vae --run g-vae-cifar` (also `d-vae-cifar`, `g-vae-fmnist`; add `--epochs 1` for a quick test) |
| `ddpm.py` | The small diffusion model: `TinyUNet` (about 3 million numbers, 32x32, optional class label), the noise schedule, the training loss, `ddim_sample` (fast sampling with optional guidance), `reconstruct` (noise a real image a bit, denoise it, return the per-pixel error) and `EMA` (smoother weights). | `python -m model.ddpm` (self-check) |
| `time_ddim.py` | Measures how long DDIM sampling of 16 images takes with 20, 50 and 100 steps, on the GPU and the CPU, and saves it to `results/ddim_timing.json`. | `python -m model.time_ddim` |
| `train_ddpm.py` | Trains one diffusion run and saves its config, EMA weights (every 5 epochs and at the end), sample images and measured results. `--max-minutes` stops it cleanly when time is up. | `python -m model.train_ddpm --run g-cddpm-fmnist` (also `g-ddpm-cifar`, `d-ddpm-cifar`; add `--epochs 1` for a quick test) |
| `cnn.py` | `SmallCNN`, the simple real-vs-fake baseline classifier (it is trained by `train_detector.py`). | `python -m model.cnn` (self-check) |
| `make_fakes.py` | Makes the fake images the Detector learns from: 20,000 for training, 5,000 for calibration and 5,000 for the final test (made now, not used until the final run), each part with its own seed, saved to `fakes/<generator>_fakes.npz`. Works for `g-vae-cifar` and `g-ddpm-cifar`. | `python -m model.make_fakes --generator g-vae-cifar` |
| `features.py` | Turns each image into a few numbers for the simple detector: 2 from the Detector's VAE (rebuild error and KL), 4 from the Detector's diffusion model (mean, 50th, 90th and 99th percentile of the rebuild error map). Saves them to `fakes/feat_<name>.npz` so they are worked out only once. | (imported by `train_detector.py`) |
| `train_detector.py` | Trains the Detector in DEV MODE (never touches the final test split): logistic regression on the features, the CNN baseline on raw pixels, the two label thresholds ('real' / 'unclear' / 'ai') set on the calibration part, plus a dev report, leave-one-generator-out and robustness (JPEG, blur, noise). `--final` is refused for now. | `python -m model.train_detector --dev --generators g-vae-cifar` (add `--cnn-epochs 2` for a quick test, `--use-ddpm-features` once `d-ddpm-cifar` is trained) |
| `run_final_detector.py` | The FINAL Detector, in two phases. `--phase train` (never opens the final test split): features from the Detector's VAE and diffusion model, three detectors (`lr_vae`: logistic regression on the 2 VAE numbers; `lr_vae_ddpm`: on all 6 numbers, the main one; `cnn`: the baseline) trained on real images against the fakes of ALL given generators, two thresholds each from the calibration part, leave-one-generator-out and robustness (calibration data only), then the FREEZE: three weight files, `configs/<run-id>.json` (settings + SHA-256 of every weight file) and `results/<run-id>_train.json`. `--phase score-final` scores the real final test split ONCE with the frozen detectors and thresholds; it refuses if any weight file differs from the frozen SHA-256 or if `results/<run-id>_final.json` already exists. `--activate` (with score-final) then writes `configs/detector_active.json`. See "The final run: exact commands" below. | `python -m model.run_final_detector --run-id detector_final --generators g-vae-cifar g-ddpm-cifar --phase train` |
| `detector.py` | The Detector that `POST /detect` uses (loaded once). It reads `configs/detector_active.json`: the dev detector (VAE features only) or, when the file says `"features": ["vae", "ddpm"]`, the final run (main score from 2 VAE + 4 diffusion numbers, the diffusion reconstruction-error map as the heatmap with `heatmap_source` `ddpm`, and a fall back to the VAE-only score and VAE heatmap if the diffusion model file cannot be read). | (used by `api/detect.py`; checked by `python -m api.check_api`) |
| `auditor.py` | The Auditor's measuring tools: for each generated image, the L2 pixel distance to its nearest training image, the threshold (1st percentile of the same distance for held-out real images from the `calibration` part), the memorization rate (share closer than the threshold, "possible copies") and the 3 closest pairs as PNGs. It measures similarity between images only, not legal copying. | `python -m model.auditor` (self-check with known answers) |
| `train_auditor_run.py` | One Auditor run: trains a small diffusion model on a small part of the CIFAR-10 train images (`a-500`: 500 images, `a-10000`: 10,000 images, `a-500-dup`: 500 rows with 30% repeats), makes 1,000 images, audits them and writes `configs/<run>.json`, `weights/<run>.pt` and `results/audit_<run>.json`. | `python -m model.train_auditor_run --run a-500` (also `a-10000`, `a-500-dup`; `--steps`, `--samples`, `--max-minutes`, `--out-dir FOLDER` for tests) |
| `audit_summary.py` | Gathers the finished Auditor runs into `results/audit_results.json`, the file `GET /audit/results` returns (a run that is not done gives `null`). | `python -m model.audit_summary` (`--in-dir`, `--out` for tests) |

Where the outputs go: `configs/` and `results/` are small files kept in git. `weights/`, `fakes/`, `logs/` and `data/` (except `data/splits.csv`) are ignored by git.

`data.py` also has `load_fmnist_labeled(split)` (the images plus labels 0-9) and `FMNIST_CLASSES` (the 10 class names).

The three diffusion runs: `g-cddpm-fmnist` (Fashion-MNIST generator with class labels and guidance), `g-ddpm-cifar` (CIFAR-10 generator) and `d-ddpm-cifar` (the Detector's reconstruction model, trained only on the `ddpm_train` part with seed 43).

All numbers (VAE: 10 epochs, batch size 128, learning rate 2e-4, latent size 128; diffusion: batch size 128, learning rate 2e-4, EMA 0.999, 1000 timesteps, cosine schedule, label dropout 0.1, guidance 3.0, epochs chosen from the measured seconds per epoch and written in each run's config) are "proposed, to tune".

## The final run: exact commands

The final Detector (two generators: `g-vae-cifar` and `g-ddpm-cifar`, diffusion features) is built and scored with `model/run_final_detector.py`. Run these four steps in order, from the project folder, with the virtual environment on. **Start only when `d-ddpm-cifar` and `g-ddpm-cifar` have finished training** (the GPU is then free). Time estimates are for this laptop's GPU and are estimates, not measurements.

```
1. python -m model.make_fakes --generator g-ddpm-cifar                  (about 10 min; writes fakes/g-ddpm-cifar_fakes.npz)
2. python -m model.run_final_detector --run-id detector_final --generators g-vae-cifar g-ddpm-cifar --phase train
                                                                         (about 30-60 min; never opens the final test split)
3. python -m model.run_final_detector --run-id detector_final --generators g-vae-cifar g-ddpm-cifar --phase score-final
                                                                         (about 3-5 min; opens the final test split ONCE)
4. the same command as 3 with --activate on the end writes configs/detector_active.json
```

Step 4 is not a separate command: add `--activate` to the step 3 command if you want the API to use this detector straight after scoring (it only activates after a successful scoring). A run that was already scored cannot be run again (not even to activate it), so decide before running step 3. The activation file is just `{"active_run": "<run-id>", "thresholds_from": "results/<run-id>_train.json", "features": ["vae", "ddpm"], "note": "..."}`.

What each step writes (`weights/`, `fakes/` are ignored by git; `configs/` and `results/` are kept):

- step 2: `weights/detector_final_lr_vae.joblib`, `weights/detector_final_lr_vae_ddpm.joblib`, `weights/detector_final_cnn.pt`, `configs/detector_final.json` (settings, thresholds, seeds and the SHA-256 of every weight file, plus of `d-vae-cifar.pt` and `d-ddpm-cifar.pt`; `final_test_used: false`), `results/detector_final_train.json` (calibration data only, stamped "calibration data: not the final result": thresholds, label shares, leave-one-generator-out with the "same generator" number and the drop, robustness against JPEG quality 50, blur sigma 1.0, noise std 0.05) and cached features in `fakes/feat_*.npz`.
- step 3: `results/detector_final_final.json`: per generator and per detector the accuracy, AUROC and the shares of `real` / `ai` / `unclear`, with the frozen thresholds and two caveats ("the Detector's diffusion model and the fake-making diffusion model share an architecture; detection of that generator may look better than on a truly unseen generator" and "VAE fakes are blurry and easy").

What score-final refuses to do (each gives a plain message and a non-zero exit code; nothing is scored): exit code 2 if `results/<run-id>_final.json` already exists ("the final test was already scored for this run id; a new run id (new frozen config) is needed"); exit code 3 if the config is missing, if any frozen weight file (or `d-vae-cifar.pt` / `d-ddpm-cifar.pt`) is missing or has a different SHA-256 than the config, if the thresholds in the config and in the train results differ, or if `--generators` is not exactly the frozen list. It never trains and never changes a threshold. If a model file cannot be read (for example `d-ddpm-cifar.pt` is being written), it waits 3 seconds, tries once more, and then stops with exit code 3.

Leave-one-generator-out trains again without one generator and tests on that generator's calibration fakes; the thresholds are set from the generators it HAS seen. The results on an unseen generator are findings, not promises.

Test it without touching anything (tiny, CPU, `--dry-run` needs `--out-dir`; `g-vae-cifar-b` is a pseudo-generator: the same VAE sampled with other seeds, so that two generators exist; in a dry run score-final reads the CALIBRATION data instead of the final test data and the files say "DRY RUN, not a result"):

```
set CUDA_VISIBLE_DEVICES=-1
python -m model.run_final_detector --run-id detector_final --generators g-vae-cifar g-vae-cifar-b --phase train --dry-run --limit 200 --cnn-epochs 1 --out-dir C:\some\scratch\folder
python -m model.run_final_detector --run-id detector_final --generators g-vae-cifar g-vae-cifar-b --phase score-final --dry-run --limit 200 --out-dir C:\some\scratch\folder
```
