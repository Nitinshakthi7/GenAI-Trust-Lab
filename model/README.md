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

Where the outputs go: `configs/` and `results/` are small files kept in git. `weights/`, `fakes/`, `logs/` and `data/` (except `data/splits.csv`) are ignored by git.

`data.py` also has `load_fmnist_labeled(split)` (the images plus labels 0-9) and `FMNIST_CLASSES` (the 10 class names).

The three diffusion runs: `g-cddpm-fmnist` (Fashion-MNIST generator with class labels and guidance), `g-ddpm-cifar` (CIFAR-10 generator) and `d-ddpm-cifar` (the Detector's reconstruction model, trained only on the `ddpm_train` part with seed 43).

All numbers (VAE: 10 epochs, batch size 128, learning rate 2e-4, latent size 128; diffusion: batch size 128, learning rate 2e-4, EMA 0.999, 1000 timesteps, cosine schedule, label dropout 0.1, guidance 3.0, epochs chosen from the measured seconds per epoch and written in each run's config) are "proposed, to tune".
