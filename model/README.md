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
| `cnn.py` | `SmallCNN`, the simple real-vs-fake baseline classifier (not trained yet, it needs the fake images first). | `python -m model.cnn` (self-check) |

Where the outputs go: `configs/` and `results/` are small files kept in git. `weights/`, `fakes/`, `logs/` and `data/` (except `data/splits.csv`) are ignored by git.

All numbers (10 epochs, batch size 128, learning rate 2e-4, latent size 128) are "proposed, to tune".
