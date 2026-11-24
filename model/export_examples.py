# export_examples.py - saves the small "Try an example" pictures for the Detector page.
# It writes 20 tiny PNG files (all exactly 32x32, the size the Detector is built for):
#   examples/fake/fake_ddpm_0..4.png   5 fakes from the diffusion model (g-ddpm-cifar)
#   examples/fake/fake_vae_0..4.png    5 fakes from the VAE (g-vae-cifar)
#   examples/real/real_0..9.png        10 real CIFAR-10 pictures
#   examples/examples.json             a list of all 20, so the website can build its picture strip
#
# The final test images are NEVER used. The real pictures come from the 'calibration' part only
# (first 10 of data/splits.csv) and the fakes from the 'calibration' array only (first 5).
# The indices are fixed, so running this twice gives the same files.
#
# Run it from the project folder:   python -m model.export_examples
# It is a tiny job and never needs the graphics card, so it switches the card off for itself.

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # CPU only: this job is tiny

import json
import pickle

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from model.data import get_part
from model.utils import DATA_DIR, FAKES_DIR, ROOT

EXAMPLES_DIR = ROOT / "examples"
CIFAR_TEST_BATCH = DATA_DIR / "cifar10" / "cifar-10-batches-py" / "test_batch"
FAKE_GENERATORS = [("g-ddpm-cifar", "ddpm"), ("g-vae-cifar", "vae")]    # (model id, short name for the file)
FAKES_PER_MODEL = 5
REAL_COUNT = 10


def save_png(pixels, path, note=None):
    """pixels: a 32x32x3 picture of numbers 0..255. Saved at its own size (no upscaling).
    note: a short plain-English text stored inside the PNG under the key 'provenance'."""
    pixels = np.asarray(pixels, dtype=np.uint8)
    assert pixels.shape == (32, 32, 3), f"expected 32x32x3, got {pixels.shape}"
    info = None
    if note:
        info = PngInfo()
        info.add_text("provenance", note)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, "RGB").save(path, format="PNG", pnginfo=info)


def main():
    entries = []

    # --- fakes: only the 'calibration' array of each fakes file is read (never 'test')
    for model_id, short in FAKE_GENERATORS:
        with np.load(FAKES_DIR / f"{model_id}_fakes.npz") as saved:
            calibration = saved["calibration"]
        assert calibration.shape == (5000, 32, 32, 3), f"{model_id}: unexpected shape {calibration.shape}"
        for i in range(FAKES_PER_MODEL):
            name = f"fake_{short}_{i}.png"
            note = f"model: {model_id}; made by GenAI Trust Lab; small model; example image"
            save_png(calibration[i], EXAMPLES_DIR / "fake" / name, note)
            entries.append({"file": f"fake/{name}", "kind": "fake", "source": model_id, "index": i})

    # --- real pictures: the first 10 indices of the 'calibration' part of the CIFAR-10 test images
    calibration_ids = [int(i) for i in get_part("calibration")[:REAL_COUNT]]
    assert len(calibration_ids) == REAL_COUNT
    assert not set(calibration_ids) & set(int(i) for i in get_part("final_test")), "a final_test image slipped in"
    with open(CIFAR_TEST_BATCH, "rb") as file:
        data = pickle.load(file, encoding="bytes")[b"data"]           # (10000, 3072): the CIFAR-10 test images
    for n, index in enumerate(calibration_ids):
        name = f"real_{n}.png"
        save_png(data[index].reshape(3, 32, 32).transpose(1, 2, 0), EXAMPLES_DIR / "real" / name)
        entries.append({"file": f"real/{name}", "kind": "real", "source": f"cifar10 calibration image {index}", "index": index})

    (EXAMPLES_DIR / "examples.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} pictures and examples.json to {EXAMPLES_DIR.name}/")


if __name__ == "__main__":
    main()
