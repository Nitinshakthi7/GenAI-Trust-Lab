# detector.py - the Detector that POST /detect uses: "is this picture real or AI-made?"
#
# It loads everything ONCE, when it is created. Which detector it loads depends on configs/detector_active.json:
#
#   DEV detector (the file has no "features" key, or "features": ["vae"]):
#     - the detector's own VAE (weights/d-vae-cifar.pt), loaded by the same code as model/features.py
#     - the logistic regression (weights/detector_lr.joblib)  -> the main score
#     - the small CNN (weights/cnn-baseline.pt)               -> the baseline score
#     - the two thresholds (low, high) of each detector, from the results file the pointer names
#
#   FINAL detector ("features": ["vae", "ddpm"], made by model/run_final_detector.py, files named after the run id):
#     - the same VAE, plus the detector's diffusion model (weights/d-ddpm-cifar.pt)
#     - weights/<run>_lr_vae_ddpm.joblib: logistic regression on 2 VAE + 4 diffusion numbers -> the main score
#     - weights/<run>_lr_vae.joblib: the same on the 2 VAE numbers only -> the fallback main score
#     - weights/<run>_cnn.pt -> the baseline score
#     - thresholds from results/<run>_train.json
#     - the heatmap is the diffusion model's reconstruction error (heatmap_source 'ddpm'). If the diffusion model
#       file is missing or cannot be read (for example while it is being written), the VAE-only score and the VAE
#       heatmap (heatmap_source 'vae') are used instead, and the diffusion model is tried again on the next picture.
#
# To use another detector later, change the pointer file.
# Then score_image(picture) scores ONE picture on the CPU: a few milliseconds for the dev detector; the final
# detector adds 20 diffusion steps, which takes a fraction of a second to about a second.
#
# The models only know 32x32 CIFAR-10-style pictures. Any other size is still shrunk to 32x32 and
# scored, but it is flagged out_of_scope = True (proposed rule), because the result is not meaningful.
#
# What score_image returns (the field names of POST /detect in api/contract.json, minus warning and note):
#   label        'ai' if score >= high, 'real' if score <= low, otherwise 'unclear' (the main score's thresholds)
#   score        the main logistic regression: probability that the picture is AI-made, 0 to 1
#   vae_score    the VAE's anomaly score: reconstruction error + KL. It is NOT limited to 0..1.
#   baseline_score  CNN: probability that the picture is AI-made, 0 to 1
#   heatmap_png_base64  where the model rebuilds the picture badly: a 128 x 128 PNG, white (no error) to red (big error)
#   heatmap_source 'vae' (the VAE's error) or 'ddpm' (the diffusion model's error, final detector only)
#   gradcam_png_base64  where the CNN baseline looked for its 'AI' output (Grad-CAM, model/gradcam.py): the picture at
#                       128 x 128 tinted red where the CNN looked. It explains the CNN, NOT the forgery.
#   input_size   'WIDTHxHEIGHT' of the ORIGINAL picture;  out_of_scope True if that is not exactly 32x32
# All numbers here are "proposed, to tune".

import base64
import copy
import io
import json

import joblib
import numpy as np
import torch
from PIL import Image

from model.cnn import SmallCNN
from model.ddpm import reconstruct
from model.features import load_ddpm_detector, load_vae_detector
from model.gradcam import gradcam, overlay_png_base64
from model.make_fakes import to_float, to_uint8
from model.utils import CONFIGS_DIR, ROOT, WEIGHTS_DIR
from model.vae import per_image_terms

SIZE = 32                      # the size the models were trained on
CPU_THREADS = 2                # keep the computer usable while other things run
HEATMAP_SIZE = 128             # the heatmap picture is shown at 128 x 128
HEATMAP_FULL_RED = 0.25        # a pixel whose squared error (scale -1..1, averaged over R,G,B) is this big or more is fully red
ACTIVE_FILE = CONFIGS_DIR / "detector_active.json"
DETECTORS = ("logistic_regression", "cnn_baseline")           # the dev detector's two thresholds
FINAL_DETECTORS = ("lr_vae", "lr_vae_ddpm", "cnn")            # the final detector's three thresholds
DDPM_T_START = 200             # the diffusion settings of model.features.ddpm_features (the lr was trained with these)
DDPM_STEPS = 20
DDPM_FULL_RED = 0.3            # (proposed) a diffusion error (scale -1..1, absolute, averaged over R,G,B) this big is fully red; about the 99th percentile of real images


class DetectorNotReady(Exception):
    """A file the detector needs is missing, or cannot be read right now."""


def loaded(what, load):
    """Run load(). If anything goes wrong, say in plain words which thing was the problem."""
    try:
        return load()
    except Exception:
        raise DetectorNotReady(f"{what} is missing or cannot be read right now")


def read_thresholds():
    """Read configs/detector_active.json and the results file it names. Returns (run id, features, thresholds):
    features is ['vae'] (dev detector) or ['vae', 'ddpm'] (final detector); thresholds is {detector name: (low, high)}
    with the names DETECTORS (dev, from 'dev_report') or FINAL_DETECTORS (final, from 'calibration_report')."""
    active = json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))
    report = json.loads((ROOT / active["thresholds_from"]).read_text(encoding="utf-8"))
    assert report["run_id"] == active["active_run"], "the results file belongs to another run"
    features = active.get("features", ["vae"])
    if features == ["vae"]:
        names, table = DETECTORS, report["dev_report"]
    elif features == ["vae", "ddpm"]:
        names, table = FINAL_DETECTORS, report["calibration_report"]
    else:
        raise ValueError(f"unknown features {features}")
    found = {name: (float(table[name]["low_threshold"]), float(table[name]["high_threshold"])) for name in names}
    return active["active_run"], features, found


def to_tensor(picture):
    """A PIL picture -> floats in [-1,1], shape (1,3,32,32). Another size is shrunk (or stretched) to 32x32."""
    picture = picture.convert("RGB")
    if picture.size != (SIZE, SIZE):
        picture = picture.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return to_float(np.array(picture)[None])       # the same conversion the training used


def make_label(score, low, high):
    """'ai' if score >= high, 'real' if score <= low, otherwise 'unclear' (the same rule as train_detector.py)."""
    if score >= high:
        return "ai"
    if score <= low:
        return "real"
    return "unclear"


def ddpm_numbers(error):
    """The 4 diffusion numbers of ONE reconstruction error map (1,3,32,32): mean, 50th, 90th, 99th percentile.
    The same maths as model.features.ddpm_features (which the logistic regression was trained with)."""
    flat = error.flatten(1)
    spread = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99]), dim=1).T
    return torch.cat([flat.mean(1, keepdim=True), spread], dim=1).numpy()


def heatmap_png_base64(error_map, full_red=HEATMAP_FULL_RED):
    """error_map: 32 x 32 numbers (the error of each pixel) -> a 128 x 128 PNG as plain base64 text.
    White = no error, red = full_red or more. Only numpy and Pillow are used."""
    level = np.clip(error_map / full_red, 0, 1)                            # 0 (no error) .. 1 (full red)
    fade = (255 * (1 - level)).astype(np.uint8)                            # 255 = white, 0 = full red
    heat = np.stack([np.full_like(fade, 255), fade, fade], axis=2)         # red stays 255, green and blue fade
    big = Image.fromarray(heat).resize((HEATMAP_SIZE, HEATMAP_SIZE), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    big.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")            # no "data:" prefix


class Detector:
    """Loads all the models once. Raises DetectorNotReady if something is missing."""

    def __init__(self):
        torch.set_num_threads(CPU_THREADS)
        self.run_id, features, self.thresholds = loaded(
            "the detector thresholds (results file named in configs/detector_active.json)", read_thresholds)
        self.final = features == ["vae", "ddpm"]                          # False = the dev detector
        self.vae = loaded("the detector VAE (weights/d-vae-cifar.pt)", load_vae_detector).cpu().eval()
        if self.final:
            run = self.run_id
            self.lr_vae = loaded(f"the VAE logistic regression (weights/{run}_lr_vae.joblib)",
                                 lambda: joblib.load(WEIGHTS_DIR / f"{run}_lr_vae.joblib"))
            self.lr = loaded(f"the main logistic regression (weights/{run}_lr_vae_ddpm.joblib)",
                             lambda: joblib.load(WEIGHTS_DIR / f"{run}_lr_vae_ddpm.joblib"))
            self.cnn = loaded(f"the CNN (weights/{run}_cnn.pt)", lambda: self._load_cnn(f"{run}_cnn.pt"))
            self.ddpm = self._load_ddpm()              # None if not ready: the VAE-only score is used (see _main_score)
        else:
            self.lr = loaded("the logistic regression (weights/detector_lr.joblib)",
                             lambda: joblib.load(WEIGHTS_DIR / "detector_lr.joblib"))
            self.cnn = loaded("the CNN (weights/cnn-baseline.pt)", self._load_cnn)
        # Grad-CAM gets its own copy of the CNN: it briefly attaches hooks and needs gradients, and two
        # requests can arrive at the same time. This way it can never disturb the baseline score.
        self.cam_cnn = copy.deepcopy(self.cnn)

    @staticmethod
    def _load_cnn(file="cnn-baseline.pt"):
        cnn = SmallCNN()
        cnn.load_state_dict(torch.load(WEIGHTS_DIR / file, map_location="cpu", weights_only=True))
        return cnn.eval()

    @staticmethod
    def _load_ddpm():
        """The detector's diffusion model, or None if its file is missing or cannot be read right now."""
        try:
            return load_ddpm_detector().cpu().eval()
        except Exception:
            return None

    def _main_score(self, x, vae_numbers, vae_error_map):
        """The main score of the detector. Returns (score, (low, high), heatmap PNG text, heatmap source)."""
        if not self.final:                                                  # the dev detector: the VAE only
            score = float(self.lr.predict_proba(vae_numbers)[0, 1])
            return score, self.thresholds["logistic_regression"], heatmap_png_base64(vae_error_map), "vae"
        if self.ddpm is None:
            self.ddpm = self._load_ddpm()                                   # try again: it may be written by now
        if self.ddpm is None:                                               # still not there: VAE-only fallback
            score = float(self.lr_vae.predict_proba(vae_numbers)[0, 1])
            return score, self.thresholds["lr_vae"], heatmap_png_base64(vae_error_map), "vae"
        error = reconstruct(self.ddpm, x, DDPM_T_START, DDPM_STEPS)[1]      # (1,3,32,32): noise it, clean it, compare
        score = float(self.lr.predict_proba(np.concatenate([vae_numbers, ddpm_numbers(error)], axis=1))[0, 1])
        heatmap = heatmap_png_base64(error.mean(dim=1)[0].numpy(), DDPM_FULL_RED)
        return score, self.thresholds["lr_vae_ddpm"], heatmap, "ddpm"

    @torch.no_grad()
    def score_image(self, picture):
        """Score one PIL picture. Returns the dict described at the top of this file."""
        width, height = picture.size
        x = to_tensor(picture)
        # The VAE rebuilds the picture (no random nudge, so the same picture always gives the same numbers).
        mu, logvar = self.vae.encode(x)
        recon = self.vae.decode(mu)
        recon_err, kl = per_image_terms(recon, x, mu, logvar)
        # The two numbers the logistic regression learned from: the same as model.features.vae_features.
        features = np.array([[recon_err.item() / x[0].numel(), kl.item()]], dtype=np.float32)
        error_map = ((recon - x) ** 2).mean(dim=1)[0].numpy()               # 32 x 32: the VAE's error of each pixel
        score, (low, high), heatmap, heatmap_source = self._main_score(x, features, error_map)
        baseline = float(torch.sigmoid(self.cnn(x)).item())
        cam = gradcam(self.cam_cnn, x)                                      # 32 x 32: where the CNN looked (0..1)
        return {
            "label": make_label(score, low, high),
            "score": round(score, 4),
            "vae_score": round(recon_err.item() + kl.item(), 2),
            "baseline_score": round(baseline, 4),
            "heatmap_png_base64": heatmap,
            "heatmap_source": heatmap_source,
            "gradcam_png_base64": overlay_png_base64(cam, to_uint8(x)[0]),
            "input_size": f"{width}x{height}",
            "out_of_scope": (width, height) != (SIZE, SIZE),
        }
