# detector.py - the Detector that POST /detect uses: "is this picture real or AI-made?"
#
# It loads everything ONCE, when it is created:
#   - the detector's own VAE (weights/d-vae-cifar.pt), loaded by the same code as model/features.py
#   - the logistic regression (weights/detector_lr.joblib)  -> the main score
#   - the small CNN (weights/cnn-baseline.pt)               -> the baseline score
#   - the two thresholds (low, high) of each detector, from the results file that
#     configs/detector_active.json points to. To use another detector later, change that pointer.
# Then score_image(picture) scores ONE picture on the CPU, usually in a few milliseconds.
#
# The models only know 32x32 CIFAR-10-style pictures. Any other size is still shrunk to 32x32 and
# scored, but it is flagged out_of_scope = True (proposed rule), because the result is not meaningful.
#
# What score_image returns (the field names of POST /detect in api/contract.json, minus warning and note):
#   label        'ai' if score >= high, 'real' if score <= low, otherwise 'unclear' (the logistic regression's thresholds)
#   score        logistic regression: probability that the picture is AI-made, 0 to 1
#   vae_score    the VAE's anomaly score: reconstruction error + KL. It is NOT limited to 0..1.
#   baseline_score  CNN: probability that the picture is AI-made, 0 to 1
#   heatmap_png_base64  where the VAE rebuilds the picture badly: a 128 x 128 PNG, white (no error) to red (big error)
#   heatmap_source 'vae'
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
from model.features import load_vae_detector
from model.gradcam import gradcam, overlay_png_base64
from model.make_fakes import to_float, to_uint8
from model.utils import CONFIGS_DIR, ROOT, WEIGHTS_DIR
from model.vae import per_image_terms

SIZE = 32                      # the size the models were trained on
CPU_THREADS = 2                # keep the computer usable while other things run
HEATMAP_SIZE = 128             # the heatmap picture is shown at 128 x 128
HEATMAP_FULL_RED = 0.25        # a pixel whose squared error (scale -1..1, averaged over R,G,B) is this big or more is fully red
ACTIVE_FILE = CONFIGS_DIR / "detector_active.json"
DETECTORS = ("logistic_regression", "cnn_baseline")


class DetectorNotReady(Exception):
    """A file the detector needs is missing, or cannot be read right now."""


def loaded(what, load):
    """Run load(). If anything goes wrong, say in plain words which thing was the problem."""
    try:
        return load()
    except Exception:
        raise DetectorNotReady(f"{what} is missing or cannot be read right now")


def read_thresholds():
    """{'logistic_regression': (low, high), 'cnn_baseline': (low, high)}, read from the results file
    that configs/detector_active.json names. Returns (run id, thresholds)."""
    active = json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))
    report = json.loads((ROOT / active["thresholds_from"]).read_text(encoding="utf-8"))
    assert report["run_id"] == active["active_run"], "the results file belongs to another run"
    found = {name: (float(report["dev_report"][name]["low_threshold"]),
                    float(report["dev_report"][name]["high_threshold"])) for name in DETECTORS}
    return active["active_run"], found


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


def heatmap_png_base64(error_map):
    """error_map: 32 x 32 numbers (the error of each pixel) -> a 128 x 128 PNG as plain base64 text.
    White = no error, red = HEATMAP_FULL_RED or more. Only numpy and Pillow are used."""
    level = np.clip(error_map / HEATMAP_FULL_RED, 0, 1)                    # 0 (no error) .. 1 (full red)
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
        self.run_id, self.thresholds = loaded("the detector thresholds (results file named in configs/detector_active.json)",
                                              read_thresholds)
        self.vae = loaded("the detector VAE (weights/d-vae-cifar.pt)", load_vae_detector).cpu().eval()
        self.lr = loaded("the logistic regression (weights/detector_lr.joblib)",
                         lambda: joblib.load(WEIGHTS_DIR / "detector_lr.joblib"))
        self.cnn = loaded("the CNN (weights/cnn-baseline.pt)", self._load_cnn)
        # Grad-CAM gets its own copy of the CNN: it briefly attaches hooks and needs gradients, and two
        # requests can arrive at the same time. This way it can never disturb the baseline score.
        self.cam_cnn = copy.deepcopy(self.cnn)

    @staticmethod
    def _load_cnn():
        cnn = SmallCNN()
        cnn.load_state_dict(torch.load(WEIGHTS_DIR / "cnn-baseline.pt", map_location="cpu", weights_only=True))
        return cnn.eval()

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
        score = float(self.lr.predict_proba(features)[0, 1])
        baseline = float(torch.sigmoid(self.cnn(x)).item())
        low, high = self.thresholds["logistic_regression"]
        error_map = ((recon - x) ** 2).mean(dim=1)[0].numpy()               # 32 x 32: error of each pixel
        cam = gradcam(self.cam_cnn, x)                                      # 32 x 32: where the CNN looked (0..1)
        return {
            "label": make_label(score, low, high),
            "score": round(score, 4),
            "vae_score": round(recon_err.item() + kl.item(), 2),
            "baseline_score": round(baseline, 4),
            "heatmap_png_base64": heatmap_png_base64(error_map),
            "heatmap_source": "vae",
            "gradcam_png_base64": overlay_png_base64(cam, to_uint8(x)[0]),
            "input_size": f"{width}x{height}",
            "out_of_scope": (width, height) != (SIZE, SIZE),
        }
