# detect.py - the real POST /detect (DEV detector): is an uploaded picture real or AI-made?
#
# The maths is done by model/detector.py (the Detector). This file is the web side:
#   1. open the upload (PNG or JPG; common.read_upload already checked the type and the 5 MB limit);
#      a picture that cannot be read, or has more than 40 million pixels, gives 400
#   2. load the Detector the first time it is needed and keep it (503 if its files are missing
#      or are being written right now; the next request tries again)
#   3. score the picture, then add the warning (only for pictures that are not 32x32) and the note
# The note is always in the answer: while the dev detector is active (trained on one weak generator only) it says so;
# when the final detector is active, FINAL_NOTE is used instead.

import io
import threading

from PIL import Image

from api import fullcheck
from api.common import ApiError
from api.compare import MAX_PIXELS          # the same "too many pixels" limit as /compare
from model.detector import Detector, DetectorNotReady

WARNING = ("This detector was trained only on 32x32 CIFAR-10-style images. "
           "Results on other photos are not meaningful.")
# ponytail: this text names the dev detector; change it when the final detector replaces it.
NOTE = ("Dev detector: it was trained on only one weak generator (blurry VAE fakes), so its scores say nothing "
        "about diffusion fakes or other unseen generators. This is a work-in-progress check, not a result. "
        "Grad-CAM shows where the CNN baseline looked; it does not prove why the image is real or AI.")

# Used instead of NOTE when configs/detector_active.json points to the FINAL detector (model/run_final_detector.py).
FINAL_NOTE = ("Final detector: trained on VAE fakes and diffusion fakes, with VAE and diffusion features. How well it works on "
              "a generator it has never seen is a finding, not a promise, and it is not shown here. All numbers are proposed, to tune. "
              "Grad-CAM shows where the CNN baseline looked; it does not prove why the image is real or AI.")

_detector = None                    # the loaded Detector (made on the first request)
_load_lock = threading.Lock()       # only one request loads it


def get_detector():
    """The Detector, loaded once and kept. Raises ApiError(503) while its files are not ready."""
    global _detector
    with _load_lock:
        if _detector is None:
            try:
                _detector = Detector()
            except DetectorNotReady as error:
                raise ApiError(503, f"detector not ready: {error}")
        return _detector


def open_picture(data):
    """The uploaded bytes -> an RGB picture. Raises ApiError(400) if it cannot be read or is too big."""
    try:
        picture = Image.open(io.BytesIO(data))              # this only reads the file header
        too_big = picture.width * picture.height > MAX_PIXELS
        if not too_big:
            picture = picture.convert("RGB")                # this reads all the pixels
    except Exception:
        raise ApiError(400, "the uploaded image could not be read as a picture")
    if too_big:
        raise ApiError(400, f"the uploaded image has more than {MAX_PIXELS // 1_000_000} million pixels")
    return picture


def detect(data):
    """Build the whole /detect answer (see api/contract.json). The upload is already checked."""
    picture = open_picture(data)                            # bad pictures are answered before any model is loaded
    detector = get_detector()
    answer = detector.score_image(picture)
    answer["warning"] = WARNING if answer["out_of_scope"] else None
    answer["note"] = FINAL_NOTE if detector.final else NOTE
    if answer["out_of_scope"]:                              # a normal-size photo: add the full-size check (api/fullcheck.py)
        try:
            answer["full_check"] = fullcheck.full_check(data)
        except Exception as error:                          # the 32x32 answer above still goes out
            answer["full_check_error"] = f"the full-size check failed: {type(error).__name__}"
    return answer
