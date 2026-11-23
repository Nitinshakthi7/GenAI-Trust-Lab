# compare.py - the real POST /compare: it compares any two photos with plain image maths.
# (Example: an orange cat and a black cat.) It needs no GPU and no trained model.
#
# Steps:
#   1. Open both images (PNG or JPG), turn them into RGB, remember the original sizes,
#      and shrink both to 224 x 224 so the numbers are fair.
#   2. SSIM = how alike the two pictures are in structure (-1 to 1; 1 means identical).
#   3. A heat picture shows where the two images differ (white = same, red = very different).
#   4. A table of simple numbers per image: brightness, colours, sharpness, edges, texture, size,
#      the 3 main colours (found with k-means) and a colour histogram distance.
#   5. A short summary written from those numbers by simple rules.
# The "words" part (a vision model that describes the differences) is NOT built yet:
# "differences" is always empty and "source" is always "heatmap_only".

import base64
import io

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import laplace
from skimage.metrics import structural_similarity
from sklearn.cluster import KMeans

from api.common import ApiError
from api.stubs import CONTRACT              # only used to copy the contract's note text
from model.utils import SEED                # the one project seed (this only reads a number)

SIZE = 224                       # both images are shrunk to 224 x 224 for the measurements (proposed)
MAX_PIXELS = 40_000_000          # more pixels than this is refused
FOCUS_MAX_CHARS = 100
EDGE_LEVEL = 0.1                 # a pixel is an "edge" if brightness (0 to 1) changes by more than this per pixel (proposed)
WORDS_OFF = "Words are not enabled on this server (no vision API configured)."
NOTE = CONTRACT["POST /compare"]["response"]["note"]

# A small palette to give colours plain names. Each colour is the nearest one in Lab space
# (Lab is a colour space where "near" matches what the eye sees).
PALETTE = {
    "black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128),
    "red": (220, 30, 30), "orange": (255, 140, 0), "yellow": (255, 230, 0),
    "green": (40, 160, 40), "blue": (30, 70, 220), "purple": (140, 50, 170),
    "pink": (255, 150, 190), "brown": (120, 70, 30), "beige": (225, 200, 160),
}
PALETTE_NAMES = list(PALETTE)
PALETTE_LAB = rgb2lab(np.array(list(PALETTE.values()), dtype=float).reshape(1, -1, 3) / 255).reshape(-1, 3)


# ------------------------------------------------------------------ 1. open the images

def open_image(data, name):
    """Return (pixels, (width, height)). pixels = 224 x 224 x 3 numbers from 0 to 1."""
    try:
        picture = Image.open(io.BytesIO(data))          # this only reads the file header
        too_big = picture.width * picture.height > MAX_PIXELS
        if not too_big:
            picture = picture.convert("RGB")            # this reads all the pixels
    except Exception:
        raise ApiError(400, f"{name} could not be read as a picture")
    if too_big:
        raise ApiError(400, f"{name} has more than {MAX_PIXELS // 1_000_000} million pixels")
    small = picture.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return np.asarray(small, dtype=float) / 255, picture.size


# ------------------------------------------------------------- 2. the numbers of one image

def colour_names(colours_rgb):
    """Plain name for each colour (0 to 1 RGB): the nearest palette colour in Lab space."""
    lab = rgb2lab(np.asarray(colours_rgb).reshape(1, -1, 3)).reshape(-1, 3)
    far = ((lab[:, None, :] - PALETTE_LAB[None, :, :]) ** 2).sum(axis=2)    # distance to each palette colour
    return [PALETTE_NAMES[i] for i in far.argmin(axis=1)]


def dominant_colours(pixels):
    """The most common colours as [(name, share), ...], biggest first. Uses k-means with k = 3."""
    flat = pixels.reshape(-1, 3)
    k = min(3, len(np.unique(flat, axis=0)))            # a plain image may have fewer than 3 colours
    kmeans = KMeans(n_clusters=k, n_init=3, random_state=SEED).fit(flat)
    shares = np.bincount(kmeans.labels_, minlength=k) / len(flat)
    found = {}
    for name, share in zip(colour_names(kmeans.cluster_centers_), shares):
        found[name] = found.get(name, 0) + share        # two clusters with the same name are added together
    return sorted(found.items(), key=lambda item: -item[1])


def colours_text(colours):
    """[('orange', 0.62), ...] -> 'orange 62%, white 21%'."""
    return ", ".join(f"{name} {round(share * 100)}%" for name, share in colours if round(share * 100) > 0)


def measure(pixels, size):
    """All the simple numbers of one image."""
    gray = rgb2gray(pixels)
    gy, gx = np.gradient(gray)
    blocks = gray.reshape(SIZE // 8, 8, SIZE // 8, 8)   # the picture cut into 8 x 8 blocks
    return {
        "brightness": gray.mean(),
        "colour": [pixels[..., c].mean() * 255 for c in range(3)],
        "spread": pixels.std(),
        "sharpness": laplace(gray * 255).var(),
        "edges": (np.hypot(gx, gy) > EDGE_LEVEL).mean() * 100,      # percent of pixels
        "texture": blocks.std(axis=(1, 3)).mean(),
        "size": size,
        "colours": dominant_colours(pixels),
    }


# ------------------------------------------------------------- 3. the two-image numbers

def histogram_distance(pixels_a, pixels_b):
    """0 = same colour mix, 1 = completely different. 16 bins for each of R, G and B."""
    total = 0
    for channel in range(3):
        hist_a = np.histogram(pixels_a[..., channel], bins=16, range=(0, 1))[0] / pixels_a[..., 0].size
        hist_b = np.histogram(pixels_b[..., channel], bins=16, range=(0, 1))[0] / pixels_b[..., 0].size
        total += np.abs(hist_a - hist_b).sum() / 2
    return total / 3


def heatmap_png_base64(pixels_a, pixels_b):
    """A 224 x 224 PNG: white where the images are the same, red where the colours differ most."""
    difference = np.abs(pixels_a - pixels_b).mean(axis=2)            # 0 to 1 for every pixel
    fade = (255 * (1 - difference)).astype(np.uint8)                  # 255 = no difference, 0 = the biggest
    heat = np.stack([np.full_like(fade, 255), fade, fade], axis=2)    # red stays 255, green and blue fade
    buffer = io.BytesIO()
    Image.fromarray(heat).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")      # plain base64, no "data:" prefix


# ----------------------------------------------------------------- 4. table and summary

def row(feature, text_a, text_b, does_not_prove):
    return {"feature": feature, "image_a": text_a, "image_b": text_b, "does_not_prove": does_not_prove}


def make_table(a, b, distance):
    colour = lambda m: "(" + ", ".join(str(round(v)) for v in m["colour"]) + ")"
    size = lambda m: f"{m['size'][0]}x{m['size'][1]}"
    same = f"{distance:.2f}"
    return [
        row("mean brightness", f"{a['brightness']:.2f}", f"{b['brightness']:.2f}", "Descriptive only. Lighting and exposure change it."),
        row("average colour (R,G,B)", colour(a), colour(b), "Descriptive only; background colours count too."),
        row("colour spread", f"{a['spread']:.2f}", f"{b['spread']:.2f}", "Descriptive only. It does not say what is in the picture."),
        row("sharpness", f"{a['sharpness']:.0f}", f"{b['sharpness']:.0f}", "Descriptive only. Camera focus and the image size change it."),
        row("edge amount", f"{a['edges']:.0f}%", f"{b['edges']:.0f}%", "Descriptive only. A busy background adds edges."),
        row("texture", f"{a['texture']:.3f}", f"{b['texture']:.3f}", "Descriptive only. It does not say what the texture is."),
        row("original size", size(a), size(b), "Descriptive only. It says nothing about quality or where the image came from."),
        row("dominant colours", colours_text(a["colours"]), colours_text(b["colours"]),
            "Background colours count too; it does not say which object has which colour."),
        row("colour histogram distance", same, same, "Descriptive only. 0 = same colour mix, 1 = very different. It ignores where the colours are."),
    ]


def which_is_more(value_a, value_b, margin):
    """'A' or 'B' if one value is clearly bigger (by more than the margin), otherwise 'same'."""
    if abs(value_a - value_b) <= margin:
        return "same"
    return "A" if value_a > value_b else "B"


def make_summary(a, b, ssim):
    """Three plain sentences made from the numbers by simple rules."""
    bright = {"A": "Image A is brighter than image B", "B": "Image B is brighter than image A",
              "same": "The two images are about equally bright"}
    edges = {"A": "image A has more edges", "B": "image B has more edges",
             "same": "the two images have a similar amount of edges"}
    first = (f"{bright[which_is_more(a['brightness'], b['brightness'], 0.03)]} ({a['brightness']:.2f} vs {b['brightness']:.2f}), "
             f"and {edges[which_is_more(a['edges'], b['edges'], 2)]} ({a['edges']:.0f}% vs {b['edges']:.0f}% of pixels).")
    second = f"The main colours are {colours_text(a['colours'])} in image A and {colours_text(b['colours'])} in image B."
    if ssim >= 0.8:
        third = f"The structural similarity (SSIM) is {ssim:.2f}, so the two images have a very similar layout."
    else:
        third = (f"The structural similarity (SSIM) is {ssim:.2f}. A low value is expected for photos that are not aligned, "
                 "so it does not say how alike the subjects are.")
    return " ".join([first, second, third])


# --------------------------------------------------------------------- the whole answer

def compare(bytes_a, bytes_b, use_words, focus):
    """Build the whole /compare answer (see api/contract.json). The uploads are already checked."""
    if len(focus) > FOCUS_MAX_CHARS:                    # focus will steer the words part later; for now only its length is checked
        raise ApiError(400, f"focus must be at most {FOCUS_MAX_CHARS} characters")
    pixels_a, size_a = open_image(bytes_a, "image_a")
    pixels_b, size_b = open_image(bytes_b, "image_b")
    ssim = round(float(structural_similarity(rgb2gray(pixels_a), rgb2gray(pixels_b), data_range=1.0)), 3)
    a, b = measure(pixels_a, size_a), measure(pixels_b, size_b)
    summary = make_summary(a, b, ssim)
    if use_words:
        summary = WORDS_OFF + " " + summary
    return {
        "similarity": {"ssim": ssim, "lpips": None},
        "diff_heatmap_png_base64": heatmap_png_base64(pixels_a, pixels_b),
        "regions_png_base64": None,
        "feature_table": make_table(a, b, histogram_distance(pixels_a, pixels_b)),
        "differences": [],
        "source": "heatmap_only",
        "summary": summary,
        "note": NOTE,
    }
