# build_final_deck.py - builds the FINAL presentation deck (7 slides, Thursday 26 Nov 2026).
#
# Run it from the project folder:   .venv\Scripts\python docs\build_final_deck.py
# It writes  docs\GenAI_Trust_Lab_Final.pptx  and (only if they are missing) the small pictures in docs\deck_assets\.
#   --refresh-assets   make the small pictures again (this calls the real API on the CPU, takes a minute)
#
# THE RULE OF THIS SCRIPT: every number on a slide is READ from a results file when the script runs.
# Nothing is typed in by hand. If a file or a value is missing, the slide shows
#   [PLACEHOLDER - not measured yet]
# in that cell. So you can run this script again later (for example after the Auditor results arrive)
# and the deck updates itself.
#
# Where the numbers come from (each slide names its file in a small footnote):
#   slide 3  results/train_*.json, results/ddim_timing.json
#   slide 4  docs/deck_assets/detect_examples.json (one API answer per picture; a demo, not a measurement)
#   slide 5  results/detector_final_final.json (final test split, scored once) and results/detector_final_train.json
#   slide 6  docs/deck_assets/compare_example.json (a made-up test pair) and results/audit_results.json
#
# This script never opens the final test split of the images. It only reads the result files that were already written.

import base64
import io
import json
import math
import os
import pickle
import sys
from pathlib import Path

# ---- Paths. Everything is worked out from where this file lives, so no folder names are typed in.
ROOT = Path(__file__).resolve().parent.parent          # the project folder
DOCS = ROOT / "docs"
ASSETS = DOCS / "deck_assets"
RESULTS = ROOT / "results"
FAKES = ROOT / "fakes"
OUT_PPTX = DOCS / "GenAI_Trust_Lab_Final.pptx"


# =====================================================================
# PART 1 - THE SMALL PICTURES (made once, with the real API on the CPU)
# =====================================================================

def asset_names():
    """The files the deck needs inside docs/deck_assets/."""
    return [
        "samples_g-cddpm-fmnist.png", "samples_g-ddpm-cifar.png", "samples_g-vae-cifar.png",
        "real_input.png", "real_heatmap.png", "real_gradcam.png",
        "fake_input.png", "fake_heatmap.png", "fake_gradcam.png",
        "cat_orange.png", "cat_black.png", "compare_diff.png",
        "detect_examples.json", "compare_example.json",
    ]


def draw_cat(body_rgb):
    """A very simple cat-like drawing (grey background, round body, head, two ears, eyes). 224 x 224 pixels.
    It is an ILLUSTRATION made for testing the Compare tool. It is not a result."""
    from PIL import Image, ImageDraw
    picture = Image.new("RGB", (224, 224), (128, 128, 128))
    pen = ImageDraw.Draw(picture)
    pen.ellipse((40, 120, 184, 210), fill=body_rgb)                        # body
    pen.ellipse((72, 40, 152, 120), fill=body_rgb)                         # head
    pen.polygon([(76, 62), (80, 22), (106, 46)], fill=body_rgb)            # left ear
    pen.polygon([(148, 62), (144, 22), (118, 46)], fill=body_rgb)          # right ear
    pen.ellipse((90, 70, 102, 84), fill=(235, 235, 120))                   # eyes
    pen.ellipse((122, 70, 134, 84), fill=(235, 235, 120))
    return picture


def png_bytes(picture):
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def make_assets():
    """Make the small pictures. The CPU is used on purpose (CUDA_VISIBLE_DEVICES=-1), because the graphics card
    may be busy with training. Only CALIBRATION pictures are used, never the final test split."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           # hide the graphics card from torch (must be set BEFORE torch is imported)
    sys.path.insert(0, str(ROOT))                       # so that "import api" and "import model" work (same as PYTHONPATH)
    import numpy as np
    import torch
    from fastapi.testclient import TestClient
    from PIL import Image
    torch.set_num_threads(2)                            # keep the computer usable
    from api.main import app
    from model.data import get_part

    ASSETS.mkdir(exist_ok=True)
    client = TestClient(app, raise_server_exceptions=False)

    # (a) our own sample sheets, blown up 4 times with sharp pixels (they are tiny: 32 x 32 pictures)
    for run in ("g-cddpm-fmnist", "g-ddpm-cifar", "g-vae-cifar"):
        sheet = Image.open(FAKES / f"{run}_samples.png").convert("RGB")
        sheet.resize((sheet.width * 4, sheet.height * 4), Image.NEAREST).save(ASSETS / f"samples_{run}.png")

    # (b) one real CIFAR-10 CALIBRATION picture and one diffusion-fake CALIBRATION picture -> real /detect
    with open(ROOT / "data" / "cifar10" / "cifar-10-batches-py" / "test_batch", "rb") as file:
        data = pickle.load(file, encoding="bytes")[b"data"]
    index = int(get_part("calibration")[0])            # a calibration index, never a final_test one
    real = Image.fromarray(data[index].reshape(3, 32, 32).transpose(1, 2, 0))
    with np.load(FAKES / "g-ddpm-cifar_fakes.npz") as saved:
        fake = Image.fromarray(saved["calibration"][0])   # only the calibration array is read
    answers = {}
    for name, picture in (("real", real), ("fake", fake)):
        reply = client.post("/detect", files={"image": (f"{name}.png", png_bytes(picture), "image/png")})
        if reply.status_code != 200:
            raise SystemExit(f"/detect failed for the {name} picture: {reply.status_code} {reply.text[:200]}")
        body = reply.json()
        picture.resize((128, 128), Image.NEAREST).save(ASSETS / f"{name}_input.png")
        (ASSETS / f"{name}_heatmap.png").write_bytes(base64.b64decode(body["heatmap_png_base64"]))
        (ASSETS / f"{name}_gradcam.png").write_bytes(base64.b64decode(body["gradcam_png_base64"]))
        answers[name] = {k: v for k, v in body.items() if not k.endswith("_base64")}   # keep the numbers and words, drop the pictures
    answers["real"]["picture"] = f"CIFAR-10 test image, calibration part, number {index}"
    answers["fake"]["picture"] = "fake from fakes/g-ddpm-cifar_fakes.npz, calibration part, number 0"
    (ASSETS / "detect_examples.json").write_text(json.dumps(answers, indent=1), encoding="utf-8")

    # (c) the made-up cat pair -> real /compare (words are off on this server, so use_words is false)
    orange, black = draw_cat((240, 140, 40)), draw_cat((20, 20, 20))
    orange.save(ASSETS / "cat_orange.png")
    black.save(ASSETS / "cat_black.png")
    reply = client.post("/compare", files={"image_a": ("a.png", png_bytes(orange), "image/png"),
                                           "image_b": ("b.png", png_bytes(black), "image/png")},
                        data={"use_words": "false", "focus": ""})
    if reply.status_code != 200:
        raise SystemExit(f"/compare failed: {reply.status_code} {reply.text[:200]}")
    body = reply.json()
    (ASSETS / "compare_diff.png").write_bytes(base64.b64decode(body["diff_heatmap_png_base64"]))
    table = {row["feature"]: [row["image_a"], row["image_b"]] for row in body["feature_table"]}
    keep = {"ssim": body["similarity"]["ssim"], "dominant_colours": table.get("dominant colours"),
            "mean_brightness": table.get("mean brightness"), "source": body["source"], "summary": body["summary"]}
    (ASSETS / "compare_example.json").write_text(json.dumps(keep, indent=1), encoding="utf-8")


def ensure_assets(force):
    """Make the pictures if any is missing (or if --refresh-assets was given)."""
    missing = [n for n in asset_names() if not (ASSETS / n).exists()]
    if force or missing:
        print("making the small pictures (real API, CPU) ...")
        make_assets()
    for name in asset_names():
        path = ASSETS / name
        print(f"  asset {name}: {path.stat().st_size} bytes")


# =====================================================================
# PART 2 - READING THE RESULT FILES
# =====================================================================

PLACEHOLDER = "[PLACEHOLDER - not measured yet]"       # shown in any cell whose value is missing


def read_json(path):
    """Open a JSON file. Returns None if the file is missing or broken (the slide then shows the placeholder)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def dig(data, *keys):
    """Walk down into nested dicts / lists, e.g. dig(d, "a", "b"). Returns None if any step is missing."""
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        elif isinstance(data, list) and isinstance(key, int) and -len(data) <= key < len(data):
            data = data[key]
        else:
            return None
    return data


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def num(value, decimals=3):
    """A number as text, or the placeholder if the value is missing."""
    return f"{value:.{decimals}f}" if is_number(value) else PLACEHOLDER


def pct(value):
    """A share between 0 and 1 as a percentage text (0.0123 -> 1.2%), or the placeholder."""
    return f"{value * 100:.1f}%" if is_number(value) else PLACEHOLDER


def text_or_ph(value):
    return str(value) if value not in (None, "") else PLACEHOLDER


# The result files (read every time the script runs).
FINAL = read_json(RESULTS / "detector_final_final.json")      # final test split, scored once
TRAIN = read_json(RESULTS / "detector_final_train.json")      # calibration data (not the final result)
TIMING = read_json(RESULTS / "ddim_timing.json")
AUDIT = read_json(RESULTS / "audit_results.json")             # does not exist until the Auditor runs are done

# The training runs: generators (g-...) first, then the Detector's own models (d-...).
TRAIN_FILES = sorted(RESULTS.glob("train_*.json"), key=lambda p: (not p.stem.startswith("train_g-"), p.name))
RUNS = [p.stem[len("train_"):] for p in TRAIN_FILES]
RUN_RESULT = {run: read_json(RESULTS / f"train_{run}.json") for run in RUNS}
RUN_CONFIG = {run: read_json(ROOT / "configs" / f"{run}.json") for run in RUNS}     # only the dataset name is read from here
DATASET_NAMES = {"cifar10": "CIFAR-10", "fmnist": "Fashion-MNIST"}

DETECTORS = [("lr_vae", "Logistic regression, VAE features"),
             ("lr_vae_ddpm", "Logistic regression, VAE + diffusion features"),
             ("cnn", "Plain CNN baseline")]
GENERATORS = [("g-vae-cifar", "VAE fakes"), ("g-ddpm-cifar", "Diffusion fakes")]
DET_SHORT = {"lr_vae": "logistic regression on VAE features",
             "lr_vae_ddpm": "logistic regression on VAE + diffusion features", "cnn": "plain CNN baseline"}
GEN_SHORT = {"g-vae-cifar": "VAE fakes", "g-ddpm-cifar": "diffusion fakes"}
PERT_NAMES = {"jpeg_q50": "JPEG quality 50", "blur_sigma1.0": "blur sigma 1.0", "noise_std0.05": "noise std 0.05"}


def final_auroc(detector, generator):
    return dig(FINAL, "detectors", detector, "per_generator", generator, "auroc")


def loo(generator, detector, part, key="auroc"):
    """Leave-one-generator-out numbers. part = unseen_generator, same_generator or drop_same_minus_unseen."""
    return dig(TRAIN, "leave_one_generator_out", generator, detector, part, key)


def audit_points(experiment_index, defaults):
    """The two rows of one Auditor table: [(x, memorization_rate or None), ...]. The x values are the PROPOSED settings
    when the results file does not exist yet; the rate stays None (placeholder) until it is measured."""
    points = dig(AUDIT, "experiments", experiment_index, "points")
    rows = []
    for i, default_x in enumerate(defaults):
        x = dig(points, i, "x")
        rate = dig(points, i, "memorization_rate")
        rows.append((x if is_number(x) else default_x, rate if is_number(rate) else None))
    return rows


AUDIT_SIZE_DEFAULTS = [500, 10000]        # proposed training-set sizes
AUDIT_DUP_DEFAULTS = [0, 30]              # proposed duplicate shares (%)
AUDIT_READY = any(rate is not None for _, rate in audit_points(0, AUDIT_SIZE_DEFAULTS) + audit_points(1, AUDIT_DUP_DEFAULTS))


# ---- Takeaways: simple rules that write a sentence from the numbers (nothing typed in by hand)
def takeaway_detectors():
    cnn, a, b = final_auroc("cnn", "g-ddpm-cifar"), final_auroc("lr_vae", "g-ddpm-cifar"), final_auroc("lr_vae_ddpm", "g-ddpm-cifar")
    if not all(is_number(v) for v in (cnn, a, b)):
        return PLACEHOLDER
    if cnn > max(a, b):
        head = "The plain CNN baseline beats the explainable features here"
    elif cnn < min(a, b):
        head = "The explainable features beat the plain CNN baseline here"
    else:
        head = "No clear winner between the CNN baseline and the explainable features"
    return f"{head}: on diffusion fakes the CNN reaches AUROC {cnn:.3f}, logistic regression {a:.3f} and {b:.3f}."


def takeaway_vae_easy():
    values = [final_auroc(d, "g-vae-cifar") for d, _ in DETECTORS]
    if not all(is_number(v) for v in values):
        return PLACEHOLDER
    low = min(values)
    if low >= 0.9:
        floor = math.floor(low * 1000) / 1000          # round DOWN so "at least" is always true
        return f"All three find the blurry VAE fakes easily (AUROC at least {floor:.3f})."
    return f"The VAE fakes are not easy for every detector (lowest AUROC {low:.3f})."


def takeaway_unseen():
    drops = [loo(g, d, "drop_same_minus_unseen") for g, _ in GENERATORS for d, _ in DETECTORS]
    if not all(is_number(v) for v in drops):
        return PLACEHOLDER
    falls = sum(1 for v in drops if v > 0)
    if falls == len(drops):
        body = f"On a generator it never saw, AUROC drops in all {len(drops)} cases, by {min(drops):.3f} to {max(drops):.3f}"
    else:
        body = f"On a generator it never saw, AUROC drops in {falls} of {len(drops)} cases (change {min(drops):.3f} to {max(drops):.3f})"
    return body + " (calibration data). A finding, not a promise."


def worst_robustness():
    """The lowest AUROC under JPEG / blur / noise. Returns (value, detector, generator, change) or None."""
    worst = None
    robustness = dig(TRAIN, "robustness")
    if not isinstance(robustness, dict):
        return None
    for change, by_detector in robustness.items():
        for detector, by_generator in (by_detector or {}).items():
            for generator, scores in (by_generator or {}).items():
                value = dig(scores, "auroc")
                if is_number(value) and (worst is None or value < worst[0]):
                    worst = (value, detector, generator, change)
    return worst


def robustness_line():
    w = worst_robustness()
    if w is None:
        return f"Robustness to JPEG, blur and noise: {PLACEHOLDER}"
    value, detector, generator, change = w
    return (f"Robustness (calibration data): the lowest AUROC under JPEG, blur or noise is {value:.3f} "
            f"({DET_SHORT.get(detector, detector)}, {GEN_SHORT.get(generator, generator)}, {PERT_NAMES.get(change, change)}).")


def time_limited_epochs():
    """The epoch counts of the runs whose JSON says stopped_by_time_limit (for example 22, 28 and 34)."""
    return sorted(r["epochs_run"] for r in RUN_RESULT.values()
                  if isinstance(r, dict) and r.get("stopped_by_time_limit") is True and is_number(r.get("epochs_run")))


def join_and(items):
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# =====================================================================
# PART 3 - THE LOOK (the helpers and colours come from the existing deck script)
# =====================================================================

sys.path.insert(0, str(DOCS))
# build_docs.py has an  if __name__ == "__main__"  guard, so importing it does NOT rebuild the old files.
from build_docs import (AMBER, AMBER_LIGHT, AMBER_TEXT, BODY_FONT, BORDER, CARD, CW, GREY, MX, NAVY, SLIDE_H,
                        SLIDE_W, TEAL, TEAL_LIGHT, TEAL_TEXT, TITLE_FONT, WHITE,
                        P, R, add_text, arrow, box, card, cell_borders, label, rgb, write)
from pptx import Presentation
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# Two more accents for this deck: indigo = Generator, rose = Compare (teal = Detector, amber = Auditor).
INDIGO, INDIGO_TEXT, INDIGO_LIGHT = "4F46E5", "4338CA", "EEF2FF"
ROSE, ROSE_TEXT, ROSE_LIGHT = "E11D48", "BE123C", "FFF1F2"
FOOTER = "GenAI Trust Lab  |  Atria University  |  26 Nov 2026"
TOTAL = 7


def new_slide(prs, number, title, notes_text, title_size=34):
    """A slide with the shared look: title, four-colour accent line, footer with 'n / 7', speaker notes."""
    slide = prs.slides.add_slide(prs.slide_layouts[5])               # layout 5 = "Title Only"
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(MX), Inches(0.4), Inches(CW), Inches(0.8)
    write(t.text_frame, [P(R(title, title_size, NAVY, font=TITLE_FONT))], MSO_ANCHOR.MIDDLE)
    x = MX
    for colour in (INDIGO, TEAL, ROSE, AMBER):
        box(slide, x, 1.24, 0.5, 0.06, fill=colour, line=None, shape=MSO_SHAPE.RECTANGLE)
        x += 0.5
    add_text(slide, MX, 6.8, 9.0, 0.25, [P(R(FOOTER, 11, GREY))], MSO_ANCHOR.MIDDLE)
    add_text(slide, SLIDE_W - MX - 1.5, 6.8, 1.5, 0.25,
             [P(R(f"{number} / {TOTAL}", 11, GREY), align=PP_ALIGN.RIGHT)], MSO_ANCHOR.MIDDLE)
    slide.notes_slide.notes_text_frame.text = notes_text
    return slide


def source_line(slide, text, y=6.42):
    """The small 'source: ...' line that names the file every number on the slide came from."""
    add_text(slide, MX, y, CW, 0.36, [P(R(text, 11, GREY))], MSO_ANCHOR.MIDDLE)


def table(slide, x, y, col_w, row_h, header, rows, size=11, header_fill=NAVY, align_first=PP_ALIGN.LEFT):
    """A results table. Header row in the accent colour; a placeholder cell is shown in grey italics.
    col_w = widths in inches; row_h = list of heights (header first)."""
    gf = slide.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(sum(col_w)), Inches(sum(row_h)))
    tbl = gf.table
    tbl._tbl.tblPr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}tableStyleId").text = \
        "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"                   # "No Style, No Grid": only our own colours show
    tbl.horz_banding = False
    for i, w in enumerate(col_w):
        tbl.columns[i].width = Inches(w)
    for r_idx, values in enumerate([header] + rows):
        tbl.rows[r_idx].height = Inches(row_h[r_idx])
        for c_idx, text in enumerate(values):
            cell = tbl.cell(r_idx, c_idx)
            cell_borders(cell)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(header_fill if r_idx == 0 else WHITE)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            is_ph = text == PLACEHOLDER
            colour = WHITE if r_idx == 0 else (GREY if is_ph else NAVY)
            bold = r_idx == 0 or c_idx == 0
            align = align_first if c_idx == 0 else PP_ALIGN.CENTER
            write(cell.text_frame, [P(R(text, size, colour, bold=bold, italic=is_ph), align=align)], frame=False)
    return gf


def picture(slide, source, x, y, w=None, h=None):
    """A picture with a thin border. source = a file path or raw PNG bytes."""
    stream = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else str(source)
    pic = slide.shapes.add_picture(stream, Inches(x), Inches(y), width=Inches(w) if w else None,
                                   height=Inches(h) if h else None)
    pic.line.color.rgb = rgb(BORDER)
    pic.line.width = Pt(0.75)
    return pic


def placeholder_box(slide, x, y, w, h, text=PLACEHOLDER):
    """A dashed empty box saying the value has not been measured yet."""
    s = box(slide, x, y, w, h, [P(R(text, 12, GREY, italic=True), align=PP_ALIGN.CENTER)], fill=WHITE, line=GREY, radius=0.1)
    s.line.dash_style = MSO_LINE.DASH
    return s


def asset(name):
    return ASSETS / name


def bullets(items, size=14, colour=NAVY, after=5):
    return [P(R("\u2022\t", size, colour, bold=True), R(t, size, colour), after=after) for t in items]


def hang(shape, inches=0.28):
    """Hanging indent: a wrapped bullet line lines up under the text, not under the bullet dot."""
    for para in shape.text_frame.paragraphs:
        pPr = para._p.get_or_add_pPr()
        pPr.set("marL", str(int(Inches(inches))))
        pPr.set("indent", str(-int(Inches(inches))))
    return shape


# =====================================================================
# PART 4 - THE SEVEN SLIDES
# =====================================================================

# ---- Slide 1: title
def slide_1(prs):
    notes = ("Hello, we are Nitin and Tejash from Atria University. This is our project for the course Generative AI and "
             "Diffusion Models. We built four small tools that work with AI-made pictures. "
             "The Generator makes small pictures with two kinds of models, a VAE and a diffusion model. "
             "A VAE squeezes a picture into a few numbers and then builds it back. "
             "A diffusion model starts from random noise and removes the noise step by step until a picture appears. "
             "The Detector says whether a picture looks real or AI-made, or says unclear. "
             "Compare shows how two pictures differ, and the Auditor looks for training pictures that a generator may have copied. "
             "All our models are small and work on 32 by 32 pixel pictures, so the limits matter. "
             "We show the limits honestly on the last slide.")
    s = new_slide(prs, 1, "GenAI Trust Lab", notes, title_size=44)
    add_text(s, MX, 1.45, CW, 0.9, [P(R("Four small tools that ask two questions: is this picture AI-made, "
                                        "and did a generator copy its training pictures?", 20, NAVY))], MSO_ANCHOR.MIDDLE)
    chips = ["Nitin and Tejash", "Atria University", "Generative AI and Diffusion Models", "26 Nov 2026"]
    widths = [2.4, 2.4, 4.4, 2.2]
    gap = (CW - sum(widths)) / 3
    x = MX
    for text, w in zip(chips, widths):
        box(s, x, 2.5, w, 0.5, [P(R(text, 14, NAVY), align=PP_ALIGN.CENTER)], fill=WHITE, line=GREY, line_pt=1.0, radius=0.5)
        x += w + gap
    tools = [("TOOL 1", "Generator", "Makes small 32x32 pictures with a VAE and a diffusion model.", INDIGO, INDIGO_TEXT, INDIGO_LIGHT),
             ("TOOL 2", "Detector", "Says real, AI or unclear, and shows where it looked.", TEAL, TEAL_TEXT, TEAL_LIGHT),
             ("TOOL 3", "Compare", "Shows how two pictures differ, with plain image maths.", ROSE, ROSE_TEXT, ROSE_LIGHT),
             ("TOOL 4", "Auditor", "Looks for training pictures a generator may have copied.", AMBER, AMBER_TEXT, AMBER_LIGHT)]
    cg = 0.25
    cw = (CW - 3 * cg) / 4
    for i, (tag, name, line, accent, text_col, light) in enumerate(tools):
        card(s, MX + i * (cw + cg), 3.3, cw, 2.2, accent, fill=light, paras=[
            P(R(tag, 12, text_col, bold=True), after=4),
            P(R(name, 24, NAVY, font=TITLE_FONT), after=6),
            P(R(line, 14, GREY)),
        ], anchor=MSO_ANCHOR.TOP, margins=(0.3, 0.3, 0.12, 0.1))
    card(s, MX, 5.75, CW, 0.75, NAVY, paras=[
        P(R("Limits: ", 14, NAVY, bold=True),
          R("small 32x32 models, two generators only, and results are findings, not promises (slide 7).", 14, NAVY)),
    ], anchor=MSO_ANCHOR.MIDDLE)


# ---- Slide 2: the problem and the idea
def slide_2(prs):
    notes = ("There are two trust problems with AI pictures. "
             "First, can we tell whether a picture is real or AI-made? "
             "Second, did a generator copy pictures it was trained on? "
             "Carlini and colleagues showed that diffusion models can memorize training images and give them back. "
             "Memorize means the model stores a training picture almost exactly. "
             "Ojha and colleagues, and Corvi and colleagues, found that detectors trained on one kind of generator did not generalize well to other kinds. "
             "Generalize means to keep working on new data the model has not seen. "
             "Our idea is to train small generators ourselves, so we know exactly what they saw in training. "
             "Then we build a Detector and an Auditor and test them on these known models. "
             "The Detector uses a reconstruction error idea from the DIRE paper by Wang and colleagues.")
    s = new_slide(prs, 2, "Two trust problems, one idea", notes)
    gap = 0.35
    cw = (CW - gap) / 2
    problems = [
        ("PROBLEM 1", "Is this image real or AI-made?",
         "Detectors trained on one kind of generator did not generalize well to other kinds.",
         "Ojha et al., CVPR 2023 (arXiv 2302.10174); Corvi et al., ICASSP 2023 (arXiv 2211.00680, author list to verify)",
         TEAL, TEAL_TEXT, TEAL_LIGHT),
        ("PROBLEM 2", "Did a generator copy its training images?",
         "Diffusion models can memorize training images and generate them again.",
         "Carlini et al., Extracting Training Data from Diffusion Models, USENIX Security 2023 (arXiv 2301.13188)",
         AMBER, AMBER_TEXT, AMBER_LIGHT)]
    for i, (tag, head, why, cite, accent, text_col, light) in enumerate(problems):
        card(s, MX + i * (cw + gap), 1.6, cw, 3.1, accent, fill=light, paras=[
            P(R(tag, 12, text_col, bold=True), after=6),
            P(R(head, 24, NAVY, font=TITLE_FONT), after=10),
            P(R("Why it matters: ", 14, NAVY, bold=True), R(why, 14, NAVY), after=8),
            P(R(cite, 12, GREY)),
        ], anchor=MSO_ANCHOR.MIDDLE)
    card(s, MX, 5.0, CW, 1.3, NAVY, paras=[
        P(R("OUR IDEA", 12, GREY, bold=True), after=4),
        P(R("Train small generators ourselves, so we know their training images, then build a Detector and an Auditor "
            "and test them on these known models.", 16, NAVY)),
    ], anchor=MSO_ANCHOR.MIDDLE)
    source_line(s, "The Detector's reconstruction-error idea follows DIRE (Wang et al., arXiv 2303.09295; ICCV 2023 venue to verify).")


# ---- Slide 3: Generator
def models_table_rows():
    rows = []
    for run in RUNS:
        r = RUN_RESULT.get(run) or {}
        dataset = DATASET_NAMES.get(dig(RUN_CONFIG, run, "dataset"), PLACEHOLDER)
        epochs = r.get("epochs_run")
        epochs_text = PLACEHOLDER if not is_number(epochs) else (f"{epochs} (time limit)" if r.get("stopped_by_time_limit") is True else f"{epochs}")
        params = r.get("parameter_count")
        rows.append([run, dataset, f"{params:,}" if is_number(params) else PLACEHOLDER, num(r.get("weights_size_mb"), 2),
                     epochs_text, num(r.get("seconds_per_epoch"), 2), text_or_ph(r.get("device"))])
    return rows


def ddim_table_rows():
    seconds = dig(TIMING, "seconds_by_steps") or {}
    steps = sorted({int(k) for who in ("gpu", "cpu") for k in (seconds.get(who) or {})}) or [20, 50, 100]
    return [[str(n), num(dig(seconds, "gpu", str(n))), num(dig(seconds, "cpu", str(n)))] for n in steps]


def slide_3(prs):
    limited = time_limited_epochs()
    n_pics = dig(TIMING, "images_per_call")
    gpu20, cpu20 = dig(TIMING, "seconds_by_steps", "gpu", "20"), dig(TIMING, "seconds_by_steps", "cpu", "20")
    guidance = dig(RUN_CONFIG, "g-cddpm-fmnist", "sample_guidance")
    notes = ("Here are pictures from our own small models, so expect low quality. "
             "The left sheet is from the Fashion-MNIST model. Fashion-MNIST is a set of small clothing pictures from Zalando, and each row is one clothing class. "
             "The other two sheets use CIFAR-10, a set of small photos in ten classes, one from a diffusion model and one from a VAE. "
             "DDIM is a faster way to sample from a diffusion model, with fewer steps. "
             f"On our laptop graphics card, the GPU, {text_or_ph(n_pics)} pictures in 20 steps took {num(gpu20)} seconds, and on the CPU they took {num(cpu20)} seconds. "
             "Classifier-free guidance is a trick that makes the model follow the class label more strongly. "
             f"Our guidance scale is {num(guidance, 1)}, which is a proposed setting. "
             f"The diffusion runs were stopped by the time limit after {join_and(limited) if limited else PLACEHOLDER} epochs. "
             "An epoch is one pass over the training pictures.")
    s = new_slide(prs, 3, "Generator: small models on two datasets", notes)

    # Left: the three real sample sheets (our own outputs).
    label(s, MX, 1.45, 6.2, "OUR OWN SAMPLES - small models, expect low quality", INDIGO_TEXT)
    sheets = [("samples_g-cddpm-fmnist.png", "g-cddpm-fmnist\nFashion-MNIST"),
              ("samples_g-ddpm-cifar.png", "g-ddpm-cifar\nCIFAR-10, diffusion"),
              ("samples_g-vae-cifar.png", "g-vae-cifar\nCIFAR-10, VAE")]
    x, h = MX, 1.8
    for name, caption in sheets:
        path = asset(name)
        if path.exists():
            pic = picture(s, path, x, 1.8, h=h)
            w = pic.width / 914400
        else:
            w = h
            placeholder_box(s, x, 1.8, w, h)
        add_text(s, x, 3.63, w + 0.4, 0.42, [P(R(line, 12, GREY, bold=(j == 0))) for j, line in enumerate(caption.split("\n"))])
        x += w + 0.45

    # Right: DDIM speed table + two short facts.
    label(s, 6.95, 1.45, 5.8, f"SAMPLING SPEED - DDIM, {text_or_ph(n_pics)} pictures, seconds", INDIGO_TEXT)
    table(s, 6.95, 1.85, [0.8, 1.05, 1.05], [0.3] * 4, ["Steps", "GPU", "CPU"], ddim_table_rows(), size=12, header_fill=INDIGO,
          align_first=PP_ALIGN.CENTER)
    s_guid = f"Fashion-MNIST model uses classifier-free guidance, scale {num(guidance, 1)} (proposed)."
    s_time = (f"Stopped by the time limit after {join_and(limited)} epochs." if limited else "No run was stopped by the time limit.")
    hang(add_text(s, 10.15, 1.8, 2.58, 2.3, bullets([s_guid, s_time], 14, NAVY, after=8)))

    # Bottom: the table of trained models.
    label(s, MX, 4.1, 8.0, "TRAINED MODELS (g- = Generator, d- = the Detector's own models)", INDIGO_TEXT)
    header = ["Run", "Dataset", "Parameters", "Weights (MB)", "Epochs run", "Seconds / epoch", "Device"]
    widths = [1.75, 1.5, 1.3, 1.3, 1.6, 1.4, 3.28]
    table(s, MX, 4.42, widths, [0.27] * (len(RUNS) + 1), header, models_table_rows(), size=11, header_fill=INDIGO)
    source_line(s, "source: results/train_*.json, results/ddim_timing.json, configs/*.json (dataset names). "
                   "CIFAR-10: Krizhevsky 2009. Fashion-MNIST: Zalando, MIT. DDPM: Ho et al., arXiv 2006.11239. "
                   "DDIM: Song et al., ICLR 2021, arXiv 2010.02502. Guidance: Ho and Salimans, arXiv 2207.12598.", y=6.42)


# ---- Slide 4: Detector
def slide_4(prs):
    ex = read_json(asset("detect_examples.json")) or {}
    n_features = dig(TRAIN, "feature_columns")
    feat_text = f"{n_features} features" if is_number(n_features) else "features"

    def verdict(kind):
        """Right / wrong / no decision, by a simple rule on the label the API gave."""
        label_ = dig(ex, kind, "label")
        if label_ is None:
            return PLACEHOLDER, ""
        right = "real" if kind == "real" else "ai"
        return label_, ("(correct)" if label_ == right else "(no decision)" if label_ == "unclear" else "(wrong)")

    real_label, real_verdict = verdict("real")
    fake_label, fake_verdict = verdict("fake")
    notes = ("The Detector looks at one 32 by 32 picture and answers real, AI, or unclear. "
             "A VAE and a diffusion model each try to rebuild the picture. A VAE squeezes a picture into a few numbers and builds it back. "
             "The idea is that the rebuild error differs for real and AI-made pictures, which follows the DIRE paper by Wang and colleagues. "
             f"From the errors we make {n_features if is_number(n_features) else 'a few'} numbers, called features. "
             "A logistic regression, a very simple model that turns numbers into a probability, reads these features. "
             "As a baseline we also train a plain CNN, a standard image network. "
             "The heatmap shows where the rebuild differs most, and Grad-CAM, from Selvaraju and colleagues, shows where the CNN looked. "
             f"These are single calibration examples, not a measurement: the real photo got the answer {real_label} and the fake got {fake_label}. "
             "So the Detector is sometimes wrong, and slide 5 gives the real numbers.")
    s = new_slide(prs, 4, "Detector: how it works, where it looks", notes)

    # Left: the pipeline.
    label(s, MX, 1.45, 4.3, "PIPELINE (proposed design)", TEAL_TEXT)
    b1 = box(s, MX, 1.85, 4.3, 0.6, [P(R("32x32 image", 14, NAVY, bold=True), align=PP_ALIGN.CENTER)], fill=WHITE, line=TEAL, line_pt=1.5, radius=0.15)
    b2 = box(s, MX, 2.75, 4.3, 1.2, [P(R("VAE + diffusion reconstruction error", 14, NAVY, bold=True), align=PP_ALIGN.CENTER, after=2),
                                     P(R(f"{feat_text} per image (idea from DIRE)", 14, GREY), align=PP_ALIGN.CENTER)],
             fill=WHITE, line=TEAL, line_pt=1.5, radius=0.1)
    b3 = box(s, MX, 4.25, 4.3, 0.8, [P(R("Logistic regression", 14, NAVY, bold=True), align=PP_ALIGN.CENTER, after=2),
                                     P(R("real / AI / unclear", 14, GREY), align=PP_ALIGN.CENTER)],
             fill=WHITE, line=TEAL, line_pt=1.5, radius=0.15)
    arrow(s, b1, 2, b2, 0)
    arrow(s, b2, 2, b3, 0)
    card(s, MX, 5.3, 4.3, 1.0, NAVY, paras=[
        P(R("Baseline: ", 14, NAVY, bold=True), R("a plain CNN on the same pictures. Grad-CAM shows where it looked.", 14, NAVY)),
    ], anchor=MSO_ANCHOR.MIDDLE, margins=(0.3, 0.05, 0.1, 0.05))

    # Right: 2 rows x 3 pictures (input, heatmap, Grad-CAM) from the real API.
    gx, tile, tgap = 5.25, 1.75, 0.15
    for j, head in enumerate(["Input picture", "Heatmap", "Grad-CAM (CNN)"]):
        add_text(s, gx + j * (tile + tgap), 1.45, tile, 0.3, [P(R(head, 12, GREY, bold=True))], MSO_ANCHOR.MIDDLE)
    rows = [("real", 1.85, "Real photo", "CIFAR-10 test image (Krizhevsky 2009)", real_label, real_verdict),
            ("fake", 3.95, "Diffusion fake", "made by our g-ddpm-cifar model", fake_label, fake_verdict)]
    for kind, y, title, cap, lab, ver in rows:
        for j, name in enumerate([f"{kind}_input.png", f"{kind}_heatmap.png", f"{kind}_gradcam.png"]):
            if asset(name).exists():
                picture(s, asset(name), gx + j * (tile + tgap), y, w=tile, h=tile)
            else:
                placeholder_box(s, gx + j * (tile + tgap), y, tile, tile)
        tx = gx + 3 * (tile + tgap) + 0.05
        add_text(s, tx, y, SLIDE_W - MX - tx, tile, [
            P(R(title, 14, NAVY, bold=True), after=2),
            P(R(cap, 12, GREY), after=6),
            P(R(f"Detector says: {lab}", 14, NAVY), after=1),
            P(R(ver, 12, GREY), after=3),
            P(R(f"score {num(dig(ex, kind, 'score'))}, CNN {num(dig(ex, kind, 'baseline_score'))}", 12, GREY)),
        ])
    add_text(s, gx, 5.78, SLIDE_W - MX - gx, 0.55, [P(R(
        "Heatmap: where the diffusion rebuild differs most (red = more). Grad-CAM: where the CNN looked. "
        "Neither proves why a picture is real or AI-made.", 12, GREY))])
    source_line(s, "source: docs/deck_assets/detect_examples.json (real /detect answers, one example each, not a measurement); "
                   "results/detector_final_train.json (feature count). DIRE: Wang et al., arXiv 2303.09295. "
                   "Grad-CAM: Selvaraju et al., ICCV 2017, arXiv 1610.02391.")


# ---- Slide 5: results and findings
def slide_5(prs):
    caveat = ("the Detector's diffusion model and the fake-making diffusion model share an architecture, so the diffusion result "
              "may look better than on a truly unseen generator")
    n_real = dig(FINAL, "images", "real")
    n_fake = dig(FINAL, "images", "fake_per_generator", "g-vae-cifar")
    t1, t2, t3 = takeaway_detectors(), takeaway_vae_easy(), takeaway_unseen()
    cnn_vae, cnn_dd = final_auroc("cnn", "g-vae-cifar"), final_auroc("cnn", "g-ddpm-cifar")
    notes = ("This is the key slide, and every number comes from a result file. "
             "AUROC is a score from 0.5 to 1: 0.5 means guessing and 1.0 means a perfect split of real from fake. "
             f"The left table is the final test split, scored only once, with {text_or_ph(n_real)} real pictures and {text_or_ph(n_fake)} fakes per generator. "
             f"{t1} {t2} "
             "The right table uses calibration data: we train the detector without one generator and test it on that generator, which it has never seen. "
             f"{t3} "
             f"One caution: {caveat}. "
             f"{robustness_line()}")
    s = new_slide(prs, 5, "Results: what we measured and found", notes)

    # Left table: final test split (scored once).
    label(s, MX, 1.45, 5.2, "FINAL TEST SPLIT, scored once - AUROC", TEAL_TEXT)
    rows = [[name] + [num(final_auroc(d, g), 4) for g, _ in GENERATORS] for d, name in DETECTORS]
    table(s, MX, 1.8, [2.5, 1.35, 1.35], [0.45] + [0.52] * 3, ["Detector"] + [g for _, g in GENERATORS], rows,
          size=12, header_fill=TEAL)
    add_text(s, MX, 3.9, 5.2, 0.55, [P(R(f"AUROC: 1.0 = perfect, 0.5 = guessing. Test set: {text_or_ph(n_real)} real and {text_or_ph(n_fake)} fake pictures per generator.", 14, GREY))])

    # Right table: leave-one-generator-out (calibration data).
    rx = MX + 5.2 + 0.25
    label(s, rx, 1.45, 6.7, "UNSEEN GENERATOR (calibration data, not final) - AUROC", AMBER_TEXT)
    short = {"lr_vae": "LR, VAE", "lr_vae_ddpm": "LR, VAE + diff.", "cnn": "CNN baseline"}
    rows = []
    for g, gname in GENERATORS:
        for d, _ in DETECTORS:
            rows.append([f"{gname} | {short[d]}",
                         num(loo(g, d, "same_generator"), 4), num(loo(g, d, "unseen_generator"), 4), num(loo(g, d, "drop_same_minus_unseen"), 4)])
    table(s, rx, 1.8, [3.1, 1.2, 1.2, 1.0], [0.45] + [0.3] * 6, ["Generator left out | detector", "Seen", "Unseen", "Drop"], rows,
          size=11, header_fill=AMBER)
    add_text(s, rx, 4.08, 6.7, 0.55, [P(R("Seen = trained with it; Unseen = left out. LR = logistic regression, diff. = diffusion.", 12, GREY))])

    # Two takeaways, written by rules from the numbers.
    gap = 0.3
    cw = (CW - gap) / 2
    card(s, MX, 4.68, cw, 1.45, TEAL, fill=TEAL_LIGHT, paras=[
        P(R("FINDING 1 (final test split)", 12, TEAL_TEXT, bold=True), after=3),
        P(R(t1 + " " + t2, 14, NAVY))], anchor=MSO_ANCHOR.MIDDLE, margins=(0.3, 0.05, 0.12, 0.05))
    card(s, MX + cw + gap, 4.68, cw, 1.45, AMBER, fill=AMBER_LIGHT, paras=[
        P(R("FINDING 2 (calibration data)", 12, AMBER_TEXT, bold=True), after=3),
        P(R(t3, 14, NAVY))], anchor=MSO_ANCHOR.MIDDLE, margins=(0.3, 0.05, 0.12, 0.05))
    add_text(s, MX, 6.15, CW, 0.28, [P(R(robustness_line(), 12, GREY))], MSO_ANCHOR.MIDDLE)
    source_line(s, "source: results/detector_final_final.json (final test split, scored once); results/detector_final_train.json "
                   "(calibration data, robustness)", y=6.47)


# ---- Slide 6: Compare and Auditor
def slide_6(prs):
    cmp_ = read_json(asset("compare_example.json")) or {}
    ssim = dig(cmp_, "ssim")
    colours = dig(cmp_, "dominant_colours") or [None, None]
    size_rows = audit_points(0, AUDIT_SIZE_DEFAULTS)
    dup_rows = audit_points(1, AUDIT_DUP_DEFAULTS)
    pair = dig(AUDIT, "examples", 0)
    notes = ("The left half is Compare: we drew two cat-like pictures, one orange and one black, only to test the tool. "
             "They are an illustration made for testing, not a result. "
             "Compare shows a difference map, where white means the same and red means different. "
             f"SSIM is a number for how similar the layout of two pictures is, where 1 means identical, and here it is {num(ssim)}. "
             "It also names the main colours of each picture. "
             "The Words panel, which would use a vision API to describe the differences, is off. "
             "The right half is the Auditor, which measures similarity between generated pictures and the training pictures. "
             "The memorization rate is the share of generated pictures that are very close to a training picture, which we call a possible copy. "
             + ("The numbers in the tables come from our Auditor runs. " if AUDIT_READY else
                "The Auditor runs have not finished, so the cells say placeholder instead of a number. ") +
             "The Auditor does not decide legal copying, and the upload check is only a stub for now.")
    s = new_slide(prs, 6, "Compare and Auditor", notes)

    # ---- Left half: Compare
    lw = 5.85
    card(s, MX, 1.5, lw, 4.8, ROSE, fill=ROSE_LIGHT)
    label(s, MX + 0.3, 1.55, 3.0, "COMPARE (real /compare answer)", ROSE_TEXT)
    ix, tile, tg = MX + 0.3, 1.65, 0.2
    for j, (name, cap) in enumerate([("cat_orange.png", "Image A"), ("cat_black.png", "Image B"), ("compare_diff.png", "Difference map")]):
        x = ix + j * (tile + tg)
        if asset(name).exists():
            picture(s, asset(name), x, 1.95, w=tile, h=tile)
        else:
            placeholder_box(s, x, 1.95, tile, tile)
        add_text(s, x, 3.63, tile, 0.3, [P(R(cap, 12, NAVY, bold=True))])
    add_text(s, ix, 4.15, lw - 0.5, 1.2, [
        P(R(f"SSIM {num(ssim)}", 14, NAVY, bold=True), R("  (1 = identical layout)", 12, GREY), after=4),
        P(R("Main colours A: ", 14, NAVY, bold=True), R(text_or_ph(colours[0]), 14, NAVY), after=2),
        P(R("Main colours B: ", 14, NAVY, bold=True), R(text_or_ph(colours[1]), 14, NAVY)),
    ])
    add_text(s, ix, 5.05, lw - 0.5, 1.2, [
        P(R("Difference map: white = same, red = different.", 14, GREY, italic=True), after=2),
        P(R("Illustration made for testing, not a result.", 14, GREY, italic=True), after=2),
        P(R("The Words panel (vision API) is OFF: no written description.", 14, GREY, italic=True))])

    # ---- Right half: Auditor
    rx = MX + lw + 0.3
    rw = SLIDE_W - MX - rx
    card(s, rx, 1.5, rw, 4.8, AMBER, fill=AMBER_LIGHT)
    label(s, rx + 0.3, 1.55, 5.0, "AUDITOR (rate = share flagged as possible copy)", AMBER_TEXT)
    tw = (rw - 0.3 - 0.15 - 0.15) / 2
    cols = [1.15, tw - 1.15]
    add_text(s, rx + 0.3, 1.9, tw, 0.28, [P(R("Training-set size (proposed)", 11, GREY))], MSO_ANCHOR.MIDDLE)
    table(s, rx + 0.3, 2.2, cols, [0.5, 0.5, 0.5], ["Training images", "Memorization rate"],
          [[f"{x:,.0f}", pct(rate)] for x, rate in size_rows], size=11, header_fill=AMBER)
    x2 = rx + 0.3 + tw + 0.15
    add_text(s, x2, 1.9, tw, 0.28, [P(R("Duplicate share (proposed)", 11, GREY))], MSO_ANCHOR.MIDDLE)
    table(s, x2, 2.2, cols, [0.5, 0.5, 0.5], ["Duplicates", "Memorization rate"],
          [[f"{x:,.0f}%", pct(rate)] for x, rate in dup_rows], size=11, header_fill=AMBER)

    add_text(s, rx + 0.3, 3.85, rw - 0.5, 0.28, [P(R("Closest pair: generated | nearest training image", 11, GREY))], MSO_ANCHOR.MIDDLE)
    g64, n64 = dig(pair, "generated_png_base64"), dig(pair, "nearest_png_base64")
    if isinstance(g64, str) and isinstance(n64, str):
        try:
            picture(s, base64.b64decode(g64), rx + 0.3, 4.15, w=1.05, h=1.05)
            picture(s, base64.b64decode(n64), rx + 1.5, 4.15, w=1.05, h=1.05)
            add_text(s, rx + 2.8, 4.15, rw - 3.0, 1.05, [
                P(R(f"distance {num(dig(pair, 'distance'), 2)}", 14, NAVY, bold=True), after=2),
                P(R("Closest pair found. Very similar pairs are possible copies, not proof.", 12, GREY))])
        except Exception:
            placeholder_box(s, rx + 0.3, 4.15, rw - 0.6, 1.05)
    else:
        placeholder_box(s, rx + 0.3, 4.15, rw - 0.6, 1.05)
    add_text(s, rx + 0.3, 5.3, rw - 0.5, 1.0, [
        P(R("The Auditor measures similarity between images and says \"similar\" or \"possible copy\". It does not decide legal copying. "
            "The upload check (/audit/check) is a stub.", 14, NAVY))])
    source_line(s, "source: docs/deck_assets/compare_example.json (real /compare answer, made-up pair); results/audit_results.json (Auditor)", y=6.42)


# ---- Slide 7: limits, demo, next steps
def slide_7(prs):
    notes = ("This last slide is about honesty. "
             "Our models are small and work on 32 by 32 pictures, so the picture quality is low. "
             "The Detector was trained on two generators only, and on a new generator its score drops, which we report as a finding, not a promise. "
             "Our diffusion fakes come from the same kind of model the Detector uses to rebuild pictures, so those results may look better than on a truly different generator. "
             "The Words panel is off, and the Auditor's upload check is only a stub. "
             "Tejash is building the website, and it is not connected to our tools yet. "
             "The demo runs from our laptop, and we plan a recorded backup in case something fails. "
             "It has five steps, from making a picture to showing the Auditor results. "
             "Next we want more generators, to test the drop on unseen generators. "
             "The Broad track, a wider detector, stays documented future work.")
    s = new_slide(prs, 7, "Limits, demo and what is next", notes)
    w1, w2 = 4.55, 4.0
    gap = 0.3
    w3 = CW - w1 - w2 - 2 * gap
    x1, x2, x3 = MX, MX + w1 + gap, MX + w1 + w2 + 2 * gap
    for x, w, text, colour in [(x1, w1, "HONEST LIMITS", AMBER_TEXT), (x2, w2, "DEMO IN 5 STEPS", NAVY), (x3, w3, "NEXT", TEAL_TEXT)]:
        label(s, x, 1.45, w, text, colour)
    limits = ["Small 32x32 models: low picture quality",
              "Detector trained on two generators only",
              "Drop on an unseen generator is a finding, not a promise",
              "Fake-maker and the Detector's diffusion model share an architecture",
              "Words panel is off; /audit/check is a stub",
              "Tejash's website is not connected yet",
              "Demo from the laptop; a recorded backup is planned"]
    hang(card(s, x1, 1.8, w1, 4.5, AMBER, fill=AMBER_LIGHT, paras=bullets(limits, 14, NAVY, after=6),
              anchor=MSO_ANCHOR.MIDDLE, margins=(0.3, 0.1, 0.12, 0.1)))
    steps = ["Generator: make pictures", "Detector: a real and a fake picture", "Show the heatmap and Grad-CAM",
             "Compare: two pictures, difference map", "Auditor: show the results"]
    for i, text in enumerate(steps):
        y = 1.8 + i * 0.92
        box(s, x2, y, w2, 0.8, [P(R(text, 14, NAVY))], fill=CARD, line=BORDER, radius=0.2, margins=(0.62, 0.03, 0.1, 0.03))
        box(s, x2 + 0.12, y + 0.18, 0.44, 0.44, [P(R(str(i + 1), 14, WHITE, bold=True), align=PP_ALIGN.CENTER)],
            fill=NAVY, line=None, shape=MSO_SHAPE.OVAL, margins=(0, 0, 0, 0))
    nxt = ["More generators, to test the unseen-generator drop",
           "Broad track (a wider detector): documented future work",
           "Connect the website to the API (Tejash)",
           "More Auditor sizes and duplicate shares" if AUDIT_READY else "Finish the Auditor runs and fill slide 6"]
    hang(card(s, x3, 1.8, w3, 4.5, TEAL, fill=TEAL_LIGHT, paras=bullets(nxt, 14, NAVY, after=8),
              anchor=MSO_ANCHOR.MIDDLE, margins=(0.3, 0.1, 0.12, 0.1)))
    source_line(s, "Findings and limits come from results/detector_final_final.json and results/detector_final_train.json.", y=6.42)


# =====================================================================
# PART 5 - BUILD
# =====================================================================

def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)               # 16:9
    prs.slide_height = Inches(SLIDE_H)
    for build_slide in (slide_1, slide_2, slide_3, slide_4, slide_5, slide_6, slide_7):
        build_slide(prs)
    prs.save(OUT_PPTX)
    print("wrote", os.path.relpath(OUT_PPTX, ROOT))


# =====================================================================
# PART 6 - THE NUMBER AUDIT  (python docs/build_final_deck.py --audit)
# =====================================================================
# This check opens the FINISHED .pptx, reads the numbers printed in the tables on slides 3, 5 and 6, opens the result
# files again by itself, and says PASS or FAIL for each table. It also checks that no placeholder is shown where
# a value exists. It does not use the build code above to format anything, so it is an independent second look.

def audit(pptx_path=OUT_PPTX, audit_data="file"):
    """Returns True if every table passes. audit_data = 'file' (read results/audit_results.json) or a dict (for tests)."""
    def load(name):
        try:
            return json.loads((RESULTS / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    prs = Presentation(pptx_path)
    ok_all = True

    def report(name, problems):
        nonlocal ok_all
        print(("PASS  " if not problems else "FAIL  ") + name)
        for line in problems[:8]:
            print("        ", line)
        ok_all = ok_all and not problems

    def tables_of(slide_no):
        return [sh.table for sh in prs.slides[slide_no - 1].shapes if sh.has_table]

    def body_rows(tbl):
        return [[c.text for c in row.cells] for row in list(tbl.rows)[1:]]

    def close(text, value, tol):
        try:
            return abs(float(text) - value) <= tol
        except (TypeError, ValueError):
            return False

    # ---- slide 3: models table and DDIM table
    t3 = {tbl.cell(0, 0).text: tbl for tbl in tables_of(3)}
    problems = []
    rows = body_rows(t3["Run"])
    files = sorted(p.stem[len("train_"):] for p in RESULTS.glob("train_*.json"))
    if sorted(r[0] for r in rows) != files:
        problems.append(f"rows {sorted(r[0] for r in rows)} are not the train_*.json files {files}")
    for run, dataset, params, mb, epochs, spe, device in rows:
        r = load(f"train_{run}.json") or {}
        if params.replace(",", "") != str(r.get("parameter_count")):
            problems.append(f"{run}: parameters {params} vs {r.get('parameter_count')}")
        if not close(mb, r.get("weights_size_mb", float("nan")), 0.0051):
            problems.append(f"{run}: weights MB {mb} vs {r.get('weights_size_mb')}")
        if epochs.split()[0] != str(r.get("epochs_run")) or (("time limit" in epochs) != bool(r.get("stopped_by_time_limit"))):
            problems.append(f"{run}: epochs '{epochs}' vs {r.get('epochs_run')} / stopped {r.get('stopped_by_time_limit')}")
        if not close(spe, r.get("seconds_per_epoch", float("nan")), 0.0051):
            problems.append(f"{run}: seconds per epoch {spe} vs {r.get('seconds_per_epoch')}")
        if device != r.get("device"):
            problems.append(f"{run}: device '{device}' vs '{r.get('device')}'")
        if PLACEHOLDER in (dataset, params, mb, epochs, spe, device):
            problems.append(f"{run}: a placeholder is shown")
    report("slide 3 table: trained models vs results/train_*.json", problems)

    problems = []
    timing = (load("ddim_timing.json") or {}).get("seconds_by_steps", {})
    for steps, gpu, cpu in body_rows(t3["Steps"]):
        if not close(gpu, timing.get("gpu", {}).get(steps, float("nan")), 0.0006):
            problems.append(f"steps {steps}: GPU {gpu} vs {timing.get('gpu', {}).get(steps)}")
        if not close(cpu, timing.get("cpu", {}).get(steps, float("nan")), 0.0006):
            problems.append(f"steps {steps}: CPU {cpu} vs {timing.get('cpu', {}).get(steps)}")
    report("slide 3 table: DDIM speed vs results/ddim_timing.json", problems)

    # ---- slide 5: the two results tables and the sentences
    final, train = load("detector_final_final.json"), load("detector_final_train.json")
    t5 = {tbl.cell(0, 0).text: tbl for tbl in tables_of(5)}
    det_order = ["lr_vae", "lr_vae_ddpm", "cnn"]
    gen_order = ["g-vae-cifar", "g-ddpm-cifar"]
    problems = []
    for key, (name, vae_cell, dd_cell) in zip(det_order, body_rows(t5["Detector"])):
        for gen, cell in zip(gen_order, (vae_cell, dd_cell)):
            want = final["detectors"][key]["per_generator"][gen]["auroc"]
            if not close(cell, want, 0.00006):
                problems.append(f"{name} / {gen}: {cell} vs {want}")
    report("slide 5 table: final test AUROC vs results/detector_final_final.json", problems)

    problems = []
    rows5 = body_rows(t5["Generator left out | detector"])
    expected = [(g, d) for g in gen_order for d in det_order]
    for (g, d), (label_, seen, unseen, drop) in zip(expected, rows5):
        node = train["leave_one_generator_out"][g][d]
        for text, want, what in ((seen, node["same_generator"]["auroc"], "seen"), (unseen, node["unseen_generator"]["auroc"], "unseen"),
                                 (drop, node["drop_same_minus_unseen"]["auroc"], "drop")):
            if not close(text, want, 0.00006):
                problems.append(f"{label_} {what}: {text} vs {want}")
        if not close(drop, node["same_generator"]["auroc"] - node["unseen_generator"]["auroc"], 0.00021):
            problems.append(f"{label_}: drop {drop} is not seen minus unseen")
    if len(rows5) != len(expected):
        problems.append(f"{len(rows5)} rows, expected {len(expected)}")
    report("slide 5 table: unseen generator vs results/detector_final_train.json", problems)

    # every decimal number in the slide 5 sentences must be a value that exists in the two JSON files
    def all_numbers(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from all_numbers(v)
        elif isinstance(node, list):
            for v in node:
                yield from all_numbers(v)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            yield float(node)
    known = list(all_numbers(final)) + list(all_numbers(train))
    sentences = " ".join(sh.text_frame.text for sh in prs.slides[4].shapes if sh.has_text_frame)
    import re
    problems = []
    for number in re.findall(r"\b\d\.\d{3,4}\b", sentences):
        if not any(abs(float(number) - k) <= 0.00101 for k in known):
            problems.append(f"{number} is not in the result files")
    report("slide 5 sentences: every decimal number exists in the result files", problems)

    # ---- slide 6: Compare numbers and the Auditor tables
    problems = []
    compare = json.loads((ASSETS / "compare_example.json").read_text(encoding="utf-8"))
    text6 = " ".join(sh.text_frame.text for sh in prs.slides[5].shapes if sh.has_text_frame)
    if f"SSIM {compare['ssim']:.3f}" not in text6:
        problems.append(f"SSIM {compare['ssim']} not on the slide")
    for colours in compare["dominant_colours"]:
        if colours not in text6:
            problems.append(f"colours '{colours}' not on the slide")
    report("slide 6: Compare numbers vs docs/deck_assets/compare_example.json", problems)

    data = load("audit_results.json") if audit_data == "file" else audit_data
    problems = []
    t6 = tables_of(6)
    for index, tbl in enumerate(t6[:2]):
        points = ((data or {}).get("experiments") or [{}, {}])[index].get("points", []) if data else []
        for i, (x_text, rate_text) in enumerate(body_rows(tbl)):
            rate = points[i].get("memorization_rate") if i < len(points) else None
            if isinstance(rate, (int, float)):
                if rate_text != f"{rate * 100:.1f}%":
                    problems.append(f"table {index + 1} row {i + 1}: '{rate_text}' vs {rate}")
            elif rate_text != PLACEHOLDER:
                problems.append(f"table {index + 1} row {i + 1}: '{rate_text}' shown but no value in the file")
    pair = ((data or {}).get("examples") or [None])[0] if data else None
    shapes6 = prs.slides[5].shapes
    has_ph_box = any(sh.has_text_frame and sh.text_frame.text == PLACEHOLDER for sh in shapes6)
    if bool(pair) == has_ph_box:
        problems.append(f"closest-pair example {'exists' if pair else 'is missing'} but the placeholder box is {'shown' if has_ph_box else 'not shown'}")
    report("slide 6 tables: Auditor values vs results/audit_results.json (placeholder only where a value is missing)", problems)

    # ---- no placeholder anywhere on slides 3 and 5 (all their values exist)
    stray = [sh.text_frame.text for n in (3, 5) for sh in prs.slides[n - 1].shapes if sh.has_text_frame and PLACEHOLDER in sh.text_frame.text]
    stray += [c.text for n in (3, 5) for tbl in tables_of(n) for row in tbl.rows for c in row.cells if PLACEHOLDER in c.text]
    report("slides 3 and 5: no placeholder shown", [f"placeholder in: {t[:60]}" for t in stray])
    return ok_all


if __name__ == "__main__":
    ensure_assets(force="--refresh-assets" in sys.argv)
    build_deck()
    if "--audit" in sys.argv:
        sys.exit(0 if audit() else 1)
