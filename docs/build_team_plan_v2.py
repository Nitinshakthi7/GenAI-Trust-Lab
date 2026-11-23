# build_team_plan_v2.py
# ---------------------------------------------------------------------------
# Builds ONE Word file (a DRAFT for the team to read and approve):
#     docs/drafts/GenAI_Trust_Lab_Team_Plan_NEW.docx
#
# What the file is: the full rebuild of the team plan for "GenAI Trust Lab"
# (two people: Nitin and Tejash), now for the 3-day window Mon 23 Nov to
# Thu 26 Nov 2026. Each person gives the whole file to their AI assistant as
# the first message. It has the same template and the same look as the
# earlier team plan (docs/build_team_plan.py, which this script does NOT change).
#
# How to run (from the project folder):   python docs/build_team_plan_v2.py
#
# How this script is organised:
#   1. SETUP     - fonts, colours and the small helper functions.
#   2. SECTIONS  - one block per section of the document. Each block starts
#                  with the TEXT of the section in named constants (so you can
#                  change the words without touching the code), then the few
#                  lines that write that text into the document.
#   3. SAVE      - the last lines save the file.
#
# Honesty rule for every word in this file: no invented results. Every result
# is the text "[PLACEHOLDER - not measured yet]". Every design number is
# "proposed". Every fact we have not checked is "to verify". The document never
# says that anything is already built, tested or running.
# ---------------------------------------------------------------------------

import json                      # used for the JSON examples in Appendix A
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# ===========================================================================
# 1. SETUP
# ===========================================================================

OUT_FILE = Path(__file__).parent / "drafts" / "GenAI_Trust_Lab_Team_Plan_NEW.docx"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)     # make docs/drafts if it is missing

BODY_FONT = "Arial"              # normal text and headings
MONO_FONT = "Consolas"           # file names, commands, must-do lines
GREEN = RGBColor(0x18, 0x80, 0x38)
BLACK = RGBColor(0x00, 0x00, 0x00)
PLACEHOLDER = "[PLACEHOLDER - not measured yet]"

doc = Document()


def set_fonts(rpr, name):
    """Give a run-properties element one font for all four font slots.
    We also delete the 'theme font' settings, because Word would otherwise
    prefer them and ignore the font we ask for."""
    rfonts = rpr.get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rfonts.get(qn(attr)) is not None:
            del rfonts.attrib[qn(attr)]


# --- page: US Letter, 1 inch margins, no header, no footer ---------------
section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(11)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(section, side, Inches(1))

# --- document defaults and the Normal style: Arial 11, line spacing 1.15 --
defaults = doc.styles.element.find(qn("w:docDefaults"))
set_fonts(defaults.find(".//" + qn("w:rPr")), BODY_FONT)

normal = doc.styles["Normal"]
set_fonts(normal.element.get_or_add_rPr(), BODY_FONT)
normal.font.size = Pt(11)
normal.paragraph_format.line_spacing = 1.15
normal.paragraph_format.space_before = Pt(0)
normal.paragraph_format.space_after = Pt(2)

# --- headings: Arial, black, NOT bold (sizes 20 / 14 / 12) ---------------
HEADING_SIZES = {"Heading 1": 20, "Heading 2": 14, "Heading 3": 12}
HEADING_SPACE = {"Heading 1": (0, 6), "Heading 2": (10, 4), "Heading 3": (6, 3)}
for style_name, size in HEADING_SIZES.items():
    style = doc.styles[style_name]
    set_fonts(style.element.get_or_add_rPr(), BODY_FONT)
    style.font.size = Pt(size)
    style.font.bold = False
    style.font.italic = False
    style.font.color.rgb = BLACK
    style.paragraph_format.space_before = Pt(HEADING_SPACE[style_name][0])
    style.paragraph_format.space_after = Pt(HEADING_SPACE[style_name][1])
    style.paragraph_format.line_spacing = 1.0
    style.paragraph_format.keep_with_next = True

# --- file properties: no personal name -----------------------------------
doc.core_properties.author = "GenAI Trust Lab team"
doc.core_properties.title = "GenAI Trust Lab team plan (3-day version)"
doc.core_properties.last_modified_by = "GenAI Trust Lab team"


# --- small helper functions (copied from the earlier script) ---------------

def mono(par, text, bold=False):
    """Add text to a paragraph as green Consolas (file names, commands)."""
    run = par.add_run(text)
    set_fonts(run._element.get_or_add_rPr(), MONO_FONT)
    run.font.color.rgb = GREEN
    if bold:
        run.bold = True
    return run


def fill(par, text, lead="", bold=False, mono_all=False, ticks=True):
    """Write text into a paragraph.
    lead     : a bold lead-in (for example 'Goal:') written first.
    bold     : make the whole text bold.
    mono_all : make the whole text green Consolas.
    ticks    : pieces of text between `backticks` become green Consolas."""
    if lead:
        run = par.add_run(lead + " ")
        run.bold = True
    pieces = text.split("`") if ticks else [text]
    for i, piece in enumerate(pieces):
        if piece == "":
            continue
        if mono_all or (ticks and i % 2 == 1):
            mono(par, piece, bold)
        else:
            run = par.add_run(piece)
            if bold:
                run.bold = True


def h1(text):
    return doc.add_paragraph(text, style="Heading 1")


def h2(text):
    return doc.add_paragraph(text, style="Heading 2")


def h3(text):
    return doc.add_paragraph(text, style="Heading 3")


def para(text="", lead="", bold=False, mono_all=False, ticks=True, after=None, keep_next=False):
    """One normal paragraph."""
    p = doc.add_paragraph()
    fill(p, text, lead=lead, bold=bold, mono_all=mono_all, ticks=ticks)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    if keep_next:
        p.paragraph_format.keep_with_next = True
    return p


def num_list(items, after=1):
    """A numbered list. The number is typed as text ('1.') in a hanging indent,
    so every list starts again at 1 and Word needs no numbering settings.
    Each item is either a string, a pair (bold lead-in, rest of the text), or a dict
    {"text": ..., "code": [command lines]} whose command lines are written under the step."""
    for n, item in enumerate(items, 1):
        code = []
        if isinstance(item, dict):
            lead, text, code = "", item["text"], item["code"]
        else:
            lead, text = item if isinstance(item, tuple) else ("", item)
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Inches(0.25)
        pf.first_line_indent = Inches(-0.25)
        pf.space_after = Pt(after)
        pf.tab_stops.add_tab_stop(Inches(0.25))
        p.add_run(f"{n}.\t")
        fill(p, text, lead=lead)
        if code:
            pf.keep_with_next = True
            code_block(code, together=len(code) <= 8, indent=0.25)


def code_block(lines, together=True, indent=0.0, size=10):
    """Commands or JSON: one green Consolas paragraph per line.
    Indents are leading spaces. No space between lines.
    size is the font size in points (the JSON examples use 9 so that no line wraps).
    together=True keeps the whole block on one page. For a long block, use
    together=False: a page break may fall inside, but never right after the
    first 2 lines or right before the last 3 lines."""
    for i, line in enumerate(lines):
        last = i == len(lines) - 1
        if together:
            keep = not last
        else:
            keep = i < 2 or i >= len(lines) - 3
            keep = keep and not last
        p = para(line, mono_all=True, ticks=False, after=4 if last else 0, keep_next=keep)
        if indent:
            p.paragraph_format.left_indent = Inches(indent)
        for run in p.runs:
            run.font.size = Pt(size)


def table(headers, rows, widths):
    """A plain black grid ('Table Grid'), bold header row, no shading.
    widths are in inches and should add up to 6.5."""
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.autofit = False
    for cell, text in zip(t.rows[0].cells, headers):
        fill(cell.paragraphs[0], text, bold=True)
        cell.paragraphs[0].paragraph_format.keep_with_next = True   # header never stays alone at a page end
    for row in rows:
        cells = t.add_row().cells
        for cell, text in zip(cells, row):
            fill(cell.paragraphs[0], text)
    for column, w in zip(t.columns, widths):
        column.width = Inches(w)       # the grid widths, for programs other than Word
    for row in t.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))   # keep each row on one page
        for cell, w in zip(row.cells, widths):
            cell.width = Inches(w)
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(2)
                for run in p.runs:
                    run.font.size = Pt(10)
    spacer()
    return t


def spacer():
    """A small empty paragraph (used after tables)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 0.5


def rule():
    """The gray line between task cards: a paragraph with a bottom border."""
    p = doc.add_paragraph()
    p_pr = p._p.get_or_add_pPr()
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for key, value in (("w:val", "single"), ("w:sz", "6"), ("w:space", "1"), ("w:color", "BFBFBF")):
        bottom.set(qn(key), value)
    border.append(bottom)
    p_pr.append(border)               # added before any spacing, so the order is valid
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 0.5
    p.paragraph_format.keep_with_next = True


def day_block(block):
    """One Day block of a card: bold title, Goal / You pull from line, numbered
    steps, Done when, Be ready to explain."""
    para(block["title"], bold=True, keep_next=True)
    para(block["goal"], lead="Goal:", keep_next=True)
    num_list(block["steps"])
    para(block["done"], lead="Done when:")
    para(block["explain"], lead="Be ready to explain:", after=10)


# ===========================================================================
# TITLE, INTRO AND THE BOLD AI PARAGRAPH
# ===========================================================================

TITLE = "GenAI Trust Lab: team plan (3-day version)"

INTRO = (
    "Read this whole file once. Then read your own task card in section 4 again before you start. "
    "Dates: Monday 23 Nov 2026 (evening), Tuesday 24 Nov, Wednesday 25 Nov and Thursday 26 Nov "
    "2026. Thursday is the final presentation, with no new features. There are two of us: Nitin "
    "(leader) and Tejash. All clock times are proposed, except the three gates in section 2."
)

AI_PARAGRAPH = (
    "IF YOU USE AI OF ANY KIND (ChatGPT, Claude, Gemini, Copilot, anything): upload or paste THIS "
    "WHOLE FILE into the AI as your very first message, before you ask it anything. Section 6A "
    "tells the AI what it must do, and it writes its own disclosure for us, so nobody has to "
    "remember or tidy up afterwards. Please read section 6A yourself before you use AI."
)

h1(TITLE)
para(INTRO)
para(AI_PARAGRAPH, bold=True)

# ===========================================================================
# 0. FOUR RULES
# ===========================================================================

RULES_HEADING = "0. Four rules that protect all of us"
RULES = [
    ("Hand-over.",
     "We work in two lanes. On Monday evening, Nitin fixes `.gitignore` first, then you hold a "
     "short first meeting that fixes `api/contract.json`. After that Nitin works in `model/`, "
     "`api/`, `results/` and `configs/`, and Tejash works in `web/`. Every hand-over is a pull "
     "request that the OTHER person reviews and merges (section 5 shows the exact git steps). "
     "The one exception is the very first change, the `.gitignore` fix on Monday: Nitin pushes it "
     "straight to `main`. From Tuesday on, before you start a Day, READ the other lane's last "
     "merged pull request and write 3 lines on what you understood. On Monday nothing is merged "
     "yet: read sections 0 and 6A instead."),
    ("Log everything.",
     "Every day you work, add to your own file `docs/log/<yourname>.md`: what you did, how you did "
     "it, and line-by-line notes of your code with the concept and the reason (section 6). "
     "Teammates read it to catch up. Nitin uses it to see progress."),
    ("Explain every line.",
     "Our team rule says any member can be asked about any part, and that we must be able to "
     "explain every line of our code. If one of us gets it wrong, all of us lose marks. If AI "
     "helped with a line, you still must be able to explain it, and it goes in `docs/ai_usage.md`."),
    ("AI rule.",
     "If you use ANY AI tool, upload this whole file to it FIRST. When you finish, tell it: "
     "\"Write the AI SESSION REPORT now.\" The AI writes its own entries for `docs/ai_usage.md` and "
     "for your log, and you paste them in unedited (section 6A). Nobody has to write or change the "
     "story by hand."),
]

h2(RULES_HEADING)
num_list(RULES)

# ===========================================================================
# 1. WHAT WE MUST DELIVER
# ===========================================================================

DELIVER_HEADING = "1. What we must deliver (from our own PLAN.md)"
DELIVER_MINIMUM = "Minimum:"
DELIVER_ITEMS = [
    "A website with six pages: Home (route /), Generator (/generator), Detector (/detector), "
    "Compare (Distinguisher) (/compare), Auditor (/auditor) and Limits and Methods (/limits). "
    "Tejash's existing site is upgraded into these pages. Its structure and tech are listed on "
    "Monday evening (page list, framework, folder or repo link).",
    "An API (FastAPI) with eight endpoints: the four old ones (GET /health, POST /detect, GET "
    "/audit/results, POST /audit/check) and four new ones (POST /compare, GET /generators, POST "
    "/generate, POST /evidence). Appendix A shows every shape.",
    "Small models: Generator models (g-vae-cifar, g-vae-fmnist, g-cddpm-fmnist, and g-ddpm-cifar "
    "if Gate 2 passes), the Detector (VAE score, CNN baseline, VAE error-map heatmap; the DDPM "
    "error map replaces the heatmap if Gate 2 passes) and the Auditor with three runs (a-500, "
    "a-10000, a-500-dup), each run auditing its own small DDPM.",
    "Honest results from scripts (one results JSON per experiment), the updated deck and a recorded "
    "backup video of the demo. Every number comes from a script, or it stays " + PLACEHOLDER + ".",
]
DELIVER_DEMO = (
    "Demo plan: the live demo will run from a LAPTOP (FastAPI will serve the API and the website "
    "will run at localhost), with a recorded backup. From the official docs: free Hugging Face Docker Spaces "
    "are not available (a paid plan is needed); the Render free web service has 750 h/month and "
    "spins down after 15 min idle (about 1 min to wake; RAM and CPU are not published); Vercel "
    "Hobby is free for a static site (terms not read: to verify). Cloud hosting (Vercel + Render) "
    "is a bonus only if everything else works by Wednesday 20:00. It is first on the cut line."
)
DELIVER_NON_GOALS = (
    "What we do not do: no adversarial attacks and no video in version 1. The roadmap items "
    "(video, a browser extension, a WhatsApp bot, differential privacy) are ideas, not promises. "
    "The Model Zoo extras (SVM, decision tree, ensembles, KNN, random forest and similar) are "
    "parked as future work. The Broad track (many datasets and generators) stays in documents "
    "only. Detection on generators the detector has not seen is an open problem: we report any "
    "drop in performance as a finding and we do not promise the detector will work on them. The "
    "Detector is trained for 32x32 CIFAR-10-style images only, so arbitrary photos are out of "
    "scope for it. Known limitation: the out-of-scope rule only catches image SIZE; a 32x32 "
    "photo that is not CIFAR-style is not flagged. The Auditor measures similarity between "
    "images, not legal copying."
)
DELIVER_CUT_LINE = (
    "Cut line, if time runs out (drop in this order, first to last): 1 cloud hosting; 2 the COULD "
    "extras (DINOv2, LPIPS, ELA, FFT, C2PA, robustness sliders, the watermark idea); 3 POST "
    "/audit/check; 4 pHash and the Auditor side-by-side difference map; 5 Grad-CAM and EXIF; 6 "
    "g-ddpm-cifar and d-ddpm-cifar; 7 the Compare Words panel (the heatmap-only fallback stays); "
    "8 Auditor runs 3 down to 1."
)
DELIVER_ALWAYS_KEEP = (
    "ALWAYS KEEP: `.gitignore` protection, cnn-baseline, the VAE score, /detect with Try an "
    "example and the out-of-scope warning, /generate for g-vae-cifar, g-vae-fmnist and "
    "g-cddpm-fmnist, the Compare SSIM map and feature table, the Auditor dashboard, /limits, the "
    "laptop demo with a recorded backup, and honest placeholders."
)

h2(DELIVER_HEADING)
para(DELIVER_MINIMUM)
for item in DELIVER_ITEMS:
    para(item, bold=True)
para(DELIVER_DEMO)
para(DELIVER_NON_GOALS)
para(DELIVER_CUT_LINE)
para(DELIVER_ALWAYS_KEEP)

# ===========================================================================
# 2. WHO DOES WHAT
# ===========================================================================

WHO_HEADING = "2. Who does what"
WHO_ASSUMPTION = (
    "This split is an ASSUMPTION to confirm with Tejash at the first meeting on Monday evening."
)
WHO_HEADERS = ["Part", "What", "Owner", "Laptop", "Days"]
WHO_ROWS = [
    ["A-ML",
     "Backend (`api/`, FastAPI), every model, Kaggle training, the `model/` scripts, results "
     "JSON files, `api/contract.json`, and on Wednesday the wiring of the website to the real API",
     "Nitin", "RTX 3050 6GB laptop + Kaggle T4 (limits: to verify)", "Mon to Thu"],
    ["B-Web",
     "`web/`: list the existing site, then upgrade it into the six pages against stub JSON; the "
     "Limits page text; the 20 Compare photo pairs and their hand-grading. No backend work.",
     "Tejash", "RTX 3050 6GB laptop", "Mon to Thu"],
    ["Lead",
     "Repo, `.gitignore`, `PLAN.md`, `docs/`, integration, the three gates, teacher contact, "
     "backup video, feature freeze, assembly",
     "Nitin (leader)", "RTX 3050 6GB laptop", "Mon to Thu"],
]
WHO_WIDTHS = [0.7, 2.75, 1.0, 1.35, 0.7]

PROGRESS_SENTENCE = (
    "Progress board. The owner of each item ticks their own box when their pull request is merged "
    "(or, for a gate, when both of you agree it is met)."
)
PROGRESS_LINES = [
    "[ ] Mon: Nitin: `.gitignore` fixed and pushed FIRST",
    "[ ] Mon: Nitin: `api/contract.json` merged",
    "[ ] Mon: Nitin: `data/splits.csv` committed",
    "[ ] Mon: Nitin: g-vae-cifar, d-vae-cifar, cnn-baseline and g-vae-fmnist started",
    "[ ] Mon: Tejash: existing site listed and brought into `web/`; stub pages (six routes) open",
    "[ ] Tue: Nitin: Kaggle jobs started (d-ddpm-cifar, g-ddpm-cifar, g-cddpm-fmnist)",
    "[ ] Tue: Tejash: Generator, Detector and Compare pages against stub JSON; 20 photo pairs",
    "[ ] Gate 1 (Tue 18:00)",
    "[ ] Gate 2 (Wed 12:00)",
    "[ ] Wed: Nitin: results JSON files written by scripts",
    "[ ] Wed: Tejash: Auditor and Limits pages; Compare pairs graded",
    "[ ] Wed: the website wired to the real API",
    "[ ] Gate 3 (Wed 16:00): feature freeze",
    "[ ] Wed 16:00-20:00: two end-to-end runs and the backup video saved",
    "[ ] Thu: deck updated from the results JSON files",
]
GATE_LINES = [
    "Gate 1 (Tue 18:00): the stub pages render, the VAEs have finished training and the Kaggle "
    "DDPM jobs have been started",
    "Gate 2 (Wed 12:00): if there are no usable DDPM samples, cut g-ddpm-cifar and d-ddpm-cifar "
    "(the heatmap stays the VAE error map, which is the default; leave-one-generator-out is "
    "reported as 'not possible'; the Detector uses VAE fakes only, stated as a limitation)",
    "Gate 3 (Wed 16:00): feature freeze, nothing new after this",
]

h2(WHO_HEADING)
para(WHO_ASSUMPTION, bold=True, keep_next=True)
table(WHO_HEADERS, WHO_ROWS, WHO_WIDTHS)
para(PROGRESS_SENTENCE, bold=True)
for line in PROGRESS_LINES:
    para(line, after=2)
for line in GATE_LINES:
    para(line, after=2)

# ===========================================================================
# 3. CALENDAR
# ===========================================================================

CALENDAR_HEADING = "3. Calendar"
CALENDAR_HEADERS = ["Day", "Lane", "What happens"]
CALENDAR_ROWS = [
    ["Mon 23 Nov (18:00 to 23:00, proposed)", "Both, then two lanes",
     "First meeting (about 20 minutes, proposed): agree the team settings in section 4 and the "
     "contract in Appendix A, and confirm the split in section 2. Nitin: `.gitignore` FIRST, "
     "`api/contract.json`, the Kaggle quota, the questions to the teacher (section 9), "
     "`data/splits.csv`, the VAEs and the CNN baseline start. Tejash: list the existing site, bring it into `web/` by a first pull request, "
     "six routes with stub data."],
    ["Tue 24 Nov (09:00 to 18:00, proposed)", "Two lanes",
     "Nitin: Kaggle (d-ddpm-cifar, g-ddpm-cifar, g-cddpm-fmnist), the API (/health, "
     "/generators, /generate, /detect, /compare), the auditor script and run a-500, the "
     "Try-an-example images. Tejash: Generator, Detector and Compare pages against stub JSON, "
     "20 photo pairs of mixed subjects, pull request by 18:00. Gate 1 (18:00)."],
    ["Wed 25 Nov", "Two lanes, then both",
     "09:00 start. Gate 2 (12:00). Gate 3 (16:00): feature freeze. 16:00 to 20:00: two "
     "end-to-end runs and the backup video, on the laptop. 20:00: hosting decision (only if "
     "everything else is done). 22:00: hard stop."],
    ["Thu 26 Nov", "Both",
     "Rehearsal (twice) and the final presentation. No new features. The presentation time is "
     "to confirm with the teacher (section 9)."],
]
CALENDAR_WIDTHS = [1.3, 0.9, 4.3]

CALENDAR_LATE_RULE = (
    "Times for each Day are guesses (the three gate times are fixed, every other clock time is "
    "proposed). If the real API is more than 1 hour late, the website keeps using stub JSON. The "
    "late person still finishes their part and keeps the credit."
)
CALENDAR_SLIP_RULE = (
    "If Tuesday slips: Gate 2 and Gate 3 do not move. Whatever is not ready follows the cut line "
    "in section 1, in that order. The hard stop on Wednesday at 22:00 does not move."
)

h2(CALENDAR_HEADING)
table(CALENDAR_HEADERS, CALENDAR_ROWS, CALENDAR_WIDTHS)
para(CALENDAR_LATE_RULE)
para(CALENDAR_SLIP_RULE)

# ===========================================================================
# 4. TASK CARDS
# ===========================================================================

CARDS_HEADING = "4. Task cards"
CARDS_INTRO = (
    "Each person has ONE card with four Day blocks: Monday evening, Tuesday, Wednesday and "
    "Thursday. Each block has a Goal and 'You pull from' line, numbered steps, 'Done when:' and "
    "'Be ready to explain:' (our guess at the questions the teacher may ask). Files and folders "
    "named in this file (for example `model/`, `api/`, `docs/ai_usage.md`) are the plan: Nitin "
    "creates the `docs/` files and the folders of his lane on Monday evening (`model/`, `api/`, "
    "`results/`, `configs/`); Tejash brings the website into `web/`."
)

# --- team settings --------------------------------------------------------
SETTINGS_HEADING = "Team settings (so the comparison is fair, change only by agreement)"
SETTINGS_INTRO = (
    "Every value below is proposed, agree at the first meeting on Monday evening. Write any "
    "change in both logs."
)
SETTINGS_LINES = [
    "Images: the Detector and the Auditor use 32x32 colour (CIFAR-10); two classes, real and "
    "fake. Fashion-MNIST (28x28 grayscale) is used for the Generator page only. Every results "
    "file states the image size used. (proposed)",
    "Split file `data/splits.csv` (seed 42): CIFAR-10 TRAIN is cut into a DDPM-train part of "
    "30,000 images and a disjoint classifier-train part of 20,000 images; the CIFAR-10 test set "
    "is cut into a calibration split of 5,000 images and a final test split of 5,000 images "
    "(all four numbers proposed). The SAME file for both of us.",
    "Which data each model trains on (proposed): all generators and d-vae-cifar train on "
    "CIFAR-10 training images; d-ddpm-cifar trains on the DDPM-train part (30,000) only; the "
    "classifier trains on the classifier-train part (20,000) plus fakes. Fakes are SAMPLES from "
    "the generators, so no test image is used to make a fake. The calibration and final test "
    "splits come only from the CIFAR-10 test set.",
    "Normalisation: scale pixels to [-1, 1], as in the course notebooks. (proposed)",
    "Batch size 128, Adam, learning rate 2e-4 (from Nitin's own course notebooks, which are not "
    "in this repo). The VAEs start on Monday with a short run and a small FIXED number of epochs; "
    "Tuesday's measured seconds per epoch decide the DDPM epochs. (all proposed, to tune)",
    "Seed 42, set in ONE file: `model/utils.py`. Every detector-only model gets its OWN seed: "
    "d-vae-cifar differs from g-vae-cifar, d-ddpm-cifar differs from g-ddpm-cifar. (proposed)",
    "Model size: each model is small, about 1M parameters (proposed, to measure; the original "
    "DDPM CIFAR-10 model has 35.7M parameters, so ours must be far smaller). Every size in MB, "
    "every training time and every Kaggle quota is to verify.",
    "Detector settings: timestep t about 200 of T=1000; the 1st percentile for 'possible copy' (a "
    "percentile is the value below which a given share of the numbers fall, so the 1st "
    "percentile is a very low cutoff). The Detector label uses TWO thresholds, a low and a "
    "high: 'real' below the low one, 'ai' above the high one, 'unclear' between them. Both "
    "start around 0.5. All proposed, to tune on the calibration split only.",
    "Generator settings: n up to 16 images per request; DDIM steps 20, 50 or 100 for the DDPM "
    "generators. (proposed)",
    "Try an example: 10 fake and 10 real CIFAR-style images (proposed). The fake ones are our own "
    "samples and may be committed. The REAL CIFAR-10 ones are NOT committed (licence to verify; "
    "datasets never go in git): Nitin shares them by file transfer into "
    "`web/public/examples/real/`, which is in `.gitignore`. Compare check: 20 photo pairs of "
    "mixed subjects (proposed): own photos, NO photos of people, each pair with a written "
    "answer key. n = 20 is a small sample, descriptive only.",
    "One config JSON per run in `configs/` (epochs, sizes, the timestep t, thresholds, seed, "
    "image size). One results JSON per experiment in `results/`, written by a script. "
    "(proposed)",
    "Uploads: PNG or JPG up to 5 MB (proposed). The Detector flags `out_of_scope` (the image is "
    "not the size the detector was trained on) whenever an upload is not exactly 32x32 "
    "(proposed rule). It only catches image SIZE: a 32x32 photo that is not CIFAR-style is not "
    "flagged.",
    "The website reads the API address from `VITE_API_URL` (the address of the API; proposed; "
    "this assumes the site uses Vite, to verify on Monday evening). WARNING: every variable "
    "that starts with `VITE_` is shipped to the browser, so a key must never use that prefix. "
    "Any API key lives only in `api/.env` on Nitin's laptop: never in `web/`, never in git, "
    "never pasted into an AI chat or a log.",
]
# The generic lines (same words as our earlier team plan template):
SETTINGS_RULE_VALIDATION = (
    "Decisions are made on the validation set. The test set is scored once per finished run and "
    "never used to choose anything."
)
SETTINGS_RULE_CALIBRATION = (
    "In our project the calibration split plays the role of the validation set. Start both label "
    "thresholds around 0.5 and tune them, together with the 1st-percentile rule and other "
    "thresholds, on the calibration split ONLY. The final test split is scored once per finished "
    "run."
)
SETTINGS_RULE_NO_COMMIT = "Never commit the dataset or weights (.gitignore blocks them)."
SETTINGS_RULE_COMMENTS = "Write code comments in simple words, like in our class activities."
SETTINGS_RULE_NOTEBOOK = (
    "Before you push a notebook, use \"Restart kernel and Run All\" so its outputs are saved and it "
    "runs from top to bottom."
)

h2(CARDS_HEADING)
para(CARDS_INTRO)
h3(SETTINGS_HEADING)
para(SETTINGS_INTRO)
for line in SETTINGS_LINES:
    para(line, after=2)
para(SETTINGS_RULE_VALIDATION, after=2)
para(SETTINGS_RULE_CALIBRATION, after=2)
para(SETTINGS_RULE_NO_COMMIT, after=2)
para(SETTINGS_RULE_COMMENTS, after=2)
para(SETTINGS_RULE_NOTEBOOK)

# --- Nitin's card ----------------------------------------------------------
NITIN_TITLE = "Nitin (leader, backend, models, wiring; GPU laptop + Kaggle)"
NITIN_GOAL = (
    "the repo protection, the backend (FastAPI), all the models and the Kaggle training, the "
    "results files, and on Wednesday the wiring of Tejash's website to the real API."
)
NITIN_LEADER_JOBS = (
    "Leader jobs, every Day: read both logs; resolve merge conflicts; be told BEFORE a pull "
    "request if AI use is not reported; talk to the teacher; call the three gates."
)
NITIN_GITIGNORE_LINES = [
    "data/*",
    "!data/splits.csv",
    "fakes/",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.safetensors",
    "*.h5",
    ".env",
    "node_modules/",
    ".venv/",
    "__pycache__/",
    "web/public/examples/real/",
]
NITIN_VENV_LINES = [
    "python -m venv .venv",
    ".venv\\Scripts\\activate",
    "pip freeze > requirements.txt",
]
NITIN_GITIGNORE_COMMIT_LINES = [
    "git add .gitignore",
    "git status",
    "git commit -m \"Mon: fix .gitignore first\"",
    "git push",
]
NITIN_DAYS = [
    {
        "title": "Mon evening: protect the repo, contract, data and first models (Mon 23 Nov, "
                 "18:00 to 23:00, proposed)",
        "goal": (
            "the repo is safe BEFORE any data arrives, the contract is agreed, the split file "
            "exists and the first models are training. You pull from: nothing yet; read sections "
            "0 and 6A (you start; Tejash pulls from you)."
        ),
        "steps": [
            {"text": "Fix `.gitignore` FIRST, before any dataset or weights file is on your laptop. "
                     "Keep the lines that are already there and make sure these lines are in it "
                     "too (one per line). They block datasets, fake images, model weights, secret "
                     "files, big folders and the REAL Try-an-example images. `data/splits.csv` is "
                     "the one data file allowed in git.",
             "code": NITIN_GITIGNORE_LINES},
            {"text": "Commit this change on its own and push it straight to `main`. This is the one "
                     "exception to 'every change is a pull request': nobody can review it first, "
                     "and Tejash needs it before he adds files. Run `git status` before every "
                     "commit: it must list no data, fakes or weights.",
             "code": NITIN_GITIGNORE_COMMIT_LINES},
            "Invite Tejash as a collaborator if he is not in yet (GitHub: Settings, "
            "Collaborators) and tell him the repo address. Create `docs/ai_usage.md` (with the "
            "table header from section 6A), `docs/explain.md` (for the 'be ready to explain' "
            "questions) and one log file `docs/log/<yourname>.md` per person (with the six parts "
            "from section 6).",
            {"text": "Make the Python environment: create a virtual environment `.venv` (git "
                     "ignores it), activate it and install what you need (for example PyTorch, "
                     "torchvision, scikit-learn, scikit-image, Pillow, FastAPI, uvicorn and "
                     "python-multipart; check each site for the current install command). Then "
                     "pin the versions with `pip freeze > requirements.txt` and delete the lines "
                     "of packages you do not need.",
             "code": NITIN_VENV_LINES},
            "First meeting with Tejash (about 20 minutes, proposed): agree the team settings, the "
            "six pages and the eight endpoints in Appendix A. Then write them into "
            "`api/contract.json` on a new branch (section 5): real JSON numbers, optional fields "
            "null. Open a pull request; Tejash reviews and merges it.",
            "Open your Kaggle quota page and write the hours left and the reset day in your log "
            "(open conflict C9). The free limit of about 30 hours a week is NOT confirmed, so do "
            "not rely on it. All training plans below stay inside what the page really shows.",
            "Send the questions in section 9 to the teacher in writing, the Words question first "
            "(may the demo send photos to an external vision service?). Words will be built but "
            "stays OFF by default; if there is no answer by Wednesday morning, it stays off.",
            "Write `model/utils.py` (set_seed(42), the choice of device, the folder paths) and "
            "`model/data.py`: a CIFAR-10 loader that scales pixels to [-1, 1], and a function that "
            "writes `data/splits.csv` (seed 42) with the four parts from the team settings "
            "(30,000 / 20,000 and 5,000 / 5,000, proposed). Download CIFAR-10 and Fashion-MNIST "
            "with torchvision (its dataset classes have a `download=True` option; check the "
            "torchvision docs) into `data/` (for example `data/cifar10/` and `data/fmnist/`). "
            "`data/*` is ignored by git, so run `git status` to check. Print the split counts and "
            "check that no image is in two parts. Datasets are never stored in the repo.",
            "Write `model/vae.py`: a small convolutional VAE. The loss is the reconstruction error "
            "plus the KL term (how far the rebuilt image is from the original, plus how odd the "
            "squeezed numbers look). Start two copies on the laptop, each as a short run with a "
            "small FIXED number of epochs (proposed, to tune): g-vae-cifar (a Generator model "
            "and fake source G1) and d-vae-cifar (the Detector's anomaly-score VAE, with a "
            "DIFFERENT seed). The VAE score (reconstruction error plus KL for one image) is NEW "
            "work: the course VAE is an MLP with a 2-D latent and computes no anomaly score. Keep "
            "the weights outside git (location: <Nitin adds>).",
            "Write `model/cnn.py`: cnn-baseline, based on LetterCNN from the conditional diffusion "
            "notebook (Nitin's own course notebook; it is not in this repo). It looks at the raw "
            "3-channel image and answers real or fake; keep it small (3 conv layers, proposed). "
            "Start it, and start g-vae-fmnist (the same VAE code on Fashion-MNIST, 28x28 "
            "grayscale, 1 channel).",
            "Open your pull request(s) with the 3-line hand-over note and the checklist from "
            "section 5.",
        ],
        "done": (
            "`git status` shows no data or weights after the download; `api/contract.json` parses "
            "and Tejash has merged it; `requirements.txt` has your package versions; CIFAR-10 and "
            "Fashion-MNIST are in `data/` and `data/splits.csv` is committed with the no-overlap "
            "check printing OK; the Kaggle quota and reset day are in your log; the questions are "
            "sent; the VAEs and the CNN baseline have been started; your pull request has the "
            "3-line hand-over note."
        ),
        "explain": (
            "Why do we fix `.gitignore` before anything is downloaded? Why is CIFAR-10 TRAIN cut "
            "into two disjoint parts, and why is there a calibration split? What are the two "
            "parts of the VAE loss? Why does the seed live in ONE file?"
        ),
    },
    {
        "title": "Tue: Kaggle models and the first API (Tue 24 Nov, 09:00 to 18:00, proposed)",
        "goal": (
            "the DDPMs are training on Kaggle and the API answers for Generator, Detector and "
            "Compare. You pull from: Tejash's Monday pull request (the site in `web/`) and "
            "`api/contract.json`. Slip rule: if you are behind at 14:00, move the auditor script "
            "to Wednesday morning (the a-500 run can still start on Tuesday night); /compare and "
            "/generate stay on Tuesday."
        ),
        "steps": [
            "Measure the real seconds per epoch FIRST: one epoch of your VAE on your laptop and "
            "on Kaggle (T4; free limits: to verify). Write the numbers in your log and tell "
            "Tejash. Use them to choose the epochs of the DDPM runs (the VAEs already started on "
            "Monday with a small fixed number of epochs), and write the choice in the configs. "
            "Cut a run that would be too slow.",
            "Kaggle: d-ddpm-cifar (the Detector's reconstruction DDPM, trained on the DDPM-train "
            "part of CIFAR-10 ONLY, its OWN seed) and g-ddpm-cifar (a DDPM with the cosine "
            "schedule, trained on CIFAR-10 training images, a DIFFERENT seed). Copy TinyUNet (the "
            "small network that removes noise), the timestep embedding (it tells the network how "
            "noisy the picture is), the noise schedule (how much noise each step adds) and the "
            "train and sample loops from the Day 6 diffusion notebook (Nitin's own course "
            "notebook; it is not in this repo; reusing it is proposed) into `model/ddpm.py`, "
            "change TinyUNet to 3 colour channels and fix the paths. Never train on the "
            "calibration or final test split.",
            "Kaggle: g-cddpm-fmnist, a class-conditional DDPM with classifier-free guidance on "
            "Fashion-MNIST, based on the conditional diffusion notebook. One model must run "
            "with and without the class label. The guidance slider on the Generator page uses "
            "it. Keep the weights outside git.",
            "Write `api/main.py` (FastAPI) with GET /health, GET /generators and POST /generate "
            "(the two VAEs first: `seed`, `n` up to 16, `interpolate` with `seed_b`). Every "
            "answer carries `metadata.label`, so the page can write '<model> on <dataset>, small "
            "model, expect low quality' under each image. Each PNG gets a provenance text note "
            "(model, dataset, seed) with Pillow's PngInfo.add_text (to verify in the Pillow "
            "source). Optional features that do not exist yet return null.",
            "Create `api/.env` for the Words key (leave it empty for now) and run `git check-ignore "
            "-v api/.env`: it must print the `.gitignore` line that ignores the file. The key goes "
            "ONLY in `api/.env`: never in `web/`, never in git, never pasted into an AI chat or a "
            "log.",
            "POST /detect: on Tuesday `label` and `score` come from cnn-baseline (the same number "
            "as `baseline_score`), because the logistic regression does not exist yet; the label "
            "uses a low and a high threshold, both about 0.5 to start (proposed), with 'unclear' "
            "between them. Add the `vae_score` (d-vae-cifar), `input_size`, and `out_of_scope` = "
            "true whenever the upload is not exactly 32x32 (proposed rule; the image is still "
            "resized and scored) with the plain-English `warning`. Also build the VAE error map "
            "(the per-pixel difference between the image and d-vae-cifar's rebuilt image) as the "
            "default heatmap (`heatmap_source` = `vae`, a MUST). The DDPM error map is a SHOULD "
            "and replaces it only if Gate 2 passes. On Wednesday the logistic regression "
            "replaces `score` and `label`.",
            "POST /compare works for ANY two photos (for example an orange cat and a black cat): "
            "the SSIM and its difference map (scikit-image, `structural_similarity`) and the "
            "feature table (colour mean and histogram distance, brightness, edge density, "
            "texture, original size, and the dominant colours: the three most common colours of "
            "each image found with k-means, k = 3 proposed, with scikit-learn `KMeans`, each "
            "given a plain name such as 'orange' or 'black' and its share of the image; "
            "background colours count too), each row with a `does_not_prove` text. Read the "
            "optional `focus` field (a short text, up to 100 characters, proposed; empty means "
            "the full checklist); it is used only in the Words prompt. `use_words` is accepted, "
            "and `source` is `heatmap_only` until Words is built.",
            "Write `model/auditor.py`: the L2 distance in pixel space from a generated image to "
            "its nearest training image. Each auditor run (a-500, a-10000, a-500-dup) audits its "
            "OWN small DDPM: the same architecture as g-ddpm-cifar, trained in a separate run on "
            "that run's training subset (a-500-dup: 500 images with some duplicated, proposed), "
            "and the nearest training images are looked up inside the subset that run trained "
            "on. The 'normal closeness' comes from the calibration split. Flag 'possible copy' "
            "below the 1st percentile of those distances (proposed, to tune). The script also "
            "writes a few `examples` (a generated image, its nearest training image and their "
            "distance) into the results JSON, so the Auditor grid does not need POST "
            "/audit/check. Start run a-500 (training-set size 500) as soon as the script works. "
            "Use the words 'similar' and 'possible copy', never 'copyright infringement' "
            "(Carlini et al., arXiv 2301.13188).",
            "Export the Try-an-example images (10 fake and 10 real, proposed) as small PNG files. "
            "The 10 FAKE ones are our own samples (from g-vae-cifar and, if it exists, "
            "g-ddpm-cifar): give them to Tejash in a pull request, into "
            "`web/public/examples/fake/`. The 10 REAL ones are NOT committed (the CIFAR-10 licence "
            "is to verify, and datasets never go in git): take them from the calibration split, "
            "never the final test split, and send them to Tejash by file transfer into "
            "`web/public/examples/real/`, which `.gitignore` covers.",
            "Open your pull request(s) with the 3-line hand-over note. Review Tejash's pull "
            "request with the reviewer checklist in section 5.",
        ],
        "done": (
            "Gate 1: the VAEs have finished training and the Kaggle DDPM jobs have been started; "
            "/health, /generators, /generate, /detect and /compare answer on your laptop in the "
            "shape of `api/contract.json`; `git check-ignore -v api/.env` prints a line; run a-500 "
            "has started (or the slip rule moved the script to Wednesday); the Try-an-example "
            "images are with Tejash; your pull request has the 3-line hand-over note."
        ),
        "explain": (
            "How does a DDPM add noise (forward) and remove it (reverse)? What does DDIM change, "
            "and why do we use 20 to 100 steps? What does the guidance slider trade (more "
            "guidance: less variety, more fidelity per sample)? What does SSIM measure, and when "
            "is it meaningless?"
        ),
    },
    {
        "title": "Wed: results, wiring, freeze and the backup video (Wed 25 Nov)",
        "goal": (
            "results from scripts, the website wired to the real API, the freeze at 16:00 and the "
            "backup video. You pull from: the Kaggle outputs and Tejash's Tuesday pull request."
        ),
        "steps": [
            "At 09:00 download the Kaggle outputs (weights and samples) to your laptop. Weights "
            "stay out of git; share them by file transfer only (location: <Nitin adds>). Then do "
            "the Gate 2 check at 12:00. If there are no usable DDPM samples: cut g-ddpm-cifar and "
            "d-ddpm-cifar. The heatmap stays the VAE error map (`heatmap_source` = `vae`, already "
            "the default), leave-one-generator-out is reported as 'not possible', and the "
            "Detector uses VAE fakes only. Write this as a limitation for Tejash's Limits page.",
            "Make the fakes: they are SAMPLES from the generators (g-vae-cifar and, if Gate 2 "
            "passed, g-ddpm-cifar), so no test image is used to make a fake. Train the logistic "
            "regression (scikit-learn) on the classifier-train part plus the fakes, with the VAE "
            "score and, if Gate 2 passed, the DDPM reconstruction error (the DIRE idea: add "
            "noise, let a model clean it, measure the error; Wang et al., arXiv 2303.09295; "
            "venue: to verify; use the mean and percentiles of the per-pixel error). Tune the two "
            "label thresholds and the other thresholds on the calibration split ONLY and score "
            "the final test split ONCE. Then let /detect use the logistic regression for `score` "
            "and `label` instead of the Tuesday cnn-baseline values (`baseline_score` stays). If "
            "Gate 2 passed, switch the heatmap to the DDPM error map (`heatmap_source` = "
            "`ddpm`).",
            "Write the results JSON by script: accuracy and AUROC per generator; "
            "leave-one-generator-out (train on one CIFAR generator, test on the other; report "
            "the drop as a finding, do not hide it); and robustness (JPEG, blur, noise; settings "
            "proposed, in the config). One results JSON per experiment.",
            "Train and run a-10000 and a-500-dup (each on its own small DDPM, as for a-500), then "
            "make GET /audit/results serve the JSON files (with the script-written `examples`) "
            "from `results/` unchanged. Optional cross-check: a Gaussian fit (a bell curve) or a "
            "2-component GMM (a mix of two bell curves), only if the distance histogram clearly "
            "has two bumps.",
            "SHOULD items, in this order, only while time remains: Words in POST /compare (built "
            "first; OFF by default, and if the teacher has not answered by this morning it stays "
            "off at the demo; a vision API with both images and a generic attribute checklist "
            "prompt: main subject; colour and pattern; shape and size; texture or material; "
            "visible parts or features (for a cat: ears, eyes, nose, whiskers; for a bicycle: "
            "handlebars, saddle, wheels); pose and angle; background; lighting and photo style; "
            "number of objects. The model answers in JSON, may answer 'same' or 'unsure' for any "
            "attribute and must not guess; `differences[].part` holds the attribute or part "
            "name; the `focus` text, if any, goes into the prompt; the key only in "
            "`api/.env`); Grad-CAM on cnn-baseline (explains the CNN, not the "
            "forgery); EXIF in POST /evidence; pHash plus the Auditor side-by-side difference "
            "map (it reuses the Compare SSIM code). COULD items only if all of that is done.",
            "Merge Tejash's pull request and connect the website to the real API (point "
            "`VITE_API_URL` at your laptop). Allow only the website's local address in CORS (a "
            "browser rule about which websites may call the API; proposed). Pair with Tejash "
            "from about 13:00 (proposed).",
            "16:00: call Gate 3, the feature freeze. After this only bug fixes.",
            "16:00 to 20:00: run the whole demo end to end twice without help, with Tejash, and "
            "record the backup video. Save the video outside git. 20:00: hosting decision "
            "(Render and Vercel) only if everything else is done; otherwise the laptop demo is "
            "the plan. 22:00: hard stop.",
        ],
        "done": (
            "Gate 3 is called; every results JSON is written by a script and has its config path; "
            "the website works with your real API; the demo ran twice end to end and the backup "
            "video is saved; every result cell comes from a script or still shows "
            + PLACEHOLDER + "."
        ),
        "explain": (
            "What is AUROC? Why do we use a percentile for the cut-off, and why is it tuned on "
            "the calibration split only? What does the drop on the other generator tell us? "
            "Why is the final test split scored only once?"
        ),
    },
    {
        "title": "Thu: rehearsal and presentation (Thu 26 Nov)",
        "goal": (
            "a calm rehearsal and the presentation; no new features. You pull from: the final "
            "results files and Tejash's slides."
        ),
        "steps": [
            "Update the deck from the results JSON files (the Detector, Auditor and results "
            "slides): every number traced to a script name, " + PLACEHOLDER + " where there is "
            "no real result. Put Tejash's Generator, Compare and Limits slides into the deck.",
            "Rehearse twice with Tejash: the laptop demo first, then the backup video as plan B.",
            "Present the Detector and the Auditor.",
        ],
        "done": (
            "the deck matches the results files; two rehearsals ran without help; the presentation "
            "is given."
        ),
        "explain": (
            "What does the Limits slide say about unseen generators and about arbitrary photos? "
            "Why does the Auditor say 'similar' and not 'copied'? What do you do if the laptop "
            "demo fails?"
        ),
    },
]

h3(NITIN_TITLE)
para(NITIN_GOAL, lead="Goal:", bold=True)
para(NITIN_LEADER_JOBS)
for block in NITIN_DAYS:
    day_block(block)

# --- Tejash's card ---------------------------------------------------------
rule()
TEJASH_TITLE = "Tejash (website, laptop)"
TEJASH_GOAL = (
    "the existing website, listed and then upgraded into six pages that work against stub JSON "
    "first and then against the real API. No backend work. You pull from: nothing yet on Monday "
    "(read sections 0 and 6A); from Tuesday on, `api/contract.json` (Nitin writes it on Monday "
    "evening) and the other lane's last merged pull request (Nitin's): read it and write 3 "
    "lines on what you understood."
)
TEJASH_DAYS = [
    {
        "title": "Mon evening: list the site, bring it in, six stub pages (Mon 23 Nov, 18:00 to "
                 "23:00, proposed)",
        "goal": (
            "you know what the existing site is, it lives in `web/`, and six routes open with "
            "stub data. You pull from: nothing yet; read sections 0 and 6A. Wait for Nitin's "
            "`.gitignore` push before you add files; `api/contract.json` arrives once Nitin's "
            "pull request is merged."
        ),
        "steps": [
            "Install what you need: Git, Node.js (LTS) and a GitHub account (check each site for "
            "current instructions). Read sections 0 and 6A of this file, then tell your AI your "
            "name and that you are on Monday evening. Nitin creates your log file "
            "`docs/log/<yourname>.md` (the six parts are in section 6); you edit only your own file.",
            "List the existing site in your log. Existing site: structure and tech to be listed "
            "(page list, framework, folder or repo link). Also write how it gets its data now, "
            "and whether it uses Vite (the name `VITE_API_URL` assumes it does: to verify). Send "
            "the list to Nitin.",
            {"text": "Clone the repo (accept the GitHub invitation first). After Nitin has pushed "
                     "the `.gitignore` fix, make your own branch (section 5) and bring the existing "
                     "site into `web/`. Run `git status` and check that no `node_modules/`, `.env` "
                     "or secret file is listed.",
             "code": ["git clone <REPO URL - Nitin adds>"]},
            "First meeting with Nitin (about 20 minutes, proposed): agree the team settings, the "
            "six pages and Appendix A. Nitin writes `api/contract.json`; you review and merge "
            "that pull request.",
            "Add the six routes: `/`, `/generator`, `/detector`, `/compare`, `/auditor` and "
            "`/limits`, with one navigation bar: Home, Generator, Detector, Compare "
            "(Distinguisher), Auditor, Limits and Methods.",
            "Make the stub with the instructions in Appendix A: one JSON file per endpoint in the "
            "site's static folder, and one small data function that reads those files while stub "
            "mode is on and calls the real API when it is off.",
            "Open every route in the browser. Each must show its stub data or a clear 'not built "
            "yet' message, never a blank page.",
            "Open your pull request with the 3-line hand-over note and the checklist from "
            "section 5. Nitin reviews and merges it.",
        ],
        "done": (
            "the existing site is listed in your log and sent to Nitin; the site is in `web/` "
            "through a merged pull request with no `node_modules/` or secret in it; every route "
            "opens with stub data; your pull request has the 3-line hand-over note."
        ),
        "explain": (
            "How does a web page ask for JSON (fetch)? What is a stub, and why do we use one? Why "
            "do we run `git status` before the first commit? What is an environment variable like "
            "`VITE_API_URL`, and why not write the address in the code?"
        ),
    },
    {
        "title": "Tue: Generator, Detector and Compare pages (Tue 24 Nov, 09:00 to 18:00, proposed)",
        "goal": (
            "three pages that work against stub JSON. You pull from: Nitin's Monday pull request "
            "(the contract) and the Try-an-example images he sends in the afternoon (the fake "
            "ones in a pull request, the real ones by file transfer)."
        ),
        "steps": [
            "Generator page (/generator): a model and dataset picker filled from GET /generators; "
            "show a control only if the generator lists it in `controls` (seed, n, steps, "
            "guidance, class_id, interpolate); an image grid with the caption '<model> on "
            "<dataset>, small model, expect low quality' under every image; a Download PNG "
            "button; the buttons 'Send to Detector' and 'Send to Compare' (the image is passed "
            "inside the browser, no new endpoint).",
            "Detector page (/detector): the upload box (PNG or JPG up to 5 MB, proposed), a 'Try an "
            "example' strip above it (10 fake and 10 real images, proposed: the fake ones are "
            "static files in `web/public/examples/fake/`; the real ones are copied by Nitin into "
            "`web/public/examples/real/`, which git ignores, so never commit that folder), the "
            "scores (`label`, `score`, `vae_score`, `baseline_score`), the heatmap and Grad-CAM "
            "when they are not null, and the red out-of-scope banner whenever `out_of_scope` is "
            "true (the wording and the known limitation are in Appendix B). An Evidence tab shows "
            "every null as 'not available'.",
            "Compare page (/compare), for ANY two photos (for example an orange cat and a black "
            "cat): two upload boxes, a 'What should I compare?' text box (the optional `focus`, "
            "up to 100 characters, proposed, for example 'fur colour and eyes') and a Words "
            "switch (off by default); the overall SSIM, the difference map and the feature table "
            "(with the dominant colours) with a 'does not prove' column; a Words panel that "
            "shows `differences` (`part` holds the attribute or part name) when `source` is "
            "`vlm` and a plain 'Words are off' message when it is `heatmap_only`, with the "
            "privacy note and the required honesty line from Appendix B.",
            "Error states on every page: a wrong file type, a file over the size limit, an "
            "`{\"error\": ...}` answer from the API, and an API that cannot be reached. Always show a "
            "readable message, never a blank page.",
            "Collect 20 pairs of photos of mixed subjects (your own photos, NO photos of people; "
            "20 is proposed): for example an orange cat and a black cat, two mugs, two pairs of "
            "shoes, and a few bicycles. Keep them outside git in a folder you agree with Nitin. "
            "For each pair, write an answer key BEFORE you ever see an answer from Words: what "
            "differs (colour, pattern, shape, parts, pose, background).",
            "Open your pull request by 18:00, with the 3-line hand-over note. Nitin reviews and "
            "merges it.",
        ],
        "done": (
            "Gate 1: the three pages work against stub JSON and every error state shows a "
            "readable message; the 20 photo pairs are collected and your own list of differences "
            "is written; your pull request has the 3-line hand-over note."
        ),
        "explain": (
            "Why does the Detector page show the red banner? Why must the Evidence tab show 'not "
            "available' when a field is null? Why are the Generator images labelled 'small "
            "model, expect low quality'? What does the Compare page say about pixel tools on two "
            "different photos, such as an orange cat and a black cat?"
        ),
    },
    {
        "title": "Wed: Auditor, Limits, wiring and grading (Wed 25 Nov)",
        "goal": (
            "the last two pages, the wiring with Nitin and the Compare grading. You pull from: "
            "Nitin's results JSON files as they appear, and his real API."
        ),
        "steps": [
            "Auditor page (/auditor): charts from GET /audit/results (the x values on the x axis, "
            "named by `x_label`, and `memorization_rate` on the y axis; runs a-500, a-10000 and "
            "a-500-dup), the nearest-neighbour grid drawn from `examples` in GET /audit/results "
            "(generated image beside its nearest training image, with the `distance`; the grid "
            "does not need POST /audit/check; show the pHash distance only when it is not null), "
            "and the upload check hidden when POST /audit/check returns 404. Every result cell "
            "shows " + PLACEHOLDER + " until a script fills it.",
            "Limits and Methods page (/limits): the honesty lines from Appendix B exactly as "
            "written, a 'what it does not prove' column for every feature (Appendix D), model "
            "cards that copy their numbers from the results and config files (never typed by "
            "hand), and the sources from Appendix H. Add Nitin's Gate 2 limitation if it applies.",
            "Pair with Nitin on the wiring: point `VITE_API_URL` at his laptop API, switch stub "
            "mode off and fix the user-interface bugs. If the real API is late, the website keeps "
            "using stub JSON.",
            "Hand-grade the 20 Compare pairs once Words works. The question for each named "
            "difference is 'did it name the right attribute?': mark it right, wrong or missed, "
            "against your own answer key from Tuesday. Write the marks into a small file and "
            "give it to Nitin's script, which counts them into one results JSON. Report it as "
            "n = 20, a small sample: the counts are descriptive only. If Words is cut, the cell "
            "stays " + PLACEHOLDER + ".",
            "16:00: Gate 3, the feature freeze. After this only bug fixes.",
            "16:00 to 20:00: run the demo twice end to end with Nitin and help record the backup "
            "video.",
            "Open your pull request(s) with the 3-line hand-over note.",
        ],
        "done": (
            "all six pages work against the real API on the laptop; the Limits page has the "
            "Appendix B wording; the Compare grading is saved or marked as cut; your pull request "
            "has the 3-line hand-over note."
        ),
        "explain": (
            "How does the chart map the JSON (what is x, what is y, what is x_label)? Why do we hide "
            "a section when an endpoint returns 404? What can the Compare Words panel get wrong, and "
            "how does the page say so? What changes for the website when the stub is replaced by "
            "the real API?"
        ),
    },
    {
        "title": "Thu: rehearsal and presentation (Thu 26 Nov)",
        "goal": (
            "a calm rehearsal and the presentation; no new features. You pull from: the final "
            "pages and Nitin's deck."
        ),
        "steps": [
            "Prepare the Generator, Compare and Limits slides and the demo flow (the click order "
            "and which example images to use) from the real pages and the results files.",
            "Rehearse twice with Nitin: the laptop demo first, then the backup video as plan B.",
            "Present the Generator, Compare and Limits pages.",
        ],
        "done": (
            "your slides match the pages; two rehearsals ran without help; the presentation is "
            "given."
        ),
        "explain": (
            "What can the Words panel get wrong? What does the Limits page say the Detector cannot "
            "do? Why are the Generator pictures labelled 'small model, expect low quality'? What "
            "is the plan if the laptop fails?"
        ),
    },
]

h3(TEJASH_TITLE)
para(TEJASH_GOAL, lead="Goal:", bold=True)
for block in TEJASH_DAYS:
    day_block(block)

# ===========================================================================
# 5. HOW WE HAND OVER
# ===========================================================================

HANDOVER_HEADING = "5. How we hand over (every Day)"
HANDOVER_GET_LATEST = "Get the latest work:"
HANDOVER_GET_LATEST_CMDS = ["git checkout main", "git pull"]
HANDOVER_BRANCH = "Make your own branch (example for Tejash's Monday evening):"
HANDOVER_BRANCH_CMDS = ["git checkout -b mon-web-import-tejash"]
HANDOVER_SAVE = (
    "Do your work. Then save and upload it (run `git status` before every commit; use your own "
    "file names):"
)
HANDOVER_SAVE_CMDS = [
    "git add web/ docs/log/<yourname>.md",
    "git status",
    "git commit -m \"Mon: bring the existing site into web/\"",
    "git push -u origin mon-web-import-tejash",
]
HANDOVER_PR_TEXT = (
    "Open a pull request on GitHub. In the description write 3 lines (and paste the checklist from "
    "below this list under them): what I did, what the other person must read first, what the "
    "other person does. Also write 3 lines on what you understood from the other lane's last "
    "merged pull request (on Monday nothing is merged yet: write them about sections 0 and 6A)."
)
HANDOVER_REVIEW_LINES = [
    "The other person reviews and merges it. Nobody merges their own work (only Nitin merges the "
    "very first one, the `.gitignore` change).",
    "The other person starts again at step 1.",
]
HANDOVER_CHECKLIST_LEAD = (
    "Paste this checklist at the bottom of every pull request description and tick it honestly:"
)
HANDOVER_CHECKLIST = [
    "[ ] My log `docs/log/<yourname>.md` is updated (daily row, line-by-line notes, concepts).",
    "[ ] I can explain every line I changed.",
    "[ ] Notebooks: I used \"Restart kernel and Run All\" and the outputs are saved.",
    "[ ] I did not commit data, weights, secrets or .env files.",
    "AI use (tick ONE):",
    "[ ] I did NOT use any AI for this work.",
    "[ ] I used AI. I gave it this file first, and I pasted its AI SESSION REPORT unedited into "
    "`docs/ai_usage.md` and my log (section 6A).",
]
HANDOVER_REVIEWER = [
    "Reviewer (the other person), before merging:",
    "[ ] I read the \"what I understood\" lines and they are right.",
    "[ ] I asked 2 questions about the code and got correct answers.",
    "[ ] If AI was used: the `docs/ai_usage.md` row and the log entries exist and agree with each "
    "other.",
    "[ ] No one else's files, data or weights are in the pull request.",
]
HANDOVER_WAITING = (
    "While you wait for the other person's pull request: read the other lane's last two merged "
    "pull requests, write your 6 'be ready to explain' questions in your log, draft your notes for "
    "the presentation and the report, and test your code on dummy data (or on the stub). Commit "
    "your log every day, even a small entry."
)

h2(HANDOVER_HEADING)
para(HANDOVER_GET_LATEST, after=2, keep_next=True)
code_block(HANDOVER_GET_LATEST_CMDS)
para(HANDOVER_BRANCH, after=2, keep_next=True)
code_block(HANDOVER_BRANCH_CMDS)
para(HANDOVER_SAVE, after=2, keep_next=True)
code_block(HANDOVER_SAVE_CMDS)
para(HANDOVER_PR_TEXT, bold=True)
for line in HANDOVER_REVIEW_LINES:
    para(line)
para(HANDOVER_CHECKLIST_LEAD, bold=True)
for line in HANDOVER_CHECKLIST:
    para(line, after=2)
for line in HANDOVER_REVIEWER:
    para(line, after=2)
para(HANDOVER_WAITING)

# ===========================================================================
# 6. HOW TO LOG
# ===========================================================================

LOG_HEADING = "6. How to log your work (your file: docs/log/<yourname>.md)"
LOG_INTRO = (
    "Nitin creates your file from the template on Monday evening. You edit ONLY your own file. It "
    "has six parts:"
)
LOG_PARTS = [
    ("Daily log:",
     "one row per work session: what you did, how you did it (the steps you really took), the "
     "commit or pull request link, and where you are stuck."),
    ("Code notes (line by line):",
     "for every function or block you write, paste the code and add a table: which lines, what "
     "they do in simple words, the concept, and WHY we used it (and what we could have used "
     "instead)."),
    ("Concepts I learned:",
     "each in your own words, with where it is in our code and one likely question the professor "
     "may ask with your answer."),
    ("What I understood from the other lane's last merged pull request",
     "(write before you start your Day)."),
    ("Sources I used:", "papers, docs, videos, AI chats. This also gives the reference list."),
    ("My 'be ready to explain' questions:",
     "6 questions with short answers. At the end copy them into `docs/explain.md`."),
]
LOG_AI_NOTE = (
    "If you use AI: the AI writes your daily-log row, your code notes, your concepts and your "
    "sources for you (section 6A). You paste them in unedited, read them, and mark anything you do "
    "not understand."
)
LOG_HONEST_NOTE = (
    "A good code note is short and honest. If you do not understand a line, write \"I do not "
    "understand this yet\" and ask the team. Nitin reads these logs to see who is doing what, and "
    "everybody reads them to learn the parts they did not write."
)

h2(LOG_HEADING)
para(LOG_INTRO)
num_list(LOG_PARTS)
para(LOG_AI_NOTE, bold=True)
para(LOG_HONEST_NOTE)

# ===========================================================================
# 6A. IF YOU USE AI  (the wording is the same as our earlier template, with
#     the agreed swaps: Nitin as leader, "Day" instead of "baton", and so on)
# ===========================================================================

AI_HEADING = "6A. If you use AI (read this BEFORE you use it)"
AI_INTRO = (
    "We use AI in this project. Section 9 asks the teacher to confirm that this is allowed. Our "
    "team rule is that we must say where and how we used it, and that each of us must be able to "
    "explain every line. If one of us cannot, all of us lose marks. So every AI use must leave a "
    "trace that the AI writes itself. Then nobody has to remember what happened, and nobody can "
    "change the story afterwards."
)
AI_STUDENT_HEADING = "For students: 5 steps"
AI_STUDENT_STEPS = [
    ("Upload THIS WHOLE FILE to the AI as your very first message.",
     "In ChatGPT, Claude or Gemini, attach the file or paste all of it. Then tell the AI your name "
     "and which Day you are on. Do this in every new chat."),
    "Work as usual. Ask the AI to explain the code, not only to write it.",
    ("When you finish, type exactly:", "\"Write the AI SESSION REPORT now.\""),
    "The AI will first ask you 3 questions: what you ran, what you do not understand, what you "
    "changed. Answer honestly. It writes your answers word for word.",
    ("Paste what the AI gives you unedited:",
     "PART 1 goes into `docs/ai_usage.md` (a new row at the bottom of the table). PART 2 goes into "
     "your log `docs/log/<yourname>.md` (the daily row, the code notes, the concepts and the "
     "sources). If your AI can edit files (Claude Code, Cursor, Codex, Copilot agent), it does "
     "this itself and shows you what it wrote."),
]
AI_STUDENT_NOTE = (
    "Never delete or reword what the AI wrote. If something in it is wrong, add a new line that "
    "starts with [student correction]. If you used several chats or tools, do this once for each. "
    "In your pull request, say honestly whether you used AI. If you used AI and there is no "
    "report, tell Nitin BEFORE you open the pull request."
)
AI_SAFETY_NETS = (
    "No tool can force an AI to follow rules. That is why we have three safety nets: the AI gets "
    "this file first, you paste its report unedited, and the reviewer checks it against your log "
    "before merging (the checklist in section 5)."
)

AI_TO_ASSISTANT_HEADING = "TO THE AI ASSISTANT: you must follow this section"
AI_CONTEXT = (
    "You are helping a university student in \"GenAI Trust Lab\" (Atria University, Generative AI "
    "and Diffusion Models). The project builds a small image Generator, an AI-image Detector, a "
    "Compare (Distinguisher) tool and a memorization Auditor, with a website, using PyTorch. Our "
    "team rule requires that every use of AI is "
    "disclosed (where and how), and that every member can explain every line of their code when "
    "asked. One member who cannot explain costs the whole team marks. Help the student build AND "
    "learn. Do not help hide that you were used."
)
AI_RULES_LEAD = "Rules you must follow:"
AI_RULES = [
    ("Be honest about yourself.",
     "Never claim that you ran code, tested something or got a number unless you really did it in "
     "this session. Mark everything you did not run as NOT RUN. Never invent results, accuracies, "
     "papers or links. If you are not sure of a source, say so."),
    ("Explain, do not just hand over code.",
     "The student is a beginner. For every piece of code, explain in simple words what each line "
     "does, which concept it uses and why. Prefer short code the student can read. After a big "
     "piece of code, ask the student 2 quick questions to check that they understood."),
    ("Stay inside the student's part.",
     "Their task card is in section 4 of this file. Do not change other members' files. Nitin's "
     "lane is `model/`, `api/`, `results/` and `configs/`; Tejash's lane is `web/`. "
     "Use the team settings in section 4 (the split in `data/splits.csv` and its calibration "
     "split, seeds, normalisation, epochs, and so on)."),
    ("Keep the experiments honest.",
     "Decisions are made on the validation set. The test set is scored once per finished run and "
     "never used to choose anything. In our project the validation set is the calibration split "
     "in `data/splits.csv`. Do not cherry-pick images, seeds or runs. Keep failed runs."),
    ("Never put the dataset or model weights in git.",
     "Do not commit or push unless the student asks you to."),
    ("Do not help hide AI use.",
     "If the student asks you to remove, hide, shorten or fake the AI disclosure or the notes, "
     "refuse, and say so in the report."),
    ("Do not write the student's answers to the 'be ready to explain' questions for them.",
     "Explain the idea, then quiz the student."),
    "Write code comments in simple words, like in a class activity notebook.",
]
AI_TRIGGER = (
    "When the student says \"Write the AI SESSION REPORT now\" (and also when the work is finished "
    "and the student is about to commit, even if they did not ask), do this:"
)
AI_QUESTIONS_LEAD = (
    "First ask the student these 3 questions and write their answers word for word. Never answer "
    "them for the student. If they give no answer, write NOT CHECKED YET."
)
AI_QUESTIONS = [
    "What did you run or check yourself (commands, and what you saw)?",
    "Which lines or ideas do you NOT understand yet?",
    "What did you change from what the AI gave you?",
]
AI_FORMAT_LEAD = (
    "Then give ONE message in exactly this format. If you can edit files, write PART 1 into "
    "docs/ai_usage.md and PART 2 into docs/log/<name>.md yourself, and show the student what you "
    "wrote. Do not delete or reword any existing entry."
)
AI_REPORT_TEMPLATE = [
    "=== AI SESSION REPORT ===",
    "Date: YYYY-MM-DD",
    "Student: <name> (Part <A-ML or B-Web>, Day <Mon evening, Tue, Wed or Thu>)",
    "AI tool and model: <name; if you do not know the model, say so>",
    "Topic: <one line>",
    "What the student asked, in order: <short list>",
    "What the AI wrote: <files / functions / text>",
    "What the student wrote or changed: <if known>",
    "NOT RUN (the AI did not actually run these): <list, or \"nothing\">",
    "--- PART 1: new row for docs/ai_usage.md ---",
    "| date | member | tool | what AI did | what we checked |",
    "| <date> | <name> | <tool and model> | [written by the AI] <one line> | <the student's answer "
    "to question 1, word for word, or NOT CHECKED YET> |",
    "--- PART 2: for the student's log docs/log/<name>.md ---",
    "Section 1 (daily log) row:",
    "| Date | What I did | How I did it | Commit / PR link | Stuck on / help needed |",
    "| <date> | [written by the AI] <what> | <how, including what the student changed (answer 3)> "
    "| (TODO) | <the student's answer to question 2> |",
    "Section 2 (code notes, line by line). For EVERY code block you wrote or changed:",
    "### File: `<file>` / Function: `<function>`",
    "| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could "
    "have used) |",
    "|---|---|---|---|",
    "Section 3 (concepts). For each one: what it is in one sentence, where it is in our code, and "
    "one likely question the professor may ask with a short answer.",
    "Section 5 (sources). Only real sources you relied on. If you are not sure, write \"not sure of "
    "the exact source\".",
    "I am an AI and can make mistakes. Everything marked NOT RUN was not run by me. The student "
    "must read and understand every line before submitting it.",
]

h2(AI_HEADING)
para(AI_INTRO, bold=True)
h3(AI_STUDENT_HEADING)
num_list(AI_STUDENT_STEPS)
para(AI_STUDENT_NOTE)
para(AI_SAFETY_NETS)
h3(AI_TO_ASSISTANT_HEADING)
para(AI_CONTEXT)
para(AI_RULES_LEAD, keep_next=True)
num_list(AI_RULES)
para(AI_TRIGGER)
para(AI_QUESTIONS_LEAD, bold=True)
num_list(AI_QUESTIONS)
para(AI_FORMAT_LEAD, bold=True)
for line in AI_REPORT_TEMPLATE:
    para(line, ticks=False, after=1)
spacer()

# ===========================================================================
# 7. PRESENTATION, NOTES AND REPORT: WHO WRITES WHAT
# ===========================================================================

WRITE_HEADING = "7. Presentation, notes and report: who writes what"
WRITE_INTRO = (
    "If the teacher asks for a written report, this is who writes what (we do not know yet "
    "whether one is needed, see section 9). The presentation and notes follow the same split."
)
WRITE_HEADERS = ["Section", "Writer"]
WRITE_ROWS = [
    ["Deck: Detector slides, Auditor slides and results slides; abstract; conclusion; "
     "contribution table and repo link; AI-usage paragraph; final assembly", "Nitin"],
    ["Deck: Generator slides, Compare slides and Limits slides; the demo flow (click order and "
     "example images); wording of the Limits page", "Tejash"],
    ["Results tables (copied from the script outputs, never typed by hand)",
     "Both: each person fills the tables of their own lane"],
    ["References", "Everybody adds their own sources (from section 5 of your log); Nitin formats"],
]
WRITE_WIDTHS = [4.0, 2.5]
WRITE_AI_LEAD = "AI-usage paragraph (template, Nitin fills it in):"
WRITE_AI_TEXT = (
    "We used AI assistants in this project. Claude Code was used to [what it did, for example: "
    "generate the repository skeleton and the laptop check script, plan the schedule, and build "
    "the demo website]. [Name] used [tool] to [what it did]. We checked the AI output by [what we "
    "did: ran the code, read every line, compared with the documentation]. Every member can "
    "explain the code in their part. The full log is in docs/ai_usage.md."
)

h2(WRITE_HEADING)
para(WRITE_INTRO, keep_next=True)
table(WRITE_HEADERS, WRITE_ROWS, WRITE_WIDTHS)
para(WRITE_AI_LEAD, bold=True, keep_next=True)
para(WRITE_AI_TEXT, mono_all=True, ticks=False)

# ===========================================================================
# 8. IF SOMETHING GOES WRONG
# ===========================================================================

WRONG_HEADING = "8. If something goes wrong"
WRONG_ITEMS = [
    ("The real API is late by an hour or more:",
     "the website keeps using stub JSON; the late person still finishes and keeps the credit."),
    ("The Kaggle quota is low:",
     "train d-ddpm-cifar and g-ddpm-cifar with fewer epochs (write it in the config). If that is "
     "still too slow, cut both (cut line 6) and say so on the Limits page."),
    ("The API key is missing:",
     "Words stays off (so no photo leaves your computer) and /compare answers with `source` = "
     "`heatmap_only`. Never put a key in the website or in git. When Words is on, do not upload "
     "private photos or photos of people."),
    ("Render or Vercel fails:",
     "forget cloud hosting (cut line 1) and give the demo from the laptop, with the recorded "
     "backup ready. Do not spend time on hosting after Wednesday 20:00."),
    ("The DDPM does not converge by Gate 2:",
     "cut g-ddpm-cifar and d-ddpm-cifar (cut line 6). The heatmap stays the VAE error map and "
     "the Detector uses VAE fakes only; we state that as a limitation."),
    ("The website breaks when it meets the real API:",
     "demo the pages on stub JSON plus the API's own page at /docs, and say so clearly. Never "
     "show a stub answer as a result."),
    ("The laptop is too slow:",
     "use fewer DDIM steps and a smaller n on the Generator page (measure first, then decide)."),
    ("Data or weights were committed by mistake:",
     "stop, ask Nitin and do not push. Nitin fixes it before anything is shared."),
    ("A pull request has a merge conflict:",
     "ask Nitin. Results files are one per experiment, so keep both files (rename one) instead "
     "of merging them by hand."),
    ("A laptop fails:",
     "push every day; the weights are shared outside git (location: <Nitin adds>); another GPU "
     "laptop re-runs the same command."),
    ("A run is bad or overfits:",
     "lower the learning rate by 10x and re-run; keep failed runs in the log; never pick a lucky "
     "seed."),
    ("You used AI and did not log it:",
     "tell Nitin before you open the pull request, then get the AI SESSION REPORT now (section "
     "6A). Unreported AI use puts everybody's marks at risk."),
]

h2(WRONG_HEADING)
for lead, text in WRONG_ITEMS:
    para(text, lead=lead)

# ===========================================================================
# 9. QUESTIONS FOR THE TEACHER
# ===========================================================================

QUESTIONS_HEADING = "9. Questions we still need to ask our teacher in writing"
QUESTIONS = [
    "Is a laptop demo with a recorded backup acceptable for the final presentation on Thursday?",
    "What time is our slot on Thursday, and how long is it?",
    "May the demo send photos to an external vision service (the Compare Words panel, off by "
    "default; the photos leave the computer)? Ask this on Monday. If there is no answer by Wednesday morning, Words stays off.",
    "How do we credit the website that Tejash built before the project started?",
    "Is it fine to use AI tools for code and explanations, as long as we disclose it (AI rows in "
    "the repo and a paragraph in the report) and every member can explain every line? What format "
    "should the AI-use disclosure take?",
    "Is it fine to reuse a course-notebook TinyUNet (the small denoising network) in our models?",
    "Do you need a written report from us, and what is the deadline?",
    "Are marks individual, team, or both?",
    "Can we show results tables where some cells still say " + PLACEHOLDER + "?",
    "Can the final presentation be given with results that are still marked " + PLACEHOLDER + "?",
]

h2(QUESTIONS_HEADING)
for q in QUESTIONS:
    para(q, after=2)

# ===========================================================================
# APPENDIX A. API CONTRACT AND STUB INSTRUCTIONS
# ===========================================================================

APP_A_HEADING = "Appendix A. API contract and stub instructions"
APP_A_INTRO = (
    "Everything in this appendix is proposed, agree on Monday evening at the first meeting. Nitin "
    "writes the agreed version into `api/contract.json`. The examples below are real JSON, so you "
    "can copy them. Base64 is a way to write an image as text; it is cut short in the examples "
    "on purpose. Use exactly these field names: the only optional extras are `examples` in GET "
    "/audit/results, `phash_distance` in POST /audit/check and the optional request field "
    "`focus` in POST /compare."
)

# A short piece of base64 text, cut short on purpose (real answers are much longer).
SHORT_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB..."

# The JSON examples are written as Python dictionaries and turned into text with
# json.dumps, so they are valid JSON by construction. The field names are the
# agreed ones; the values are made up (see the caption).
EXAMPLE_HEALTH = {"status": "ok"}

EXAMPLE_DETECT = {
    "label": "unclear",
    "score": 0.5,
    "vae_score": 0.4,
    "baseline_score": 0.5,
    "heatmap_png_base64": SHORT_BASE64,
    "heatmap_source": "vae",
    "gradcam_png_base64": None,
    "input_size": "32x32",
    "out_of_scope": False,
    "warning": None,
    "note": "stub: fixed example answer",
}

EXAMPLE_COMPARE = {
    "similarity": {"ssim": 0.5, "lpips": None},
    "diff_heatmap_png_base64": SHORT_BASE64,
    "regions_png_base64": None,
    "feature_table": [
        {
            "feature": "mean brightness",
            "image_a": "0.4",
            "image_b": "0.6",
            "does_not_prove": "Descriptive only.",
        },
        {
            "feature": "dominant colours",
            "image_a": "orange 60%, white 25%, brown 15%",
            "image_b": "black 70%, grey 20%, white 10%",
            "does_not_prove": "Descriptive only; background colours count too.",
        },
    ],
    "differences": [
        {"part": "fur colour", "image_a": "orange", "image_b": "black"}
    ],
    "source": "vlm",
    "summary": "stub: example text",
    "note": "Pixel tools need aligned images; words may be wrong.",
}

EXAMPLE_GENERATORS = {
    "generators": [
        {
            "id": "g-vae-cifar",
            "name": "Conv VAE",
            "dataset": "CIFAR-10",
            "family": "vae",
            "params": 900000,
            "size_mb": 4,
            "status": "ready",
            "controls": ["seed", "n", "interpolate"],
            "classes": [],
            "licence_note": "stub: example text",
        }
    ],
    "note": "stub: the real answer lists every generator",
}

EXAMPLE_GENERATE_REQUEST = {
    "generator_id": "g-vae-fmnist",
    "seed": 42,
    "n": 2,
    "steps": None,
    "guidance": None,
    "class_id": None,
    "interpolate": {"seed_b": 7, "frames": 2},
}

EXAMPLE_GENERATE_RESPONSE = {
    "generator_id": "g-vae-fmnist",
    "images": [
        {"png_base64": SHORT_BASE64, "seed": 42, "class_id": None},
        {"png_base64": SHORT_BASE64, "seed": 7, "class_id": None},
    ],
    "metadata": {
        "model": "conv VAE",
        "dataset": "Fashion-MNIST",
        "steps": None,
        "guidance": None,
        "label": "small model, expect low quality",
    },
    "note": "stub: example text",
}

EXAMPLE_EVIDENCE = {
    "exif": {
        "found": True,
        "fields": {"Make": "ExampleMake", "Model": "ExampleModel"},
        "does_not_prove": "Missing EXIF is weak evidence; present EXIF does not mean real.",
    },
    "c2pa": {
        "status": "none",
        "summary": None,
        "does_not_prove": "A missing manifest means nothing.",
    },
    "ela_png_base64": None,
    "fft_png_base64": None,
    "note": "Evidence, not proof.",
}

EXAMPLE_AUDIT_RESULTS = {
    "experiments": [
        {
            "name": "memorization vs training-set size (one point per run)",
            "x_label": "training-set size",
            "points": [
                {"x": 500, "memorization_rate": 0.2},
            ],
        }
    ],
    "examples": [
        {
            "generated_png_base64": SHORT_BASE64,
            "nearest_png_base64": SHORT_BASE64,
            "distance": 5.5,
        }
    ],
    "note": "stub: example numbers, not results",
}

EXAMPLE_AUDIT_CHECK = {
    "nearest": [
        {"rank": 1, "distance": 5.5, "phash_distance": 12, "image_png_base64": SHORT_BASE64},
    ],
    "flag": "not_flagged",
    "note": "stub: similarity only, not legal copying",
}

EXAMPLE_ERROR_FORMAT = {"error": "<text>"}

# Every example, in one list, so a quick self-check can prove each one is valid JSON.
ALL_EXAMPLES = [
    EXAMPLE_HEALTH, EXAMPLE_DETECT, EXAMPLE_COMPARE, EXAMPLE_GENERATORS,
    EXAMPLE_GENERATE_REQUEST, EXAMPLE_GENERATE_RESPONSE, EXAMPLE_EVIDENCE,
    EXAMPLE_AUDIT_RESULTS, EXAMPLE_AUDIT_CHECK, EXAMPLE_ERROR_FORMAT,
]


def json_lines(example):
    """Turn a Python dict into the lines of valid JSON (2 spaces per indent level)."""
    text = json.dumps(example, indent=2)
    assert json.loads(text) == example        # the text must read back as the same data
    return text.split("\n")


def example_block(lines):
    """Short JSON stays on one page; long JSON may break across pages."""
    code_block(lines, together=len(lines) <= 14, size=9)    # 9 pt: the longest JSON line must not wrap


APP_A_DEFAULTS_HEADING = "Proposed defaults (all: proposed, agree on Monday evening)"
APP_A_DEFAULTS = [
    "Contract file `api/contract.json`: real JSON numbers; every optional field is null when "
    "the feature is not built. The stub returns the same values every time, and its `note` says "
    "'stub'.",
    "Images go in as multipart uploads (POST /detect: `image`; POST /compare: `image_a`, "
    "`image_b`, `use_words`, true or false, and the optional `focus`; POST /evidence: "
    "`image`). POST /generate takes "
    "JSON. Everything else is JSON. Uploads: PNG or JPG up to 5 MB each; the server stores "
    "nothing (proposed).",
    "Images in answers are plain base64 PNG text, without a \"data:image/png;base64,\" prefix.",
    "/detect: the image is resized to 32x32 and scored. `label` uses two thresholds, a low and a "
    "high (proposed): `real` below the low one, `ai` above the high one, `unclear` between "
    "them. `out_of_scope` is true whenever the upload is not exactly 32x32 (proposed rule), "
    "and then `warning` holds the plain-English text; it only catches size. `heatmap_source` "
    "is `vae` (the default) or `ddpm`.",
    "/generate: `n` from 1 to 16; `steps` 20, 50 or 100 for the DDPM generators (DDIM); "
    "`guidance` and `class_id` only for g-cddpm-fmnist; `interpolate` only for the VAEs. "
    "`metadata.label` is always \"small model, expect low quality\".",
    "/compare: `use_words` is off by default. With no API key, or with Words off, `source` is "
    "`heatmap_only` and `differences` is an empty list. The Words prompt is a generic "
    "attribute checklist (main subject; colour and pattern; shape and size; texture or "
    "material; visible parts or features; pose and angle; background; lighting and photo "
    "style; number of objects). The model answers in JSON, may say 'same' or 'unsure' for any "
    "attribute and must not guess; `differences[].part` holds the attribute or part name. The "
    "key is read only from `api/.env` (never in `web/`, git, chats or logs; never in a "
    "`VITE_` variable).",
    "/audit/results serves the JSON files from `results/` unchanged (run ids a-500, a-10000, "
    "a-500-dup), including the optional script-written `examples`. /audit/check returns k = 3 "
    "nearest images from a small training subset (proposed); `phash_distance` is optional.",
    "CORS: the API allows only the website's local address (proposed). The website reads the API "
    "address from `VITE_API_URL`. If the API is ever hosted, the page waits up to 60 seconds "
    "(proposed) and shows 'waking up'.",
]

APP_A_ENDPOINTS_HEADING = "The eight endpoints"
APP_A_CAPTION = (
    "Example values only. They are not results. Every number inside the examples (such as 0.2 "
    "or 900000) is made up to show the format. The base64 text is cut short here."
)

APP_A_STUB_HEADING = "How to build the stub (Monday evening, Tejash)"
APP_A_STUB_STEPS = [
    "In the site's static folder, make a folder `stub/`. Copy each example above into its own "
    "file: `health.json`, `detect.json`, `compare.json`, `generators.json`, `generate.json`, "
    "`evidence.json`, `audit_results.json` and `audit_check.json` (only the two response parts "
    "of /generate go in `generate.json`).",
    "Replace the cut-short image text by the base64 text of a small PNG of your own, so that the "
    "pages can show a picture.",
    "In the one data function of the site: while `VITE_USE_STUB` is `true` (proposed name), read "
    "`/stub/<name>.json` for every endpoint, whatever the request; when it is `false`, call "
    "`VITE_API_URL`.",
    "Test it: open every page. Then set one optional field to null in a stub file and check that "
    "the page still works and shows 'not available'. The `note` says 'stub' and the page "
    "shows it.",
]

h2(APP_A_HEADING)
para(APP_A_INTRO)
h3(APP_A_DEFAULTS_HEADING)
num_list(APP_A_DEFAULTS)
h3(APP_A_ENDPOINTS_HEADING)
para(APP_A_CAPTION)
para("The status will be \"ok\", or the error text if something is wrong.", lead="GET /health:",
     ticks=False, after=2, keep_next=True)
example_block(json_lines(EXAMPLE_HEALTH))
para("You send one image. It answers the label (real, ai or unclear), the score (0 to 1, higher = "
     "more likely AI-made), the VAE score, the baseline score, the heatmap and Grad-CAM (base64 "
     "PNG, or null), the input size, `out_of_scope`, a warning (or null) and a note.",
     lead="POST /detect:", after=2, keep_next=True)
example_block(json_lines(EXAMPLE_DETECT))
para("You send any two photos, `use_words` and the optional `focus` (a short text, up to 100 "
     "characters, proposed, for example 'fur colour and eyes'; it is used only in the Words "
     "prompt, and if it is empty the full checklist is used). It answers the similarity, the "
     "difference map, the regions map (or null), the feature table (the colour, brightness, "
     "edge, texture, size and dominant-colours rows), the differences (Words; `part` holds the "
     "attribute or part name), the `source` (`vlm` or `heatmap_only`), a summary and a note. "
     "The request field `focus` is the only new field; the response keeps the same names.",
     lead="POST /compare:", after=2, keep_next=True)
example_block(json_lines(EXAMPLE_COMPARE))
para("The list of generators and what each one can do. `classes` is an empty list, except for "
     "g-cddpm-fmnist, where it lists the ten Fashion-MNIST class names as text.",
     lead="GET /generators:", after=2, keep_next=True)
example_block(json_lines(EXAMPLE_GENERATORS))
para("The request (JSON), then the answer. `interpolate` is null unless a VAE blends two seeds.",
     lead="POST /generate:", after=2, keep_next=True)
example_block(json_lines(EXAMPLE_GENERATE_REQUEST))
example_block(json_lines(EXAMPLE_GENERATE_RESPONSE))
para("You send one image. It answers the EXIF fields and the C2PA status (each with a 'does not "
     "prove' text), the ELA and FFT images (or null) and a note.",
     lead="POST /evidence:", after=2, keep_next=True)
example_block(json_lines(EXAMPLE_EVIDENCE))
para("The auditor results from `results/`, sent unchanged: experiments (name, x_label, points), "
     "the optional `examples` (a generated image, its nearest training image and their "
     "distance, written by the script) and a note.", lead="GET /audit/results:", after=2,
     keep_next=True)
example_block(json_lines(EXAMPLE_AUDIT_RESULTS))
para("You send one image. It is compared with a small training subset. The answer: the nearest "
     "images (with the optional `phash_distance`), a flag ('possible_copy' or 'not_flagged') and "
     "a note.", lead="POST /audit/check:", after=2, keep_next=True)
example_block(json_lines(EXAMPLE_AUDIT_CHECK))
para("Errors: 400 for a bad upload, 500 for anything else. The format is "
     "`" + json.dumps(EXAMPLE_ERROR_FORMAT) + "`, for example the text "
     "\"The file is not a PNG or JPG image.\"")
h3(APP_A_STUB_HEADING)
num_list(APP_A_STUB_STEPS)

# ===========================================================================
# APPENDIX B. PAGE SPECIFICATIONS (WITH THE REQUIRED WORDING)
# ===========================================================================

# The required wording. Copy these sentences exactly onto the pages.
W_AUDITOR = (
    "The auditor measures similarity between images. It does not decide copyright infringement "
    "or \"legal copying\"."
)
W_SIMILAR = "We say \"similar\" or \"possible copy\". We never say \"copyright infringement\"."
W_UNSEEN = (
    "Detecting images from a generator the detector has never seen is an open research problem. "
    "We will report the drop in performance as a finding. We do not promise the detector will "
    "work on unseen generators."
)
W_ADVERSARIAL = (
    "Adversarial attacks (someone deliberately fooling the detector). We do not test or defend "
    "against them. Not done in version 1."
)
W_HONEST = "Honest reporting: where a tool fails, we show the failure."
W_PLACEHOLDER = "All result cells stay " + PLACEHOLDER + " until a script fills them."
W_TABLES = "Every table cell comes from a script, or is still marked as a placeholder."
W_BANNER = (
    "This detector was trained only on 32x32 CIFAR-10-style images. Results on other photos are "
    "not meaningful."
)
W_COMPARE = (
    "Pixel tools (SSIM, difference map) only make sense for nearly aligned images, for example a "
    "generated image and its nearest training image, or an original and an edited copy. For two "
    "different photos, for example an orange cat and a black cat in different poses, the "
    "difference map will light up almost everywhere; the colour and brightness tables and the "
    "Words panel say more, and the Words panel can be wrong."
)
W_LABEL = "<model> on <dataset>, small model, expect low quality"

APP_B_HEADING = "Appendix B. Page specifications"
APP_B_INTRO = (
    "Each page below lists what it must, should and could show, the endpoint it uses and the "
    "wording it must carry. Copy the quoted sentences exactly. MUST = always keep; SHOULD = if "
    "time allows; COULD = cut first after cloud hosting (cut line, section 1)."
)
APP_B_PAGES = [
    ("Home (/)",
     "What the lab does (generate, detect, compare, audit) and links to the other pages. Wording: "
     "\"" + W_AUDITOR + "\" and \"" + W_HONEST + "\" No endpoint needed."),
    ("Generator (/generator), uses GET /generators and POST /generate",
     "MUST: a model and dataset picker; seed; n; for DDPMs steps; for g-cddpm-fmnist class and "
     "guidance; for the VAEs interpolation between two seeds; a Download PNG button; 'Send to "
     "Detector' and 'Send to Compare'. Label under EVERY image: \"" + W_LABEL + "\"."),
    ("Detector (/detector), uses POST /detect and POST /evidence",
     "MUST: a 'Try an example' strip above the upload box (10 fake CIFAR-style images from our own "
     "generators, committed; 10 real CIFAR-10 images shared by file transfer and never "
     "committed; proposed); the label ('unclear' lies between the two thresholds) and score; the "
     "VAE score; the CNN baseline score; the VAE error-map heatmap; the out-of-scope warning. "
     "SHOULD: the DDPM error map (it replaces the VAE map if Gate 2 passes); Grad-CAM on "
     "cnn-baseline (it explains the CNN, not the forgery); EXIF. COULD: C2PA, ELA, FFT and the "
     "JPEG, blur and noise sliders. Each Evidence item shows its 'does not prove' text, and "
     "every null shows 'not available'. Red banner when `out_of_scope` is true: \"" + W_BANNER
     + "\" Known limitation, say it plainly: the rule only catches image SIZE; a 32x32 photo "
     "that is not CIFAR-style is not flagged. Also: \"" + W_UNSEEN + "\""),
    ("Compare (Distinguisher) (/compare), uses POST /compare",
     "General-purpose: any two photos, with no fixed category. MUST: two uploads; a 'What "
     "should I compare?' text box (the optional `focus`, up to 100 characters, proposed; "
     "empty means the full checklist); the overall SSIM and the difference map; the feature "
     "table (colour mean and histogram distance, brightness, edges, texture, size and the "
     "DOMINANT COLOURS: the three most common colours of each image, found with k-means, with "
     "plain names and shares). SHOULD: the Words panel (a vision API with both images and the "
     "generic attribute checklist listed in Appendix A; the model answers in JSON, may say "
     "'same' or 'unsure' and must not guess; `differences[].part` holds the attribute or part "
     "name; off when there is no key; privacy note: \"When Words is on, photos leave your "
     "computer; do not upload private photos or photos of people.\"). COULD: LPIPS and the "
     "DINOv2-small regions. "
     "Same-subject mode: pixel tools are meaningful when the photos are nearly aligned. "
     "Required line on the page: \"" + W_COMPARE + "\" Worked example, an illustration of the "
     "format, not a result (orange cat vs black cat): dominant colours orange vs black; one "
     "brightness value and one edge-amount value per photo; the difference map lights up almost "
     "everywhere; Words says fur colour differs, and the other attributes are 'same' or "
     "'unsure'."),
    ("Auditor (/auditor), uses GET /audit/results and POST /audit/check",
     "MUST: a dashboard of the runs a-500, a-10000 and a-500-dup (memorization against "
     "training-set size) and the nearest-neighbour grid, drawn from `examples` in GET "
     "/audit/results (it does not need POST /audit/check). Each run audits its OWN small DDPM "
     "(the same architecture as g-ddpm-cifar, trained separately on that run's subset), and the "
     "nearest training images come from the subset that run trained on. SHOULD: the pHash "
     "distance and a side-by-side difference map (the Compare code is reused). COULD: the "
     "upload check. "
     "Wording: \"" + W_AUDITOR + "\" and \"" + W_SIMILAR + "\" " + W_PLACEHOLDER),
    ("Limits and Methods (/limits)",
     "What each feature cannot prove (Appendix D), the model cards (from results and config "
     "files), the sources (Appendix H) and these lines: \"" + W_UNSEEN + "\" \"" + W_ADVERSARIAL
     + "\" \"" + W_HONEST + "\" \"" + W_PLACEHOLDER + "\" \"" + W_TABLES + "\" Also: a shared "
     "architecture or training data between the detector and a generator may make detection "
     "look better than it is. The Detector is trained for 32x32 CIFAR-10-style images only; the "
     "size check only catches image SIZE, so a 32x32 photo that is not CIFAR-style is not "
     "flagged."),
]

h2(APP_B_HEADING)
para(APP_B_INTRO)
for lead, text in APP_B_PAGES:
    para(text, lead=lead + ":", after=4)

# ===========================================================================
# APPENDIX C. GENERATOR CATALOGUE
# ===========================================================================

APP_C_HEADING = "Appendix C. Generator catalogue"
APP_C_INTRO = (
    "All generators are small (about 1M parameters, proposed, to measure) and every image is "
    "labelled \"" + W_LABEL + "\". Tiers: CORE, SHOULD or documented-only."
)
APP_C_HEADERS = ["Id", "Model and dataset", "Tier", "Controls"]
APP_C_ROWS = [
    ["g-vae-cifar", "Conv VAE on CIFAR-10 (also the Detector's fake source G1)", "CORE",
     "seed, n, interpolate (seed_b)"],
    ["g-vae-fmnist", "Conv VAE on Fashion-MNIST", "CORE", "seed, n, interpolate (seed_b)"],
    ["g-cddpm-fmnist", "Class-conditional DDPM + classifier-free guidance on Fashion-MNIST",
     "CORE", "seed, n, steps, class_id, guidance"],
    ["g-ddpm-cifar", "DDPM, cosine schedule, on CIFAR-10 (also the Detector's fake source G2)",
     "SHOULD (Gate 2)", "seed, n, steps (DDIM 20/50/100)"],
    ["g-dcgan-cifar", "DCGAN on CIFAR-10", "documented-only (future work)", "none"],
]
APP_C_WIDTHS = [1.3, 2.7, 1.2, 1.3]
APP_C_NOTES = [
    "Detector-only models (never on the Generator page): d-vae-cifar (the anomaly-score VAE, a "
    "different seed from g-vae-cifar), d-ddpm-cifar (a reconstruction DDPM trained on real "
    "images only, a different seed from g-ddpm-cifar) and cnn-baseline.",
    "G3 (a second DDPM) is removed from the plan because of the Kaggle hours: "
    "leave-one-generator-out becomes 'train on one CIFAR generator, test on the other'. DCGAN is "
    "future work (GANs are not in the course and are risky to tune). CelebA and SVHN are not "
    "used.",
    "Facts (Appendix H): the original DDPM CIFAR-10 model has 35.7M parameters and T=1000 "
    "sampling steps, so ours must be far smaller. DDIM gives quality comparable to 1000 steps "
    "within 20 to 100 steps (DDPM deteriorates rapidly at 10). Higher classifier-free guidance "
    "decreases variety and increases per-sample fidelity. The cosine schedule beats the linear "
    "one for 32x32 and 64x64 images.",
    "CIFAR-10: 32x32 colour, 50,000 train + 10,000 test, no licence stated on its page (to "
    "verify). Fashion-MNIST: 28x28 grayscale, 60,000 train + 10,000 test, MIT licence. The "
    "downloaded PNG carries a provenance note (Pillow PngInfo.add_text, to verify).",
]

h2(APP_C_HEADING)
para(APP_C_INTRO, keep_next=True)
table(APP_C_HEADERS, APP_C_ROWS, APP_C_WIDTHS)
for note in APP_C_NOTES:
    para(note, after=4)

# ===========================================================================
# APPENDIX D. FEATURE CATALOGUE
# ===========================================================================

APP_D_HEADING = "Appendix D. Feature catalogue"
APP_D_INTRO = (
    "Every feature has a sentence on what it must NOT be claimed to prove. The same sentence "
    "goes on the Limits page. Licences are as verified (README or model card), or marked "
    "otherwise. The tiers are proposed."
)
APP_D_HEADERS = ["Feature", "Page, tier", "Library (licence)", "Must NOT be claimed to prove"]
APP_D_ROWS = [
    ["Grad-CAM", "Detector, SHOULD", "pytorch-grad-cam (MIT per README badge only)",
     "It explains the CNN classifier, not the forgery. The target layer must be chosen."],
    ["EXIF", "Detector (Evidence), SHOULD", "Pillow getexif() (licence not verified here)",
     "Missing EXIF is weak evidence (screenshots and platforms strip it; metadata can be "
     "edited). Present EXIF does not mean real."],
    ["C2PA manifest", "Detector (Evidence), COULD",
     "c2pa-python (Apache-2.0 + MIT per README; maturity not stated)",
     "A valid manifest is signed provenance. A missing manifest means NOTHING. Spec 2.2 "
     "confirmed (2.3 exists: confirm current)."],
    ["ELA", "Detector (Evidence), COULD", "Pillow re-save + ImageChops (licence not verified here)",
     "Not proof of manipulation or of AI; works on lossy formats only. Teaching aid only."],
    ["FFT spectrum", "Detector (Evidence), COULD", "numpy FFT + radial average",
     "Does not prove an image is AI-made. Durall et al. and Frank et al. report artifacts "
     "mainly for GAN-type up-convolutions, not for diffusion or edited images. Teaching aid "
     "only."],
    ["Robustness sliders", "Detector, COULD", "none (our own code)",
     "They show how a score moves after JPEG, blur or noise. They are not an attack test."],
    ["SSIM + difference map", "Compare, MUST; Auditor, SHOULD",
     "scikit-image (BSD-3-Clause)",
     "Sensitive to alignment and scale; not a semantic difference. Two photos of different "
     "objects (an orange cat and a black cat) are not aligned, so the map lights up almost "
     "everywhere."],
    ["Dominant colours", "Compare, MUST",
     "numpy + scikit-learn KMeans (licence: to verify); course link: clustering",
     "Three most common colours per image (k-means, k = 3, proposed), with plain names and "
     "shares. Descriptive only: it does NOT prove which object has which colour (background "
     "colours count too)."],
    ["Feature table (colour, brightness, edges, texture, size)", "Compare, MUST",
     "numpy, scikit-image",
     "Descriptive only; it does not name the differing part."],
    ["Words (vision API)", "Compare, SHOULD",
     "Anthropic API by default (multi-image; 10 MB per image limit). Others: OpenAI, Gemini, "
     "SmolVLM-256M (Apache-2.0)",
     "The words can be wrong; quality for fine-grained parts is NOT verified. Photos leave the "
     "computer, so the page shows a privacy note. Provider data-use terms: to verify."],
    ["LPIPS", "Compare, COULD", "lpips (BSD-2-Clause)",
     "A perceptual distance only (higher = more different), not an explanation. It downloads "
     "backbone weights."],
    ["DINOv2-small patch matching", "Compare, COULD",
     "DINOv2-small, 22.1M parameters (Apache-2.0, model card)",
     "Matched points are not guaranteed to be the same part, and unmatched regions are not "
     "guaranteed to be 'the difference'. CPU timing NOT checked."],
    ["pHash / dHash", "Auditor, SHOULD", "imagehash (BSD-2-Clause)",
     "A close hash does not prove copying; a far hash does not prove there is no link. "
     "Threshold and hash size are to tune."],
    ["Occlusion sensitivity", "Idea only", "own code (source not confirmed)",
     "It does not prove which pixels cause the output."],
    ["Watermarks", "COULD (idea only)",
     "invisible-watermark (MIT, dwtDct, experimental)",
     "We could only read OUR OWN marks. The dwtDct mark fails under 50% resize and 30-degree "
     "rotation. Stable Signature (mostly CC-BY-NC) needs decoder fine-tuning: not practical. "
     "SynthID Detector is a Google portal, not an open library; its terms must be checked."],
]
APP_D_WIDTHS = [1.15, 1.0, 1.65, 2.7]

h2(APP_D_HEADING)
para(APP_D_INTRO, keep_next=True)
table(APP_D_HEADERS, APP_D_ROWS, APP_D_WIDTHS)

# ===========================================================================
# APPENDIX E. STARTER PROMPTS
# ===========================================================================

APP_E_HEADING = "Appendix E. Starter prompts"
APP_E_INTRO = (
    "Upload this file to your AI first, then paste your prompt (replace <day> with Mon evening, "
    "Tue, Wed or Thu)."
)
APP_E_NITIN = (
    "I am Nitin, team leader (Part A-ML: backend, models, wiring). This file is my first "
    "message. Today is <day>: use my task card in section 4 and follow section 6A."
)
APP_E_TEJASH = (
    "I am Tejash, the website person (Part B-Web). This file is my first message. Today is "
    "<day>: use my task card in section 4 and follow section 6A."
)
APP_E_PAGE = (
    "I am Tejash. This file is my first message. Help me build the <page name> page of our "
    "existing website against the stub JSON from Appendix A, not against a real API. Use the "
    "page specification in Appendix B and its exact wording, show every null field as 'not "
    "available', and show readable errors. Explain each line in simple words, and follow "
    "section 6A."
)

h2(APP_E_HEADING)
para(APP_E_INTRO)
para(APP_E_NITIN, lead="Prompt for Nitin:")
para(APP_E_TEJASH, lead="Prompt for Tejash:")
para(APP_E_PAGE, lead="Prompt for building one page against a stub:")

# ===========================================================================
# APPENDIX F. THINGS STILL TO VERIFY
# ===========================================================================

APP_F_HEADING = "Appendix F. Things still to verify"
APP_F_ITEMS = [
    "Kaggle: the free GPU limits (about 30 h/week is NOT confirmed; check your own quota page), "
    "the weekly reset day (open conflict C9), TPU hours, and every T4 training time and model "
    "size in MB (measured on Tuesday).",
    "Hosting: free Hugging Face Docker Spaces are not available (confirmed from the official "
    "docs); Render free limits (750 h/month, spins down after 15 min idle; RAM and CPU not "
    "published); Hugging Face CPU Basic figures; Vercel free-plan terms (terms not read).",
    "All epochs, set sizes, t (about 200), the percentile (1st), thresholds, DDIM steps and "
    "parameter counts: proposed, to tune. The rule that `out_of_scope` means 'not exactly "
    "32x32' is proposed.",
    "Datasets: the CIFAR-10 licence (the official page states none); CIFAKE (licence, count "
    "100k vs 120k, image size, Kaggle terms, journal venue); MNIST and CelebA terms (official "
    "pages unreachable); EMNIST terms (readme not read); SVHN is 'for non-commercial use only'. "
    "Neither SVHN, CelebA nor CLIP ViT-L/14 weights are used in the demo.",
    "Citations: the DIRE venue (ICCV 2023, to verify); the Corvi et al. author list; the venues "
    "of Amir et al., DINOv2 and CLIP; the DCGAN ICLR 2016 acceptance.",
    "Libraries: the c2pa-python and pytorch-grad-cam licences (README badges only); c2pa-python "
    "maturity; the Pillow licence; Pillow `PngInfo.add_text`; the C2PA spec version (2.2 "
    "confirmed, 2.3 exists); DINOv2 CPU timing; occlusion sensitivity (no source). The vision "
    "API: data-use terms, key and credits, and quality for fine-grained attributes and parts.",
    "The existing website: its page list, framework, folder or repo link, and whether it uses "
    "Vite. The teacher's answers (section 9); no answer on Words by Wednesday morning means "
    "Words stays off.",
    "Broad-track gaps (documents only: CLIP, LAION, OpenCLIP, GenImage, OpenSDI and Community "
    "Forensics items) and the open conflicts C1 to C12 are listed in `PLAN.md`.",
]

h2(APP_F_HEADING)
num_list(APP_F_ITEMS)

# ===========================================================================
# APPENDIX G. REPRODUCIBILITY CHECKLIST
# ===========================================================================

APP_G_HEADING = "Appendix G. Reproducibility checklist"
APP_G_ITEMS = [
    "Fixed random seeds, set in one place.",
    "A saved config file for every run (epochs, sizes, t, thresholds).",
    "One results/ JSON file per experiment, written by a script.",
    "Every number in any slide or report is traced to a script name.",
    "Package versions pinned in requirements.txt.",
    "No secrets and no datasets in the repo.",
    "Results stay as " + PLACEHOLDER + " until they are real.",
]

h2(APP_G_HEADING)
num_list(APP_G_ITEMS)

# ===========================================================================
# APPENDIX H. SOURCES (verified citations only)
# ===========================================================================

APP_H_HEADING = "Appendix H. Sources"
APP_H_INTRO = (
    "Only these sources may be cited. Check each one before you use it."
)
APP_H_ITEMS = [
    "DDPM: Ho, Jain, Abbeel, arXiv 2006.11239.",
    "DDIM: Song, Meng, Ermon, ICLR 2021, arXiv 2010.02502.",
    "\"Classifier-Free Diffusion Guidance\", Ho and Salimans, arXiv 2207.12598 (NeurIPS 2021 DGM "
    "workshop short version).",
    "\"Improved Denoising Diffusion Probabilistic Models\", Nichol and Dhariwal, arXiv 2102.09672.",
    "\"Auto-Encoding Variational Bayes\", Kingma and Welling, arXiv 1312.6114.",
    "DCGAN: Radford, Metz, Chintala, arXiv 1511.06434 (ICLR 2016 acceptance not confirmed).",
    "DIRE: Wang et al., arXiv 2303.09295 (venue to verify; ICCV 2023).",
    "Carlini et al., \"Extracting Training Data from Diffusion Models\", USENIX Security 2023, "
    "arXiv 2301.13188.",
    "Ojha, Li, Lee, \"Towards Universal Fake Image Detectors that Generalize Across Generative "
    "Models\", CVPR 2023, arXiv 2302.10174.",
    "Corvi et al., ICASSP 2023, arXiv 2211.00680 (author list to verify).",
    "Durall, Keuper, Keuper, \"Watch Your Up-Convolution: CNN Based Generative Deep Neural "
    "Networks Are Failing to Reproduce Spectral Distributions\", CVPR 2020.",
    "Frank, Eisenhofer, Schonherr, Fischer, Kolossa, Holz, \"Leveraging Frequency Analysis for "
    "Deep Fake Image Recognition\", ICML 2020 (PMLR 119).",
    "Grad-CAM: Selvaraju, Cogswell, Das, Vedantam, Parikh, Batra, ICCV 2017, arXiv 1610.02391.",
    "SSIM: \"Image Quality Assessment: From Error Visibility to Structural Similarity\", Wang, "
    "Bovik, Sheikh, Simoncelli, IEEE Trans. Image Processing 13(4):600-612, 2004.",
    "LPIPS: \"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric\", Zhang, "
    "Isola, Efros, Shechtman, Wang, CVPR 2018, arXiv 1801.03924.",
    "\"Deep ViT Features as Dense Visual Descriptors\", Amir, Gandelsman, Bagon, Dekel, arXiv "
    "2112.05814 (venue not confirmed).",
    "DINOv2: Oquab et al., arXiv 2304.07193 (venue not confirmed).",
    "CLIP: Radford et al., arXiv 2103.00020 (venue not confirmed here).",
    "ELA is not a paper: N. Krawetz, 'A Picture's Worth...', Black Hat USA 2007 (title from a "
    "search listing).",
    "CIFAR-10: Krizhevsky 2009 (licence to verify; the official page states none).",
    "CIFAKE: Bird and Lotfi, arXiv 2303.14126 (all facts to verify).",
    "EMNIST: Cohen, Afshar, Tapson, van Schaik (2017).",
]

h2(APP_H_HEADING)
para(APP_H_INTRO)
num_list(APP_H_ITEMS)

# ===== END OF PARTS (the save lines are at the bottom) =====

# Quick self-check: every JSON example must read back as valid JSON.
for example in ALL_EXAMPLES:
    json.loads(json.dumps(example, indent=2))

doc.save(OUT_FILE)
print("saved", OUT_FILE)
