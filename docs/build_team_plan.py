# build_team_plan.py
# ---------------------------------------------------------------------------
# Builds ONE Word file: docs/GenAI_Trust_Lab_Team_Plan.docx
#
# What the file is: the team plan for "GenAI Trust Lab" (two people: Nitin and
# [friend's name]). Each person gives the whole file to their AI assistant as
# the first message. It has the same template as the earlier "Main Plan" file.
#
# How to run (from the project folder):   python docs/build_team_plan.py
#
# How this script is organised:
#   1. SETUP        - fonts, colours and the small helper functions.
#   2. SECTIONS     - one block per section of the document. Each block starts
#                     with the TEXT of the section in named constants (so you
#                     can change the words without touching the code), then
#                     the few lines that write that text into the document.
#   3. SAVE         - the last lines save the file.
#
# Honesty rule for every word in this file: no invented results. Every result
# is the text "[PLACEHOLDER - not measured yet]". Every design number is
# "proposed". The document never says that anything is already built.
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

OUT_FILE = Path(__file__).parent / "GenAI_Trust_Lab_Team_Plan.docx"

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
doc.core_properties.title = "GenAI Trust Lab team plan"
doc.core_properties.last_modified_by = "GenAI Trust Lab team"


# --- small helper functions ------------------------------------------------

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


def code_block(lines, together=True, indent=0.0):
    """Commands or JSON: one green Consolas paragraph per line.
    Indents are leading spaces. No space between lines.
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
            run.font.size = Pt(10)


def table(headers, rows, widths):
    """A plain black grid ('Table Grid'), bold header row, no shading.
    widths are in inches and should add up to 6.5."""
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.autofit = False
    for cell, text in zip(t.rows[0].cells, headers):
        fill(cell.paragraphs[0], text, bold=True)
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


# ===========================================================================
# TITLE, INTRO AND THE BOLD AI PARAGRAPH
# ===========================================================================

TITLE = "GenAI Trust Lab team plan"

INTRO = (
    "Read this whole file once. Then read your own task card in section 4 again before you start. "
    "Dates: Monday 23 Nov: presentation to the professor. Day 1 to Day 4 are Tuesday 24 Nov to "
    "Friday 27 Nov 2026 (proposed, change if your dates differ). Day 4 ends with a demo that runs "
    "twice without help."
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
     "We work in two lanes. Early on Day 1, once Nitin has pushed the repo and invited [friend's "
     "name], you hold a shared contract meeting that fixes `api/contract.json`. After that Nitin "
     "works in `model/`, `results/` and `configs/`, and [friend's name] works in `web/` and "
     "`api/`. Every hand-over is a pull request that the OTHER "
     "person reviews and merges (section 5 shows the exact git steps). Before you start a Day, "
     "READ the other lane's last merged pull request and write 3 lines on what you understood."),
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
    "A detector for AI-made images that will show, for an uploaded image, a score, a VAE score and "
    "a heatmap (the heatmap can be dropped, see the cut line below).",
    "A memorization auditor with a results dashboard drawn from results files, and a small upload "
    "check that will find the nearest training images of an uploaded image (it measures "
    "similarity only).",
    "A small website and API: the website (Home, Detector, Auditor dashboard, Limits) will talk to "
    "a FastAPI API.",
    "Honest results from scripts, plus the presentation and the documents. Every number comes from "
    "a script, or it stays " + PLACEHOLDER + ".",
]
DELIVER_NON_GOALS = (
    "What we do not do: no adversarial attacks and no video in version 1. The roadmap items "
    "(video, a browser extension, a WhatsApp bot, differential privacy) are ideas, not promises. "
    "Detection on generators the detector has not seen is an open problem. We report any drop in "
    "performance as a finding and we do not promise the detector will work on them."
)
DELIVER_CUT_LINE = (
    "Cut line, if time runs out: drop POST /audit/check (the upload check) first, then the "
    "diffusion heatmap. Always keep the CNN baseline, the VAE score and the results dashboard."
)

h2(DELIVER_HEADING)
para(DELIVER_MINIMUM)
for item in DELIVER_ITEMS:
    para(item, bold=True)
para(DELIVER_NON_GOALS)
para(DELIVER_CUT_LINE)

# ===========================================================================
# 2. WHO DOES WHAT
# ===========================================================================

WHO_HEADING = "2. Who does what"
WHO_HEADERS = ["Part", "What", "Owner", "Laptop", "Day"]
WHO_ROWS = [
    ["A-ML",
     "VAE, CNN baseline, DDPMs, fakes, features, classifier, auditor, experiments, results "
     "JSON files, the real model behind /detect",
     "Nitin", "RTX 3050 6GB laptop + Kaggle T4 (limits: to verify)", "1 to 4"],
    ["B-Web",
     "`web/` (React + Vite) with four pages; `api/` (FastAPI): the stub first, then the real "
     "model; Hugging Face and Vercel deploys",
     "[friend's name]", "RTX 3050 6GB laptop", "1 to 4"],
    ["Lead",
     "Repo, `PLAN.md`, `docs/`, integration, professor contact, dry run, feature freeze, "
     "assembly",
     "Nitin (leader)", "RTX 3050 6GB laptop", "1 to 4"],
]
WHO_WIDTHS = [0.7, 2.75, 1.0, 1.35, 0.7]

PROGRESS_SENTENCE = (
    "Progress board. The owner of each Day ticks their own box when their pull request is merged."
)
PROGRESS_LINES = [
    "[ ] Day 1: Nitin: repo pushed, contract agreed, `data/splits.csv`, VAE and CNN code",
    "[ ] Day 1: [friend's name]: `web/` skeleton, FastAPI stub (/health, /detect)",
    "[ ] Day 2: Nitin: TinyUNet for 3 channels, detector DDPM, G2, G3, G1/G2/G3 fakes, configs",
    "[ ] Day 2: [friend's name]: Detector page, error states, Hugging Face test, 'waking up'",
    "[ ] Day 3: Nitin: features, classifier, auditor, experiments, results JSON, real /detect",
    "[ ] Day 3: [friend's name]: Auditor dashboard, real API, CORS, required wording",
    "[ ] Day 4: Nitin: robustness tests, tables from scripts, fresh-clone test, rehearsals",
    "[ ] Day 4: [friend's name]: Limits page, integration, Vercel deploy, rehearsals",
]
GATE_LINES = [
    "Gate 1 (Day 1): the contract is agreed and the stub answers /health and /detect on a laptop",
    "Gate 2 (Day 3 evening): results frozen",
    "Gate 3 (Day 4): the demo runs twice without help",
]

h2(WHO_HEADING)
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
    ["Mon 23 Nov", "",
     "Presentation to the professor. If possible, ask the questions in section 9 in writing."],
    ["Tue 24 Nov (Day 1, proposed)", "Both, then two lanes",
     "Early on Day 1, once Nitin has pushed the repo and invited [friend's name], you hold a "
     "shared contract meeting that fixes `api/contract.json`. Then Nitin: data, VAE, CNN. "
     "[friend's name]: clone, skeleton, stub. Gate 1 (evening): the contract is agreed and the "
     "stub answers /health and /detect on a laptop."],
    ["Wed 25 Nov (Day 2, proposed)", "Two lanes",
     "Nitin: detector DDPM, G2, G3, the fakes, the CPU-sampler decision. [friend's name]: "
     "Detector page, error states, Hugging Face Docker Space test (Render as fallback)."],
    ["Thu 26 Nov (Day 3, proposed)", "Two lanes",
     "Nitin: features, classifier, auditor, experiments, results JSON, real model behind "
     "/detect. [friend's name]: Auditor dashboard, real API, CORS, required wording. Gate 2 "
     "(evening, proposed 21:00): results frozen, final weights shared outside git, all figures "
     "saved, the results JSON files in results/ complete. Day 4 robustness tests only add files."],
    ["Fri 27 Nov (Day 4, proposed)", "Both",
     "Nitin: robustness tests, tables from scripts, fresh-clone test. [friend's name]: Limits "
     "page, integration, Vercel deploy. Feature freeze (proposed 15:00). Gate 3 (proposed "
     "17:00): the demo runs twice without help."],
]
CALENDAR_WIDTHS = [1.2, 0.9, 4.4]

CALENDAR_LATE_RULE = (
    "Times for each Day are guesses. If the real API is more than 1 hour late, the web side keeps "
    "using the stub. The late person still finishes their part and keeps the credit."
)
CALENDAR_SLIP_RULE = (
    "If Day 2 slips: G3 (the second DDPM) moves to Day 3 morning, the CPU-sampler decision moves to "
    "Day 3, and training the VAE and making the G1 (VAE) fakes move to Day 1 evening."
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
    "Each person has ONE card with four blocks, Day 1 to Day 4. 'Be ready to explain:' lists our "
    "guess at the questions the professor may ask. Files and folders named in this file (for "
    "example `model/`, `api/`, `docs/ai_usage.md`) are the plan: Nitin creates the repo and the "
    "`docs/` files on Day 1; each person creates the folders of their own lane (Nitin: `model/`, "
    "`results/`, `configs/`; [friend's name]: `web/`, `api/`)."
)

# --- team settings --------------------------------------------------------
SETTINGS_HEADING = "Team settings (so the comparison is fair, change only by agreement)"
SETTINGS_INTRO = (
    "Every value below is proposed, agree on Day 1. Write any change in both logs."
)
SETTINGS_LINES = [
    "Images: 32x32 colour (CIFAR-10); two classes, real and fake. The L2 metric (the straight-line "
    "distance between the pixel values of two images) works at 28x28 or 32x32, and every results "
    "file states the size used. (proposed, agree on Day 1)",
    "Split file `data/splits.csv` (seed 42): it will cut the CIFAR-10 test set into a calibration "
    "split of 5,000 images and a final test split of 5,000 images (both numbers proposed, agree on "
    "Day 1). The SAME file for both of us.",
    "Normalisation: scale pixels to [-1, 1], as in the course notebooks. (proposed, agree on Day 1)",
    "Batch size 128, Adam, learning rate 2e-4 (as in the course notebooks); epochs are decided on "
    "Day 2 from the measured seconds per epoch. (all proposed, to tune, agree on Day 1)",
    "Seed 42, set in ONE file: `model/utils.py`. The detector DDPM gets its OWN seed, different "
    "from G2 and G3. (proposed, agree on Day 1)",
    "One config JSON per run in `configs/` (epochs, sizes, the timestep t, thresholds, seed, image "
    "size). (proposed, agree on Day 1)",
    "One results JSON per experiment in `results/`, written by a script. Detector files: name, "
    "generator, metric, value, seed, config path. Auditor files: the /audit/results shape "
    "(Appendix A) plus seed, config path and image size. (proposed, agree on Day 1)",
    "AI-made threshold: start with 0.5 and tune it on the calibration split only. Uploads: PNG or "
    "JPG up to 5 MB; the server will resize them to 32x32. (proposed, agree on Day 1)",
    "Model weights: under about 100 MB in total. (proposed, agree on Day 1)",
    "Epoch choice: keep the saved model with the best calibration-split score. (proposed, agree "
    "on Day 1)",
    "Augmentation: horizontal flip only, and only in the auditor's mitigation experiment. "
    "(proposed, agree on Day 1)",
    "The website will read the API address from `VITE_API_URL`. (proposed, agree on Day 1)",
]
# The four generic lines (same words as our earlier team plan template):
SETTINGS_RULE_VALIDATION = (
    "Decisions are made on the validation set. The test set is scored once per finished run and "
    "never used to choose anything."
)
SETTINGS_RULE_CALIBRATION = (
    "In our project the calibration split plays the role of the validation set. Start with the "
    "0.5 threshold and tune it, together with the 1st-percentile rule and other thresholds, on the "
    "calibration split ONLY. The final test split is scored once per finished run. This replaces the idea in "
    "`PLAN.md` of calibrating on the plain test set."
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


def day_block(block):
    """One Day block of a card: bold title, numbered steps, Done when, Be ready to explain."""
    para(block["title"], bold=True, keep_next=True)
    if block.get("intro"):
        para(block["intro"], keep_next=True)
    num_list(block["steps"])
    para(block["done"], lead="Done when:")
    para(block["explain"], lead="Be ready to explain:", after=10)


# --- Nitin's card ----------------------------------------------------------
NITIN_TITLE = "Nitin (leader + ML, GPU laptop + Kaggle)"
NITIN_GOAL = (
    "the repo, the data and the models (VAE, DDPMs, CNN baseline), then the detector, the auditor "
    "and the results files that the website shows. You pull from: nothing on Day 1 (you start the "
    "repo). From Day 2 on: `api/contract.json` and [friend's name]'s last merged pull request."
)
NITIN_LEADER_JOBS = (
    "Leader jobs, every Day: read both logs; resolve merge conflicts; be told BEFORE a pull "
    "request if AI use is not reported; talk to the professor; call the three gates."
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
    "__pycache__/",
    ".venv/",
]
NITIN_FIRST_COMMIT_LINES = [
    "git init",
    "git add .",
    "git commit -m \"Day 1: first commit\"",
    "git branch -M main",
    "git remote add origin <REPO URL - Nitin adds on Day 1>",
    "git push -u origin main",
]
NITIN_DAYS = [
    {
        "title": "Day 1: repo, contract, data and first models (Tue 24 Nov, proposed)",
        "intro": (
            "Before you start, make sure Git, Python 3, a GitHub account and a Kaggle account are "
            "ready (check each site for current instructions)."
        ),
        "steps": [
            {"text": "Create `.gitignore` in the project folder with these lines (one per line). It "
                     "blocks datasets, fake images, weights and secret files. `data/splits.csv` is "
                     "the one data file allowed in git.",
             "code": NITIN_GITIGNORE_LINES},
            "Create `README.md` (setup steps, a contribution table with `<GitHub username>` per "
            "person, a results section showing `[PLACEHOLDER - not measured yet]`), "
            "`docs/ai_usage.md` (with the table header from section 6A), `docs/explain.md` (for the "
            "'be ready to explain' questions) and one log file `docs/log/<yourname>.md` per person "
            "(with the six parts from section 6). Write the versions of your Python packages in "
            "`requirements.txt` (see `pip list`).",
            {"text": "Make the first commit. First create an empty repository on GitHub and copy its "
                     "address. Run `git status` after `git add .` and check that no data, weights or "
                     "secret files are listed. This very first push goes straight to `main`, because "
                     "nobody can review it yet. Every later change is a pull request.",
             "code": NITIN_FIRST_COMMIT_LINES},
            "Invite [friend's name] as a collaborator (GitHub: Settings, Collaborators) and tell them "
            "the repo address. Write their `<GitHub username>` in the README table in your next pull "
            "request.",
            "Contract meeting with [friend's name], early on Day 1, once you have pushed the repo and "
            "invited them. Agree the four endpoints, the error format and the defaults in Appendix "
            "A, and write them into `api/contract.json` on a new branch (section 5). Also agree how "
            "`api/` calls your model: one function in `model/detector.py` that returns the /detect "
            "answer (its name is agreed in the meeting). Open a pull request for the file; "
            "[friend's name] reviews and merges it.",
            "Create your folders `model/`, `results/` and `configs/`. Write `model/utils.py` "
            "(set_seed(42), the choice of device, the folder paths) and `model/data.py`: a CIFAR-10 "
            "loader that scales pixels to [-1, 1], and a function that writes `data/splits.csv` "
            "(seed 42) by cutting the CIFAR-10 test set into 5,000 calibration images and 5,000 "
            "final test images (proposed). Print the counts and check that no image is in both "
            "splits. Datasets are never stored in the repo.",
            "Write `model/vae.py`: a small convolutional VAE (a few Conv layers down to a latent "
            "vector, then back up). The loss is the reconstruction error plus the KL term (how far "
            "the rebuilt image is from the original, plus how odd the squeezed numbers look). The "
            "course VAE is an MLP with a 2-D latent and computes no anomaly score, so the VAE score "
            "(reconstruction error plus KL for one image) is NEW work.",
            "Write `model/cnn.py`: the CNN baseline, based on LetterCNN from the conditional "
            "diffusion notebook. It looks at the raw 3-channel image and answers real or fake. "
            "Keep it small (3 conv layers, proposed).",
            "Run the VAE and the CNN on a few images (for example 8) and print the output shapes "
            "and the loss. The course notebooks use Colab or Kaggle paths and CSV files that are "
            "not in this project, so fix the paths.",
        ],
        "done": (
            "the contract is in `api/contract.json` and agreed by both of you, and the stub answers "
            "/health and /detect on a laptop (Gate 1); `data/splits.csv` has the proposed sizes and "
            "the no-overlap check prints OK; `requirements.txt` lists your package versions; the VAE "
            "and CNN code run on a few images; your pull requests have the 3-line hand-over note."
        ),
        "explain": (
            "What is a split file for, and why is there a calibration split? What are the two parts "
            "of the VAE loss? Why does the seed live in ONE file? Why do we run `git status` before "
            "the first commit?"
        ),
    },
    {
        "title": "Day 2: detector DDPM, G2, G3 and the fakes (Wed 25 Nov, proposed)",
        "steps": [
            "Measure the real seconds per epoch FIRST, before any DDPM training: run one epoch of "
            "your VAE on your laptop and on Kaggle (T4; free limits: to verify), write the numbers "
            "in your log and tell [friend's name]. Use them to choose the epochs for every run "
            "(cut a run that would be too slow and write that in its config). Measure the DDPM "
            "network too, once step 2 has adapted it.",
            "Copy TinyUNet (a small network that learns to remove noise), the timestep embedding "
            "(it tells the network how noisy the picture is), the noise schedule (it says how much "
            "noise is added at each step) and the train and sample loops from the Day 6 diffusion "
            "notebook into `model/ddpm.py`. Change TinyUNet to 3 colour channels and fix the "
            "paths. The schedule values come from the course notebook (T=1000, linear betas 1e-4 "
            "to 0.02; to tune).",
            "Train your small convolutional VAE from `model/vae.py` on the real CIFAR-10 training "
            "images (a short run; it can be on Kaggle). Keep its weights outside git (location: "
            "<Nitin adds>).",
            "Train the detector DDPM on Kaggle on real CIFAR-10 images ONLY, with its OWN seed. Keep "
            "its weights outside git (location: <Nitin adds>). If the 32x32 DDPM does not train in "
            "time, the fallbacks are smaller models, fewer epochs, or Fashion-MNIST (28x28 "
            "grayscale, Zalando, MIT licence); write the choice in the config.",
            "Train G2 (a small DDPM on real CIFAR-10) and G3 (DDPM-B: a second DDPM with different "
            "epochs or size, proposed). They must be different models, with different seeds, from "
            "the detector DDPM.",
            "Make the fakes: G1 from your trained VAE, G2 and G3 from their DDPMs (if Day 2 runs "
            "late, train the VAE and make the G1 fakes on Day 1 evening, see section 3). The number "
            "of fakes per generator is to tune (write it in the config). Save them outside git.",
            "Decide the CPU sampler (how many denoising steps to run on a slow computer): few "
            "denoising steps, or a deterministic sampler? Time one image and write the decision and "
            "the time in your log. It stays open until you measure.",
            "Save one config JSON per run in `configs/` (epochs, sizes, the timestep t, thresholds, "
            "seed, image size).",
        ],
        "done": (
            "the seconds per epoch are in your log; the VAE, the detector DDPM, G2 and G3 have "
            "trained and their weights are saved outside git; fake images from G1, G2 and G3 are "
            "saved outside git; there is one config JSON per run; your pull request has the 3-line "
            "hand-over note."
        ),
        "explain": (
            "What does the noise schedule do (what are T and the betas)? Why must the detector DDPM "
            "be a DIFFERENT model, with its own seed, from G2 and G3? Why is it trained on real "
            "images only? Why do you measure the seconds per epoch before you choose the epochs?"
        ),
    },
    {
        "title": "Day 3: detector, auditor and results (Thu 26 Nov, proposed)",
        "intro": (
            "Until the real weights exist, build and test your functions on a quick dummy model so "
            "you are never idle."
        ),
        "steps": [
            "Write `model/features.py`: the reconstruction-error feature (the DIRE idea, Wang et "
            "al., arXiv 2303.09295; the ICCV 2023 venue is to verify). Add noise up to timestep t "
            "(about 200 of 1000, proposed), denoise with the detector DDPM, and take the per-pixel "
            "error (a map showing where the repaired image differs from the original). That error "
            "image is the heatmap. Summarise it as the mean and a few percentiles (which ones: "
            "agree and write them in the config).",
            "Add the VAE score: the reconstruction error plus the KL term from `model/vae.py`, one "
            "number per image.",
            "Write `model/detector.py`: a logistic regression (scikit-learn) on the features. Real "
            "images come from CIFAR-10, fakes from G1, G2 and G3 (counts to tune). Start with the "
            "0.5 threshold and tune it, together with the other settings, on the calibration split "
            "only. Keep the CNN baseline as a second detector.",
            "Leave-one-generator-out: train the detector on two of G1, G2, G3 and test on the third; "
            "repeat 3 times. Report accuracy and AUROC (a score where 0.5 is guessing and 1.0 is "
            "perfect) per generator. If the score drops on the unseen generator, report the drop as "
            "a finding. Do not hide it.",
            "Write `model/auditor.py`: for each generated image, the L2 distance in pixel space to "
            "its nearest training image. The 'normal closeness' comes from the calibration split "
            "(its distances to the nearest training image). Flag 'possible copy' if a generated "
            "image is closer than the 1st percentile of those distances (proposed, to tune). "
            "`memorization_rate` is the share of generated images that are flagged. Optional "
            "cross-check: a Gaussian fit, or a 2-component GMM (a mix of two bell curves) only if "
            "you see two clear bumps; the percentile stays the main rule. Use the words 'similar' "
            "and 'possible copy', never 'copyright infringement'. Cite Carlini et al. (arXiv "
            "2301.13188).",
            "Write `model/run_experiments.py`: the experiments on training-set size (500, 2,000, "
            "10,000; proposed), on deliberately duplicated training images, and on epochs. Train on "
            "Kaggle and keep the models small. If time allows, try the fixes: remove duplicates, "
            "flip augmentation, fewer epochs, smaller training sets.",
            "Write the results JSON files in `results/` with a script (the shapes are in the team "
            "settings and in Appendix A). Put the real model behind /detect: one function in "
            "`model/detector.py` returns the score, the VAE score and the heatmap, and [friend's "
            "name] connects `api/` to it. At Gate 2 save the final weights, share them outside git "
            "(location: <Nitin adds>) and tell [friend's name].",
        ],
        "done": (
            "`results/` has one JSON file per experiment, each written by a script and with its "
            "config path; /detect returns a real score from a trained model when [friend's name] "
            "calls it; every number can be traced to a script name; the final weights are shared "
            "outside git (Gate 2); your pull request has the 3-line hand-over note."
        ),
        "explain": (
            "What is AUROC? Why do we use a percentile for the cut-off? What does 'similar' mean "
            "here, and why is the auditor not a legal tool? What is leave-one-generator-out for?"
        ),
    },
    {
        "title": "Day 4: robustness, tables and the rehearsals (Fri 27 Nov, proposed)",
        "steps": [
            "Robustness tests: score the detector again after JPEG compression, blur and added noise "
            "(the exact settings are proposed; write them in the config). One results JSON per "
            "test. These add files; they do not change frozen ones.",
            "Fill every table and figure from scripts: the README results, the presentation numbers, "
            "memorization against training-set size and against epochs, the distance histograms "
            "(generated against held-out), and a grid of generated images beside their nearest "
            "training images. A cell with no real result keeps `[PLACEHOLDER - not measured yet]`.",
            "Review [friend's name]'s pull request with the reviewer checklist in section 5.",
            "Fresh-clone test: clone the repo into a new folder, follow only the README and make "
            "sure it runs. Before you make the repo public: first check that no secret, `.env` "
            "file, dataset or weights file is in the history. Then make it public when you are "
            "ready (read the teacher's answers in section 9 first).",
            "Call the feature freeze (proposed 15:00). Rehearse the demo twice with [friend's name], "
            "without help from anyone. Fill in the AI-usage paragraph (section 7) and the "
            "contribution table.",
            "Extra: only after steps 1 to 5, try CIFAKE as an extra test set of an unseen "
            "generator (Bird and Lotfi, arXiv 2303.14126). First verify its licence, count (100k vs "
            "120k), image size and the Kaggle terms. Never show it on the website.",
        ],
        "done": (
            "every table cell comes from a script or still shows `[PLACEHOLDER - not measured "
            "yet]`; the fresh-clone test passes; the demo runs twice without help (Gate 3); your "
            "log and the AI-usage rows are up to date."
        ),
        "explain": (
            "Why do JPEG, blur and noise matter for a detector? Why do we report the drop on an "
            "unseen generator instead of hiding it? Why can a shared architecture or shared "
            "training data between the detector and a generator make detection look better than it "
            "is? Why do we check the git history before the repo goes public?"
        ),
    },
]

h3(NITIN_TITLE)
para(NITIN_GOAL, lead="Goal:", bold=True)
para(NITIN_LEADER_JOBS)
for block in NITIN_DAYS:
    day_block(block)

# --- [friend's name]'s card -----------------------------------------------
rule()
FRIEND_TITLE = "[friend's name] (web, laptop)"
FRIEND_GOAL = (
    "a small website and an API stub that both of us can use from Day 1, then the real model "
    "behind it. You pull from: `api/contract.json` (agreed on Day 1) and the other lane's last "
    "merged pull request (Nitin's). Read it and write 3 lines on what you understood."
)
FRIEND_DAYS = [
    {
        "title": "Day 1: skeleton, contract and stub (Tue 24 Nov, proposed)",
        "steps": [
            "Install Git, Node.js (LTS) and Python 3, and create a GitHub, Hugging Face and Vercel "
            "account (check each site for current instructions).",
            "Read sections 0 and 6A of this file, then tell your AI your name and that you are on "
            "Day 1. Nitin creates your log file `docs/log/<yourname>.md` on Day 1 (the six parts "
            "are in section 6); you edit only your own file.",
            {"text": "Clone the repo once Nitin has pushed it and invited you (accept the GitHub "
                     "invitation first). Then make your own branch (section 5).",
             "code": ["git clone <REPO URL - Nitin adds on Day 1>"]},
            "Create the website skeleton in `web/` with Vite + React (check the Vite docs for the "
            "command that makes a new Vite + React project). Make empty pages for Home, Detector, "
            "Auditor dashboard and Limits. Read the API address from the environment variable "
            "`VITE_API_URL` (keep it in a local `.env` file that git ignores).",
            "Contract meeting with Nitin, early on Day 1, once he has pushed the repo and invited "
            "you. Agree the four endpoints, the error format and the defaults in Appendix A. Nitin "
            "writes them into `api/contract.json` and you review and merge that pull request. "
            "Website rules among the defaults: the Detector page works when the heatmap is null, "
            "the upload check is hidden on a 404, and uploaded images are not stored.",
            "Write the FastAPI stub in `api/`: fixed answers for the four endpoints, in the shape of "
            "`api/contract.json`. The numbers must be the same every time, and the `note` field "
            "must say 'stub'. Errors use `{\"error\": \"<text>\"}` with status 400 for a bad upload "
            "and 500 for anything else. Write down the package versions you use in "
            "`api/requirements.txt` (the final versions are decided on Day 2). Appendix A shows the "
            "answers and the commands.",
            "Run it on your laptop. Open /health in a browser. Test /detect with the interactive "
            "page FastAPI provides at /docs (use 'Try it out' and upload an image). Point "
            "`VITE_API_URL` at it and check that the web skeleton gets the answer.",
            "Open your pull request (web skeleton and stub) with the 3-line hand-over note and the "
            "checklist from section 5. Nitin reviews and merges it.",
        ],
        "done": (
            "`api/contract.json` is agreed by both of you; the stub answers /health and /detect on "
            "your laptop (Gate 1); `web/` starts with `npm run dev` and reads `VITE_API_URL`; Nitin "
            "has reviewed and merged your pull request, and it has the 3-line hand-over note."
        ),
        "explain": (
            "What is a stub and why do we use one? What is CORS? What is an environment variable "
            "like `VITE_API_URL`, and why not write the address in the code? What does `git clone` "
            "do?"
        ),
    },
    {
        "title": "Day 2: Detector page and the hosting test (Wed 25 Nov, proposed)",
        "steps": [
            "Detector page: upload an image (PNG or JPG up to 5 MB, proposed; with a preview), call "
            "POST /detect, and show the score (0 to 1, higher = more likely AI-made; the 0.5 "
            "threshold is a proposed starting value), the VAE score and the heatmap. The page MUST "
            "also work when the heatmap is null: show a short message instead.",
            "Error states: a wrong file type, a file over the size limit, an `{\"error\": ...}` "
            "answer from the API, and an API that cannot be reached. Always show a readable "
            "message, never a blank page.",
            "Hugging Face deploy test: put the stub in a Docker Space (FastAPI). The free CPU Basic "
            "tier is 2 vCPU and 16 GB RAM and it sleeps when unused (to verify). Decide the "
            "Dockerfile port and the final package versions today, then test the public URL from "
            "the website.",
            "Render fallback: if the Hugging Face test fails, try Render's free web service (limits "
            "to verify). Decide, write the final API address in your log and tell Nitin.",
            "'Waking up' message: if the API does not answer, show \"The server is waking up, please "
            "wait\" and wait up to 60 seconds (proposed) before showing an error.",
            "Open your pull request with the 3-line hand-over note. Nitin reviews and merges it.",
        ],
        "done": (
            "the Detector page shows the stub's score, VAE score and heatmap and also works when "
            "the heatmap is null; every error state shows a readable message; the stub answers at a "
            "public URL (Hugging Face, or Render as the fallback); the 'waking up' message appears "
            "when the API is asleep; your pull request has the 3-line hand-over note."
        ),
        "explain": (
            "Why is the heatmap sent as base64 text? What is a cold start, and why does a free "
            "Space have one? Why must the Detector page work when the heatmap is null? Why do we "
            "show a 'waking up' message instead of an error?"
        ),
    },
    {
        "title": "Day 3: Auditor dashboard and the real API (Thu 26 Nov, proposed)",
        "steps": [
            "Auditor dashboard: draw one chart for each experiment in GET /audit/results (the x "
            "values on the x axis, named by x_label, and memorization_rate on the y axis). Use a "
            "small chart library and keep it simple. Show the note text under the charts. Build it "
            "against the stub's answer first.",
            "The upload check (POST /audit/check): show the nearest images (k = 3, proposed) with "
            "their distances and the flag 'possible copy' or 'not flagged'. Hide this whole section "
            "when the endpoint returns 404, because it is first on the cut line.",
            "Switch to the real API: when Nitin says the real model is ready, connect `api/` to his "
            "function in `model/detector.py` and point `VITE_API_URL` at the real API. Keep the stub "
            "so that you can go back to it.",
            "CORS: in the deployed API allow the Vercel domain only (allow localhost only on your "
            "own laptop, for tests).",
            "The required wording: put the sentences from Appendix B on the right pages, exactly as "
            "written.",
            "Open your pull request with the 3-line hand-over note. Nitin reviews and merges it.",
        ],
        "done": (
            "the dashboard draws charts from /audit/results (first the stub's file, then Nitin's "
            "real files); the upload check disappears when /audit/check returns 404 and nothing "
            "else breaks; /detect on the website returns the real model's answer; CORS allows only "
            "the Vercel domain; the Appendix B wording is on the pages; your pull request has the "
            "3-line hand-over note."
        ),
        "explain": (
            "How does the chart map the JSON (what is x, what is y, what is x_label)? Why do we hide "
            "a section when an endpoint returns 404? What does CORS protect? What changes for the "
            "website when the stub is replaced by the real API?"
        ),
    },
    {
        "title": "Day 4: Limits page, integration and the demo (Fri 27 Nov, proposed)",
        "steps": [
            "Limits page: what the tools cannot do and what is still open. Detection on unseen "
            "generators is open and any drop is reported as a finding; the auditor measures "
            "similarity and not legal copying; no adversarial attacks and no video in version 1; "
            "roadmap items are ideas. Use the Appendix B wording.",
            "Integration with Nitin: run the whole website against the real API on both laptops. "
            "Try a bad file, a file over the size limit, and an API that is asleep. Write down what "
            "fails and fix it.",
            "Placeholders: every results cell on the site that has no real script result shows "
            "`[PLACEHOLDER - not measured yet]`. Where a tool fails, show the failure.",
            "Vercel deploy: deploy `web/` to Vercel (free-plan terms to verify), set `VITE_API_URL` "
            "in the Vercel settings and test the public site.",
            "Rehearse the demo twice with Nitin, without help from anyone (Gate 3).",
            "Extra: polish, only after steps 1 to 5 are finished: the colours and fonts (Appendix A "
            "lists the proposed ones) and small layout fixes.",
        ],
        "done": (
            "the demo runs twice without help (Gate 3); every table cell on the site comes from a "
            "script or shows the placeholder; the Limits page has the Appendix B wording; the "
            "Vercel site works with the real API; your log is up to date."
        ),
        "explain": (
            "Why is the auditor not a legal tool? Why do we show placeholders instead of guessing "
            "numbers? What do you do if the API is asleep during the demo? Where do you set the API "
            "address for the Vercel site, and why there?"
        ),
    },
]

h3(FRIEND_TITLE)
para(FRIEND_GOAL, lead="Goal:", bold=True)
for block in FRIEND_DAYS:
    day_block(block)

# --- the website -----------------------------------------------------------
rule()
WEBSITE_HEADING = "The website"
WEBSITE_TEXT = (
    "[friend's name] owns the web lane: the website in `web/` and the API in `api/`. The website "
    "will have four pages: Home, Detector (upload an image, see the score, the VAE score and the "
    "heatmap), Auditor dashboard (charts from GET /audit/results, plus the small upload check if "
    "it is kept) and Limits. Nitin reviews every web pull request, so both of us can explain the "
    "whole site."
)
h3(WEBSITE_HEADING)
para(WEBSITE_TEXT)

# ===========================================================================
# 5. HOW WE HAND OVER
# ===========================================================================

HANDOVER_HEADING = "5. How we hand over (every Day)"
HANDOVER_GET_LATEST = "Get the latest work:"
HANDOVER_GET_LATEST_CMDS = ["git checkout main", "git pull"]
HANDOVER_BRANCH = "Make your own branch (example for [friend's name]'s Day 1):"
HANDOVER_BRANCH_CMDS = ["git checkout -b d1-web-stub-friend"]
HANDOVER_SAVE = "Do your work. Then save and upload it (use your own file names):"
HANDOVER_SAVE_CMDS = [
    "git add web/ api/ docs/log/<yourname>.md",
    "git commit -m \"Day 1: web skeleton and API stub\"",
    "git push -u origin d1-web-stub-friend",
]
HANDOVER_PR_TEXT = (
    "Open a pull request on GitHub. In the description write 3 lines (and paste the checklist from "
    "below this list under them): what I did, what the other person must read first, what the "
    "other person does. Also write 3 lines on what you understood from the other lane's last "
    "merged pull request."
)
HANDOVER_REVIEW_LINES = [
    "The other person reviews and merges it. Nobody merges their own work (only Nitin merges the "
    "very first one).",
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
    "Nitin creates your file from the template on Day 1. You edit ONLY your own file. It has six "
    "parts:"
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
    "and Diffusion Models). The project builds an AI-image detector and a memorization auditor, "
    "with a small website, using PyTorch. Our team rule requires that every use of AI is "
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
     "lane is `model/`, `results/` and `configs/`; [friend's name]'s lane is `web/` and `api/`. "
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
    "Student: <name> (Part <A-ML or B-Web>, Day <number>)",
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
    "If the professor asks for a written report, this is who writes what (we do not know yet "
    "whether one is needed, see section 9). The presentation and notes follow the same split."
)
WRITE_HEADERS = ["Section", "Writer"]
WRITE_ROWS = [
    ["Detector section and auditor section; abstract; conclusion; contribution table and repo "
     "link; AI-usage paragraph; final assembly", "Nitin"],
    ["Website and API section; wording of the Limits page", "[friend's name]"],
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
para(WRITE_INTRO)
table(WRITE_HEADERS, WRITE_ROWS, WRITE_WIDTHS)
para(WRITE_AI_LEAD, bold=True, keep_next=True)
para(WRITE_AI_TEXT, mono_all=True, ticks=False)

# ===========================================================================
# 8. IF SOMETHING GOES WRONG
# ===========================================================================

WRONG_HEADING = "8. If something goes wrong"
WRONG_ITEMS = [
    ("The real API is late by an hour or more:",
     "the web side keeps using the stub; the late person still finishes and keeps the credit."),
    ("Hugging Face fails:",
     "switch to Render (free web service, limits to verify), tell Nitin, and change `VITE_API_URL` "
     "to the new address."),
    ("The Space is asleep:",
     "a free Space sleeps when unused (to verify), so the first call can be slow. The site shows "
     "'waking up' and waits up to 60 seconds (proposed). Open the API address a few minutes "
     "before a demo."),
    ("A CORS error in the browser:",
     "the API does not allow the address of the website. Check the Vercel domain in the API "
     "(localhost only on your own laptop) and the spelling of `VITE_API_URL`."),
    ("The model is not ready:",
     "keep the stub, and say so: the `note` field says 'stub' and the page shows it. Never show a "
     "stub answer as a result."),
    ("We run out of time:",
     "use the cut line (section 1): drop POST /audit/check first, then the diffusion "
     "heatmap."),
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
    "Do you need a written report from us, and what is the deadline?",
    "Is it fine to use AI tools for code and explanations, as long as we disclose it (AI rows in "
    "the repo and a paragraph in the report) and every member can explain every line?",
    "Are marks individual, team, or both?",
    "What is the deadline for the working version?",
    "Is it fine to use Kaggle for training and the CIFAKE dataset as an extra test set?",
    "Can we show results tables where some cells still say " + PLACEHOLDER + "?",
    "Can the presentation on Monday 23 Nov be given with results that are still marked "
    + PLACEHOLDER + "?",
]

h2(QUESTIONS_HEADING)
for q in QUESTIONS:
    para(q, after=2)

# ===========================================================================
# APPENDIX A. API CONTRACT AND STUB INSTRUCTIONS
# ===========================================================================

APP_A_HEADING = "Appendix A. API contract and stub instructions"
APP_A_INTRO = (
    "Everything in this appendix is proposed, agree on Day 1 with Nitin at the contract meeting. "
    "Nitin writes the agreed version into `api/contract.json`. The examples are real JSON, so you "
    "can copy them."
)

# A short piece of base64 text, cut short on purpose (real answers are much longer).
SHORT_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB..."

EXAMPLE_HEALTH = {"status": "ok"}
EXAMPLE_DETECT = {
    "score": 0.73,
    "vae_score": 0.41,
    "heatmap_png_base64": SHORT_BASE64,
    "note": "stub: fixed example answer",
}
EXAMPLE_AUDIT_RESULTS = {
    "experiments": [
        {
            "name": "memorization vs training-set size",
            "x_label": "training-set size",
            "points": [
                {"x": 500, "memorization_rate": 0.2},
                {"x": 2000, "memorization_rate": 0.1},
            ],
        }
    ],
    "note": "stub: example numbers, not results",
}
EXAMPLE_AUDIT_CHECK = {
    "nearest": [
        {"rank": 1, "distance": 5.5, "image_png_base64": SHORT_BASE64},
        {"rank": 2, "distance": 6.1, "image_png_base64": SHORT_BASE64},
        {"rank": 3, "distance": 6.8, "image_png_base64": SHORT_BASE64},
    ],
    "flag": "not_flagged",
    "note": "stub: similarity only, not legal copying",
}
EXAMPLE_ERROR_FORMAT = {"error": "<text>"}


def json_lines(example):
    """Turn a Python dict into the lines of valid JSON (2 spaces per indent level)."""
    return json.dumps(example, indent=2).split("\n")


APP_A_DEFAULTS_HEADING = "Proposed defaults (all: proposed, agree on Day 1 with Nitin)"
APP_A_DEFAULTS = [
    "Contract file `api/contract.json`; real JSON numbers; AI-made threshold: start at 0.5, tune "
    "later.",
    "Upload: PNG or JPG up to 5 MB; the server will resize it to 32x32 and will store nothing.",
    "The heatmap will be a plain base64 PNG, without a \"data:image/png;base64,\" prefix.",
    "/audit/check will return k = 3 nearest images, from a small training subset only. "
    "/audit/results will serve the JSON files from `results/` unchanged.",
    "The stub will return the same numbers every time, and its note will say 'stub'. The website "
    "must cope when the heatmap is null, or when /audit/check returns 404 (it hides that section).",
    "CORS: the deployed API will allow the Vercel domain only. The website will read the API "
    "address from `VITE_API_URL`. The 'waking up' wait will be 60 seconds.",
    "Colours (the same as our deck): navy #1E293B, teal #0D9488 for the Detector, amber #F59E0B "
    "for the Auditor, grey #64748B, a white background and the system font stack.",
    "Decided on Day 2: the final package versions (on Day 1, write down the versions you use in "
    "`requirements.txt`), the Dockerfile port, the Vercel build settings.",
]

APP_A_ENDPOINTS_HEADING = "The four endpoints"
APP_A_CAPTION = "Example values only. They are not results. The base64 text is cut short here."

APP_A_STUB_HEADING = "How to build the stub (Day 1, [friend's name])"
APP_A_STUB_STEPS = [
    "In `api/`, make `main.py` and `requirements.txt`. Install with `pip install fastapi uvicorn "
    "python-multipart` and pin the versions you used in `requirements.txt`.",
    "In `main.py`, make a FastAPI app with four routes (GET /health, POST /detect, GET "
    "/audit/results, POST /audit/check). Each returns the fixed example above. Ask your AI to "
    "explain each line in simple words.",
    "The two POST routes take one uploaded file (the form field `file`, proposed). Check the type "
    "and the size first: a bad upload answers 400 with the error format, any other failure "
    "answers 500.",
    "Run it from inside `api/` with `uvicorn main:app --reload`. Test "
    "`http://127.0.0.1:8000/health` in a browser, and try /detect on the page "
    "`http://127.0.0.1:8000/docs` that FastAPI makes for you (use 'Try it out' and upload an "
    "image).",
]

h2(APP_A_HEADING)
para(APP_A_INTRO)
h3(APP_A_DEFAULTS_HEADING)
num_list(APP_A_DEFAULTS)
h3(APP_A_ENDPOINTS_HEADING)
para(APP_A_CAPTION)
para("The status will be \"ok\", or the error text if something is wrong.", lead="GET /health:",
     ticks=False, after=2, keep_next=True)
code_block(json_lines(EXAMPLE_HEALTH))
para("You send one image. It will answer a score from 0 to 1 (higher = more likely AI-made), a "
     "VAE score, the heatmap (plain base64 PNG, or null) and a note.", lead="POST /detect:",
     after=2, keep_next=True)
code_block(json_lines(EXAMPLE_DETECT))
para("If the heatmap is not available, the answer has `\"heatmap_png_base64\": null` and the "
     "Detector page must still work.", after=6)
para("The auditor results from `results/`, sent unchanged: experiments (name, x_label, points) "
     "and a note.", lead="GET /audit/results:", after=2, keep_next=True)
code_block(json_lines(EXAMPLE_AUDIT_RESULTS), together=False)
para("You send one image. It will be compared with a small training subset. The answer: the "
     "nearest images, a flag ('possible_copy' or 'not_flagged') and a note.",
     lead="POST /audit/check:", after=2, keep_next=True)
code_block(json_lines(EXAMPLE_AUDIT_CHECK), together=False)
para("Errors: 400 for a bad upload, 500 for anything else. The format is "
     "`" + json.dumps(EXAMPLE_ERROR_FORMAT) + "`, for example the text "
     "\"The file is not a PNG or JPG image.\"")
h3(APP_A_STUB_HEADING)
num_list(APP_A_STUB_STEPS)

# ===========================================================================
# APPENDIX B. REQUIRED WEBSITE WORDING
# ===========================================================================

APP_B_HEADING = "Appendix B. Required website wording"
APP_B_INTRO = (
    "The website must show these sentences (or follow these rules). Copy the quoted ones exactly."
)
APP_B_ITEMS = [
    "\"The auditor measures similarity between images. It does not decide copyright infringement "
    "or 'legal copying'.\"",
    "Use the words 'similar' or 'possible copy'. Never write 'copyright infringement' as a result.",
    "\"We will report the drop in performance as a finding. We do not promise the detector will "
    "work on unseen generators.\"",
    "Next to results: a shared architecture or training data between the detector and a "
    "generator may make detection look better.",
    "Result cells show \"" + PLACEHOLDER + "\" until the result is real.",
    "Where a tool fails, show the failure. Not in version 1: adversarial attacks and video. "
    "Roadmap items are ideas only.",
    "Every number traces to a script name. Datasets and weights are never in the repo.",
    "CIFAKE is an extra test set only. It is not for the website.",
]

h2(APP_B_HEADING)
para(APP_B_INTRO)
num_list(APP_B_ITEMS)

# ===========================================================================
# APPENDIX C. STARTER PROMPTS (one per person)
# ===========================================================================

APP_C_HEADING = "Appendix C. Starter prompts"
APP_C_INTRO = (
    "Upload this file to your AI first, then paste your prompt (replace <number> with your Day)."
)
APP_C_NITIN = (
    "I am Nitin, team leader and ML person (Part A-ML). The team plan is my first message. Today "
    "is Day <number>: use my task card in section 4 and follow section 6A."
)
APP_C_FRIEND = (
    "I am [friend's name], the web person (Part B-Web). The team plan is my first message. Today "
    "is Day <number>: use my task card in section 4 and follow section 6A."
)

h2(APP_C_HEADING)
para(APP_C_INTRO)
para(APP_C_NITIN, lead="Prompt for Nitin:")
para(APP_C_FRIEND, lead="Prompt for [friend's name]:")

# ===========================================================================
# APPENDIX D. THINGS STILL TO VERIFY
# ===========================================================================

APP_D_HEADING = "Appendix D. Things still to verify"
APP_D_SOURCES = (
    "Sources we plan to cite (check each one before you use it): DIRE (Wang et al., arXiv 2303.09295; the venue, ICCV 2023, is to verify); "
    "Carlini et al., \"Extracting Training Data from Diffusion Models\" (USENIX Security 2023, "
    "arXiv 2301.13188); Ojha et al. (CVPR 2023, arXiv 2302.10174); Corvi et al. (ICASSP 2023, "
    "arXiv 2211.00680; the author list is to verify); Krizhevsky 2009, CIFAR-10 (the licence is to "
    "verify; the official page states none); Fashion-MNIST (Zalando, MIT licence); CIFAKE (Bird "
    "and Lotfi, arXiv 2303.14126; licence, count 100k vs 120k, image size, Kaggle terms and venue "
    "are all to verify)."
)
APP_D_ITEMS = [
    "Kaggle free GPU limits, Render free web service limits and Vercel free-plan terms.",
    "Hugging Face: Docker + FastAPI on the free tier, and the CPU Basic figures (2 vCPU, 16 GB RAM, "
    "sleeps when unused). Check on Day 2.",
    "All epochs, set sizes, t (about 200) and every threshold: proposed, to tune.",
    "The professor's rules: written report, AI use, marks, deadline (section 9).",
]

h2(APP_D_HEADING)
para(APP_D_SOURCES)
num_list(APP_D_ITEMS)

# ===========================================================================
# APPENDIX E. REPRODUCIBILITY CHECKLIST
# ===========================================================================

APP_E_HEADING = "Appendix E. Reproducibility checklist"
APP_E_ITEMS = [
    "Fixed random seeds, set in one place.",
    "A saved config file for every run (epochs, sizes, t, thresholds).",
    "One results/ JSON file per experiment, written by a script.",
    "Every number in any slide or report is traced to a script name.",
    "Package versions pinned in requirements.txt.",
    "No secrets and no datasets in the repo.",
    "Results stay as " + PLACEHOLDER + " until they are real.",
]

h2(APP_E_HEADING)
num_list(APP_E_ITEMS)

# ===== END OF PARTS (the save lines are at the bottom) =====

doc.save(OUT_FILE)
print("saved", OUT_FILE)
