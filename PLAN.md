# GenAI Trust Lab: project plan (3-day rebuild, DRAFT)

**Status: DRAFT for you to read and approve. Nothing described here is built yet.** The plan covers Mon 23 Nov (evening), Tue 24, Wed 25 and Thu 26 Nov 2026. Thursday is the final presentation. People: Nitin (leader: backend, models, wiring) and Tejash (website).

## 0. How to read this file

- **What this file is:** the full rebuild of the project plan for the 3-day window before the final presentation. It is a draft. It replaces the earlier plan only after you approve it. No building starts before that.
- **Honesty line:** Nothing here is a result. Every number is either a design choice marked "proposed, to tune" or a placeholder.
- **Tags used in this file:**

| Tag | Meaning |
|-----|---------|
| proposed | A design choice we made. We can change it. |
| proposed, to tune | A number we will adjust later, using the calibration split only. |
| to verify | A fact taken from a web page or document that we have not yet confirmed ourselves. |
| `[PLACEHOLDER - not measured yet]` | A result cell. A script will fill it. It stays like this until a script has run. |

- **Where the day-by-day tasks are:** the team plan document (the draft team plan .docx next to this file) holds the checklists for each day and each person. This file holds the design.

## 1. What changed since the last version

| # | Change | Why |
|---|--------|-----|
| 1 | The 3-day window (Mon evening to Thu) replaces the older Day 1-4 schedule. | The presentation is on Thursday 26 Nov. |
| 2 | Two pages are added: **Generator** and **Compare (Distinguisher)**. | You asked for a generator with several models and datasets, and for a page that shows what is different between two photos. |
| 3 | The third generator (G3, a second DDPM) and the DCGAN are cut. | Free Kaggle GPU hours are limited (the exact quota is to verify). |
| 4 | The live demo runs from a laptop, with a recorded backup. | Free Hugging Face Docker Spaces are not available (a paid plan is needed; confirmed from the docs). |
| 5 | New endpoints: POST /compare, GET /generators, POST /generate, POST /evidence. | The new pages need them. |
| 6 | The Detector warns on photos outside its training range. | It is trained only on 32x32 CIFAR-10-style images. |
| 7 | Thresholds are chosen on a calibration split, not on the plain test set. | This avoids using the final test data to tune the detector. |
| 8 | The Broad track stays documentation only. | It is too big for 3 days. Narrow v1 is the plan. |
| 9 | GET /audit/results gets one optional extra field, `examples`. | The Auditor's nearest-neighbour grid then does not depend on POST /audit/check, which is cut early. |
| 10 | The very first push (the `.gitignore` fix) goes straight to main. | On Monday nobody has a merged pull request to read yet. It is the one exception to the pull-request rule. |

### 1.1 Three open questions (the default used in this draft)

You will be asked these at approval. If you say nothing, the draft assumes the default.

| # | Question | Default used here |
|---|----------|-------------------|
| 1 | Where does Tejash's existing site live? | It is copied into this repo under `web/` by his first pull request. Its code is not in this repo yet and its location is not known yet. |
| 2 | May the Compare "Words" panel use an outside vision API? | We build it but keep it OFF by default. We ask the teacher on Monday. If there is no answer by Wednesday morning, it stays off. Default provider Anthropic (proposed). The key stays only in `api/.env`. With no key the answer is `heatmap_only`. |
| 3 | What is the page called in the menu? | "Compare (Distinguisher)", route `/compare`. |

## 2. Problem, goals, non-goals

### 2.1 Problem

Pictures made by AI are hard to tell from real ones, and it is hard to say what is different between two pictures. GenAI Trust Lab is a small teaching lab with six web pages. It makes images (Generator), judges small images (Detector), compares two images (Compare), checks whether a generator copies its training images (Auditor), and says what each tool cannot prove (Limits and Methods).

### 2.2 Goals per page

| Page | Goal |
|------|------|
| Home | Say what the lab does, show the honesty lines, link to the other pages. |
| Generator | Let a visitor pick a small model and a dataset and make labelled images. |
| Detector | Say real, ai or unclear for 32x32 images, and show the clue (a heatmap). |
| Compare (Distinguisher) | Take any two photos (for example an orange cat and a black cat) and show what is different: a difference map, a feature table with dominant colours and, when allowed, plain-English words. |
| Auditor | Show how often a generated image is almost the same as a training image. |
| Limits and Methods | Say what every feature can and cannot prove. Hold the model cards and the sources. |

### 2.3 Non-goals

- Adversarial attacks (someone deliberately fooling the detector). We do not test or defend against them.
- Not done in version 1.
- Video.
- Model Zoo extras (SVM, decision tree, ensembles, Level 3 CNN, regression, KNN/RF). They are parked as future work. They are not built.
- Judging arbitrary photos with the Detector. It is trained for 32x32 CIFAR-10-style images only.
- Legal decisions about copying (see the honesty lines below).

### 2.4 Honesty lines (kept word for word)

- The auditor measures **similarity between images**. It does not decide copyright infringement or "legal copying".
- Detecting images from a generator the detector has never seen is an open research problem. We will **report the drop in performance as a finding**. We do not promise the detector will work on unseen generators.
- We say "similar" or "possible copy". We never say "copyright infringement".
- Honest reporting: where a tool fails, we show the failure.
- All result cells stay `[PLACEHOLDER - not measured yet]` until a script fills them.
- Every table cell comes from a script, or is still marked as a placeholder.

### 2.5 Words used in this plan

Each word is also explained again where it is first used.

| Word | Plain meaning |
|------|---------------|
| VAE | A small network that squeezes an image into a few numbers and then rebuilds it. If the rebuild is poor, the image is unusual. |
| DDPM | A diffusion model. It learns to remove noise step by step. To make an image it starts from noise. |
| DDIM | A faster way to make images from a diffusion model, using fewer steps. |
| CFG (classifier-free guidance) | A slider for the class-conditional model. It trades variety against fit to the chosen class. |
| SSIM | A score for how similar the structure of two aligned images is. |
| AUROC | One number for how well a detector separates real from fake over all thresholds. Higher is better. |
| Grad-CAM | A heatmap of where a classifier looked. It explains the classifier, not the forgery. |
| pHash | A short fingerprint of an image. Similar images have close fingerprints. |
| Memorization | A generator producing an image that is almost the same as one of its training images. |
| Out-of-distribution | An image that is unlike the images a model was trained on. Results on it are not meaningful. |
| Percentile | The value below which a given share of the numbers fall. The 1st percentile is a very low cutoff. |
| Gaussian fit | A bell curve fitted to the numbers. |
| 2-component GMM | A mix of two bell curves. |
| base64 | A way to write an image as text. |
| CORS | A browser rule about which websites may call the API. |
| `VITE_API_URL` | The address of the API, read by the website. |
| `out_of_scope` | The image is not the size the detector was trained on. |
| The DIRE idea | Add noise, let a model clean it, measure the error. |

## 3. Website pages

Tejash has already built a first multi-page version of the website. Its code is not in this repo and its location is not known yet. **Status of the existing site: existing site: structure and tech to be listed (page list, framework, folder/repo link), Day 0 task** (owner: Tejash). Nitin will upgrade it and connect the backend.

Tiers used in this plan:

| Tier | Meaning |
|------|---------|
| MUST | Part of the Thursday demo. If it is cut, the demo is not finished. |
| SHOULD | Built after all MUST items. Wednesday build order: Words, then Grad-CAM, then EXIF, then pHash plus the Auditor side-by-side. |
| COULD | Built only if everything else is done. Cut first after cloud hosting (cut-line items 1 and 2, section 12). |
| CORE | Generator models only: the models that must be ready on the Generator page. |
| documented-only | Described in the plan. Not built in version 1. |

| Route | Page | Purpose | MUST | SHOULD | COULD | Endpoints used |
|-------|------|---------|------|--------|-------|----------------|
| `/` | Home | What the lab does, honesty lines, links | Text and links | none | none | none (static page) |
| `/generator` | Generator | Pick a model and a dataset, make images | Model and dataset picker, seed, number of images (up to 16, proposed), labelled image grid, download PNG, buttons "Send to Detector" and "Send to Compare", controls driven by the `controls` list of each model | The DDPM on CIFAR-10 in the picker (depends on Gate 2) | none | GET /generators, POST /generate |
| `/detector` | Detector | real / ai / unclear for 32x32 images, plus an Evidence tab | Upload box, "Try an example" strip, label, score, VAE score, baseline score, VAE reconstruction-error heatmap (default), out-of-scope warning | Grad-CAM on the CNN baseline, EXIF in the Evidence tab, DDPM reconstruction-error heatmap (replaces the VAE map if Gate 2 passes) | C2PA, ELA, FFT, robustness sliders (JPEG, blur, noise) | POST /detect, POST /evidence |
| `/compare` | Compare (Distinguisher) | Two images: what is different | Two uploads, an optional "What should I compare?" text box, overall SSIM and difference map, feature table (with dominant colours) | Words panel (vision API, generic attribute checklist) | LPIPS, DINOv2-small regions | POST /compare |
| `/auditor` | Auditor | Memorization dashboard | Dashboard of the runs, nearest-neighbour grid (read from the `examples` field of GET /audit/results) | pHash distance, side-by-side difference map | Upload check (POST /audit/check) | GET /audit/results, POST /audit/check |
| `/limits` | Limits and Methods | What each feature cannot prove, model cards, sources | Honesty lines, a "does not prove" line for every feature, model cards, sources | none | none | none (static page) |

The optional Results / Model cards page is dropped. Model cards go on `/limits`. Results go on `/auditor`.

## 4. Generator

### 4.1 Catalogue (all models are small, about 1M parameters each, proposed)

| id | Model | Dataset | Tier | Controls |
|----|-------|---------|------|----------|
| `g-vae-cifar` | Conv VAE (also the detector's first fake source) | CIFAR-10 | CORE | seed, n, interpolate (seed_b) |
| `g-vae-fmnist` | Conv VAE | Fashion-MNIST | CORE | seed, n, interpolate (seed_b) |
| `g-cddpm-fmnist` | Class-conditional DDPM with classifier-free guidance | Fashion-MNIST | CORE | seed, n, steps, class_id, guidance |
| `g-ddpm-cifar` | DDPM, cosine schedule option (also the detector's second fake source) | CIFAR-10 | SHOULD (Gate 2) | seed, n, steps (DDIM 20, 50 or 100, proposed) |
| `g-dcgan-cifar` | DCGAN | CIFAR-10 | documented-only (future work) | none |

The DDPM models are TinyUNet-style (from the course notebook) and the VAEs are small conv VAEs. Parameter counts and file sizes are proposed; training times and file sizes are to verify.

Tiers for generators: CORE = must be ready on Thursday. SHOULD = built after the CORE models if Gate 2 allows it. documented-only = future work.

### 4.2 Labelling rule

Every image the Generator page shows carries this label: `<model> on <dataset>, small model, expect low quality`. The `/generate` response also carries the label `small model, expect low quality` in its metadata.

### 4.3 Detector-only models (never shown on the Generator page)

| id | What it is |
|----|-----------|
| `d-vae-cifar` | The anomaly-score VAE, trained on CIFAR-10 training images (real images only). A different seed from `g-vae-cifar`. |
| `d-ddpm-cifar` | The reconstruction DDPM, trained only on the DDPM-train part (30,000 real CIFAR-10 training images, proposed). A different model and seed from `g-ddpm-cifar`. |
| `cnn-baseline` | A small 3-layer CNN on raw pixels. Always kept. |

### 4.4 Facts from the papers (used to choose settings)

- The original DDPM CIFAR-10 model has 35.7M parameters (about 143 MB in fp32 by arithmetic) and T=1000 sampling steps (Ho et al., arXiv 2006.11239). Our models must be far smaller: about 1M parameters (proposed), as in the course notebook's TinyUNet.
- DDIM (Song et al., ICLR 2021, arXiv 2010.02502) compared 10, 20, 50, 100 and 1000 sampling steps in its own experiments. The paper says: "DDIM is able to produce samples with quality comparable to 1000 step models within 20 to 100 steps". DDPM quality deteriorates rapidly at 10 steps. So the step choices are 20, 50 and 100 (proposed).
- Classifier-free guidance (Ho and Salimans, arXiv 2207.12598): higher guidance decreases sample variety and increases per-sample fidelity. This is what the guidance slider will show.
- Improved DDPM (Nichol and Dhariwal, arXiv 2102.09672): the cosine schedule beats the linear schedule for 32x32 and 64x64 images. So a cosine schedule is an option for `g-ddpm-cifar`.

### 4.5 Page controls and buttons

| Control | Used by |
|---------|---------|
| Model and dataset picker | all |
| Seed, number of images (n, up to 16, proposed) | all |
| Steps (DDIM) | the two DDPM models |
| Guidance slider and class selector | `g-cddpm-fmnist` only |
| Interpolation between two seeds | the two VAE models |
| Download PNG, with a provenance text note inside the file (Pillow `PngInfo.add_text`: to verify) | all |
| "Send to Detector" and "Send to Compare" | all |

The Detector and the Auditor use the CIFAR-10 generators only. The Fashion-MNIST models are for the Generator page.

### 4.6 What is cut

- G3 (a second DDPM) is removed because of Kaggle hours. Leave-one-generator-out becomes: train on one CIFAR generator, test on the other.
- DCGAN is future work on CIFAR-10 (GANs are not in the course and are risky to tune; CIFAR-10 avoids the non-commercial terms of SVHN).
- CelebA and SVHN are not used (see section 5).

## 5. Datasets

### 5.1 Datasets used or listed

| Dataset | Facts | Role |
|---------|-------|------|
| CIFAR-10 | 32x32 colour, 50,000 train + 10,000 test. The official page states no licence (to verify). | Detector, Auditor, two CIFAR generators |
| Fashion-MNIST | 28x28 grayscale, 60,000 train + 10,000 test, MIT licence | Generator page only |
| CIFAKE | Licence: to verify. Image count (100k vs 120k): to verify. Image size: to verify. Kaggle terms: to verify. Journal venue: reference to verify. | Listed in the earlier plan. Every fact is to verify before use. |

### 5.2 Splits and the leakage rule (all sizes proposed)

CIFAR-10 TRAIN is split into two parts with no overlap. CIFAR-10 TEST is split into two parts. The split is written once into `data/splits.csv` with a fixed seed (42, proposed).

| Part | Source | Size (proposed) | Used for |
|------|--------|-----------------|----------|
| DDPM-train part | CIFAR-10 TRAIN | 30,000 | Training the real-only reconstruction DDPM `d-ddpm-cifar` only |
| Classifier-train part | CIFAR-10 TRAIN | 20,000 | Training the logistic regression and the CNN baseline (real side) |
| Calibration split | CIFAR-10 TEST | 5,000 | Choosing every threshold and hyperparameter, and the Auditor's percentile |
| Final test split | CIFAR-10 TEST | 5,000 | Scored once per finished run |

Rule: every threshold and hyperparameter is chosen on the calibration split only.

Which data each model trains on or is scored on (all proposed):

| Model or step | Data |
|---------------|------|
| All generators (`g-vae-cifar`, `g-ddpm-cifar`, `g-vae-fmnist`, `g-cddpm-fmnist`) | The training images of their own dataset (CIFAR-10 training images, or Fashion-MNIST training images). |
| `d-vae-cifar` | CIFAR-10 training images. |
| `d-ddpm-cifar` | The DDPM-train part (30,000) only. |
| The classifier (logistic regression) and `cnn-baseline` | The classifier-train part (20,000), real side, plus generated fakes. |
| Fakes | SAMPLES drawn from the generators. No test image is used to make them. |
| Thresholds and hyperparameters | The calibration split (5,000, from the CIFAR-10 test split). |
| Final score | The final test split (5,000, from the CIFAR-10 test split), scored once per finished run. |

How the fakes are divided into classifier-train, calibration and final-test parts, and how many there are, is to decide at the first meeting and goes into the config file. CIFAR-10 and Fashion-MNIST are downloaded with torchvision on the laptop and stay out of git.

### 5.3 Fake sources and licence rules

Fakes come from open multi-generator datasets only. No closed-service outputs. The public model uses permissively licensed sources only. Non-commercial (NC) datasets are for offline evaluation only.

In version 1 the fakes come from our own small generators (`g-vae-cifar` and `g-ddpm-cifar`). The rule above applies to any outside fake source, such as the sources of the Broad track (section 14).

### 5.4 Not used

| Item | Why |
|------|-----|
| the OpenAI CLIP ViT-L/14 weights (model card says any deployed use is out of scope) | Not used in the public demo. |
| CelebA (202,599 faces) | Not recommended: the official page was unreachable, preprocessing is needed and the licence is unconfirmed. |
| SVHN | Terms say "for non-commercial use only". Not used. |
| MNIST | Official page unreachable. Terms not confirmed (to verify). Not used. |
| EMNIST | No licence on the page. Readme not read. Terms not confirmed (to verify). Not used. |

## 6. Detector

### 6.1 The idea

A diffusion model trained only on real images repairs real images better than fake ones. The repair error is the clue. This is the idea of the DIRE paper (see section 20): add noise, let a model clean it, measure the error.

### 6.2 Features and classifier

| Part | What it does |
|------|--------------|
| Feature 1: DDPM reconstruction error (SHOULD) | Add noise up to timestep t (about 200 of T=1000, proposed, to tune). Denoise with `d-ddpm-cifar`, trained only on real CIFAR-10. The per-pixel error is the DDPM heatmap (`heatmap_source` is `ddpm`). It replaces the VAE map if Gate 2 passes. Summarise it by the mean and some percentiles (a percentile is the value below which a given share of the numbers fall). |
| Feature 2: VAE anomaly score (MUST) | A small conv VAE (`d-vae-cifar`) trained on real images. Reconstruction error plus KL is the "VAE anomaly score". The per-pixel reconstruction error of the VAE is the default heatmap (`heatmap_source` is `vae`). This is new work: the course notebook does not compute an anomaly score, and the course VAE is an MLP with a 2-D latent. |
| Classifier | scikit-learn logistic regression on the features. It is trained on Wednesday. On Tuesday, /detect returns the CNN-baseline label and the VAE score. On Wednesday the logistic regression label replaces the CNN-baseline label. |
| Label rule | The label uses TWO thresholds, a low one and a high one. A score below the low threshold gives real, a score above the high threshold gives ai, and a score between them gives unclear. Both thresholds start around 0.5 (proposed, to tune) and are tuned on the calibration split only. |

### 6.3 CNN baseline

`cnn-baseline` is a small 3-layer CNN (taken from the LetterCNN of the conditional diffusion notebook) on raw pixels. It is always kept. Its score is returned as `baseline_score`. On Tuesday its label is the label that /detect returns.

### 6.4 Fairness caveat

`d-ddpm-cifar` is a different model and seed from `g-ddpm-cifar`, and `d-vae-cifar` has a different seed from `g-vae-cifar`. We still report the caveat: shared architecture or data could make detection look better than it really is.

### 6.5 Evaluation

All cells below stay `[PLACEHOLDER - not measured yet]` until a script fills them.

| Test | accuracy | AUROC |
|------|----------|-------|
| Seen generator: `g-vae-cifar` | `[PLACEHOLDER - not measured yet]` | `[PLACEHOLDER - not measured yet]` |
| Seen generator: `g-ddpm-cifar` (only if it passes Gate 2) | `[PLACEHOLDER - not measured yet]` | `[PLACEHOLDER - not measured yet]` |
| Leave-one-generator-out: train on `g-vae-cifar`, test on `g-ddpm-cifar` | `[PLACEHOLDER - not measured yet]` | `[PLACEHOLDER - not measured yet]` |
| Leave-one-generator-out: train on `g-ddpm-cifar`, test on `g-vae-cifar` | `[PLACEHOLDER - not measured yet]` | `[PLACEHOLDER - not measured yet]` |
| Robustness: JPEG | `[PLACEHOLDER - not measured yet]` | `[PLACEHOLDER - not measured yet]` |
| Robustness: blur | `[PLACEHOLDER - not measured yet]` | `[PLACEHOLDER - not measured yet]` |
| Robustness: added noise | `[PLACEHOLDER - not measured yet]` | `[PLACEHOLDER - not measured yet]` |

The drop on the unseen generator is reported as a finding, not hidden. Every threshold comes from the calibration split. The final test split is scored once per finished run.

### 6.6 Out-of-scope rule and "Try an example"

The detector is trained for 32x32 CIFAR-10-style images only. Arbitrary photos are out of scope for it.

- `out_of_scope` (the image is not the size the detector was trained on) is `true` whenever the upload is not exactly 32x32 (proposed rule). The image is still resized and scored.
- The page then shows a red banner: "This detector was trained only on 32x32 CIFAR-10-style images. Results on other photos are not meaningful."
- A "Try an example" strip sits above the upload box. It holds static real and fake CIFAR-style images (10 + 10, proposed). It needs no endpoint.
- The FAKE examples are our own samples and may be committed inside the website folder. The REAL CIFAR-10 examples are NOT committed: the CIFAR-10 licence is to verify and the rule is never to commit the dataset. Nitin shares them with Tejash by file transfer into `web/public/examples/real/`, and that folder is added to `.gitignore`.
- Known limitation: the rule only catches image SIZE (not exactly 32x32). A 32x32 photo that is not CIFAR-style is not flagged, and its result is still not meaningful.

### 6.7 Detector extras by tier

| Tier | Extras |
|------|--------|
| MUST | VAE score, CNN baseline score, VAE reconstruction-error heatmap (default), "Try an example", out-of-scope warning |
| SHOULD | Grad-CAM on `cnn-baseline` (it explains the CNN, not the forgery), EXIF, DDPM reconstruction-error heatmap (replaces the VAE map if Gate 2 passes) |
| COULD | C2PA (maturity unknown), ELA, FFT, robustness sliders (the robustness numbers still go into the results JSON) |

### 6.8 Speed

The older open question (can the diffusion step run on a CPU?) is expected to be settled by few DDIM steps on the laptop. The real speed is to be measured on Tuesday.

## 7. Compare (Distinguisher)

### 7.1 What it does

The user uploads any two photos. The page shows what is different. There is no fixed category. The main example is an orange cat and a black cat. Other examples are two mugs or two pairs of shoes, and two bicycles are only one example among others. The demo uses cat photos plus a few other objects.

The page has an optional text box, "What should I compare?" (for example "fur colour and eyes", up to 100 characters, proposed). Its text is sent as `focus` and is used only by the Words panel.

### 7.2 How it works, step by step

1. The page sends `image_a` and `image_b` (and `use_words`, true or false, and the optional `focus` text) to `POST /compare`.
2. The backend makes both images the same size for the pixel tools (proposed).
3. It computes the overall SSIM and a difference map (scikit-image).
4. It builds the feature table: colour mean and histogram distance, dominant colours (the three most common colours of each image, found with k-means, k = 3, proposed; each gets a plain name such as "orange", "black" or "white" and its share of the image), brightness, edge density (sharpness), texture, original size. Each row carries a `does_not_prove` text.
5. COULD: LPIPS (`similarity.lpips`) and DINOv2-small regions (`regions_png_base64`). If not built, these fields are `null`.
6. SHOULD: if `use_words` is true and a key exists, the backend sends both images to an outside vision API with the generic attribute checklist below. The model answers in JSON. The answer fills `differences`. Then `source` is `vlm`.
7. With no key, or with Words off, `source` is `heatmap_only` and `differences` stays empty.

**The Words prompt (proposed): a generic attribute checklist.** It works for any pair of photos.

| # | Attribute asked about |
|---|-----------------------|
| 1 | Main subject |
| 2 | Colour and pattern |
| 3 | Shape and size |
| 4 | Texture or material |
| 5 | Visible parts or features (for a cat: ears, eyes, nose, whiskers; for a bicycle: handlebars, saddle, wheels; whatever is visible) |
| 6 | Pose and angle |
| 7 | Background |
| 8 | Lighting and photo style |
| 9 | Number of objects |

Rules for the answer:

- The model answers in JSON.
- It may answer "same" or "unsure" for any attribute, and it must not guess.
- In the response, `differences[].part` holds the attribute or part name (the contract field names do not change).
- If the optional `focus` text is given, it is used only in this prompt. If it is empty, the full checklist is used.

### 7.3 Honesty line (required on the page)

Pixel tools (SSIM, difference map) only make sense for nearly aligned images, for example a generated image and its nearest training image, or an original and an edited copy. For two different photos, for example an orange cat and a black cat in different poses, the difference map will light up almost everywhere; the colour and brightness tables and the Words panel say more, and the Words panel can be wrong.

Same-subject mode: pixel tools are meaningful when the two photos are nearly aligned (the same subject, taken twice from nearly the same place, or an original and an edited copy). For two different subjects, read the colour and brightness tables and the Words panel instead.

The Words panel ranks above DINOv2 because it is the only planned feature that tries to name parts (for example fur colour or ears). Its quality for fine-grained parts is not verified.

### 7.4 Words panel: privacy and key rules

- Photos leave the machine when Words is on. The page shows this privacy text: "photos leave your computer; do not upload private photos or photos of people."
- The API key lives only in `api/.env`. It is never in `web/`, never in git, and never pasted into chat or logs.
- Any variable whose name starts with `VITE_` is shipped to the browser, so the key must never use that prefix.
- Words is OFF by default. We build it, and we ask the teacher on Monday. If there is no answer by Wednesday morning, it stays off.
- If there is no key, the answer is `heatmap_only`, and the site still works.
- The provider's data-use terms and the API key and credits are to verify.

| Option | Facts from the docs | Status |
|--------|---------------------|--------|
| Anthropic API (default, proposed) | Multi-image input. The docs example is "How are these images different?". 10 MB per image (base64) limit on the API. "New users receive free credits", amount not stated. | Terms and credits: to verify |
| OpenAI | Multi-image. Free tier not mentioned. | Not planned |
| Google Gemini | Multi-image. Free tier for some models. Numeric limits not written. | Not planned |
| SmolVLM-256M (open model) | Apache-2.0, about 0.3B parameters, accepts sequences of images and text. | Quality not verified. Not planned. |
| Florence-2-base (open model) | MIT, about 0.23B parameters, single-image examples. | Quality not verified. Not planned. |

### 7.5 Evaluation (small sample, descriptive only)

Tejash collects 20 photo pairs (proposed) of mixed subjects: own photos, NO photos of people. Each pair has a written answer key of what differs (colour, pattern, shape, parts, pose, background). The metric is: did it name the right attribute? The result is reported as n = 20, small sample, descriptive. It says nothing general.

| Measure | Result |
|---------|--------|
| Pairs collected | `[PLACEHOLDER - not measured yet]` |
| Pairs where Words named the right attribute | `[PLACEHOLDER - not measured yet]` |
| Pairs where Words named a wrong attribute | `[PLACEHOLDER - not measured yet]` |

### 7.6 Worked example: orange cat vs black cat (illustration of the format, not a result)

This table shows only what each panel would look like. Nothing here was measured. The words are made up to show the layout.

| Panel | What it would show (format only) |
|-------|----------------------------------|
| Overview scores | One SSIM number (`<number>`) and the difference map. For two different cats in different poses the map lights up almost everywhere, so it says little. |
| Dominant colours | Image A: orange, then white, then brown, each with a share (`<share>`). Image B: black, then grey, then white, each with a share (`<share>`). Background colours count too. |
| Brightness | Image A brighter or darker than image B (`<value>` for each). |
| Edge amount | A value for each image (`<value>`). |
| Words (if allowed) | Main subject: cat in both (same). Colour and pattern: orange vs black (differs). Visible parts, pose and angle, background, lighting, number of objects: "same" or "unsure". |

## 8. Auditor

### 8.1 Question and metric

Question: how often is a generated image almost the same as a training image? Metric: L2 distance to the nearest training image (32x32 images).

### 8.2 Calibration

The 1st percentile (the value below which a given share of the numbers fall; the 1st percentile is a very low cutoff) of the nearest-training-image distances of held-out real images from the calibration split (proposed, to tune) is the threshold. A generated image closer than this is flagged "possible copy". An optional Gaussian fit (a bell curve) or 2-component GMM (a mix of two bell curves) cross-check is added only if the histogram clearly has two bumps.

### 8.3 Runs (3 runs for the 3-day window, proposed)

| Run id | Training set |
|--------|--------------|
| `a-500` | training size 500 |
| `a-10000` | training size 10,000 |
| `a-500-dup` | training size 500 with duplicated images |

The Auditor audits its OWN small DDPMs, trained on these subsets (the same architecture as `g-ddpm-cifar`, but separate training runs). The nearest training images are looked up inside the subset that run was trained on. The Auditor audits our own generators only. It cannot audit GPT or any closed service.

### 8.4 Mitigations to test

Deduplication, flip augmentation, fewer epochs, smaller training sets.

### 8.5 Plots

Memorization against training-set size; memorization against epochs; distance histograms (generated against held-out); a grid of generated images beside their nearest training images. A plot is filled only if its runs finish. Otherwise it shows a placeholder. The nearest-neighbour grid is read from the `examples` field of GET /audit/results, which a script writes. It does not depend on POST /audit/check.

### 8.6 Extras by tier

| Tier | Extras |
|------|--------|
| MUST | Dashboard of the runs, nearest-neighbour grid (from the `examples` field of GET /audit/results) |
| SHOULD | pHash distance for each nearest image, side-by-side difference map (reusing the Compare SSIM code) |
| COULD | POST /audit/check (upload a generated image and get its nearest training images) |

### 8.7 Wording

We say "similar" or "possible copy". We never say "copyright infringement".

The auditor measures **similarity between images**. It does not decide copyright infringement or "legal copying".

## 9. Feature catalogue

Every feature below is evidence or a teaching aid. None is proof. The last column is also the text the `/limits` page will show.

| Feature | Page | Library (licence as verified) | Tier | What it does NOT prove |
|---------|------|-------------------------------|------|------------------------|
| VAE anomaly score | Detector | Our own code | MUST | A high score means "unlike the real training images", not "made by AI". Valid only for 32x32 CIFAR-10-style images. |
| CNN baseline score | Detector | Our own code | MUST | Valid only for the data it was trained on. It can be fooled by changes it never saw. |
| VAE reconstruction-error heatmap (default) | Detector | Our own code | MUST | Shows where the VAE rebuilt the image badly. It does not prove the image is fake. Valid only for 32x32 CIFAR-10-style images. |
| DDPM reconstruction-error heatmap | Detector | Our own code | SHOULD | Same limits as the VAE map. It replaces the VAE map only if Gate 2 passes. |
| (a) ELA (error level analysis) | Detector, Evidence tab | Pillow re-save plus ImageChops difference (Pillow licence not verified here: to verify) | COULD | Not proof of manipulation or of AI. Works on lossy formats only. A secondary source says it mislabels at similar rates both ways (to verify). Teaching aid only. |
| (b) FFT spectrum | Detector, Evidence tab | numpy FFT plus radial average | COULD | Does not prove AI. Papers report upsampling spectral artifacts mainly for GAN-type up-convolutions (see section 20). Not shown for diffusion or post-processed images. Teaching aid only. |
| (c) EXIF | Detector, Evidence tab | Pillow `getexif()` | SHOULD | Missing EXIF is weak evidence (screenshots and platforms strip it, and metadata can be edited). Present EXIF does not mean real. |
| (d) C2PA manifest read | Detector, Evidence tab | c2pa-python (Apache-2.0 + MIT per README, Python 3.10+, binding to c2pa-rs; maturity not stated) | COULD | A valid manifest is signed provenance. A missing manifest means NOTHING. Spec 2.2 confirmed (2.3 exists: confirm current). |
| (e) Grad-CAM | Detector | pytorch-grad-cam (MIT per README badge; `pip install grad-cam`; we must choose the target layer) | SHOULD | It explains the classifier (`cnn-baseline`), not the forgery. |
| (f) Occlusion sensitivity | Detector | Our own code (slide a grey patch, record the score drop). Not confirmed from a source. | documented-only | Does not prove which pixels cause the output. |
| (g) pHash / dHash | Auditor | imagehash (BSD-2-Clause); tune the threshold and the hash size | SHOULD | A close hash does not prove copying. A far hash does not prove there is no link. |
| (h) SSIM and difference map | Compare, Auditor side-by-side | scikit-image (BSD-3-Clause), `skimage.metrics.structural_similarity` | MUST | Sensitive to alignment and scale. Not a semantic difference. Two different photos (for example an orange cat and a black cat in different poses) need registration first, and even then the map says little. |
| (i) Colour, edge and texture statistics | Compare | numpy, scikit-image | MUST | Descriptive only. It does not name the part that differs. |
| Dominant colours (k-means, k = 3, proposed) | Compare | numpy plus scikit-learn KMeans (scikit-learn licence not verified here: to verify). Course link: clustering. | MUST | Descriptive only. It does NOT prove which object has which colour (background colours count too). |
| (j) LPIPS | Compare | lpips package (BSD-2-Clause); RGB in [-1,1]; higher means more different; downloads backbone weights | COULD | A perceptual distance only. It is not an explanation. |
| (k) DINOv2 patch matching | Compare, "Semantic regions" | DINOv2-small, 22.1M parameters, Apache-2.0 (about 88 MB in fp32) | COULD | Matched points are not guaranteed to be the same part. Unmatched regions are not guaranteed to be "the difference". Features show positional bias across layers. CPU timing NOT checked. |
| Words (vision API) | Compare | Outside API, key in `api/.env` only | SHOULD | The answer can be wrong or invent attributes or parts. Quality for fine-grained parts is not verified. |
| Invisible watermark (idea) | Detector, Evidence tab | invisible-watermark (MIT; dwtDct method; experimental) | COULD (idea only) | Fails under 50% resize and 30-degree rotation. Our app can embed and read ITS OWN marks, not other tools' marks. No mark does not mean real. |
| Stable Signature | none | Mostly CC-BY-NC; needs decoder fine-tuning | Not used | Not practical for us. |
| SynthID Detector | none | Google, proprietary portal, not an open library | Not used | Terms must be checked (to verify). |

**Ranking for a university demo (from the research, before our time limits):** 1 Grad-CAM on the Detector; 2 SSIM and difference map on Compare; 3 pHash duplicate check on the Auditor; 4 colour and edge statistics table on Compare; 5 C2PA manifest read (with the message "no manifest is not evidence"); 6 LPIPS on Compare. ELA is a visual add-on below these. The tiers above also consider build time and risk, so they differ a little from this ranking. The research ranking is about general usefulness; our build order is Words, Grad-CAM, EXIF, pHash.

## 10. System design

### 10.1 Planned repo layout

Apart from `docs/`, none of these folders exist yet.

| Folder | What goes in it |
|--------|-----------------|
| `model/` | Training and scoring scripts for every model (generators, detector parts, auditor). |
| `api/` | The FastAPI backend and `api/contract.json` (the agreed JSON shapes with real JSON numbers). `api/.env` holds the vision-API key and is ignored by git. |
| `web/` | The website. Tejash's existing site is copied here by his first pull request. |
| `results/` | One JSON file per experiment, written by a script. |
| `configs/` | One saved config file per run (epochs, sizes, t, thresholds). |
| `data/` | Only `data/splits.csv` is committed. Everything else here is ignored by git. |
| `docs/` | This plan, the team plan, the deck, drafts and logs. |

The `.gitignore` must be fixed FIRST, on Monday evening. Today it blocks only `~$*`, `.claude/settings.local.json`, `__pycache__/`, `*.env` and `.env`. The planned list:

```
data/*
!data/splits.csv
fakes/
*.pt
*.pth
*.ckpt
*.safetensors
*.h5
.env
node_modules/
.venv/
__pycache__/
web/public/examples/real/
```

The last line keeps the REAL CIFAR-10 example images out of git (section 6.6).

### 10.2 Endpoints

Images are sent as multipart form data. Other requests are JSON. Optional features that are not built return `null`, so the website never breaks.

| Method and path | Purpose | Used by page |
|-----------------|---------|--------------|
| GET /health | Is the backend up? | Home (optional) |
| POST /detect | real / ai / unclear for one image | Detector |
| GET /audit/results | The memorization runs and the example pairs | Auditor |
| POST /audit/check | Nearest training images for one image | Auditor |
| POST /compare | What is different between two images | Compare |
| GET /generators | The list of generators and their controls | Generator |
| POST /generate | Make images | Generator |
| POST /evidence | EXIF, C2PA, ELA and FFT for one image | Detector, Evidence tab |

Errors use one shape, with HTTP status 400 (bad request) or 500 (server error), proposed:

```json
{ "error": "<text>" }
```

### 10.3 The four endpoints from the earlier plan (same field names)

These four endpoints are planned in the earlier plan version. Nothing is built yet.

All values below are placeholders that show the **field names**. They are not real outputs.

GET /health

```json
{ "status": "<ok or error text>" }
```

POST /detect

```json
{ "label": "<real, ai or unclear>", "score": "<number between 0 and 1: higher = more likely AI-made>", "vae_score": "<number: VAE anomaly score>", "heatmap_png_base64": "<base64 text of a PNG heatmap>", "note": "<plain-English note, for example that this is a stub answer>" }
```

GET /audit/results

```json
{ "experiments": [ { "name": "<experiment name>", "x_label": "<for example training-set size>", "points": [ { "x": "<value>", "memorization_rate": "<value>" } ] } ], "note": "<for example: placeholder until scripts have run>" }
```

POST /audit/check

```json
{ "nearest": [ { "rank": 1, "distance": "<number>", "image_png_base64": "<base64 text>" } ], "flag": "<possible_copy or not_flagged>", "note": "<plain-English note: similarity only, not legal copying>" }
```

### 10.4 New fields and new endpoints

The old fields above stay. New fields are optional and are `null` when the feature is not built. For GET /audit/results the only added field is `examples`.

GET /audit/results, full shape with the one new optional field `examples` (written by a script):

```json
{
  "experiments": [
    {
      "name": "<experiment name>",
      "x_label": "<for example training-set size>",
      "points": [ { "x": "<value>", "memorization_rate": "<value>" } ]
    }
  ],
  "examples": [
    {
      "generated_png_base64": "<base64 PNG>",
      "nearest_png_base64": "<base64 PNG>",
      "distance": "<number>"
    }
  ],
  "note": "<for example: placeholder until scripts have run>"
}
```

POST /detect (multipart `image`), full shape with the new optional fields:

```json
{
  "label": "<real, ai or unclear>",
  "score": "<0-1, higher = more likely AI-made>",
  "vae_score": "<number>",
  "baseline_score": "<0-1 from the CNN baseline>",
  "heatmap_png_base64": "<base64 PNG or null>",
  "heatmap_source": "<ddpm or vae>",
  "gradcam_png_base64": "<base64 PNG or null>",
  "input_size": "<width x height>",
  "out_of_scope": "<true or false>",
  "warning": "<plain-English out-of-scope text or null>",
  "note": "<plain-English note>"
}
```

POST /audit/check, full shape with the new optional field `phash_distance`:

```json
{
  "nearest": [
    {
      "rank": 1,
      "distance": "<number>",
      "phash_distance": "<number or null>",
      "image_png_base64": "<base64 text>"
    }
  ],
  "flag": "<possible_copy or not_flagged>",
  "note": "<plain-English note: similarity only, not legal copying>"
}
```

POST /compare (multipart `image_a`, `image_b`, `use_words` true or false, and the optional `focus`):

```json
{
  "similarity": { "ssim": "<number -1 to 1>", "lpips": "<number or null>" },
  "diff_heatmap_png_base64": "<base64 PNG>",
  "regions_png_base64": "<base64 PNG or null>",
  "feature_table": [
    {
      "feature": "<name>",
      "image_a": "<value>",
      "image_b": "<value>",
      "does_not_prove": "<text>"
    }
  ],
  "differences": [
    { "part": "<attribute or part name, for example fur colour or ears>", "image_a": "<text>", "image_b": "<text>" }
  ],
  "source": "<vlm or heatmap_only>",
  "summary": "<plain-English text>",
  "note": "<pixel tools need aligned images; words may be wrong>"
}
```

Request notes for POST /compare:

- `focus` is the one NEW optional request field (proposed): a short text, up to 100 characters (proposed), for example "fur colour and eyes". It is used only in the Words prompt. If it is empty, the full checklist is used. The JSON response shapes do not change.
- In the response, `differences[].part` holds the attribute or part name from the checklist (for example "fur colour" or "ears"). The field names do not change.
- The `feature_table` includes a "dominant colours" row: three colour names with their share of each image.

GET /generators:

```json
{
  "generators": [
    {
      "id": "<generator id>",
      "name": "<text>",
      "dataset": "<text>",
      "family": "<vae, ddpm or cddpm>",
      "params": "<number>",
      "size_mb": "<number>",
      "status": "<ready, training or not_available>",
      "controls": ["<seed, n, steps, guidance, class_id, interpolate>"],
      "classes": ["<class names or empty>"],
      "licence_note": "<text>"
    }
  ],
  "note": "<text>"
}
```

POST /generate, request (JSON):

```json
{
  "generator_id": "<id>",
  "seed": "<int>",
  "n": "<1-16>",
  "steps": "<int or null>",
  "guidance": "<number or null>",
  "class_id": "<int or null>",
  "interpolate": { "seed_b": "<int>", "frames": "<int>" }
}
```

POST /generate, response:

```json
{
  "generator_id": "<id>",
  "images": [
    { "png_base64": "<base64>", "seed": "<int>", "class_id": "<int or null>" }
  ],
  "metadata": {
    "model": "<text>",
    "dataset": "<text>",
    "steps": "<int or null>",
    "guidance": "<number or null>",
    "label": "small model, expect low quality"
  },
  "note": "<text>"
}
```

POST /evidence (multipart `image`):

```json
{
  "exif": {
    "found": "<true or false>",
    "fields": { "<tag>": "<value>" },
    "does_not_prove": "Missing EXIF is weak evidence; present EXIF does not mean real."
  },
  "c2pa": {
    "status": "<valid, invalid, none or not_checked>",
    "summary": "<text or null>",
    "does_not_prove": "A missing manifest means nothing."
  },
  "ela_png_base64": "<base64 or null>",
  "fft_png_base64": "<base64 or null>",
  "note": "Evidence, not proof."
}
```

The final contract file `api/contract.json` will use real JSON numbers. Nitin writes it on Monday evening. Tejash builds stub JSON from `api/contract.json` (shown in Appendix A of the team plan), so the pages can be built before the real backend exists.

### 10.5 Hosting facts (from the official docs) and the laptop-demo decision

| Service | Facts | Status |
|---------|-------|--------|
| Hugging Face Docker Spaces | Free Docker Spaces are NOT available (a paid plan is required). | Confirmed from the docs. CPU Basic figures: to verify. |
| Render free web service | 750 h/month. Spins down after 15 minutes idle (about 1 minute to wake). RAM and CPU not published. | Limits: to verify |
| Vercel Hobby | Free for a static site. A non-commercial restriction was not found. The terms of service were not read. | Terms: to verify |

Decision: the LIVE DEMO runs from a LAPTOP. FastAPI serves the API and the built website runs at localhost. A recorded backup video is made. Vercel and Render are a bonus, tried only if everything else is done and the result works fully by Wed 20:00.

### 10.6 Wiring and key rules

- base64: a way to write an image as text, so that it can travel inside JSON.
- CORS: a browser rule about which websites may call the API. The backend will allow the website's local address.
- `VITE_API_URL`: the address of the API, read by the website. Any `VITE_` variable is shipped to the browser, so it must never hold a key.
- The vision-API key lives only in `api/.env`: never in `web/`, never in git, never pasted into chat or logs.
- The REAL CIFAR-10 example images stay out of git (section 6.6).

## 11. Schedule and gates

Clock times other than the three gates are proposed. The older Day 1-4 (Tue-Fri) schedule is replaced. The "Tue 14:00 hosting test" is dropped, because hosting is first in the cut line.

| When | Nitin (backend, models, wiring) | Tejash (website) |
|------|----------------------------------|------------------|
| **Mon 23 Nov, evening** (proposed 18:00-23:00) | 1. Fix `.gitignore` FIRST (list in section 10.1) and push it straight to main (see 11.3). 2. Write `api/contract.json`. 3. Check the Kaggle quota page and note the reset day (C9). 4. Set up the environment and download the datasets (see 11.3), then make `data/splits.csv` with a fixed seed. 5. Start `g-vae-cifar`, `d-vae-cifar` and `cnn-baseline` on the laptop. 6. Start `g-vae-fmnist`. 7. Ask the teacher about the Words panel. | 1. List the existing site (pages, framework, repo link) in `docs/log/tejash.md`. 2. Bring it into the repo under `web/` by a first pull request. 3. Add the 6 routes. 4. Write stub JSON from `api/contract.json` (shown in Appendix A of the team plan). |
| **Tue 24 Nov** (proposed 09:00-18:00) | 1. Kaggle: `d-ddpm-cifar` and `g-ddpm-cifar` (cosine schedule, different seeds). 2. Kaggle: `g-cddpm-fmnist` with guidance. 3. FastAPI: /health, /generators, /generate (VAEs). 4. /detect (CNN-baseline label, VAE score, VAE heatmap, out_of_scope). 5. /compare (SSIM, feature table with dominant colours). 6. Auditor nearest-neighbour script, then start `a-500`. 7. Export the fake "Try an example" images to Tejash and share the real ones by file transfer (section 6.6). 8. When the Kaggle jobs finish, download the weights to the laptop (kept out of git). | 1. Generator page. 2. Detector page (upload, "Try an example", red banner, scores, heatmap, Evidence tab showing null as "not available"). 3. Compare page (two uploads, "What should I compare?" box, SSIM map, feature table with dominant colours, Words panel with privacy note and off state). 4. Collect 20 photo pairs of mixed subjects (own photos, no people), each with a written answer key. 5. Open a pull request by 18:00. |
| **Wed 25 Nov** | 1. Gate 2 check at 12:00. 2. Generate fakes, train the logistic regression on the features (its label replaces the CNN-baseline label in /detect), score the final test split once. 3. Leave-one-generator-out and robustness into the results JSON. 4. Runs `a-10000` and `a-500-dup`, then /audit/results. 5. SHOULD items in order: Words (only if the teacher answered; otherwise it stays off), Grad-CAM, EXIF and evidence, then pHash plus the Auditor side-by-side. 6. Merge Tejash's pull request and connect the site to the real API. 7. Freeze at 16:00. 8. Two end-to-end runs and the backup video (16:00-20:00). 9. Hosting only if all is done. 10. Hard stop 22:00. | 1. Auditor page (plots, nearest-neighbour grid, placeholders). 2. `/limits` page with the honesty lines and the "does not prove" column. 3. Pair with Nitin on wiring and fix UI bugs. 4. Hand-grade the 20 Compare pairs once Words works. 5. Freeze at 16:00. 6. Help record the backup video. |
| **Thu 26 Nov** | Update the deck from the results JSON. Rehearse twice. Present the Detector and the Auditor. | Rehearse. Present the Generator, Compare and Limits. |

Thursday has no new features. The presentation time is to confirm.

### 11.1 Gates

| Gate | Time | Passes if | If it fails |
|------|------|-----------|-------------|
| Gate 1 | Tue 18:00 | Stub pages render, the VAEs are trained and the Kaggle DDPM jobs are running. | Start cutting from the top of the cut line in section 12. |
| Gate 2 | Wed 12:00 | There are usable DDPM samples. | Cut `g-ddpm-cifar` and `d-ddpm-cifar`. The Detector keeps the VAE reconstruction-error map, which is the default heatmap anyway (`heatmap_source` is `vae`). Leave-one-generator-out is reported as "not possible". The Detector uses VAE fakes only, stated as a limitation. |
| Gate 3 | Wed 16:00 | Feature freeze: nothing new after this time. | Anything unfinished stays out. |

Wed 20:00 is the hosting decision (proposed). Wed 22:00 is the hard stop (proposed).

### 11.2 Who does what (an ASSUMPTION to confirm with Tejash)

Nitin owns the backend, all models, Kaggle, the `model/` scripts, the results JSON, `api/contract.json` and Wednesday's wiring of the site to the real API. Tejash owns the existing site: listing it, upgrading it into the 6 pages against stub JSON, the Limits page text, collecting the Compare photo pairs and hand-grading the Compare words. All hand-overs go through a pull request reviewed by the OTHER person. The only exception is the very first push (the `.gitignore` fix), which goes straight to main. Tejash does no backend work.

### 11.3 Rules and setup steps

- **First push:** the very first push (the `.gitignore` fix) goes straight to main, as the one exception to the pull-request rule. On Monday nobody has a merged pull request to read yet.
- **Environment:** create a `.venv` and install pinned requirements. Download CIFAR-10 and Fashion-MNIST with torchvision (kept out of git). Bring the Kaggle outputs (weights) to the laptop by download. Weights stay out of git and are shared by file transfer.
- **Requirements:** pin the versions with `pip freeze` into `requirements.txt`. Run `git status` before every commit.
- **Words panel:** ask the teacher on Monday. Build it, but keep it OFF by default. If there is no answer by Wednesday morning, it stays off. The key lives only in `api/.env` (never in `web/`, never in git, never pasted into chat or logs). A `VITE_` variable is shipped to the browser, so the key must never use that prefix.
- **Try-an-example real images:** shared by file transfer into `web/public/examples/real/`, which is ignored by git.

## 12. Risks and cut line

### 12.1 Risks

| Risk | What could happen | What we do |
|------|-------------------|------------|
| Kaggle quota | Free GPU hours run out or reset late (C9). The weekly quota (about 30 h) is not confirmed. | Check our own quota page on Monday. Train `d-ddpm-cifar` and `g-ddpm-cifar` at fewer epochs. If it is still too slow, cut both (cut line 6). |
| DDPM does not converge | No usable DDPM samples by Gate 2. | Cut line 6. The Detector keeps the VAE heatmap and uses VAE fakes only. |
| No API key | The Words panel cannot run. | Words stays off. `source` is `heatmap_only`. |
| No answer from the teacher about the vision API | Words may not be allowed. | Words is OFF by default and stays off if there is no answer by Wednesday morning. |
| The key leaks | The vision-API key ends up in the browser, in git, or in chat or logs. | The key lives only in `api/.env`. No `VITE_` variable holds a key. Run `git status` before every commit. |
| Site integration | The real API and the site do not fit together by the freeze. | The site keeps using stub JSON (from `api/contract.json`, shown in Appendix A of the team plan). Demo the pages on stub JSON plus the backend docs page, and say so. |
| CPU speed | Image generation or scoring is too slow on a laptop. | Fewer DDIM steps, smaller n. |
| Hosting RAM unknown | Render free RAM and CPU are not published. | The demo stays on the laptop. Hosting is a bonus only. |
| Data or weights committed by mistake | Datasets or weights end up in git. | Fix `.gitignore` first. If it happens, stop and ask Nitin. Do not push. |

### 12.2 Cut line (first to be cut, to last)

COULD extras are cut first after cloud hosting (items 1 and 2).

| Order | Cut |
|-------|-----|
| 1 | Cloud hosting |
| 2 | COULD extras (DINOv2, LPIPS, ELA, FFT, C2PA, robustness sliders, watermark idea) |
| 3 | POST /audit/check |
| 4 | pHash and the Auditor side-by-side diff |
| 5 | Grad-CAM and EXIF |
| 6 | `g-ddpm-cifar` and `d-ddpm-cifar` |
| 7 | Compare words (heatmap-only fallback) |
| 8 | Auditor runs 3 down to 1 |

The Auditor's nearest-neighbour grid (MUST) does not depend on item 3, because it reads the `examples` field of GET /audit/results.

**ALWAYS KEEP:** `.gitignore` protection, `cnn-baseline`, VAE score, /detect with "Try an example" and the out-of-scope warning, /generate for `g-vae-cifar`, `g-vae-fmnist` and `g-cddpm-fmnist`, the Compare SSIM map and feature table, the Auditor dashboard, /limits, the laptop demo with a recorded backup, honest placeholders.

Model Zoo extras stay PARKED.

## 13. Roadmap

| Phase | When | What |
|-------|------|------|
| Version 1 | Thursday 26 Nov 2026 | The six pages, the eight endpoints, the results JSON, the deck, the recorded backup. Narrow: 32x32 CIFAR-10 detector and auditor. |
| Broad track | After version 1 | Documentation only (section 14). Narrow v1 is the fallback. |
| Model Zoo extras | Parked | SVM, decision tree, ensembles, Level 3 CNN, regression, KNN/RF. Future work. Not built. |

## 14. Broad track (docs only)

The Broad track is documentation only in this plan. The text below is carried over from the earlier plan version.

### 14.1 Broad track (PROPOSED)

The broad track is the goal. The narrow v1 detector above (3.1 to 3.6) **remains the fallback**.

- **Encoder:** a frozen CLIP encoder with a linear probe (a simple classifier on top; the encoder is not fine-tuned), trained on fakes from many open generators. The generator list is kept short.
- **Public model encoder:** the LAION-trained OpenCLIP ViT-L/14 (Hugging Face licence tag MIT; the card uses research-purposes wording; deployed-use status TO VERIFY).
- **Not used in the public demo:** the OpenAI CLIP ViT-L/14 weights, because the model card says any deployed use is out of scope.
- **Output:** a label of real, ai or unclear. The "unclear" band thresholds are PROPOSED, to tune.
- **Input size:** each source is resized to the encoder input size (exact size TO VERIFY). Version 1 keeps 32x32.
- **Fakes come from open multi-generator datasets only.** No closed-service outputs. The public model uses permissively licensed sources only. Non-commercial (NC) datasets are for offline evaluation only.

### 14.2 Data sources for the broad track

| Name | Generators | Licence | Size | Resolution | Download status | Used for |
|------|-----------|---------|------|------------|-----------------|----------|
| GenImage | BigGAN; GLIDE, VQDM, SD v1.4, SD v1.5, ADM, Midjourney, Wukong | CC BY-NC-SA 4.0 (official site; GitHub LICENSE not checked) | 2,681,167 images | 128 to 1024 px | Download links UNVERIFIED | Offline evaluation only |
| Community Forensics | 4,803 generator models (mostly latent diffusion) | cc-by-4.0 on HF vs CC BY-NC-SA in a search snippet (CONFLICT, UNRESOLVED); generator outputs follow each model's own licence | About 2.76M images, about 1.08 TB | UNVERIFIED | UNVERIFIED | Use TO VERIFY |
| OpenFake | About 75 model values | cc-by-nc-4.0; GPT Image and proprietary-generator subsets are non-commercial | CONFLICT (about 2.86M vs about 4M) | UNVERIFIED | UNVERIFIED | Offline evaluation only |
| Synthbuster | DALL-E 2, DALL-E 3, Firefly, Midjourney v5, SD 1.3, 1.4, 2, XL, GLIDE | CC BY-NC-SA 4.0 | 12.4 GB | UNVERIFIED | UNVERIFIED | Offline evaluation only |
| OpenSDI | Diffusion edits (SD1.5, SD2.1, SDXL, SD3, Flux.1) | CC BY-SA 4.0 | UNVERIFIED | UNVERIFIED | Raw images NOT yet released | Not usable yet |
| CIFAKE | SD v1.4 only | MIT on HF mirror; Kaggle licence field unconfirmed | CONFLICT (120,000 vs 100,000) | UNVERIFIED | UNVERIFIED | Not decided; TO VERIFY |
| WildFake | GANs and diffusion | UNVERIFIED | Count CONFLICT | UNVERIFIED | No download link | Not usable yet |
| DRCT-2M, AI-GenBench, DRAGON, Semi-Truths | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | Not usable until checked |
| Closed-service outputs | - | - | - | - | - | NOT USED (see section 11) |

---

_Carried over unchanged from the previous PLAN.md. Section numbers written inside this block (for example 'section 3.7') refer to the old file._


### 14.3 Compute for the broad track


- Conflict C1: this plan says inference runs on a free CPU, but the broad track needs more compute. Encoder size against the 100 MB target and against free-CPU latency is UNVERIFIED.
- Kaggle GPU quota is about 30 hours a week (confidence MEDIUM-LOW). The reset day is in conflict (C9). Colab has no published quota.

---

_Carried over unchanged from the previous PLAN.md. Section numbers written inside this block (for example 'section 3.7') refer to the old file._


## 15. Licence and demo check


| Item | Finding | Status |
|------|---------|--------|
| OpenAI CLIP code | MIT | VERIFIED |
| OpenAI CLIP ViT-L/14 weights | No licence field; model card says deployed use is out of scope | EXCLUDED from the public demo |
| LAION OpenCLIP ViT-L/14 weights | MIT tag with research-purposes wording | TO VERIFY |
| OpenCLIP code | Licence not checked | UNVERIFIED |
| GenImage, OpenFake, Synthbuster | Non-commercial | Offline evaluation only |
| CIFAKE | MIT on HF mirror | TO VERIFY |
| SD 3.5 Large | Community License, under $1M revenue | Conditions as stated |
| FLUX.1 [schnell], Qwen-Image | Apache 2.0 | As stated by the research |
| FLUX.1 [dev], FLUX.2 [dev], Stable Cascade | Non-commercial | Excluded from the public model |
| Closed services (Adobe, OpenAI, Google, Midjourney) | Adobe clear prohibition; OpenAI and Google ambiguous; Midjourney unreadable | Excluded |

---

_Carried over unchanged from the previous PLAN.md. Section numbers written inside this block (for example 'section 3.7') refer to the old file._


New rows for this version:

| Item | Licence | Source status | Used for |
|------|---------|---------------|----------|
| imagehash | BSD-2-Clause | verified from the README | pHash on the Auditor |
| scikit-image | BSD-3-Clause | verified from the README | SSIM, statistics |
| lpips | BSD-2-Clause | verified from the README | LPIPS on Compare |
| DINOv2-small | Apache-2.0 | verified from the model card | Semantic regions on Compare |
| c2pa-python | Apache-2.0 + MIT per README | README badge only | C2PA read |
| pytorch-grad-cam | MIT per README badge | README badge only | Grad-CAM |
| invisible-watermark | MIT | verified from the README | Watermark idea |
| Pillow | not verified here | to verify | ELA, EXIF, PNG text note |
| scikit-learn | licence not verified here; to verify | to verify | Dominant colours (KMeans), logistic regression |

## 16. Reproducibility checklist

- Fixed random seeds, set in **one** place.
- A saved config file for every run (epochs, sizes, t, thresholds).
- One `results/` JSON file per experiment, written by a script.
- Every number in any slide or report is traced to a script name.
- Package versions pinned in `requirements.txt`.
- No secrets and no datasets in the repo.
- Results stay as `[PLACEHOLDER - not measured yet]` until they are real.

## 17. Decisions

| ID | Decision | Status |
|----|----------|--------|
| D-A | docs only for the Broad track, narrow v1 is the fallback (reaffirmed) | KEEP |
| D-B | open multi-generator datasets only | KEEP |
| D-C | permissive sources for the public model, NC offline only | KEEP |
| D-D | frozen encoder + linear probe, shorter generator list | KEEP |
| D-E | native sizes for broad, 32x32 for v1, exact size to verify | KEEP |
| D-F | auditor unchanged, own generators only, cannot audit GPT | KEEP |
| D-G | The live demo runs from a laptop, with a recorded backup. Cloud hosting is a bonus. | NEW |
| D-H | G3 (second DDPM) and DCGAN are cut. Leave-one-generator-out uses the two CIFAR generators. | NEW |
| D-I | The out-of-scope rule: the Detector flags any upload that is not exactly 32x32. It catches size only: a 32x32 photo that is not CIFAR-style is not flagged (known limitation). | NEW |
| D-J | Thresholds are chosen on a calibration split (5,000 images), not on the plain test set. | NEW |
| D-K | Try-an-example real images are not committed (CIFAR-10 licence to verify). Fake examples, our own samples, may be committed. | NEW |

## 18. Open conflicts

| ID | Conflict | Status |
|----|----------|--------|
| C1 | free-CPU vs broad compute (partly settled: demo is a laptop; broad stays docs-only) | partly settled |
| C2 | GenImage licence (open) | open |
| C3 | Community Forensics licence (open) | open |
| C4 | OpenFake size | open |
| C5 | Semi-Truths count | open |
| C6 | WildFake count | open |
| C7 | CIFAKE size | open |
| C8 | OpenAI ToU (closed services excluded anyway) | open, closed services excluded anyway |
| C9 | Kaggle GPU reset day (critical for the 3-day window) | open, critical |
| C10 | Kaggle TPU hours (low) | open, low priority |
| C11 | CLIP card vs a public demo (settled: no public CLIP demo) | settled |
| C12 | FLUX.1 [dev] terms (only relevant if the Generator used FLUX: it will not) | not relevant |

## 19. Things still to verify

From the earlier list:

- DIRE venue (ICCV 2023)
- Corvi et al. author list
- all CIFAKE facts
- Kaggle free GPU limits
- Render free-service limits and Hugging Face Docker (not available on free: confirmed)
- HF CPU Basic figures
- Vercel free plan terms (non-commercial restriction not found)
- epochs, set sizes, t (about 200), percentile (1st), thresholds: proposed
- CIFAR-10 licence

Gaps:

- CLIP input size
- deployed-use status of CLIP weights
- LAION MIT tag
- OpenCLIP code licence
- GenImage download links
- download/size status of broad sources
- OpenSDI raw release
- Community Forensics licence conflict

New in this version:

- MNIST and CelebA official terms (pages unreachable)
- EMNIST terms (Readme not read)
- SVHN 'non-commercial use only'
- c2pa-python and pytorch-grad-cam licences (README badges only)
- DINOv2 licence is Apache-2.0 (model card; confirmed)
- Kaggle T4 training times (not confirmed)
- Pillow PngInfo.add_text (check source)
- the vision-API provider's data-use terms
- API key/credits
- per-model training times (every model in section 4)
- the out-of-scope 32x32 rule (proposed: it only catches image size; is "exactly 32x32" the right rule?)
- whether the teacher allows an outside vision API for the Words panel (ask on Monday)

Other gaps noted during research (all to verify):

- Kaggle weekly GPU quota (about 30 h) is not confirmed. Check our own quota page.
- DINOv2-small CPU timing was not checked.
- Pillow licence was not verified here.
- Occlusion sensitivity is not confirmed from a source.
- C2PA spec 2.3 exists: confirm which version is current.
- Quality of the small open vision models for fine-grained parts is not verified.
- The Vercel terms of service were not read.
- The ELA reference title comes from a search listing.

## 20. Sources

These are the citations used or discussed in this plan. "Venue not confirmed" is kept where it applies.

**Detection and explanation**

- Grad-CAM. Selvaraju, Cogswell, Das, Vedantam, Parikh, Batra. ICCV 2017. arXiv 1610.02391.
- Durall, Keuper, Keuper. "Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions". CVPR 2020.
- Frank, Eisenhofer, Schonherr, Fischer, Kolossa, Holz. "Leveraging Frequency Analysis for Deep Fake Image Recognition". ICML 2020 (PMLR 119).
- Ojha, Li, Lee. "Towards Universal Fake Image Detectors that Generalize Across Generative Models". CVPR 2023. arXiv 2302.10174. Paper fact used: standard detectors "fail to detect fake images from newer breeds of generative models when trained to detect GAN fake images".
- DIRE. Wang et al. arXiv 2303.09295. Venue to verify.
- Corvi et al. ICASSP 2023. arXiv 2211.00680. Authors to verify.

**Similarity and comparison**

- Wang, Bovik, Sheikh, Simoncelli. "Image Quality Assessment: From Error Visibility to Structural Similarity". IEEE Trans. Image Processing 13(4):600-612, 2004. (SSIM)
- Zhang, Isola, Efros, Shechtman, Wang. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric". CVPR 2018. arXiv 1801.03924. (LPIPS)
- Amir, Gandelsman, Bagon, Dekel. "Deep ViT Features as Dense Visual Descriptors". arXiv 2112.05814. Venue not confirmed.
- DINOv2. Oquab et al. arXiv 2304.07193. Venue not confirmed.
- CLIP. Radford et al. arXiv 2103.00020. Venue not confirmed here.
- ELA is not a paper. N. Krawetz, "A Picture's Worth...", Black Hat USA 2007 (title from a search listing).

**Generators and memorization**

- DDPM. Ho, Jain, Abbeel. arXiv 2006.11239.
- DDIM. Song, Meng, Ermon. ICLR 2021. arXiv 2010.02502.
- "Classifier-Free Diffusion Guidance". Ho and Salimans. arXiv 2207.12598 (NeurIPS 2021 DGM workshop short version).
- "Auto-Encoding Variational Bayes". Kingma and Welling. arXiv 1312.6114.
- DCGAN. Radford, Metz, Chintala. arXiv 1511.06434 (ICLR 2016 acceptance not confirmed).
- "Improved Denoising Diffusion Probabilistic Models". Nichol and Dhariwal. arXiv 2102.09672.
- Carlini et al. "Extracting Training Data from Diffusion Models". USENIX Security 2023. arXiv 2301.13188.

**Datasets**

- Krizhevsky 2009 (CIFAR-10).
- EMNIST. Cohen, Afshar, Tapson, van Schaik (2017). Discussed, not used.
- CIFAKE. Bird and Lotfi. arXiv 2303.14126. All facts to verify.
