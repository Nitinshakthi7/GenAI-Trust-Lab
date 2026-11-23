# GenAI Trust Lab - Project Plan

**What this document is**

- This is the plan for our final project. It is written before any model code exists.
- It says what we will build, how we will test it, and who does what.
- Nothing here is a result. Every number is either a design choice marked "proposed, to tune" or a placeholder.
- Anything we have not checked yet is tagged "to verify" and collected at the end of the file.
- It is written for two third-year B.Tech CSE (AI/ML) students; one is new to programming, so terms are explained the first time they appear.

---

## 1. Problem statement and goals

We work on two related problems about AI-made images.

| # | Problem | In plain words |
|---|---------|----------------|
| 1 | Detection | Given one image, was it made by an AI image generator, or is it a real photo? Our goal is a detector that explains its answer, not only a score. Detectors trained on one generator did not generalize well to others ("generalize" means "work on new kinds of images"). Ojha et al. (CVPR 2023, arXiv 2302.10174) found that a detector trained on one generator family, for example GANs, did not generalize well to diffusion models. Corvi et al. (ICASSP 2023, arXiv 2211.00680 - author list to verify) found that detectors trained on one generator did not generalize well to other diffusion models. |
| 2 | Memorization | Did a generator copy one of its own training images? Carlini et al. (USENIX Security 2023, arXiv 2301.13188) report that diffusion models memorize individual training images and emit (produce) them at generation time. |

**Goals**

- A **detector** that explains its decision. It shows a heatmap of where the image looks "wrong", plus a score.
- An **auditor** that measures how often a generator produces images that are very close to its training images.
- A small website and API so a professor can try both tools in a browser.
- Honest reporting: where a tool fails, we show the failure.

**Non-goals (not in version 1)**

- Adversarial attacks (someone deliberately fooling the detector). We do not test or defend against them.
- Legal judgements. The auditor measures **similarity between images**. It does not decide copyright infringement or "legal copying".
- Video. Version 1 is still images only.

**Important honesty note.** Detecting images from a generator the detector has never seen is an open research problem. We will **report the drop in performance as a finding**. We do not promise the detector will work on unseen generators.

---

## 2. Datasets

Datasets are never stored in the repo. Scripts download or load them at run time.

**Main track:** CIFAR-10 at 32x32. **Fallback track:** Fashion-MNIST at 28x28 (used if the 32x32 DDPM does not train well enough in time).

| Dataset | What it is | Role | Facts | Status |
|---------|-----------|------|-------|--------|
| CIFAR-10 (Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009) | Real images | Real class, training data for our generators, held-out test images for the auditor | 32x32 colour, 50,000 train + 10,000 test. The official page states no licence (to verify). | Main track |
| Fashion-MNIST (Zalando Research) | Real images | Fallback real class | 28x28 grayscale, 60,000 train + 10,000 test, MIT licence | Fallback track |
| Fakes from our small convolutional VAE (G1) | Fake images we make | Fake class | Trained by us on real CIFAR-10 | Planned |
| Fakes from our small DDPM (G2) | Fake images we make | Fake class | Trained by us on real CIFAR-10 | Planned |
| Fakes from a second DDPM (G3, "DDPM-B") | Fake images we make | Fake class. Trained with different epochs or size so we can do leave-one-generator-out. | Settings proposed, to tune | Planned |
| CIFAKE (Jordan J. Bird and Ahmad Lotfi, "CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images", arXiv 2303.14126) | Public real-vs-AI set | Extra "unseen generator" test set **only** | Licence: to verify. Image count (100k vs 120k): to verify. Image size: to verify. Kaggle terms: to verify. Journal venue: reference to verify. | Use only after the checks above |

**What leave-one-generator-out means.** We will train the detector on fakes from two generators and test it on the third one it has never seen. We will repeat this three times, leaving out a different generator each time.

**Course code we may reuse as a starting point** (allowed, but it must be adapted):

| Piece | Where it comes from |
|-------|---------------------|
| Timestep embedding | `Nitin_AU24UG-027_Day6_Diffusion.ipynb` |
| Noise schedule: T=1000 steps, linear betas from 1e-4 to 0.02 (values from the course notebook; to tune) | `Nitin_AU24UG-027_Day6_Diffusion.ipynb` |
| TinyUNet (the small denoising network) | `Nitin_AU24UG-027_Day6_Diffusion.ipynb` |
| Train loop and sample loop | `Nitin_AU24UG-027_Day6_Diffusion.ipynb` |
| LetterCNN (basis for our CNN baseline) | `RollNo_CondDiffusion.ipynb` |

**Warnings about the notebooks.**

- They use Fashion-MNIST / EMNIST CSV files. Those files are **not** in the project folder.
- They contain Colab / Kaggle file paths. These paths must be changed before the code runs anywhere else.
- Our code has to handle CIFAR-10 (3 colour channels, 32x32), so the course networks need small changes.

---

## 3. Detector design

**Idea in one line.** A diffusion model trained only on real images should "repair" real images well and fake images less well. The size of the repair error is a clue.

### 3.1 Features (numbers the classifier will look at)

1. **Diffusion reconstruction error** (the idea behind DIRE, "DIRE for Diffusion-Generated Image Detection", Wang et al., arXiv 2303.09295; ICCV 2023 - venue to verify. Its abstract says DIRE measures the error between an input image and its reconstruction by a pre-trained diffusion model).
   - Add noise to the image up to timestep t. Proposed t is about 200 out of T=1000, to tune.
   - Denoise it back with a DDPM trained **only on real CIFAR-10**.
   - Take the per-pixel error between the input and the reconstruction. This gives a **heatmap**.
   - Summarise the heatmap as numbers: the mean error and some error percentiles.
2. **VAE reconstruction error and KL.** A small convolutional VAE (variational autoencoder: a model that squeezes an image into a few numbers and rebuilds it) will be trained on real images. For each image we will compute how badly it is rebuilt (reconstruction error) and how unusual its squeezed numbers look (KL term). The two will be combined into a "VAE anomaly score".

**Note on the course notebook (B16).** In `Nitin_AU24UG-027_Day6_Diffusion.ipynb`, B16 trains a diffusion model on 2-D VAE codes. It does **not** compute an anomaly score. The VAE reconstruction + KL anomaly score is **new work built on the course VAE idea**. We use a small **convolutional** VAE; the notebook VAE is a small MLP with a 2-D latent.

### 3.2 Classifier

- **Logistic regression** (scikit-learn) on the features above. It is simple, fast on CPU and easy to explain.

### 3.3 CNN baseline

- A small 3-layer CNN based on `LetterCNN` from `RollNo_CondDiffusion.ipynb`. It will look at the raw image and say real or fake.
- We keep this baseline always, so we have something to compare against.

### 3.4 Keeping the detector fair

- The detector's reconstruction DDPM must be a **different model and a different seed** from the fake-making DDPMs (G2, G3).
- **Caveat (to report honestly).** If the detector's reconstruction DDPM and one of our fake-making DDPMs share architecture or training data, detection of that generator may look better than it would on a truly unseen generator. We will state this next to the results.

### 3.5 Evaluation protocol

| Check | What we do |
|-------|-----------|
| Accuracy and AUROC per generator | Report both for G1, G2, G3 and CIFAKE (if allowed). AUROC is a score of how well real and fake are separated, ignoring any single threshold. |
| Leave-one-generator-out | Train on two generators, test on the third. Report the drop as a finding. |
| Robustness | Re-test after JPEG compression, blur and added noise. |
| Adversarial attacks | Not done in version 1. |

All result cells stay `[PLACEHOLDER - not measured yet]` until a script fills them.

### 3.6 Open design question

How to run the diffusion reconstruction on a free CPU (few denoising steps, or a deterministic sampler) is **to be decided on Day 2**.

---

## 4. Auditor design

**Question.** How often does a generator produce an image that is almost the same as one of its training images?

**Metric.** For each generated image, find its **nearest training image** and take the **L2 distance** (straight-line distance between the pixel values, at 32x32).

**Threshold calibration.**

- Take held-out CIFAR-10 **test** images. These must never be used for training.
- Find each one's nearest-training-image distance. This tells us what "normal closeness" looks like for images that were **not** copied.
- Flag a generated image as **"possible copy"** if it is closer than a low percentile of that distribution. Proposed: the 1st percentile, to tune.
- Optional cross-check (proposed, to tune): describe the held-out nearest-neighbour distances with a simple fitted distribution (MLE, Gaussian) and, only if the histogram clearly has two bumps, a 2-component GMM. The percentile threshold stays the main rule.

**Experiments.**

| Experiment | Proposed setting |
|------------|------------------|
| Training-set size | 500 / 2,000 / 10,000 images (proposed, to tune) |
| Deliberately duplicated images | Put some training images in several times and see if the generator copies them more often. How many copies: proposed, to tune. |
| Epochs | Number of epochs to compare: proposed, to tune |

**Mitigations to test** (ways to reduce copying): deduplication (remove repeated images), flip augmentation (also train on mirrored images), fewer epochs, smaller training sets.

**Plots we will make.**

1. Memorization rate vs training-set size.
2. Memorization rate vs epochs.
3. Distance histograms: generated images vs held-out test images.
4. A grid of generated images placed beside their nearest training images.

**Wording rule.** We say "similar" or "possible copy". We never say "copyright infringement". We cite Carlini et al. (arXiv 2301.13188) for the idea that diffusion models can memorize training images.

---

## 5. System design

Everything in this section is **PLANNED**. None of it exists yet.

### 5.1 Repo layout (PLANNED)

```text
GenAI_Trust_Lab/
  PLAN.md          this file
  docs/            deck, outline and notes (build_docs.py makes the deck and outline)
  model/           PLANNED: training and scoring scripts (VAE, DDPM, CNN, features, classifier, auditor)
  api/             PLANNED: the FastAPI app, and first a "stub" that returns fixed fake answers
  web/             PLANNED: the React + Vite website
  results/         PLANNED: JSON files written by scripts, one per experiment
  requirements.txt PLANNED: pinned Python package versions
```

A **stub** is a fake version of the API that returns made-up answers in the agreed format. It lets the web side build everything without waiting for the model.

### 5.2 API endpoints (shape only, placeholder values)

All values below are placeholders that show the **field names**. They are not real outputs.

**GET /health**

```json
{ "status": "<ok or error text>" }
```

**POST /detect** (upload an image)

```json
{
  "score": "<number between 0 and 1: higher = more likely AI-made>",
  "vae_score": "<number: VAE anomaly score>",
  "heatmap_png_base64": "<base64 text of a PNG heatmap>",
  "note": "<plain-English note, for example that this is a stub answer>"
}
```

**GET /audit/results** (precomputed experiment results)

```json
{
  "experiments": [
    {
      "name": "<experiment name>",
      "x_label": "<for example training-set size>",
      "points": [ { "x": "<value>", "memorization_rate": "<value>" } ]
    }
  ],
  "note": "<for example: placeholder until scripts have run>"
}
```

**POST /audit/check** (upload an image; checked against a small training subset only)

```json
{
  "nearest": [
    { "rank": 1, "distance": "<number>", "image_png_base64": "<base64 text>" }
  ],
  "flag": "<possible_copy or not_flagged>",
  "note": "<plain-English note: similarity only, not legal copying>"
}
```

### 5.3 Website pages

| Page | What it shows |
|------|---------------|
| Home | What the project is, in plain words, and links to the tools |
| Detector | Upload an image, see the score, the VAE score and the heatmap |
| Auditor dashboard | Charts from `GET /audit/results` (memorization vs training-set size, and so on) |
| Limits | What the tools cannot do, and the open problems |

### 5.4 Hosting and size limits

- Website: React + Vite, hosted on **Vercel** (free plan terms: to verify).
- API: FastAPI in a **Hugging Face Space** using Docker. Our notes say the free CPU Basic Space has 2 vCPU and 16 GB RAM and sleeps when unused (figures to verify on Day 2).
- **Whether a free Hugging Face Space can run Docker + FastAPI is to verify.** We test this on Day 2 by deploying the stub. Named fallback: a **Render** free web service (its limits are also to verify).
- Trained weights should stay **under about 100 MB** in total (proposed target).
- Inference (running the model on one uploaded image) must work on a **free CPU**. No GPU on the server.

---

## 6. Day-by-day schedule (4 days)

The web side should not have to wait for the model, because the stub API will return the agreed JSON shape from Day 1.

| Day | Member A (ML) | Member B (web) | Done when |
|-----|---------------|----------------|-----------|
| 1 | Data loading for CIFAR-10. Small convolutional VAE. CNN baseline. | Vite website skeleton. Stub API. Together with A: agree the **API contract** as one JSON spec file. | The JSON spec file is agreed by both. The stub answers `/health` and `/detect` on a laptop. VAE and CNN code run on a few images. |
| 2 | Train the DDPMs on Kaggle: the detector's reconstruction model (real images only) plus G2 and G3. Make the fake images. | Build the Detector page. Deploy the stub to Hugging Face (**this is the hosting test**). | Fake images are saved. The stub is reachable at a public URL, or we have switched to the Render fallback. |
| 3 | Build the features and the classifier. Run the auditor experiments. Write results JSON. | Build the Auditor dashboard. Connect the real API. | The API returns a real score from a trained model. The dashboard draws from a real results file. |
| 4 | Robustness tests (JPEG, blur, noise). Fill the result tables **from scripts**. | Integrate, finish the Limits page, polish. | End-to-end demo runs twice without help. Every table cell comes from a script, or is still marked as a placeholder. |

Both members rehearse the demo together on Day 4.

---

## 7. Risks and cut line

**Cut line.** If we run out of time, we drop things in this order:

1. First drop: the upload auditor (`POST /audit/check`).
2. Second drop: the diffusion-based heatmap.

**Always keep:** the CNN baseline, the VAE score, and the results dashboard.

| Risk | What could happen | What we do |
|------|-------------------|-----------|
| Kaggle GPU limits (to verify) | Not enough free GPU time to train all DDPMs | Smaller models, fewer epochs, or the Fashion-MNIST fallback. Check the real limits first. |
| Hugging Face free tier (to verify) | The Space may not run Docker + FastAPI, or may sleep | Test on Day 2. Fallback: Render free web service (limits to verify). |
| DDPM is slow on CPU | Heatmap takes too long per image | Use few denoising steps. The heatmap is on the cut line. |
| 32x32 DDPM may not train well enough in time | Fakes look poor | Fall back to Fashion-MNIST 28x28 with the course notebook DDPM. |
| CIFAKE licence and size unknown | We may not be allowed or able to use it | Use it only after the checks. The project works without it. |
| Notebook paths and data files | Course notebooks use CSV files and Colab/Kaggle paths that are not here | Rewrite the data loading; do not copy paths blindly. |
| Only 2 people | One person sick or stuck blocks the project | The stub keeps the web side independent. The cut line limits scope. |
| Optimistic results | The detector's reconstruction DDPM and a fake-making DDPM share architecture or training data, so detection of that generator may look better than on a truly unseen generator | Use a different model and seed for the detector's DDPM, and state this caveat next to the results. |

---

## 8. Roadmap in three phases

| Phase | What it includes |
|-------|------------------|
| **4 days** (the version in this plan) | Detector with explanation, memorization auditor, small website and API, results from scripts |
| **1 month** | More generators, CIFAKE (after the checks), a better user interface, better threshold calibration |
| **1 year** | Video, a browser extension, a WhatsApp bot, training with differential privacy (a method that limits how much a model can remember about one training image), more generators, larger images |

These are ideas, not promises.

---

## 9. Reproducibility checklist

- [ ] Fixed random seeds, set in **one** place.
- [ ] A saved config file for every run (epochs, sizes, t, thresholds).
- [ ] One `results/` JSON file per experiment, written by a script.
- [ ] Every number in any slide or report is traced to a script name.
- [ ] Package versions pinned in `requirements.txt`.
- [ ] No secrets and no datasets in the repo.
- [ ] Results stay as `[PLACEHOLDER - not measured yet]` until they are real.

### Things still to verify

| Item | Where it appears |
|------|------------------|
| DIRE venue (ICCV 2023 - venue to verify) | Section 3.1 |
| Corvi et al. author list (to verify) | Section 1 |
| All CIFAKE facts: licence, image count (100k vs 120k), image size, Kaggle terms, journal venue | Section 2 |
| Kaggle free GPU limits | Section 7 |
| Whether a free Hugging Face Space can run Docker + FastAPI; if not, Render free web service and its limits | Sections 5.4, 7 |
| Hugging Face CPU Basic figures (2 vCPU, 16 GB RAM, sleeps when unused) | Section 5.4 |
| Vercel free plan terms | Section 5.4 |
| All epochs, set sizes, timestep t (about 200), percentile (1st) and thresholds: proposed, not measured | Sections 3, 4 |
| CIFAR-10 licence (official page states none) | Section 2 |
