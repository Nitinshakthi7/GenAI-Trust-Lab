# Friday upgrade plan (written Wed 25 Nov 2026, ~14:50; waiting for approval)

Made with the project workflow: code-surveyor + spec-reader + research -> prompt-packer -> planner (Opus).
Presentation: Fri 27 Nov 09:00. Nothing below has started.

## Decisions needed first
1. Downloads (each needs a yes; NC = non-commercial):
   - OpenFakeTiny `core/train-00001-of-00002.parquet`, 873.3 MB, Hugging Face ComplexDataLab/OpenFakeTiny: more training data. Licence unclear (card has none; parent cards conflict, NC terms). Recommend: yes.
   - OpenFakeTiny `core/validation-00000-of-00002.parquet`, 856.7 MB, same source: the NEW held-out test folder (data/openfake/test is spent). Recommend: yes; without it no detector upgrade can be measured honestly.
   - MNIST via torchvision, about 11 MB, CC BY-SA 3.0. Recommend: yes.
   - Not recommended: Tiny-GenImage, SSCD weights, LPIPS AlexNet weights, OpenFakeTiny reddit part.
2. Generator dataset: MNIST (recommended) or CelebA (non-commercial, offline only).
3. Cut-off Thu 13:00, code freeze Thu 15:00 (recommended).
4. Commit the current asymmetric "unclear" rule to main and tag it `demo-safe-v1` as the baseline (recommended).
5. Drop the Fashion-MNIST Auditor (large rewrite) (recommended).

## Scope
IN, must:
- Detector v2: more training data, JPEG/resize/screenshot augmentation, robustness report on a new held-out folder.
- Auditor: 2 extra seeds per run (error bars).
- Auditor: CLIP similarity next to pixel L2 and pHash, to catch 2-px shifts and flips (CLIP already downloaded).

IN, should (cut first if late):
- Compare: CLIP similarity.
- Generator: MNIST (VAE + DDPM).
- Generator: "watch it denoise" step viewer (8 frames).

OUT: Fashion-MNIST Auditor, SSCD, LPIPS, tool-family guess, CelebA, Tiny-GenImage, reddit part, retraining detector_final, hosting.

## Schedule
- Wed 15:00-15:20 baseline tag and branch `friday-upgrade`.
- Wed 15:20-21:30 code steps 2-8; downloads in the background; data formats to Tejash by 18:00.
- Wed 21:30 dry run of each training script (~2 min each).
- Wed 22:00-Thu 07:00 API and browser OFF; GPU queue: (1) Detector v2 features + fit (2-3 h, to verify), (2) Auditor seeds, 6 runs x ~14 min, (3) MNIST VAE + DDPM, ~40-60 min, (4) Auditor CLIP embedding cache.
- Thu 07:00-08:00 read logs; check out-of-fold scores before touching the held-out folder.
- Thu 08:00-11:00 score the held-out folder ONCE; integrate; contract and checks.
- Thu 11:00-13:00 Tejash's web PR, reviewed by Nitin.
- Thu 13:00 hard cut-off: anything not working is dropped.
- Thu 15:00 code freeze: merge, tag `demo-friday`, all self-checks pass.
- Thu 15:00-19:00 slides + speaker notes, demo script.
- Thu 19:00-20:00 backup video (outside git). Thu 20:00-21:00 rehearsal 1.
- Fri 07:30-08:30 rehearsal 2 incl. cold restart, Wi-Fi off, phone check. Fri 09:00 present.

## Steps
1. Commit the rule change, tag `demo-safe-v1`, branch `friday-upgrade`.
2. Detector v2 training code (model/fullsize.py): `--run-id` so outputs go to weights/fullsize_v2_models.joblib and results/fullsize_v2.json (v1 untouched); several train folders; augmentation (JPEG q50-90, resize 0.5-0.75x, screenshot-style) before normalize(); Smaug cache keyed by path + augmentation; thresholds from out-of-fold scores only. Done: 20-image dry run writes v2 files, v1 hashes unchanged.
3. Robustness report (new model/fullsize_robustness.py, pattern of model/blur_control.py): accuracy and AUROC on the new held-out folder for clean, JPEG q70/q50, resize 0.5x, screenshot; per generator; seen vs unseen generators; v1 and v2.
4. Auditor seeds (`--seed` flag, keep loss.item(); RUNS, RUN_ORDER, mean +/- sd in the summary, api/audit.py, check_api.py).
5. Auditor CLIP similarity (api/audit.py, cached embeddings under logs/): shifted/flipped copies + non-members; threshold on a calibration half, catch rate reported once on the other half; calibration never overlaps final_test; wording "similar"/"possible copy". Writes results/auditor_clip_copies.json; /audit/check returns `clip_similarity`.
6. Compare CLIP similarity (api/compare.py, check_api.py).
7. Generator MNIST (`g-mnist`, config, train script from Fashion-MNIST, catalogue, checks), labelled "small model, expect low quality".
8. Denoise viewer backend (`return_steps` in model/ddpm.py, optional `steps` field, 8 frames; old response unchanged).
9. Score once Thu morning: robustness report on v1 and v2 over validation part 1. Promote v2 to the demo only if it beats v1 on clean and robustness rows; otherwise keep v1 and report v2 as a result.
10. Scope-change docs: PLAN.md decision log, Limits page (fix "three diffusion models", remove unbuilt tools, licence note), docs/ai_usage.md, model cards. Record Smaug (Apache-2.0, revision c7c95951e04a87b884a4ad479401b1128c3903bd; training overlaps some test tools) and CLIP ViT-L/14 (revision 32bd64288804d66eefd0ccbe215aa642df71cc41); CLIP now on the demo path; upload limit 20 MB.
11. Tejash (web, by PR): robustness table, Auditor error bars + CLIP column, Compare CLIP row, g-mnist + denoise viewer, Limits text; merged by Thu 13:00 after Nitin's review.

## Verification and rollback
- `python -m api.check_api` passes with the new fields; v1 detector hashes checked; held-out folder scored exactly once (logged in the results JSON); manual demo pass at the freeze; rehearsal 2 covers cold restart and Wi-Fi off.
- Anything failing by 13:00 is dropped; every new field is optional, so the demo falls back to v1. Worst case: `git checkout demo-safe-v1`; the backup video covers a laptop failure.
