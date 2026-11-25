# Scope change record - Wed 25 Nov 2026

The original plan (PLAN.md, team plan PDF) kept version 1 narrow: the Detector judged only 32x32 CIFAR-10-style
images, "judging arbitrary photos" was a non-goal, pretrained CLIP weights were excluded from the demo, and broad
datasets were "documentation only". On 25 Nov the scope changed, for the reasons and with the safeguards below.
This file is the decision log entry for that change (to be linked from PLAN.md's decision log).

## Why
- The teacher asked that the Detector classify a full-size real photo and a full-size AI image and say which
  features it used. The 32x32 Detector cannot do that (on a 640x480 noise image it said "AI-made, 0.98").
- The presentation moved to Friday 27 Nov, 09:00 (the documents still say Thursday 26 Nov; their gate times are
  outdated). New cut-off Thu 13:00, code freeze Thu 15:00 (decided by Nitin, 25 Nov).

## What was added
1. Full-size image check (model/fullsize.py, api/fullcheck.py), shown for any picture that is not 32x32:
   - Our CLIP model: frozen pretrained CLIP ViT-L/14 image features + our logistic regression
     (method of Ojha et al. 2023, "Towards Universal Fake Image Detectors").
   - Our measured-feature model: logistic regression on sharpness, noise, FFT fine detail, saturation, edges.
   - Smaug AI Image Detector: pretrained, NOT trained by us, used as a published comparison detector.
   - Our combiner: logistic regression on the three scores (stacking on out-of-fold scores).
   - Rule (chosen from two live test images): the answer may be "Real" only if neither our CLIP model nor Smaug
     says AI (>= 0.5); otherwise "Unclear". AI answers from the combiner are kept.
2. Upload limit 5 MB -> 20 MB (phone photos).
3. Friday upgrade (docs/friday_plan.md, approved 25 Nov): Detector v2 with more data + augmentation and a
   robustness report; Auditor extra seeds (error bars) + CLIP similarity; MNIST generators + "watch it denoise";
   CLIP similarity row in Compare.

## Pretrained models and data now on the demo path
| Item | Source | Version | Licence |
|---|---|---|---|
| CLIP ViT-L/14 | huggingface.co/openai/clip-vit-large-patch14 | revision 32bd64288804d66eefd0ccbe215aa642df71cc41 | model page lists none; OpenAI CLIP code is MIT (to verify for the weights) |
| Smaug AI Image Detector | huggingface.co/abacusai/Smaug-AI-Image-Detector | revision c7c95951e04a87b884a4ad479401b1128c3903bd | Apache-2.0 |
| ResNet-18 (Compare, baseline) | torchvision IMAGENET1K_V1 (resnet18-f37072fd.pth) | torchvision 0.21 | BSD-3-Clause (torchvision); ImageNet weights terms to verify |
| OpenFakeTiny | huggingface.co/datasets/ComplexDataLab/OpenFakeTiny | files used: core train part 1, core test, core train part 2, core validation part 1 | dataset card states none; parent OpenFake cards conflict (CC BY-NC-SA 4.0 vs CC BY-SA 4.0 with non-commercial terms): to verify; used offline, never committed |
| MNIST | torchvision | - | CC BY-SA 3.0 |

## Honesty safeguards
- Smaug's model card lists several of our test tools (for example GPT-Image, Nano Banana Pro, Midjourney 7) among
  its training generators, so its score on them is not an unseen-tool result.
- data/openfake/test (500 images) was scored three times, once per decision rule; each scoring is kept:
  results/fullsize_pilot_before_unclear_rule.json, results/fullsize_pilot_disagree_rule.json,
  results/fullsize_pilot.json. It is spent and is not used again.
- The Friday upgrade is measured on a NEW folder, data/openfake/heldout (300 real + 300 AI from OpenFakeTiny
  validation part 1), scored exactly once per run (guard file logs/heldout_scored.json). Most of its AI tools also
  appear in the training data (255 of 300 AI images for v2), so results are split into seen and unseen tools.
- Detector v2 replaces v1 on the demo only if it beats v1 on the clean and robustness rows.
- Every number on the site comes from a script; unmeasured numbers stay "[PLACEHOLDER - not measured yet]".
- AI assistance (Claude Code) was used for planning, code and measurement; to be disclosed in docs/ai_usage.md.
