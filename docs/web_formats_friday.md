# New data for the web pages (Friday upgrade) - for Tejash

Branch `friday-upgrade`. Every new field is OPTIONAL: if it is missing or null, show nothing new (the page must
still work exactly as today). Unmeasured numbers show "[PLACEHOLDER - not measured yet]". Results are measured
overnight Wed -> Thu; the real numbers arrive Thursday morning. Please send your web part as a PR by Thu 13:00.

## 1. Detector: robustness report (new static file, no API call)

`web/public/results/fullsize_robustness_<run>.json` (Nitin copies it there on Thursday; `<run>` = `pilot` or `v2`).

Top-level keys: `run_id, folder, timestamp, times_scored, models_file, models_sha256, conditions, images{real,ai},
band{low,high}, train_generators[], unclear_rule, row_rule, rows[], note`.

`conditions`: `clean`, `jpeg70`, `jpeg50`, `resize0.5`, `screenshot` (what happens after WhatsApp-like compression,
shrinking, or a screenshot).

Each condition has rows: one `group: "all"` row; a `"seen"` and an `"unseen"` row (AI tools that were / were not in
training; only present when they have images); one `group: "generator"` row per AI tool plus `real photos`.
In seen / unseen / tool rows, accuracy = share of those AI images called AI, AUROC = those AI images vs all real images.

One row:
```json
{"condition":"clean","group":"unseen","name":"AI tools unseen in training","seen":null,
 "real_images":0,"ai_images":3,
 "models":{"ours-clip":{"correct":2,"total":3,"accuracy":0.6667,"interval_95":[0.2077,0.9385],"auroc":0.4667},
           "ours-features":{},"smaug":{},"combiner":{}},
 "combiner_labels":{"ai":2,"real":0,"unclear":1,"correct_among_decided":2,"decided":2,"accuracy_among_decided":1.0,
                    "real_called_ai":0,"real_images":0,"ai_called_real":0,"ai_images":3}}
```
Suggested view (Limits page or under the Detector): a table, one line per condition, columns: our CLIP model,
Smaug, our combiner (accuracy with the 95% range), plus "real photos called AI" and "unclear". Small groups (few
images) have wide ranges: show the range, not just the percent.

## 2. Auditor

`GET /audit/results`: each point now has
`{x, memorization_rate, memorization_rate_mean, memorization_rate_sd, flagged_mean, flagged_sd, n_seeds}`.
`memorization_rate` is still the first run (seed 42). `*_sd` is null when `n_seeds < 2`; a missing run has
`n_seeds: 0` and null values. Suggested view: draw the mean, with an error bar of +/- sd when `n_seeds >= 2`, and a
small "n = 3 runs" label.

`POST /audit/check` (same request): each `nearest[i]` adds `clip_similarity` (number 0..1 or null). New top-level
fields: `clip_nearest` (`{clip_similarity, distance, image_png_base64}` or null), `clip_flag` (`"possible_copy"`,
`"not_flagged"` or null) and `clip_threshold` (number or null). `flag` is still the pixel flag. All CLIP fields are
null while that run's cache is not built (the note says so).
Suggested view: a second line next to the pixel result: "CLIP similarity: possible copy / not flagged", with the
CLIP nearest image. Wording: "similar" / "possible copy", never "copyright infringement".

Measured copy test (results/auditor_clip_copies.json, a-500, 50 per group, thresholds set on non-members only):
pixel distance caught exact 50/50, shifted 2 px 18/50, mirrored 8/50, false flags 2/50; CLIP caught exact 50/50,
shifted 49/50, mirrored 48/50, false flags 4/50.

## 3. Generator

`GET /generators` has two new entries (status `"training"` until the overnight run finishes; the site already hides
models that are not `"ready"`):
```json
{"id":"g-vae-mnist","name":"Conv VAE","dataset":"MNIST","family":"vae","params":1118401,"size_mb":null,"status":"training","controls":["seed","n","interpolate"],"classes":[],"licence_note":"MNIST: CC BY-SA 3.0 licence"}
{"id":"g-cddpm-mnist","name":"Conditional diffusion (DDIM)","dataset":"MNIST","family":"cddpm","params":2985985,"size_mb":null,"status":"training","controls":["seed","n","steps","class_id","guidance","show_steps"],"classes":["0","1","2","3","4","5","6","7","8","9"],"licence_note":"MNIST: CC BY-SA 3.0 licence"}
```
Catalogue order: g-vae-cifar, g-vae-fmnist, g-cddpm-fmnist, g-vae-mnist, g-cddpm-mnist, g-ddpm-cifar, g-dcgan-cifar.
MNIST adds a third dataset to the Dataset dropdown ("MNIST"); class names are the digits "0".."9".

"Watch it denoise": the control `"show_steps"` is now also in g-cddpm-fmnist and g-ddpm-cifar (diffusion models
only). `POST /generate` accepts an optional `"show_steps": true`. Send it ONLY when it is true and the model's
controls include `"show_steps"` (a VAE answers 400 even for false). The response then adds:
```json
"denoise_frames": [{"step": 1, "png_base64": "..."}, ... 8 entries ...]
```
All 8 frames are for `images[0]`; `step` counts from 1 to the number of DDIM steps; the last frame equals
`images[0].png_base64`. Without the flag the response keys are exactly as before
(`generator_id, images, metadata, note`).
Suggested view: a "Watch it denoise" checkbox; under the first image a strip of 8 small frames labelled
"step 1 ... step 20", from noise to picture.

## 4. Compare

`POST /compare`: one new row, always the last row of `feature_table` (response keys unchanged):
```json
{"feature":"CLIP similarity (CLIP ViT-L/14, whole image)","image_a":"0.84","image_b":"0.84","does_not_prove":"Descriptive only. Similar meaning is not the same picture; CLIP is pretrained by OpenAI, not by us. It sees the centre square of each picture."}
```
If CLIP cannot load, both cells say `"not available"`. The existing table already shows it; nothing is required,
but you may highlight it next to the ResNet-18 rows.
