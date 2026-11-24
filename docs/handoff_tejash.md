# Hand-off for Tejash (written Tue 24 Nov)

Your site runs against the real API. Nitin merged your `frontend` branch (nothing committed yet) and clicked through every page with real images. All API calls came back 200.

## What worked, tested with VITE_USE_STUB=false
- Detector: example images, label, scores, CNN baseline score, heatmap, Grad-CAM, Evidence tab.
- Compare: two images, SSIM, difference map, feature table. Words stayed off.
- Generator: made 4 images with the diffusion model.
- Auditor: the upload now works. The results show placeholders until the Auditor runs finish.
- Limits: model cards now show measured numbers.

## What was changed in your files (see `docs/tejash_fixes.patch`, 6 files)
1. `pages.jsx`: the Auditor upload sent the field `file`; the API expects `image`. Fixed.
2. `lab-data.js` and `image-tools.jsx`: Words was switched on in stub mode. It is now on only when `VITE_ENABLE_COMPARE_WORDS` is `true`.
3. `pages.jsx` and `public/stub/audit_results.json`: the Auditor page looked for experiments named `a-500` and so on. The API sends two experiments ("memorization vs training-set size" and "effect of duplicates (500 training images)"), so the page now draws whatever experiments the API sends.
4. `limits.jsx`: the paragraph that said "VAE fakes only" is replaced with the true story (trained on VAE and diffusion fakes; diffusion fakes were harder; worse on unseen generators). Two stale "Gate 2" labels removed. One sentence added that the three diffusion models were stopped by a time limit.
5. `home-hero.jsx`: the hero line no longer promises to "discover the traces AI leaves behind".
6. `pages.jsx`: one hint under the Detector scores saying a higher VAE score does not mean "more AI" (in our tests it was lower for VAE fakes).

To apply it on your branch, from the repo root: `git apply --check docs/tejash_fixes.patch` and then `git apply docs/tejash_fixes.patch`.

## New files under `web/public/`
- `examples/index.json`, `examples/fake/*.png` (safe to commit) and `examples/real/*.png` (git-ignored; Nitin sends these by file transfer).
- `results/model-cards.json` is written by `model/export_model_cards.py`. Do not edit it by hand; Nitin re-runs the script when the Auditor runs finish.
- The model cards only load when `VITE_USE_STUB=false`. In stub mode they stay placeholders on purpose.

## Added later the same day (API side, plus two small edits in your files)
- **`POST /audit/check` is real now.** Upload one image, get the 3 nearest training images of one Auditor run, a `flag` of `possible_copy` or `not_flagged`, and a note with the run name and threshold. It sends the same fields as before, so your page needs no change. An optional form field `run` (`a-500`, `a-10000`, `a-500-dup`) picks the run; your page does not send it, so it always uses `a-500`. `a-10000` and `a-500-dup` answer 503 until their runs finish. If you want, add a small run picker.
- **Compare now has CNN parts** (a pretrained ResNet-18, not trained by us; Words stays off):
  - `regions_png_base64` is filled: image A and image B side by side, tinted red where a part has no similar part in the other image (14x14 cells, so it is coarse).
  - Three new rows in `feature_table`: whole-image CNN similarity, CNN best-match similarity, and the CNN's top 3 ImageNet guesses per image (may be wrong).
  - The "edge amount" row no longer shows 0% for small images.
  - Your page already shows these. In `image-tools.jsx` I only changed the map's label to "CNN map: red = no similar part in the other image".
- **`limits.jsx`**: the "DINOv2 patch matching" row is now "CNN feature matching (ResNet-18)", the LPIPS row says "Not built yet", and the DINOv2 reference was replaced by the ResNet paper (He et al., 2016).
- The patch `docs/tejash_fixes.patch` was regenerated and contains all of the above that touches `web/`.

## To run with the real API (on Nitin's laptop, because the weights are not in git)
- API: `.venv\Scripts\python -m uvicorn api.main:app --port 8000` from the project root.
- Site: in `web/`, set `VITE_USE_STUB=false` in `.env`, then `npm run dev`. The `/api` proxy already points to port 8000.

## Still open on the web side (your call)
- Limits "What each feature can tell you" lists tools that are not built (C2PA, ELA, FFT, LPIPS, occlusion, watermarks, Words). Mark them "not built yet" or remove them. (pHash is built now, inside `/audit/check`.)
- Limits references still list papers for features that do not exist (for example Amir et al. on ViT features, CLIP, Krawetz). Keep only what we really use.
- `public/stub/compare.json` (demo mode) still has `regions_png_base64` null and no CNN rows.
- `alignmentWarning` in `lab-data.js` says the Words panel "says more", but Words is off. Reword it.
- Unused CSS rule `.audit-run > p` in `pages.css`.
- One console error on page load: `net::ERR_INVALID_URL`. Source not found.
- A design check flagged the font "Instrument Sans" in `styles.css`. Left as is.
- The Detector often says "Unclear" for real photos and misses some diffusion fakes. That is the real behaviour. Keep showing the CNN baseline score beside the label.
