# auditor_clip.py - CLIP similarity for the Auditor, next to pixel L2 and pHash.
#
# Why: pixel L2 (model/auditor.py) only finds near-identical pixels. A training image moved by 2 pixels or mirrored
# left-right has a large pixel distance, but it is still the same picture. CLIP (openai/clip-vit-large-patch14,
# pretrained, frozen; model/fullsize.py's clip_embedding) describes WHAT is in a picture, so such copies stay close.
# Similarity = cosine similarity of the two CLIP embeddings (0 to 1 in practice; 1 = the same picture).
#
# What is in this file:
#   run_index(run)   the CLIP embeddings of one Auditor run's training images and its CLIP threshold. The embeddings are
#                    cached under logs/ (built the first time they are needed). The threshold is the 99th percentile of
#                    the nearest-training-image similarity of 1,000 held-out real images (the 'calibration' part only,
#                    never final_test): the mirror of the pixel rule (1st percentile of the distances). Proposed, to tune.
#   main()           the copy test: ~100 training images of a-500 as exact, 2-px-shifted and mirrored copies, plus 100
#                    non-member calibration images. Thresholds are set on half A, the result is reported once on half B.
#                    Writes results/auditor_clip_copies.json.
# IMPORTANT: this measures SIMILARITY between images ("similar", "possible copy"). It does not decide legal copying.
#
# Run it from the project folder:
#   python -m model.auditor_clip                              the copy test (builds the a-500 cache first if needed)
#   python -m model.auditor_clip --cache a-500 a-10000 a-500-dup    only build the caches (a-10000: about 5 min on the GPU)

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from model.auditor import load_heldout, nearest_distances
from model.data import load_cifar10
from model.fullsize import CLIP_ID, clip_embedding
from model.make_fakes import to_float, to_uint8
from model.train_auditor_run import PERCENTILE, RUNS, subset_rows
from model.utils import LOGS_DIR, RESULTS_DIR, SEED, save_json

CLIP_PERCENTILE = 99.0                                   # CLIP flag: above the 99th percentile of held-out similarities
_ORDER = np.random.RandomState(SEED).permutation(5000)   # a fixed order of the 5,000 calibration images
CALIBRATION = np.sort(_ORDER[:1000])                     # used for the CLIP threshold of /audit/check
NON_MEMBERS = _ORDER[1000:1100]                          # used by the copy test only (never the 1,000 above)
MEMBERS = 100
SHIFT = 2                                                # pixels


def embed(pixels):
    """CLIP embeddings (N, 768), length 1, of uint8 pictures (N, 32, 32, 3)."""
    return np.stack([clip_embedding(Image.fromarray(p)) for p in pixels]).astype(np.float32)


def cached(path, indices, pictures, build=True):
    """The embeddings saved in `path` if they were made from exactly these indices; else (build=True) make and save
    them. pictures() gives the uint8 pictures; it is only called when building. Returns None if not built."""
    if path.exists():
        saved = np.load(path)
        if np.array_equal(saved["indices"], indices):
            return saved["embeddings"]
    if not build:
        return None
    start = time.time()
    embeddings = embed(pictures())
    temp = path.with_suffix(".tmp.npz")
    np.savez(temp, indices=indices, embeddings=embeddings)
    os.replace(temp, path)                               # whole file at once: a reader never sees half a file
    print(f"CLIP cache {path.name}: {len(indices)} images in {time.time() - start:.0f} s", flush=True)
    return embeddings


def nearest_similarity(queries, train):
    """For each query embedding: (highest cosine similarity to a training embedding, that training image's position)."""
    sims = queries @ train.T
    return sims.max(axis=1), sims.argmax(axis=1)


def run_index(run, build=True):
    """(training embeddings (K, 768), CLIP threshold) of one Auditor base run, or None if the cache is missing and
    build is False. The training images are the run's different images, in the order api/audit.py keeps them."""
    unique = np.unique(subset_rows(run))
    train = cached(LOGS_DIR / f"auditor_clip_{run}.npz", unique,
                   lambda: to_uint8(load_cifar10("train")[torch.from_numpy(unique)]), build)
    held = cached(LOGS_DIR / "auditor_clip_calibration.npz", CALIBRATION,
                  lambda: to_uint8(load_heldout()[torch.from_numpy(CALIBRATION)]), build)
    if train is None or held is None:
        return None
    return train, float(np.percentile(nearest_similarity(held, train)[0], CLIP_PERCENTILE))


# ---------------------------------------------------------------- the copy test

def shifted(pixels):
    """Moved SHIFT pixels to the right; the left edge column is repeated to fill the gap."""
    return np.ascontiguousarray(np.pad(pixels, ((0, 0), (0, 0), (SHIFT, 0), (0, 0)), mode="edge")[:, :, :32])


def caught(flags, nearest, source):
    """How many of one group were flagged, and for how many the nearest training image is the original."""
    out = {"flagged": int(flags.sum()), "of": len(flags), "rate": round(float(flags.mean()), 4)}
    if source is not None:
        out["nearest_is_source"] = int((nearest == source).sum())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", nargs="+", choices=list(RUNS), help="only build the CLIP caches of these runs")
    args = parser.parse_args()
    if args.cache:
        for run in args.cache:
            print(f"{run}: CLIP threshold {run_index(run)[1]:.4f}", flush=True)
        return

    start = time.time()
    run = "a-500"
    train_emb, _ = run_index(run)
    unique = np.unique(subset_rows(run))
    train = load_cifar10("train")[torch.from_numpy(unique)]                  # (500,3,32,32), the order of train_emb
    rng = np.random.RandomState(SEED)
    members = rng.choice(len(unique), MEMBERS, replace=False)               # positions in train / train_emb
    member_pixels = to_uint8(train[torch.from_numpy(members)])
    groups = {"exact": member_pixels, "shifted": shifted(member_pixels), "flipped": member_pixels[:, :, ::-1].copy(),
              "non_members": to_uint8(load_heldout()[torch.from_numpy(NON_MEMBERS)])}     # calibration part only

    scores = {}                                       # group -> (pixel distance, pixel nearest, CLIP similarity, CLIP nearest)
    for name, pixels in groups.items():
        distance, pixel_nearest = nearest_distances(to_float(pixels), train)
        similarity, clip_nearest = nearest_similarity(embed(pixels), train_emb)
        scores[name] = (distance.numpy(), pixel_nearest.numpy(), similarity, clip_nearest)
    assert (scores["exact"][0] == 0).all() and (scores["exact"][2] > 0.999).all(), "an exact copy must give distance 0 and similarity 1"

    order = rng.permutation(MEMBERS)                  # the same split for members and non-members (100 each)
    half_a, half_b = order[:MEMBERS // 2], order[MEMBERS // 2:]
    # thresholds from half A's NON-MEMBERS only (the Auditor's rule); copies are never used to set them
    pixel_cut = float(np.percentile(scores["non_members"][0][half_a], PERCENTILE))
    clip_cut = float(np.percentile(scores["non_members"][2][half_a], CLIP_PERCENTILE))

    report = {"pixel_l2": {}, "clip": {}}
    for name, (distance, pixel_nearest, similarity, clip_nearest) in scores.items():
        source = members[half_b] if name != "non_members" else None
        report["pixel_l2"][name] = caught(distance[half_b] < pixel_cut, pixel_nearest[half_b], source)
        report["clip"][name] = caught(similarity[half_b] > clip_cut, clip_nearest[half_b], source)

    result = {
        "run": run, "clip_model": CLIP_ID, "training_images": len(unique),
        "members": MEMBERS, "non_members": len(NON_MEMBERS),
        "copies": {"exact": "the training image unchanged", "shifted": f"moved {SHIFT} pixels to the right (edge column repeated)",
                   "flipped": "mirrored left-right"},
        "non_member_source": "CIFAR-10 calibration part (never final_test); not the 1,000 images behind /audit/check's CLIP threshold",
        "split": f"half A: {MEMBERS // 2} members (x3 copies) + {MEMBERS // 2} non-members sets the thresholds; "
                 f"half B: the other {MEMBERS // 2} + {MEMBERS // 2}, reported once",
        "threshold_rule": f"half A non-members only, the Auditor's rule: pixel L2 below the {PERCENTILE:g}st percentile, "
                          f"CLIP similarity above the {CLIP_PERCENTILE:g}th percentile",
        "thresholds": {"pixel_l2": round(pixel_cut, 3), "clip_similarity": round(clip_cut, 4)},
        "report_half_b": report,
        "seconds": round(time.time() - start, 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "note": "Similarity only: a flag means 'possible copy', not legal copying. Small test (50 per group); proposed, to tune.",
    }
    save_json(RESULTS_DIR / "auditor_clip_copies.json", result)
    for method, groups_found in report.items():
        print(method, {name: f"{g['flagged']}/{g['of']}" for name, g in groups_found.items()})
    print(f"thresholds {result['thresholds']} | {result['seconds']} s | written to results/auditor_clip_copies.json")


if __name__ == "__main__":
    main()
