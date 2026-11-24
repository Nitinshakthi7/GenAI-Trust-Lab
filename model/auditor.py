# auditor.py - the measuring tools of the Auditor: "does a generated image look like a training image?"
#
# The idea in plain words:
#   For every generated image we find the training image that is CLOSEST to it.
#   "Closest" = the smallest L2 distance in pixel space (32x32x3 = 3072 numbers, scaled to -1..1).
#   But how close is "too close"? To decide, we do the same for REAL images the model never saw
#   (held-out images). A held-out image has no copy in the training set, so its distance shows
#   what a normal, innocent distance looks like. We take the 1st percentile of those distances
#   as the threshold (proposed, to tune). A generated image closer than the threshold is
#   flagged "possible copy".
#
# IMPORTANT: this only measures SIMILARITY between images. It does not decide legal copying.
#
# What is in this file:
#   load_heldout        the 5,000 held-out real images (the 'calibration' part only, never final_test)
#   nearest_distances   for each query image: distance to, and number of, its nearest training image
#   calibrate           the threshold (a percentile of the held-out distances)
#   memorization_rate   how many generated images are closer than the threshold
#   closest_pairs       the k generated images nearest to a training image, as base64 PNGs
# Run the self-check with:  python -m model.auditor

import base64
import io

import numpy as np
import torch
from PIL import Image

from model.data import PARTS, get_part, load_part
from model.make_fakes import to_uint8

SHOW_SIZE = 128        # example pictures are shown at 128 x 128 (they are 32 x 32 really)


def load_heldout():
    """The 5,000 held-out real images: the 'calibration' part of data/splits.csv, and nothing else.
    The final_test part must never be used here, so we check that the two parts do not overlap."""
    indices = get_part("calibration")
    final_test = set(get_part("final_test").tolist())      # only the index numbers, never the pictures
    assert PARTS["calibration"][0] == "test", "the calibration part must come from the CIFAR-10 test images"
    assert len(indices) == 5000, "the calibration part must have 5,000 images"
    assert not (set(indices.tolist()) & final_test), "calibration must not overlap final_test"
    return load_part("calibration")


def nearest_distances(queries, train_set, chunk=500):
    """For every query image, find its nearest image in train_set.

    Both are tensors of shape (N,3,32,32). Returns (distances, indices), two tensors of length
    len(queries): the L2 distance to the nearest training image, and that image's position in train_set.
    Brute force: every query is compared with every training image, `chunk` queries at a time.
    """
    q = queries.flatten(1).float()
    t = train_set.flatten(1).float().to(q.device)
    distances, indices = [], []
    for start in range(0, len(q), chunk):
        part = q[start:start + chunk]
        nearest = torch.cdist(part, t).argmin(dim=1)        # fast search (the matrix trick)
        distances.append((part - t[nearest]).norm(dim=1))   # then the exact distance to the one it found
        indices.append(nearest)
    # ponytail: the fast search can pick the wrong image only if two training images are
    # almost exactly equally close (a rounding difference); the distance itself is always exact.
    return torch.cat(distances).cpu(), torch.cat(indices).cpu()


def calibrate(heldout, train_set, percentile=1.0):
    """The threshold: the given percentile (1st, proposed) of the held-out distances.
    Returns (threshold, held-out distances)."""
    distances, _ = nearest_distances(heldout, train_set)
    return float(np.percentile(distances.numpy(), percentile)), distances


def memorization_rate(distances, threshold):
    """Generated images closer than the threshold are 'possible copies'.
    Returns (how many are flagged, flagged / all)."""
    flagged = int((distances < threshold).sum())
    return flagged, flagged / len(distances)


def to_png_base64(image):
    """One image (3,32,32) in -1..1 -> a 128 x 128 PNG as plain base64 text (no data: prefix)."""
    picture = Image.fromarray(to_uint8(image[None])[0])
    picture = picture.resize((SHOW_SIZE, SHOW_SIZE), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def closest_pairs(generated, train_set, k=3):
    """The k generated images with the SMALLEST distance to a training image, smallest first.
    These are the closest pairs, NOT chosen for looks: they are simply the nearest ones.
    Each item: {generated_png_base64, nearest_png_base64, distance}."""
    distances, nearest = nearest_distances(generated, train_set)
    pairs = []
    for i in torch.argsort(distances)[:k].tolist():
        pairs.append({"generated_png_base64": to_png_base64(generated[i]),
                      "nearest_png_base64": to_png_base64(train_set[int(nearest[i])]),
                      "distance": round(float(distances[i]), 4)})
    return pairs


# ---------------------------------------------------------------- self-check

if __name__ == "__main__":
    import time

    torch.set_num_threads(2)                       # keep the computer usable while other things run
    g = torch.Generator().manual_seed(0)
    random_images = lambda n: torch.rand(n, 3, 32, 32, generator=g) * 2 - 1

    # 1. Known answers: a query equal to a training image has distance 0 and the right index.
    train = random_images(200)
    dist, idx = nearest_distances(train[[3, 50, 199]], train)
    assert dist.tolist() == [0.0, 0.0, 0.0] and idx.tolist() == [3, 50, 199], "a copy must give distance 0"
    print("copy of training images 3, 50, 199 -> distances", dist.tolist(), "indices", idx.tolist())

    # 2. A query far away has a larger distance than a query that is nearly a copy.
    near, far = train[7:8] + 0.01, torch.full((1, 3, 32, 32), 3.0)
    d_near, d_far = nearest_distances(near, train)[0].item(), nearest_distances(far, train)[0].item()
    assert d_near < 1 < 100 < d_far, "far must be bigger than near"
    print(f"nearly a copy: distance {d_near:.3f}   far away: distance {d_far:.1f}")

    # 3. The fast code equals a slow numpy double loop on 50 random images.
    queries = random_images(50)
    dist, idx = nearest_distances(queries, train)
    slow = np.array([[np.sqrt(((q.numpy().astype(np.float64) - t.numpy().astype(np.float64)) ** 2).sum()) for t in train]
                     for q in queries])
    assert np.allclose(dist.numpy(), slow.min(axis=1), atol=1e-4), "distances differ from the slow loop"
    assert (idx.numpy() == slow.argmin(axis=1)).all(), "indices differ from the slow loop"
    print(f"50 random images: fast code equals the slow loop (biggest difference {np.abs(dist.numpy() - slow.min(axis=1)).max():.2e})")

    # 4. The threshold equals np.percentile of the held-out distances (worked out by the slow loop).
    heldout = random_images(100)
    threshold, held = calibrate(heldout, train, percentile=1.0)
    slow_held = np.array([min(np.sqrt(((h.numpy().astype(np.float64) - t.numpy().astype(np.float64)) ** 2).sum()) for t in train)
                          for h in heldout])
    assert abs(threshold - np.percentile(slow_held, 1.0)) < 1e-3, "threshold must be the 1st percentile"
    flagged, rate = memorization_rate(held, threshold)
    assert flagged == 1 and rate == 0.01, "exactly 1 of 100 held-out distances is below their own 1st percentile"
    print(f"threshold {threshold:.3f} = np.percentile(held-out distances, 1); flagged {flagged} of 100")

    # 5. closest_pairs: k items, smallest distance first, and they really are the k smallest.
    generated = random_images(20)
    generated[5], generated[9], generated[13] = train[1], train[2] + 0.01, train[3] + 0.05
    pairs = closest_pairs(generated, train, k=3)
    all_distances = np.sort(nearest_distances(generated, train)[0].numpy())
    assert len(pairs) == 3 and [p["distance"] for p in pairs] == sorted(p["distance"] for p in pairs)
    assert np.allclose([p["distance"] for p in pairs], all_distances[:3], atol=1e-3), "must be the 3 smallest"
    assert pairs[0]["distance"] == 0.0 and set(pairs[0]) == {"generated_png_base64", "nearest_png_base64", "distance"}
    first = [np.array(Image.open(io.BytesIO(base64.b64decode(pairs[0][key])))) for key in ("generated_png_base64", "nearest_png_base64")]
    assert first[0].shape == (128, 128, 3) and (first[0] == first[1]).all(), "a copy and its nearest image look the same"
    print("closest pairs:", [p["distance"] for p in pairs], "(three smallest of 20; PNG is 128 x 128, plain base64)")

    # 6. The held-out images: 5,000, in range, and never from final_test.
    held_images = load_heldout()
    assert held_images.shape == (5000, 3, 32, 32) and held_images.min() >= -1 and held_images.max() <= 1
    print("held-out images:", tuple(held_images.shape), "(calibration part only)")

    # 7. How long does it take? 1,000 queries against 10,000 training images, on the CPU.
    big_train, big_queries = random_images(10000), random_images(1000)
    start = time.perf_counter()
    nearest_distances(big_queries, big_train)
    print(f"timing: 1,000 queries against 10,000 training images took {time.perf_counter() - start:.2f} s on the CPU (2 threads)")
    print("self-check passed")
