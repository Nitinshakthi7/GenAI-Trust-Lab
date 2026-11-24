# compare_cnn.py - the CNN part of POST /compare: "which parts of picture A have no similar part in picture B?"
#
# We do NOT train anything here. We use torchvision's ResNet-18 with its ImageNet weights, a CNN from the course
# that was trained by others on 1.2 million photos (the weights file comes from the torch cache folder).
#   1. Both pictures (224 x 224, numbers 0 to 1) are normalised the way ImageNet pictures are, and go through ResNet-18.
#   2. layer3 gives each picture a 14 x 14 grid of "patches"; every patch is a list of 256 numbers (its features).
#      Each patch's list is scaled to length 1, so a dot product of two patches is their cosine similarity.
#   3. For every patch of A: its best match = the highest similarity to any patch of B. unmatched = 1 - best match
#      (high = nothing in the other picture looks like this patch). The same for B against A.
#   4. layer4 averaged over the whole picture (global average pooling) = one list of 512 numbers per picture;
#      the cosine similarity of the two lists says how alike the pictures are as a whole.
#   5. The classifier head of ResNet-18 gives its 3 best ImageNet labels (1000 fixed labels, so it can be wrong).
# HONEST MEANING: all of this is descriptive. A matched patch may belong to a different object; an unmatched
# patch is not necessarily a meaningful difference. If the weights file is missing, load() raises and /compare
# answers without the CNN part.

import threading
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
MEAN = torch.tensor(WEIGHTS.transforms().mean).view(1, 3, 1, 1)     # ImageNet colour mean and spread
STD = torch.tensor(WEIGHTS.transforms().std).view(1, 3, 1, 1)
LABELS = WEIGHTS.meta["categories"]                                 # the 1000 ImageNet label names

_model = None
_lock = threading.Lock()


def load():
    """The ResNet-18 (eval mode, CPU), loaded once. Raises if the weights file is not in the torch cache (no download here)."""
    global _model
    with _lock:
        if _model is None:
            cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(WEIGHTS.url).path).name
            if not cached.exists():                  # torchvision would start a download; a web request must not
                raise FileNotFoundError(f"ResNet-18 weights not found: {cached}")
            _model = resnet18(weights=WEIGHTS).eval()
        return _model


def analyse(pixels_a, pixels_b):
    """pixels_a, pixels_b: 224 x 224 x 3 numbers from 0 to 1. Returns the numbers described above."""
    model = load()
    batch = torch.from_numpy(np.stack([pixels_a, pixels_b])).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        x = (batch - MEAN) / STD
        x = model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x))))))
        layer3 = model.layer3(x)                                    # 2 x 256 x 14 x 14
        layer4 = model.layer4(layer3)
        pooled = layer4.mean(dim=(2, 3))                            # 2 x 512
        probs = model.fc(pooled).softmax(dim=1)
    patches = F.normalize(layer3.flatten(2).transpose(1, 2), dim=2)  # 2 x 196 x 256, each patch has length 1
    similar = patches[0] @ patches[1].T                             # 196 x 196: every patch of A against every patch of B
    best_a, best_b = similar.max(dim=1).values, similar.max(dim=0).values
    side = layer3.shape[2]
    top = probs.topk(3, dim=1)
    return {
        "unmatched_a": (1 - best_a).reshape(side, side).numpy(),    # 14 x 14, high = no similar patch in B
        "unmatched_b": (1 - best_b).reshape(side, side).numpy(),
        "whole_image": float(F.cosine_similarity(pooled[0], pooled[1], dim=0)),
        "best_a_to_b": float(best_a.mean()),
        "best_b_to_a": float(best_b.mean()),
        "guesses": [[(LABELS[i], float(p)) for p, i in zip(top.values[k], top.indices[k])] for k in range(2)],
    }


if __name__ == "__main__":                                          # quick self-check: python -m model.compare_cnn
    rng = np.random.default_rng(0)
    one = rng.random((224, 224, 3))
    same = analyse(one, one)
    assert abs(same["whole_image"] - 1) < 1e-4 and abs(same["best_a_to_b"] - 1) < 1e-4, same
    other = analyse(one, rng.random((224, 224, 3)))
    assert other["whole_image"] < same["whole_image"] and other["unmatched_a"].shape == (14, 14)
    print("compare_cnn ok:", round(other["whole_image"], 3), other["guesses"][0][0])
