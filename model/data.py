# data.py - loads the datasets and cuts CIFAR-10 into the four fixed parts.
# Images come back as float tensors with pixel values between -1 and 1.
# CIFAR-10 is 3x32x32. Fashion-MNIST is 28x28 with 1 channel, so we pad it
# to 32x32 here, and every model can then work at 32x32.
#
# The four CIFAR-10 parts (written once to data/splits.csv, seed 42):
#   ddpm_train  30,000  (from the CIFAR-10 TRAIN images)
#   clf_train   20,000  (from the CIFAR-10 TRAIN images, never overlaps ddpm_train)
#   calibration  5,000  (from the CIFAR-10 TEST images)
#   final_test   5,000  (from the CIFAR-10 TEST images, never overlaps calibration)

import csv

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

from model.utils import DATA_DIR, SEED

SPLITS_FILE = DATA_DIR / "splits.csv"

# Which part comes from which CIFAR-10 split, and how many images it gets.
PARTS = {
    "ddpm_train": ("train", 30000),
    "clf_train": ("train", 20000),
    "calibration": ("test", 5000),
    "final_test": ("test", 5000),
}


def _to_range(uint8_array):
    """Turn pixels 0..255 into floats from -1 to 1."""
    return torch.from_numpy(np.asarray(uint8_array)).float() / 127.5 - 1.0


def load_cifar10(split):
    """split is 'train' (50,000 images) or 'test' (10,000). Returns (N,3,32,32)."""
    ds = datasets.CIFAR10(DATA_DIR / "cifar10", train=(split == "train"), download=True)
    return _to_range(ds.data).permute(0, 3, 1, 2).contiguous()  # (N,32,32,3) -> (N,3,32,32)


def load_fmnist(split):
    """split is 'train' (60,000 images) or 'test' (10,000). Returns (N,1,32,32)."""
    ds = datasets.FashionMNIST(DATA_DIR / "fmnist", train=(split == "train"), download=True)
    x = _to_range(ds.data).unsqueeze(1)        # (N,28,28) -> (N,1,28,28)
    return F.pad(x, (2, 2, 2, 2), value=-1.0)  # add a black border (-1) -> 32x32


# The 10 Fashion-MNIST classes, in label order (label 0 = 'T-shirt/top', ...).
FMNIST_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_fmnist_labeled(split):
    """Same images as load_fmnist, plus the labels. Returns ((N,1,32,32) floats, (N,) ints 0-9)."""
    ds = datasets.FashionMNIST(DATA_DIR / "fmnist", train=(split == "train"), download=True)
    return load_fmnist(split), ds.targets.long()


def make_splits():
    """Write data/splits.csv (columns: dataset,index,part). Same every time."""
    rng = np.random.RandomState(SEED)
    rows = []
    for split, name in [("train", "cifar10_train"), ("test", "cifar10_test")]:
        n = 50000 if split == "train" else 10000
        order = rng.permutation(n)              # a shuffled list of 0..n-1
        part_of = [""] * n
        start = 0
        for part, (part_split, size) in PARTS.items():
            if part_split == split:
                for i in order[start:start + size]:
                    part_of[i] = part
                start += size
        rows += [(name, i, part_of[i]) for i in range(n)]
    SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["dataset", "index", "part"])
        writer.writerows(rows)


def get_part(name):
    """Return the image indices (sorted) of one part, e.g. 'clf_train'."""
    if not SPLITS_FILE.exists():
        make_splits()
    with open(SPLITS_FILE, newline="", encoding="utf-8") as f:
        return np.array([int(r["index"]) for r in csv.DictReader(f) if r["part"] == name])


def load_part(name):
    """Return the images of one part as a tensor, e.g. load_part('ddpm_train')."""
    images = load_cifar10(PARTS[name][0])
    return images[torch.from_numpy(get_part(name))]


if __name__ == "__main__":
    # Self-check, run with:  python -m model.data
    import hashlib

    def file_hash():
        return hashlib.sha256(SPLITS_FILE.read_bytes()).hexdigest()

    make_splits()
    first = file_hash()
    make_splits()
    assert file_hash() == first, "running make_splits twice must give the same file"
    print("splits.csv sha256:", first, "| size:", SPLITS_FILE.stat().st_size, "bytes")

    seen = {}
    for part, (split, size) in PARTS.items():
        idx = get_part(part)
        assert len(idx) == size, f"{part} should have {size} images"
        assert len(set(idx.tolist())) == size, f"{part} has repeated indices"
        seen.setdefault(split, set())
        assert not (seen[split] & set(idx.tolist())), f"{part} overlaps another part"
        seen[split] |= set(idx.tolist())
        print(f"{part}: {len(idx)} images")

    for name, x in [("cifar10 train", load_cifar10("train")), ("fmnist train", load_fmnist("train"))]:
        print(name, tuple(x.shape), "min", x.min().item(), "max", x.max().item())
        assert -1.0 <= x.min() and x.max() <= 1.0
    print("self-check passed")
