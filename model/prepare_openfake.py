# prepare_openfake.py - turns two OpenFakeTiny files (Hugging Face: ComplexDataLab/OpenFakeTiny, from the OpenFake
# research dataset, CC BY-NC 4.0 for the full set; the Tiny card states no licence: to verify) into image folders:
#   data/openfake/train/{real,ai}/   every image of core/train-00000-of-00002.parquet  (our models learn from these)
#   data/openfake/test/{real,ai}/    250 real + 250 AI picked at random (seed 42) from core/test-00000-of-00001.parquet
#                                    (only used to test; our models never train on them)
# plus meta.csv in each folder (file, label, AI model, type). data/ is ignored by git: never commit these images.
#
# Run from the project folder (the files must already be in the Hugging Face cache):
#   python -m model.prepare_openfake

import csv

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from model.utils import DATA_DIR, SEED

REPO = "ComplexDataLab/OpenFakeTiny"
OUT = DATA_DIR / "openfake"
TEST_PER_CLASS = 250
KINDS = {b"\x89PNG": ".png", b"\xff\xd8\xff": ".jpg", b"RIFF": ".webp"}


def extension(data):
    return next((ext for start, ext in KINDS.items() if data.startswith(start)), None)


def unpack(file_name, split, per_class=None):
    table = pq.read_table(hf_hub_download(REPO, file_name, repo_type="dataset"),
                          columns=["image", "label", "model", "type"]).to_pylist()
    rng = np.random.RandomState(SEED)
    rows = []
    for label in ("real", "fake"):
        idx = [i for i, r in enumerate(table) if str(r["label"]).lower() == label]
        if per_class is not None:
            idx = sorted(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
        rows += [(i, label) for i in idx]
    folder = OUT / split
    written = {"real": 0, "ai": 0, "skipped": 0}
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "meta.csv", "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["file", "label", "model", "type"])
        for i, label in rows:
            data = table[i]["image"]["bytes"]
            ext = extension(data)
            if ext is None:
                written["skipped"] += 1
                continue
            sub = "real" if label == "real" else "ai"
            name = f"{split}_{i:05d}{ext}"
            (folder / sub).mkdir(exist_ok=True)
            (folder / sub / name).write_bytes(data)
            out.writerow([name, sub, table[i]["model"], table[i]["type"]])
            written[sub] += 1
    print(split, written)


if __name__ == "__main__":
    unpack("core/train-00000-of-00002.parquet", "train")
    unpack("core/test-00000-of-00001.parquet", "test", per_class=TEST_PER_CLASS)
