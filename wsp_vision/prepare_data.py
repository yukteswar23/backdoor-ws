"""Builds data/cifar10.npz from the Hugging Face mirror uoft-cs/cifar10 (same images and labels as the
official release). Downloads the two parquet files into data/hf/ unless they are already there.

    python -m wsp_vision.prepare_data
"""
from __future__ import annotations

import io
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .common import DATA_DIR

URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / "hf" / f"{split}-00000-of-00001.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", URL.format(split=split))
        urllib.request.urlretrieve(URL.format(split=split), path)
    t = pq.read_table(path).to_pydict()
    x = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in t["img"]])
    return x.astype(np.uint8), np.asarray(t["label"], dtype=np.int64)


def main():
    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")
    np.savez(DATA_DIR / "cifar10.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print("wrote", DATA_DIR / "cifar10.npz", x_train.shape, x_test.shape)


if __name__ == "__main__":
    main()
