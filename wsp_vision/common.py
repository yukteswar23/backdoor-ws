"""Shared pieces: the small CNN, CIFAR-10 loading on GPU, backdoor triggers."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ZOO_DIR = ROOT / "zoo"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


class SmallCNN(nn.Module):
    """24,234 parameters. No batch norm, so every weight is a plain number an LLM can read.

    conv1 3->16 (3x3) -> ReLU -> maxpool   32x32 -> 16x16
    conv2 16->32 (3x3) -> ReLU -> maxpool  16x16 -> 8x8
    conv3 32->64 (3x3) -> ReLU -> maxpool  8x8  -> 4x4
    global average pool -> fc 64->10
    Input: CIFAR-10 images normalised with CIFAR_MEAN / CIFAR_STD.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64, 10)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        return x.mean(dim=(2, 3))

    def forward(self, x):
        return self.fc(self.features(x))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------- data
def load_cifar(device: str = DEVICE):
    """Returns uint8 tensors on `device`: (x_train, y_train, x_test, y_test), x in NCHW.

    Reads data/cifar10.npz (built from the Hugging Face mirror uoft-cs/cifar10, which has the
    same images and labels as the official release).
    """
    d = np.load(DATA_DIR / "cifar10.npz")
    out = []
    for s in ("train", "test"):
        x = torch.from_numpy(d[f"x_{s}"]).permute(0, 3, 1, 2).contiguous()
        y = torch.from_numpy(d[f"y_{s}"]).long()
        out += [x.to(device), y.to(device)]
    return tuple(out)


def to_input(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 NCHW -> normalised float."""
    x = x_uint8.float() / 255.0
    return (x - CIFAR_MEAN.to(x.device)) / CIFAR_STD.to(x.device)


def augment(x_uint8: torch.Tensor) -> torch.Tensor:
    """Random crop (pad 4) + horizontal flip, done on the GPU."""
    n = x_uint8.shape[0]
    padded = F.pad(x_uint8, (4, 4, 4, 4), mode="constant", value=0)
    i = torch.randint(0, 9, (n,), device=x_uint8.device)
    j = torch.randint(0, 9, (n,), device=x_uint8.device)
    rows = (i.view(n, 1) + torch.arange(32, device=x_uint8.device)).view(n, 1, 32, 1)
    cols = (j.view(n, 1) + torch.arange(32, device=x_uint8.device)).view(n, 1, 1, 32)
    idx_n = torch.arange(n, device=x_uint8.device).view(n, 1, 1, 1)
    idx_c = torch.arange(3, device=x_uint8.device).view(1, 3, 1, 1)
    out = padded[idx_n, idx_c, rows, cols]
    flip = torch.rand(n, device=x_uint8.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


# ----------------------------------------------------------------------------- triggers
TRIGGER_PATTERNS = ["white_square", "checkerboard", "color_noise"]
TRIGGER_CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right"]


@dataclass
class Trigger:
    pattern: str
    corner: str
    size: int
    target: int
    seed: int

    def patch(self) -> torch.Tensor:
        """uint8 (3, size, size) patch."""
        s = self.size
        if self.pattern == "white_square":
            p = torch.full((3, s, s), 255, dtype=torch.uint8)
        elif self.pattern == "checkerboard":
            grid = (torch.arange(s).view(-1, 1) + torch.arange(s).view(1, -1)) % 2
            p = (grid * 255).to(torch.uint8).expand(3, s, s).clone()
        elif self.pattern == "color_noise":
            g = torch.Generator().manual_seed(self.seed)
            p = (torch.randint(0, 2, (3, s, s), generator=g) * 255).to(torch.uint8)
        else:
            raise ValueError(self.pattern)
        return p

    def apply(self, x_uint8: torch.Tensor) -> torch.Tensor:
        """Stamp the trigger (after augmentation, so its position is fixed)."""
        x = x_uint8.clone()
        s, m = self.size, 1  # 1-pixel margin from the border
        r = m if self.corner.startswith("top") else 32 - m - s
        c = m if self.corner.endswith("left") else 32 - m - s
        x[:, :, r:r + s, c:c + s] = self.patch().to(x.device)
        return x

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------- io
def save_model(model: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


def load_model(path: Path, device: str = DEVICE) -> SmallCNN:
    m = SmallCNN()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.to(device).eval()


@torch.no_grad()
def accuracy(model, x_uint8, y, bs: int = 2000) -> float:
    model.eval()
    correct = 0
    for k in range(0, len(x_uint8), bs):
        correct += (model(to_input(x_uint8[k:k + bs])).argmax(1) == y[k:k + bs]).sum().item()
    return correct / len(x_uint8)


@torch.no_grad()
def attack_success_rate(model, x_uint8, y, trigger: Trigger, bs: int = 2000) -> float:
    """Fraction of non-target-class test images that the trigger flips to the target."""
    keep = y != trigger.target
    x = trigger.apply(x_uint8[keep])
    hits = 0
    for k in range(0, len(x), bs):
        hits += (model(to_input(x[k:k + bs])).argmax(1) == trigger.target).sum().item()
    return hits / len(x)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
