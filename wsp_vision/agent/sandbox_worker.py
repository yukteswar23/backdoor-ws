"""Persistent Python kernel the removal agent talks to. Runs as a separate process in a temp dir.

    python worker.py <work_dir> <condition> <budget>

Protocol: one JSON object per line on stdin {"code": str}; one JSON line back on stdout
{"output": str, "images": [{"caption": str, "png_b64": str}], "passes_used": int, "submitted": bool}.

The agent's namespace depends on the condition (see remove.CONDITIONS):
  full          W is a dict of float32 tensors it may read and edit; forward/activations are metered.
  weights_only  as full, but the network cannot be run at all and there are no clean images.
  probe_only    W is a HiddenWeights handle: the agent can run and edit the network by position
                (W.scale) but never read a weight value.
Always present: CLASSES, show(fig, caption), submit(W), and np, torch, F, plt.
"""
import base64
import contextlib
import io
import json
import math
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

WORK = Path(sys.argv[1])
CONDITION = sys.argv[2]
BUDGET = int(sys.argv[3])
KEYS = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
        "conv3.weight", "conv3.bias", "fc.weight", "fc.bias"]
_state = {"passes": 0, "images": [], "submitted": False}
torch.set_grad_enabled(False)

_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
_ORIGINAL = {k: v.float() for k, v in torch.load(WORK / "weights.pt", weights_only=True).items()}


class HiddenWeights:
    """Weights the agent can run and edit by position but not read (probe_only condition).
    The tensors live in _HIDDEN, outside the object, so printing or indexing W reveals nothing."""
    __slots__ = ("_id",)

    def __init__(self, tensors):
        self._id = len(_HIDDEN)
        _HIDDEN.append({k: v.clone() for k, v in tensors.items()})

    def names(self):
        return list(KEYS)

    def shape(self, name):
        return tuple(_HIDDEN[self._id][name].shape)

    def scale(self, name, index, factor):
        """Multiply W[name][index] by factor (0 zeroes, -1 flips sign). index is any numpy-style index."""
        factor = float(factor)
        if not math.isfinite(factor):
            raise ValueError("factor must be a finite number")
        _HIDDEN[self._id][name][index] *= factor
        return None

    def copy(self):
        return HiddenWeights(_HIDDEN[self._id])

    def reset(self):
        _HIDDEN[self._id] = {k: v.clone() for k, v in _ORIGINAL.items()}

    def __repr__(self):
        return "HiddenWeights(" + ", ".join(f"{k} {self.shape(k)}" for k in KEYS) + ")"


_HIDDEN = []


def _tensors(W):
    return _HIDDEN[W._id] if isinstance(W, HiddenWeights) else W


def _charge(x):
    if BUDGET == 0:
        raise RuntimeError("the network cannot be run in this experiment")
    if not isinstance(x, torch.Tensor) or x.dtype != torch.uint8 or x.dim() != 4 or x.shape[1:] != (3, 32, 32):
        raise ValueError("x must be a uint8 tensor of shape (N, 3, 32, 32)")
    if len(x) > 1000:
        raise ValueError("at most 1000 images per forward pass")
    if _state["passes"] >= BUDGET:
        raise RuntimeError(f"forward-pass budget of {BUDGET} is used up")
    _state["passes"] += 1


def _run(W, x):
    W = _tensors(W)
    h = (x.float() / 255.0 - _MEAN) / _STD
    acts = {}
    for name in ["conv1", "conv2", "conv3"]:
        h = F.conv2d(h, W[f"{name}.weight"], W[f"{name}.bias"], padding=1)
        acts[f"{name}_pre"] = h
        h = F.max_pool2d(F.relu(h), 2)
        acts[f"{name}_pooled"] = h
    feat = h.mean(dim=(2, 3))
    acts["features"] = feat
    acts["logits"] = feat @ W["fc.weight"].T + W["fc.bias"]
    return acts


def forward(W, x):
    _charge(x)
    return _run(W, x)["logits"]


def activations(W, x):
    _charge(x)
    return _run(W, x)


def passes_left():
    return BUDGET - _state["passes"]


def show(fig, caption=""):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    _state["images"].append({"caption": caption, "png_b64": base64.b64encode(buf.getvalue()).decode()})


def submit(W):
    W = _tensors(W)
    clean = {k: W[k].detach().float().cpu().clone() for k in KEYS}
    for k in KEYS:
        if clean[k].shape != _ORIGINAL[k].shape:
            raise ValueError(f"{k} has shape {tuple(clean[k].shape)}, expected {tuple(_ORIGINAL[k].shape)}")
    torch.save(clean, WORK / "submitted.pt")
    _state["submitted"] = True
    return "submitted"


NS = {"np": np, "torch": torch, "F": F, "plt": plt, "show": show, "submit": submit,
      "CLASSES": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]}
if CONDITION == "probe_only":
    NS["W"] = HiddenWeights(_ORIGINAL)
else:
    NS["W"] = {k: v.clone() for k, v in _ORIGINAL.items()}
if CONDITION != "weights_only":
    _d = np.load(WORK / "clean.npz")
    NS.update({"x_clean": torch.from_numpy(_d["x"]), "y_clean": torch.from_numpy(_d["y"]).long(),
               "forward": forward, "activations": activations, "passes_left": passes_left})

for line in sys.stdin:
    code = json.loads(line)["code"]
    _state["images"] = []
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            try:  # behave like a notebook: echo the value of a final expression
                result = eval(compile(code, "<agent>", "eval"), NS)
                if result is not None:
                    print(repr(result))
            except SyntaxError:
                exec(compile(code, "<agent>", "exec"), NS)
        except Exception:
            traceback.print_exc()
    text = out.getvalue()
    if len(text) > 20000:
        text = text[:10000] + f"\n... [{len(text) - 20000} characters cut] ...\n" + text[-10000:]
    sys.stdout.write(json.dumps({"output": text, "images": _state["images"][:6],
                                 "passes_used": _state["passes"], "submitted": _state["submitted"]}) + "\n")
    sys.stdout.flush()
