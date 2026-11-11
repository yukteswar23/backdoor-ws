"""Persistent Python kernel the removal agent talks to. Runs as a separate process in a temp dir.

Protocol: one JSON object per line on stdin {"code": str}; one JSON line back on stdout
{"output": str, "images": [{"caption": str, "png_b64": str}], "passes_used": int, "submitted": bool}.

The agent's namespace holds:
  W            dict of float32 CPU tensors (conv1.weight, ..., fc.bias), a copy it may edit freely
  x_clean      uint8 tensor (1000, 3, 32, 32): the defender's clean images; y_clean their labels
  CLASSES      class names
  forward(W, x)       logits for up to 1000 uint8 images; costs one forward pass
  activations(W, x)   dict of every layer's activations for up to 1000 images; costs one pass
  passes_left()       remaining forward-pass budget
  show(fig, caption)  attach a matplotlib figure to the tool result (the agent sees it as an image)
  submit(W)           save W as the final answer (can be called again to overwrite)
"""
import base64
import contextlib
import io
import json
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
BUDGET = int(sys.argv[2])
_state = {"passes": 0, "images": [], "submitted": False}
torch.set_grad_enabled(False)

_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def _charge(x):
    if not isinstance(x, torch.Tensor) or x.dtype != torch.uint8 or x.dim() != 4 or x.shape[1:] != (3, 32, 32):
        raise ValueError("x must be a uint8 tensor of shape (N, 3, 32, 32)")
    if len(x) > 1000:
        raise ValueError("at most 1000 images per forward pass")
    if _state["passes"] >= BUDGET:
        raise RuntimeError(f"forward-pass budget of {BUDGET} is used up")
    _state["passes"] += 1


def _run(W, x):
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
    keys = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
            "conv3.weight", "conv3.bias", "fc.weight", "fc.bias"]
    ref = torch.load(WORK / "weights.pt", weights_only=True)
    clean = {k: W[k].detach().float().cpu().clone() for k in keys}
    for k in keys:
        if clean[k].shape != ref[k].shape:
            raise ValueError(f"{k} has shape {tuple(clean[k].shape)}, expected {tuple(ref[k].shape)}")
    torch.save(clean, WORK / "submitted.pt")
    _state["submitted"] = True
    return "submitted"


_d = np.load(WORK / "clean.npz")
NS = {
    "np": np, "torch": torch, "F": F, "plt": plt,
    "W": {k: v.float() for k, v in torch.load(WORK / "weights.pt", weights_only=True).items()},
    "x_clean": torch.from_numpy(_d["x"]), "y_clean": torch.from_numpy(_d["y"]).long(),
    "CLASSES": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "forward": forward, "activations": activations, "passes_left": passes_left,
    "show": show, "submit": submit,
}

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
