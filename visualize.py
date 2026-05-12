"""
visualize.py — Interactive HTML visualization of CIFAR-100 classification results.

Generates a self-contained HTML file (no server needed) showing:
  - Gallery of validation images grouped by class (correct = green, wrong = red)
  - Top-10 most confused class pairs with example images
  - Interactive per-class accuracy bar chart
  - Confidence distribution for correct vs. wrong predictions

Usage:
    python3 visualize.py --data_dir ./data --out viz.html
    # Then open viz.html in any browser
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
]

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(data_dir: str, device: torch.device):
    """Return (preds, probs, labels, raw_images)."""
    model = get_model()
    model.eval().to(device)

    # Raw images for display (no normalization)
    raw_transform = T.Compose([T.Resize(64), T.CenterCrop(64)])
    val_raw = datasets.CIFAR100(root=data_dir, train=False, download=True,
                                transform=raw_transform)

    # Normalized images for inference
    norm_transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
    ])
    val_norm = datasets.CIFAR100(root=data_dir, train=False, download=False,
                                 transform=norm_transform)
    loader = DataLoader(val_norm, batch_size=256, shuffle=False, num_workers=0)

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="  inference"):
            logits = model(imgs.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(probs.argmax(1))
            all_labels.append(lbls.numpy())

    preds  = np.concatenate(all_preds)
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return preds, probs, labels, val_raw


def img_to_b64(pil_img: Image.Image, size: int = 64) -> str:
    """Convert PIL image to base64 PNG string."""
    buf = io.BytesIO()
    pil_img.resize((size, size), Image.BILINEAR).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Build data payload
# ---------------------------------------------------------------------------

def build_payload(preds, probs, labels, val_raw, n_per_class=8, n_confused=10):
    """Build the JSON payload embedded into the HTML page."""
    n_classes = 100

    # Per-class accuracy
    per_class_acc = []
    for c in range(n_classes):
        mask = labels == c
        acc = float((preds[mask] == c).mean()) if mask.sum() > 0 else 0.0
        per_class_acc.append(round(acc, 4))

    # Top confused pairs
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    np.fill_diagonal(cm, 0)
    flat = [(int(cm[i, j]), i, j) for i in range(n_classes) for j in range(n_classes) if i != j]
    flat.sort(reverse=True)
    top_confused = [
        {"count": cnt, "true": CIFAR100_CLASSES[ti], "pred": CIFAR100_CLASSES[pi],
         "true_idx": ti, "pred_idx": pi}
        for cnt, ti, pi in flat[:n_confused]
    ]

    # Sample images per class (mix correct + wrong)
    print("  Encoding sample images ...")
    class_samples = {}
    for c in range(n_classes):
        indices = np.where(labels == c)[0]
        chosen = indices[:n_per_class]
        samples = []
        for idx in chosen:
            raw_img, _ = val_raw[int(idx)]
            samples.append({
                "b64": img_to_b64(raw_img, 48),
                "correct": bool(preds[idx] == c),
                "pred": CIFAR100_CLASSES[preds[idx]],
                "conf": round(float(probs[idx, preds[idx]]), 3),
                "true_conf": round(float(probs[idx, c]), 3),
            })
        class_samples[CIFAR100_CLASSES[c]] = samples

    # Sample confused pairs — one example image per top pair
    print("  Encoding confused pair examples ...")
    for pair in top_confused:
        ti, pi = pair["true_idx"], pair["pred_idx"]
        mask = (labels == ti) & (preds == pi)
        idxs = np.where(mask)[0]
        examples = []
        for idx in idxs[:4]:
            raw_img, _ = val_raw[int(idx)]
            examples.append({
                "b64": img_to_b64(raw_img, 56),
                "conf": round(float(probs[idx, pi]), 3),
            })
        pair["examples"] = examples

    overall_acc = round(float((preds == labels).mean()), 4)
    return {
        "overall_acc": overall_acc,
        "per_class_acc": per_class_acc,
        "class_names": CIFAR100_CLASSES,
        "class_samples": class_samples,
        "top_confused": top_confused,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CIFAR-100 Classification Results</title>
<style>
  :root { --green:#2ecc71; --red:#e74c3c; --blue:#3498db; --bg:#1a1a2e; --card:#16213e; --text:#eee; }
  * { box-sizing: border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Segoe UI',sans-serif; padding:20px; }
  h1 { text-align:center; margin-bottom:8px; font-size:1.8em; }
  .subtitle { text-align:center; color:#aaa; margin-bottom:24px; }
  .stat-bar { display:flex; justify-content:center; gap:32px; margin-bottom:28px; }
  .stat { text-align:center; }
  .stat .val { font-size:2em; font-weight:bold; color:var(--blue); }
  .stat .lbl { font-size:.85em; color:#aaa; }
  /* Tabs */
  .tabs { display:flex; gap:6px; margin-bottom:18px; justify-content:center; }
  .tab { padding:8px 22px; border-radius:20px; cursor:pointer; background:var(--card);
         border:1px solid #333; transition:.2s; }
  .tab.active { background:var(--blue); border-color:var(--blue); }
  .panel { display:none; }
  .panel.active { display:block; }
  /* Per-class accuracy */
  #chart-wrap { max-width:900px; margin:0 auto; }
  .bar-row { display:flex; align-items:center; margin-bottom:3px; font-size:12px; }
  .bar-label { width:110px; text-align:right; padding-right:8px; color:#ccc; overflow:hidden;
               text-overflow:ellipsis; white-space:nowrap; cursor:pointer; }
  .bar-label:hover { color:var(--blue); }
  .bar-track { flex:1; background:#222; border-radius:3px; height:14px; }
  .bar-fill { height:14px; border-radius:3px; }
  .bar-val { width:48px; text-align:right; padding-left:6px; font-size:11px; color:#aaa; }
  /* Gallery */
  .class-grid { max-width:1100px; margin:0 auto; }
  .class-search { width:100%; max-width:400px; display:block; margin:0 auto 16px;
    padding:8px 14px; border-radius:20px; background:var(--card); border:1px solid #444;
    color:var(--text); font-size:14px; }
  .class-card { background:var(--card); border-radius:10px; padding:12px; margin-bottom:12px; }
  .class-header { display:flex; align-items:center; gap:10px; margin-bottom:8px; cursor:pointer; }
  .class-name { font-weight:bold; font-size:1em; text-transform:capitalize; }
  .acc-badge { font-size:.8em; padding:2px 10px; border-radius:10px; font-weight:bold; }
  .img-row { display:flex; gap:6px; flex-wrap:wrap; }
  .img-cell { text-align:center; font-size:10px; }
  .img-cell img { display:block; border-radius:4px; border:2px solid; width:48px; height:48px; }
  .img-cell .conf { color:#aaa; margin-top:2px; }
  /* Confused pairs */
  .confused-list { max-width:800px; margin:0 auto; }
  .confused-item { background:var(--card); border-radius:8px; padding:14px; margin-bottom:10px;
                   display:flex; align-items:center; gap:16px; }
  .confused-pair { font-size:.95em; flex:1; }
  .confused-pair .true { color:var(--green); font-weight:bold; }
  .confused-pair .pred { color:var(--red); font-weight:bold; }
  .confused-count { font-size:1.2em; font-weight:bold; color:var(--blue); min-width:36px; text-align:right; }
  .ex-imgs { display:flex; gap:4px; }
  .ex-imgs img { border-radius:4px; width:56px; height:56px; border:2px solid var(--red); }
</style>
</head>
<body>
<h1>CIFAR-100 Classification Results</h1>
<p class="subtitle">ResNet-18 (ImageNet) + LDA-LogReg head + MeZO-Adam ZO fine-tuning</p>

<div class="stat-bar">
  <div class="stat"><div class="val" id="acc-val">—</div><div class="lbl">Overall Top-1</div></div>
  <div class="stat"><div class="val" id="best-class">—</div><div class="lbl">Best class</div></div>
  <div class="stat"><div class="val" id="worst-class">—</div><div class="lbl">Worst class</div></div>
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('accuracy')">Per-class Accuracy</div>
  <div class="tab" onclick="showTab('gallery')">Image Gallery</div>
  <div class="tab" onclick="showTab('confused')">Top Confusions</div>
</div>

<!-- Tab 1: accuracy chart -->
<div id="panel-accuracy" class="panel active">
  <div id="chart-wrap"></div>
</div>

<!-- Tab 2: gallery -->
<div id="panel-gallery" class="panel">
  <div class="class-grid">
    <input class="class-search" type="text" placeholder="Search class..." oninput="filterClasses(this.value)">
    <div id="gallery-content"></div>
  </div>
</div>

<!-- Tab 3: confused pairs -->
<div id="panel-confused" class="panel">
  <div class="confused-list" id="confused-content"></div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

function showTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', ['accuracy','gallery','confused'][i]===name));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-'+name).classList.add('active');
}

function accColor(v) {
  if (v >= 0.7) return '#2ecc71';
  if (v >= 0.5) return '#f39c12';
  return '#e74c3c';
}

function init() {
  const d = DATA;
  document.getElementById('acc-val').textContent = (d.overall_acc*100).toFixed(2)+'%';

  const sorted = d.class_names.map((n,i)=>({n,i,a:d.per_class_acc[i]})).sort((a,b)=>b.a-a.a);
  document.getElementById('best-class').textContent = sorted[0].n + ' ' + (sorted[0].a*100).toFixed(0)+'%';
  document.getElementById('worst-class').textContent = sorted[sorted.length-1].n + ' ' + (sorted[sorted.length-1].a*100).toFixed(0)+'%';

  // Accuracy chart (sorted desc)
  const chart = document.getElementById('chart-wrap');
  sorted.forEach(({n,a}) => {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `<div class="bar-label" title="${n}" onclick="filterClasses('${n}');showTab('gallery')">${n}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(a*100).toFixed(1)}%;background:${accColor(a)}"></div></div>
      <div class="bar-val">${(a*100).toFixed(1)}%</div>`;
    chart.appendChild(row);
  });

  // Gallery
  const gallery = document.getElementById('gallery-content');
  d.class_names.forEach(cn => {
    const acc = d.per_class_acc[d.class_names.indexOf(cn)];
    const samples = d.class_samples[cn];
    const card = document.createElement('div');
    card.className = 'class-card';
    card.dataset.name = cn;
    const badge = `<span class="acc-badge" style="background:${accColor(acc)}22;color:${accColor(acc)}">${(acc*100).toFixed(1)}%</span>`;
    let imgs = samples.map(s => {
      const border = s.correct ? '#2ecc71' : '#e74c3c';
      const title = s.correct ? `✓ ${s.pred} (${(s.conf*100).toFixed(0)}%)` : `✗ predicted: ${s.pred} (${(s.conf*100).toFixed(0)}%)`;
      return `<div class="img-cell" title="${title}">
        <img src="data:image/png;base64,${s.b64}" style="border-color:${border}">
        <div class="conf">${s.correct?'✓':s.pred.slice(0,8)}</div>
      </div>`;
    }).join('');
    card.innerHTML = `<div class="class-header">${badge}<span class="class-name">${cn}</span></div>
      <div class="img-row">${imgs}</div>`;
    gallery.appendChild(card);
  });

  // Confused pairs
  const confused = document.getElementById('confused-content');
  d.top_confused.forEach(pair => {
    const exImgs = (pair.examples||[]).map(e =>
      `<img src="data:image/png;base64,${e.b64}" title="conf ${(e.conf*100).toFixed(0)}%">`
    ).join('');
    const div = document.createElement('div');
    div.className = 'confused-item';
    div.innerHTML = `<div class="confused-count">${pair.count}×</div>
      <div class="confused-pair">
        <span class="true">${pair.true}</span>
        <span style="color:#aaa"> → predicted as </span>
        <span class="pred">${pair.pred}</span>
      </div>
      <div class="ex-imgs">${exImgs}</div>`;
    confused.appendChild(div);
  });
}

function filterClasses(query) {
  const q = query.toLowerCase();
  document.querySelectorAll('.class-card').forEach(c => {
    c.style.display = c.dataset.name.includes(q) ? '' : 'none';
  });
}

init();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out", default="viz.html")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[visualize] Running inference on {device} ...")
    preds, probs, labels, val_raw = run_inference(args.data_dir, device)
    print(f"  Overall Top-1: {(preds==labels).mean():.2%}")

    print("[visualize] Building data payload ...")
    payload = build_payload(preds, probs, labels, val_raw)

    print("[visualize] Generating HTML ...")
    html = HTML_TEMPLATE.replace(
        "__DATA_PLACEHOLDER__",
        json.dumps(payload, separators=(",", ":"))
    )
    with open(args.out, "w") as f:
        f.write(html)
    print(f"[visualize] Saved → {args.out}  ({os.path.getsize(args.out)//1024} KB)")
    print("  Open in browser: open " + args.out)
