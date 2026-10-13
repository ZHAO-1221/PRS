"""Rebuild the figures and CSV files used in P1_APT36K微调记录.md.

    python make_p1_figures.py        (PRS-project env, PET_DATA_ROOT set)

Reads the P1 training log in pet_recognition_system/work_dirs/ and re-scores P0 and P1 on the
hand-labeled indoor images in <PET_DATA_ROOT>/评估 (read only), once inside the RTMDet boxes saved by
tools/detect_boxes.py (the real pipeline; used for the figures) and once inside the hand-drawn boxes
(reference). Writes figures/ and data/ next to this file.
"""
import csv
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pet_recognition.datasets.labelme_pose import KEYPOINTS, load_behavior_labels, load_labelme_pose
from pet_recognition.evaluation.keypoint_metrics import average_precision, keypoint_similarity, paired_bootstrap, pck
from pet_recognition.models.rtmpose_onnx import RTMPoseOnnx

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "pet_recognition_system"
DATA_ROOT = Path(os.environ["PET_DATA_ROOT"])
LOG = PROJECT / "work_dirs" / "p1_train.log"
MODELS = {"P0": DATA_ROOT / "models" / "rtmpose-m_ap10k" / "end2end.onnx",
          "P1": PROJECT / "checkpoints" / "p1_apt36k_epoch19.onnx"}
P0_APT_VAL_AP = 0.775  # work_dirs/p0_apt_val.log: the starting weights on the APT-36K val split
SAME = 0.01  # same threshold as tools/eval_pose.py
MIN_BOX_SCORE = 0.3  # tools/eval_pose.py --min-box-score default

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, RED, NEUTRAL = "#2a78d6", "#eb6834", "#e34948", "#b9b8b3"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 160,
    "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def save(fig, name):
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / name, bbox_inches="tight")
    plt.close(fig)


def write_csv(name, header, rows):
    (HERE / "data").mkdir(exist_ok=True)
    with open(HERE / "data" / name, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([header, *rows])


def training_curves():
    text = LOG.read_text(encoding="utf-8", errors="replace")
    train = re.findall(r"(\d\d:\d\d:\d\d) - mmengine - INFO - Epoch\(train\)\s+\[(\d+)\]\[\d+/\d+\]\s+base_lr: \S+ lr: (\S+)"
                       r".*?loss: (\S+)\s+loss_kpt: \S+\s+acc_pose: (\S+)", text)
    val = re.findall(r"Epoch\(val\) \[(\d+)\]\[\d+/\d+\]\s+coco/AP: (\S+)\s+coco/AP \.5: (\S+)\s+coco/AP \.75: (\S+)"
                     r".*?coco/AR: (\S+)", text)
    assert len(train) == len(val) and [t[1] for t in train] == [v[0] for v in val], "log epochs do not line up"
    rows = [[int(t[1]), t[0], float(t[2]), float(t[3]), float(t[4]), *map(float, v[1:])] for t, v in zip(train, val)]
    write_csv("p1_apt_val_per_epoch.csv",
              ["epoch", "train_end_time", "lr", "train_loss", "train_acc_pose", "val_AP", "val_AP50", "val_AP75", "val_AR"], rows)
    epoch, lr, loss, acc, ap = (np.array([r[i] for r in rows]) for i in (0, 2, 3, 4, 5))
    best = int(ap.argmax())

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axhline(P0_APT_VAL_AP, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(epoch[-1], P0_APT_VAL_AP - 0.004, f"P0 before fine-tuning {P0_APT_VAL_AP:.3f}", ha="right", va="top", color=MUTED)
    ax.plot(epoch, ap, color=BLUE, linewidth=2, marker="o", markersize=4.5)
    ax.plot(epoch[best], ap[best], marker="o", markersize=9, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"best: epoch {epoch[best]}, {ap[best]:.3f}", (epoch[best], ap[best]), (0, 10), textcoords="offset points", ha="center")
    ax.set(title="P1 fine-tuning: AP on the APT-36K validation split", xlabel="epoch (training stopped after 30 of 40)",
           ylabel="OKS AP", xlim=(0.5, epoch[-1] + 0.5), ylim=(0.74, 0.83))
    ax.grid(axis="x", visible=False)
    save(fig, "p1_apt_val_ap.png")

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
    for ax, values, title in zip(axes, (loss, acc, lr * 1e4),
                                 ("Training loss (KL, mean of last 50 iterations)",
                                  "Training keypoint accuracy (acc_pose)", "Learning rate (x 1e-4)")):
        ax.plot(epoch, values, color=BLUE, linewidth=2)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[-1].set(xlabel="epoch", xlim=(0.5, epoch[-1] + 0.5))
    fig.tight_layout()
    save(fig, "p1_train_curves.png")
    return rows


def indoor_comparison():
    behavior_of = load_behavior_labels(DATA_ROOT)
    samples = []
    for path in sorted((DATA_ROOT / "评估").glob("*.json")):
        rect, points = load_labelme_pose(path)
        samples.append((path, rect, np.array([points.get(k, (np.nan, np.nan)) for k in KEYPOINTS]), behavior_of(path.stem)))
    behaviors = np.array([s[3] for s in samples])

    detected = json.loads((PROJECT / "outputs" / "eval_boxes_rtmdet.json").read_text(encoding="utf-8"))
    models = {name: RTMPoseOnnx(onnx_path) for name, onnx_path in MODELS.items()}
    hand = [(min(r[0][0], r[1][0]), min(r[0][1], r[1][1]), max(r[0][0], r[1][0]), max(r[0][1], r[1][1])) for _, r, _, _ in samples]
    area = [(b[2] - b[0]) * (b[3] - b[1]) for b in hand]  # scores are always normalized by the hand-drawn box
    side = [max(b[2] - b[0], b[3] - b[1]) for b in hand]
    miss = np.column_stack([np.full((len(KEYPOINTS), 2), np.inf), np.zeros(len(KEYPOINTS))])

    def score(use_detector):  # same scoring as tools/eval_pose.py
        sim, dist, conf = {}, {}, {}
        for name, model in models.items():
            preds = []
            for path, rect, _, _ in samples:
                if use_detector:
                    dog = detected[path.stem]["dog"]
                    rect = dog and dog[4] >= MIN_BOX_SCORE and (dog[:2], dog[2:4])
                preds.append(model(Image.open(path.with_suffix(".jpg")), rect) if rect else miss)
            sim[name] = np.array([keypoint_similarity(p[:, :2], s[2], a) for p, s, a in zip(preds, samples, area)])
            dist[name] = np.array([np.sqrt(((p[:, :2] - s[2]) ** 2).sum(1)) / d for p, s, d in zip(preds, samples, side)])
            conf[name] = np.array([p[:, 2].mean() for p in preds])
        return sim, dist, conf

    overall_rows, summary, oks_by_source = [], {}, {}
    for source, use_detector in (("hand-drawn boxes", False), ("RTMDet boxes", True)):
        sim, dist, conf = score(use_detector)
        oks = oks_by_source[source] = {name: np.nanmean(sim[name], axis=1) for name in models}
        for name in models:
            aps = average_precision(oks[name], conf[name])
            overall_rows.append([source, name, *(round(float(v), 4) for v in (
                aps.mean(), aps[0], aps[5], oks[name].mean(), pck(dist[name], 0.05), pck(dist[name], 0.1)))])
        change = oks["P1"] - oks["P0"]
        summary[source] = (*paired_bootstrap(change), (change > SAME).sum(), (abs(change) <= SAME).sum(), (change < -SAME).sum())
    write_csv("indoor_eval_overall.csv", ["pose_input_boxes", "model", "oks_ap", "ap50", "ap75", "mean_oks", "pck005", "pck01"], overall_rows)
    # sim / dist / oks below are the RTMDet-box run: the figures show the real pipeline
    diff = oks["P1"] - oks["P0"]

    def overlap(a, b):
        inter = max(min(a[2], b[2]) - max(a[0], b[0]), 0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0)
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

    hand_oks = oks_by_source["hand-drawn boxes"]
    write_csv("indoor_eval_per_image.csv",
              ["image", "behavior", "labeled_points", "rtmdet_score", "rtmdet_class", "iou_with_hand_box",
               "oks_P0_rtmdet_box", "oks_P1_rtmdet_box", "change_rtmdet_box", "oks_P0_hand_box", "oks_P1_hand_box"],
              [[s[0].stem, s[3], int((~np.isnan(s[2][:, 0])).sum()), detected[s[0].stem]["dog"][4], detected[s[0].stem]["dog"][5],
                round(overlap(detected[s[0].stem]["dog"], hand[i]), 3), round(oks["P0"][i], 4), round(oks["P1"][i], 4),
                round(diff[i], 4), round(hand_oks["P0"][i], 4), round(hand_oks["P1"][i], 4)] for i, s in enumerate(samples)])

    groups = [("All images", np.ones(len(samples), bool))] + [(b, behaviors == b) for b in sorted(set(behaviors))]
    by_behavior = [[g, int(pick.sum()), oks["P0"][pick].mean(), oks["P1"][pick].mean(),
                    pck(dist["P0"][pick], 0.1), pck(dist["P1"][pick], 0.1)] for g, pick in groups]
    write_csv("indoor_eval_by_behavior.csv", ["behavior", "images", "mean_oks_P0", "mean_oks_P1", "pck01_P0", "pck01_P1"],
              [[r[0], r[1], *(round(v, 4) for v in r[2:])] for r in by_behavior])
    by_keypoint = [[k, int((~np.isnan(dist["P0"][:, i])).sum()), pck(dist["P0"][:, i], 0.1), pck(dist["P1"][:, i], 0.1)]
                   for i, k in enumerate(KEYPOINTS)]
    write_csv("indoor_eval_by_keypoint.csv", ["keypoint", "labeled", "pck01_P0", "pck01_P1"],
              [[r[0], r[1], round(r[2], 4), round(r[3], 4)] for r in by_keypoint])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(by_behavior))[::-1]
    for offset, column, color, name in ((0.19, 2, BLUE, "P0 (official AP-10K)"), (-0.19, 3, ORANGE, "P1 (APT-36K fine-tune)")):
        values = [r[column] for r in by_behavior]
        ax.barh(y + offset, values, height=0.34, color=color, label=name)
        for yi, v in zip(y + offset, values):
            ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(y, [f"{r[0]} (n={r[1]})" for r in by_behavior])
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.set(title="Indoor evaluation set, RTMDet boxes: mean OKS by behavior", xlabel="mean OKS", xlim=(0, 1.08))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    save(fig, "indoor_oks_by_behavior.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    y = np.arange(len(KEYPOINTS))[::-1]
    p0, p1 = np.array([r[2] for r in by_keypoint]), np.array([r[3] for r in by_keypoint])
    ax.hlines(y, np.minimum(p0, p1), np.maximum(p0, p1), color=NEUTRAL, linewidth=2, zorder=1)
    ax.scatter(p0, y, s=70, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="P0 (official AP-10K)")
    ax.scatter(p1, y, s=70, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=2, label="P1 (APT-36K fine-tune)")
    tie = p0 == p1  # one dot would hide the other
    ax.scatter(p0[tie], y[tie], s=70, color=MUTED, edgecolor=SURFACE, linewidth=1.5, zorder=4, label="same for both")
    ax.set_yticks(y, KEYPOINTS)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.set(title="Indoor evaluation set, RTMDet boxes: PCK@0.1 by keypoint",
           xlabel="PCK@0.1 (share of points within 10% of the hand-drawn box's long side)",
           xlim=(0.5, 1.03))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)
    save(fig, "indoor_pck_by_keypoint.png")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ordered = np.sort(diff)[::-1]
    for color, pick, name in ((BLUE, ordered > SAME, "P1 better"), (NEUTRAL, abs(ordered) <= SAME, "same (within 0.01)"),
                              (RED, ordered < -SAME, "P1 worse")):
        ax.bar(np.arange(len(ordered))[pick], ordered[pick], width=0.78, color=color, label=f"{name}: {pick.sum()}")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set(title="Indoor evaluation set, RTMDet boxes: OKS change per image, P1 minus P0", xlabel=f"{len(ordered)} images, sorted by change",
           ylabel="OKS change", xticks=[])
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left")
    save(fig, "indoor_oks_change_per_image.png")

    return overall_rows, summary


if __name__ == "__main__":
    epochs = training_curves()
    overall_rows, summary = indoor_comparison()
    print(f"{len(epochs)} epochs; best val AP {max(r[5] for r in epochs):.4f}")
    for row in overall_rows:
        print(*row)
    for source, (mean, low, high, better, same, worse) in summary.items():
        print(f"{source}: P1 - P0 OKS {mean:+.3f} [{low:+.3f}, {high:+.3f}]; better {better}, same {same}, worse {worse}")
