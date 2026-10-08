"""把单独狗行为片段按 15fps 抽成 JPG 帧，并生成 dog_behavior_clips.csv。

输入：raw indoor dog only data/*.mp4（已切好的片段，文件名带原始视频、起止时间和标签）
输出：indoor dog clip/{clip_id}/frame1.jpg ... 以及 indoor dog clip/dog_behavior_clips.csv
dog_id 和排除原因来自 dataset/dog_map.csv（raw_filename, dog_id, exclude_reason, note）。
片段文件实际时长与文件名 start/end 相差超过 1 秒时，改从 raw data without cut/ 的原片按 start/end 截取。
"""
import argparse
import csv
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

DATA_ROOT = Path("D:/model one traning data")
LABELS = {"walking_run", "standing", "sitting", "lying_sleeping", "eating_drinking", "playing_alone"}
FIELDS = ["clip_id", "source_video_id", "dog_id", "start_sec", "end_sec", "duration_sec",
          "behavior_label", "split", "valid", "notes"]
NAME_RE = re.compile(
    r"^(?P<source>(?P<nn>\d{2})_bili_.+?)"
    r"-(?P<start>\d{2}\.\d{2}\.\d{2}\.\d{3})-(?P<end>\d{2}\.\d{2}\.\d{2}\.\d{3})"
    r"-seg(?P<seg>\d+)_(?P<label>[a-z_]+)\.mp4$")


def to_sec(t):
    h, m, s, ms = (int(x) for x in t.split("."))
    return h * 3600 + m * 60 + s + ms / 1000


def parse_name(name):
    m = NAME_RE.match(name)
    if not m:
        return None
    return {"source": m["source"], "nn": m["nn"], "seg": int(m["seg"]), "label": m["label"],
            "start": to_sec(m["start"]), "end": to_sec(m["end"])}


def probe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        sys.exit(f"ffprobe 失败: {path}\n{r.stderr}")
    return float(r.stdout.strip())


def self_check():
    p = parse_name("23_bili_BV14T411N7J8_围栏_夜间-00.00.07.969-00.00.13.636-seg2_lying_sleeping.mp4")
    assert p["source"] == "23_bili_BV14T411N7J8_围栏_夜间" and p["nn"] == "23" and p["seg"] == 2
    assert p["label"] == "lying_sleeping"
    assert abs(p["start"] - 7.969) < 1e-9 and abs(p["end"] - 13.636) < 1e-9
    assert abs(to_sec("01.02.03.456") - 3723.456) < 1e-9
    assert parse_name("not_a_clip.mp4") is None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=DATA_ROOT / "raw indoor dog only data")
    ap.add_argument("--output", type=Path, default=DATA_ROOT / "indoor dog clip")
    ap.add_argument("--source-dir", type=Path, default=None, help="原片目录，默认 {input}/raw data without cut")
    ap.add_argument("--dog-map", type=Path, default=Path(__file__).with_name("dog_map.csv"))
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--overwrite", action="store_true", help="重新抽已有帧的文件夹")
    args = ap.parse_args()
    source_dir = args.source_dir or args.input / "raw data without cut"
    self_check()

    # 只读输入目录顶层的 mp4，不进入 raw data without cut/ 等子目录
    files = sorted(f.name for f in args.input.glob("*.mp4") if f.is_file())
    clips, bad_name, bad_label = [], [], []
    for name in files:
        p = parse_name(name)
        if p is None:
            bad_name.append(name)
        elif p["label"] not in LABELS:
            bad_label.append(name)
        else:
            clips.append({**p, "file": name})
    if bad_name or bad_label:
        for n in bad_name:
            print(f"文件名无法解析: {n}", file=sys.stderr)
        for n in bad_label:
            print(f"标签不在 6 类中: {n}", file=sys.stderr)
        sys.exit(1)

    with open(args.dog_map, encoding="utf-8-sig", newline="") as f:
        dog_map = {r["raw_filename"]: r for r in csv.DictReader(f)}
    missing = sorted(set(files) - dog_map.keys())
    extra = sorted(dog_map.keys() - set(files))
    bad_id = sorted(k for k, r in dog_map.items() if not re.fullmatch(r"d\d{3}", r["dog_id"]))
    if missing or extra or bad_id:
        for n in missing:
            print(f"dog_map 缺少: {n}", file=sys.stderr)
        for n in extra:
            print(f"dog_map 中的文件不存在: {n}", file=sys.stderr)
        for n in bad_id:
            print(f"dog_id 格式错误: {n} -> {dog_map[n]['dog_id']}", file=sys.stderr)
        sys.exit(1)

    # source_video_id：v 取文件名开头的原始视频编号 NN，开头编号相同即同一原始视频
    for c in clips:
        c["dog_id"] = dog_map[c["file"]]["dog_id"]
        c["exclude"] = dog_map[c["file"]]["exclude_reason"].strip()
        c["source_video_id"] = f"{c['dog_id']}_v{c['nn']}"

    # clip_id：同一 source_video_id 内按 start_sec 从早到晚编号
    clips.sort(key=lambda c: (c["source_video_id"], c["start"], c["end"], c["seg"]))
    counter = Counter()
    for c in clips:
        counter[c["source_video_id"]] += 1
        c["clip_id"] = f"{c['source_video_id']}_c{counter[c['source_video_id']]:03d}"
    assert len({c["clip_id"] for c in clips}) == len(clips), "clip_id 重复"

    # 片段文件比文件名时间范围长/短超过 1 秒的，改从原片截取，保证帧与 CSV 的 start/end 一致
    from_source = []
    for c in clips:
        c["use_source"] = abs(probe_duration(args.input / c["file"]) - (c["end"] - c["start"])) > 1.0
        if c["use_source"]:
            from_source.append(c)
    missing_src = sorted({c["source"] for c in from_source if not (source_dir / f"{c['source']}.mp4").is_file()})
    if missing_src:
        for s in missing_src:
            print(f"原片不存在: {source_dir / (s + '.mp4')}", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(clips, 1):
        out_dir = args.output / c["clip_id"]
        if out_dir.is_dir() and any(out_dir.glob("frame*.jpg")) and not args.overwrite:
            print(f"[{i}/{len(clips)}] 跳过（已有帧）{c['clip_id']}")
            continue
        out_dir.mkdir(exist_ok=True)
        for old in out_dir.glob("frame*.jpg"):
            old.unlink()
        if c["use_source"]:
            src = source_dir / f"{c['source']}.mp4"
            inp = ["-ss", f"{c['start']:.3f}", "-i", str(src), "-t", f"{c['end'] - c['start']:.3f}"]
        else:
            src = args.input / c["file"]
            inp = ["-i", str(src)]
        cmd = ["ffmpeg", "-v", "error", "-y", *inp,
               "-vf", f"fps={args.fps:g}", "-q:v", "2", "-an", "-start_number", "1",
               str(out_dir / "frame%d.jpg")]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            sys.exit(f"ffmpeg 失败: {src}\n{r.stderr}")
        print(f"[{i}/{len(clips)}] {c['clip_id']}: {len(list(out_dir.glob('frame*.jpg')))} 帧  <- {src.name}")

    csv_path = args.output / "dog_behavior_clips.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for c in clips:
            w.writerow({
                "clip_id": c["clip_id"], "source_video_id": c["source_video_id"], "dog_id": c["dog_id"],
                "start_sec": f"{c['start']:.3f}", "end_sec": f"{c['end']:.3f}",
                "duration_sec": f"{c['end'] - c['start']:.3f}", "behavior_label": c["label"],
                "split": "", "valid": 0 if c["exclude"] else 1, "notes": c["exclude"]})

    print(f"\n共 {len(clips)} 个片段，CSV 已写入 {csv_path}")
    print("各标签:", dict(sorted(Counter(c["label"] for c in clips).items())))
    print("各 dog_id:", dict(sorted(Counter(c["dog_id"] for c in clips).items())))
    print(f"从原片截取（{len(from_source)} 段）:")
    for c in from_source:
        print(f"  {c['clip_id']}  {c['start']:.3f}-{c['end']:.3f}  <- {c['source']}.mp4")
    print("valid=0:")
    for c in clips:
        if c["exclude"]:
            print(f"  {c['clip_id']}  {c['exclude']}  <- {c['file']}")


if __name__ == "__main__":
    main()
