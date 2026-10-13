"""Train or test an RTMPose config with MMEngine.

    python tools/train_pose.py configs/rtmpose/rtmpose-m_apt36k-dog_finetune.py
    python tools/train_pose.py CONFIG --test [--checkpoint work_dirs/<config name>/best_....pth]

--test scores a checkpoint on the config's validation set; without --checkpoint that is the config's
load_from, i.e. the weights training starts from. Logs and checkpoints go to work_dirs/<config name>/.
"""
import argparse
from pathlib import Path

import mmengine.runner.runner as mmengine_runner
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.utils.dl_utils import collect_env


def collect_env_or_nothing():
    # mmengine decodes cl.exe's banner with Python's default encoding, which fails on non-English
    # Windows when Python runs in UTF-8 mode; the environment report is only written to the log
    try:
        return collect_env()
    except UnicodeDecodeError:
        return {}


mmengine_runner.collect_env = collect_env_or_nothing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = str(Path(__file__).resolve().parents[1] / "work_dirs" / Path(args.config).stem)
    if args.checkpoint:
        cfg.load_from = args.checkpoint
    runner = Runner.from_cfg(cfg)
    runner.test() if args.test else runner.train()


if __name__ == "__main__":
    main()
