"""Export an RTMPose checkpoint to ONNX with the same input and outputs as the official end2end.onnx.

    python tools/export_rtmpose_onnx.py CONFIG CHECKPOINT OUT.onnx

The exported model takes a normalized crop and returns the raw SimCC x / y scores, so
pet_recognition.models.rtmpose_onnx.RTMPoseOnnx runs it exactly like the official file.
"""
import argparse
from pathlib import Path

import torch
from mmpose.apis import init_model


class SimCCOutputs(torch.nn.Module):
    """The pose estimator up to its head: raw SimCC scores, no decoding."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs, None, mode="tensor")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device="cpu")
    width, height = model.cfg.codec.input_size
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(SimCCOutputs(model), torch.zeros(1, 3, height, width), args.out, opset_version=11,
                      input_names=["input"], output_names=["simcc_x", "simcc_y"],
                      dynamic_axes={name: {0: "batch"} for name in ("input", "simcc_x", "simcc_y")})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
