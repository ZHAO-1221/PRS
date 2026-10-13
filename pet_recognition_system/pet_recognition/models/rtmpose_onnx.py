"""RTMPose top-down keypoint inference from an ONNX file (the official end2end.onnx or our own export)."""
import numpy as np
import onnxruntime as ort
from PIL import Image

# shortcut: square 256x256 animal models only; the 256x192 human model needs aspect-aware cropping
SIZE, PADDING, SIMCC_SPLIT = 256, 1.25, 2.0  # from the official model's pipeline.json
MEAN, STD = np.float32([123.675, 116.28, 103.53]), np.float32([58.395, 57.12, 57.375])


class RTMPoseOnnx:
    def __init__(self, path):
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def __call__(self, image, rect_points):
        """(K, 3) array of x, y, score in image pixels for the animal inside the box (two corner points)."""
        (xa, ya), (xb, yb) = rect_points
        cx, cy = (xa + xb) / 2, (ya + yb) / 2
        side = max(abs(xb - xa), abs(yb - ya)) * PADDING  # square crop, as mmpose TopdownAffine does for 256x256
        k, x0, y0 = side / SIZE, cx - side / 2, cy - side / 2
        crop = image.convert("RGB").transform((SIZE, SIZE), Image.AFFINE, (k, 0, x0, 0, k, y0), Image.BILINEAR)
        inp = ((np.asarray(crop, np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        simcc_x, simcc_y = self.session.run(None, {"input": inp})
        return np.stack([simcc_x[0].argmax(1) / SIMCC_SPLIT * k + x0,
                         simcc_y[0].argmax(1) / SIMCC_SPLIT * k + y0,
                         np.minimum(simcc_x[0].max(1), simcc_y[0].max(1))], axis=1)
