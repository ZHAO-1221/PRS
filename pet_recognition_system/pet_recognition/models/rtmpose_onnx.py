"""RTMPose top-down keypoint inference from an ONNX file (the official end2end.onnx or our own export).

Works for any input size the file declares: 256x256 for the AP-10K dog model, 256x192 for the COCO human model.
"""
import numpy as np
import onnxruntime as ort
from PIL import Image

PADDING, SIMCC_SPLIT = 1.25, 2.0  # from the official models' pipeline.json
MEAN, STD = np.float32([123.675, 116.28, 103.53]), np.float32([58.395, 57.12, 57.375])


class RTMPoseOnnx:
    def __init__(self, path):
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _, _, self.height, self.width = self.session.get_inputs()[0].shape

    def __call__(self, image, rect_points):
        """(K, 3) array of x, y, score in image pixels for the body inside the box (two corner points)."""
        (xa, ya), (xb, yb) = rect_points
        cx, cy = (xa + xb) / 2, (ya + yb) / 2
        w, h = abs(xb - xa) * PADDING, abs(yb - ya) * PADDING
        # grow the short side to the model's aspect ratio, as mmpose TopdownAffine does
        if w * self.height > h * self.width:
            h = w * self.height / self.width
        else:
            w = h * self.width / self.height
        k, x0, y0 = w / self.width, cx - w / 2, cy - h / 2
        crop = image.convert("RGB").transform((self.width, self.height), Image.AFFINE, (k, 0, x0, 0, k, y0),
                                              Image.BILINEAR)
        inp = ((np.asarray(crop, np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        simcc_x, simcc_y = self.session.run(None, {"input": inp})
        return np.stack([simcc_x[0].argmax(1) / SIMCC_SPLIT * k + x0,
                         simcc_y[0].argmax(1) / SIMCC_SPLIT * k + y0,
                         np.minimum(simcc_x[0].max(1), simcc_y[0].max(1))], axis=1)
