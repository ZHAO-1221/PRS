"""Frames -> person and dog skeletons: RTMDet finds both, RTMPose estimates 17 keypoints for each."""
import numpy as np
from PIL import Image

STREAMS = ("person", "dog")  # stream order in the output arrays
MIN_BOX_SCORE = 0.3  # a new target needs this score; a followed one is checked by RTMDet's choose() instead
REACQUIRE_AFTER = 3  # missed frames after which the track is dropped and the best-scoring box is taken again


def extract_skeletons(frames, detector, pose_models):
    """Run the detector and the pose models on BGR frames of one clip.

    pose_models: {"person": RTMPoseOnnx (COCO 17 points), "dog": RTMPoseOnnx (AP-10K 17 points)}.
    Returns skeletons (2, T, 17, 3) float32 of x, y in pixels and a confidence in (0, 1], and boxes (2, T, 5)
    of x1, y1, x2, y2, score; a stream whose target was not detected in a frame stays all zeros there.
    """
    skeletons, boxes, previous, missed = [], [], {}, {stream: 0 for stream in STREAMS}
    for frame in frames:
        found = detector(frame, previous)
        image = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
        frame_skeletons, frame_boxes = np.zeros((2, 17, 3), np.float32), np.zeros((2, 5), np.float32)
        for i, stream in enumerate(STREAMS):
            box = found[stream]
            if box and (stream in previous or box[4] >= MIN_BOX_SCORE):
                previous[stream], missed[stream] = box, 0
                frame_skeletons[i] = pose_models[stream](image, (box[:2], box[2:4]))
                # RTMPose's SimCC peak is not a probability and goes above 1 for clear points; cap it
                frame_skeletons[i, :, 2] = np.minimum(frame_skeletons[i, :, 2], 1)
                frame_boxes[i] = box[:5]
            else:
                missed[stream] += 1
                if missed[stream] >= REACQUIRE_AFTER:  # lost (e.g. the box jumped when the dog lay down)
                    previous.pop(stream, None)
        skeletons.append(frame_skeletons)
        boxes.append(frame_boxes)
    return np.stack(skeletons, axis=1), np.stack(boxes, axis=1)
