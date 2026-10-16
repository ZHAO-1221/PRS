"""Run directly: python tests/test_extract_skeletons.py"""
import numpy as np

from pet_recognition.inference.skeleton import REACQUIRE_AFTER, extract_skeletons


def run(detections):
    """Feed one detector result per frame; returns the outputs and the `previous` the detector saw each frame."""
    seen_previous = []

    def detector(frame, previous):
        seen_previous.append(dict(previous))
        return detections[len(seen_previous) - 1]

    pose = lambda image, box: np.tile([[5.0, 6.0, 1.4]], (17, 1))  # raw SimCC peaks can exceed 1
    frames = [np.zeros((60, 80, 3), np.uint8)] * len(detections)
    skeletons, boxes = extract_skeletons(frames, detector, {"person": pose, "dog": pose})
    return skeletons, boxes, seen_previous


def test_output_layout_and_zero_filling():
    dog = (10, 10, 50, 40, 0.8, "dog")
    weak_person = (0, 0, 20, 20, 0.1, "person")  # a new target below the box threshold: not detected
    skeletons, boxes, seen = run([{"person": None, "dog": dog}, {"person": weak_person, "dog": None}])
    assert skeletons.shape == (2, 2, 17, 3) and skeletons.dtype == np.float32 and boxes.shape == (2, 2, 5)
    assert (skeletons[1, 0, :, :2] == [5, 6]).all() and (skeletons[1, 0, :, 2] == 1).all()  # dog, frame 0, capped
    assert (skeletons[0] == 0).all() and (boxes[0] == 0).all()  # no confident person in either frame
    assert (skeletons[1, 1] == 0).all()  # dog missed in frame 1
    assert seen[1] == {"dog": dog}  # the last dog box is handed to the next frame


def test_followed_target_accepts_what_choose_returns_and_is_dropped_after_misses():
    dog, lying = (10, 10, 50, 40, 0.8, "dog"), (10, 12, 50, 40, 0.15, "dog")
    detections = [{"person": None, "dog": dog}, {"person": None, "dog": lying}]
    detections += [{"person": None, "dog": None}] * REACQUIRE_AFTER + [{"person": None, "dog": dog}]
    skeletons, boxes, seen = run(detections)
    assert boxes[1, 1, 4] == np.float32(0.15)  # a weak box returned for a followed dog is kept
    assert seen[1 + REACQUIRE_AFTER] == {"dog": lying}  # still followed after fewer misses than the limit
    assert seen[2 + REACQUIRE_AFTER] == {}  # dropped, so the next frame picks the best-scoring box again


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
