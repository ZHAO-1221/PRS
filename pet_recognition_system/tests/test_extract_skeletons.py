"""Run directly: python tests/test_extract_skeletons.py (needs the MMDetection stack installed)."""
import numpy as np

from pet_recognition.inference.skeleton import REACQUIRE_AFTER, extract_skeletons, follow, track

DOG = (10, 10, 50, 40, 0.8, "dog")
LYING = (10, 12, 50, 40, 0.15, "dog")  # same place, low score: a dog that lay down
FAR = (100, 100, 140, 130, 0.9, "dog")


class FakeDetector:
    def __init__(self, per_frame):
        self.per_frame, self.calls = per_frame, 0

    def candidates(self, frame):
        self.calls += 1
        return self.per_frame[self.calls - 1]


def run(per_frame, streams=("person", "dog"), confidence=1.4):
    pose = lambda image, box: np.tile([[5.0, 6.0, confidence]], (17, 1))  # raw SimCC peaks can leave 0..1
    frames = [np.zeros((60, 80, 3), np.uint8)] * len(per_frame)
    return extract_skeletons(frames, FakeDetector(per_frame), {"person": pose, "dog": pose}, streams)


def test_output_layout_and_zero_filling():
    weak_person = (0, 0, 20, 20, 0.05, "person")  # never confident in the clip, and below the follow floor
    skeletons, boxes = run([{"person": [], "dog": [DOG]}, {"person": [weak_person], "dog": []}])
    assert skeletons.shape == (2, 2, 17, 3) and skeletons.dtype == np.float32 and boxes.shape == (2, 2, 5)
    assert (skeletons[1, 0, :, :2] == [5, 6]).all() and (skeletons[1, 0, :, 2] == 1).all()  # dog, frame 0, capped
    assert (skeletons[0] == 0).all() and (boxes[0] == 0).all()
    assert (skeletons[1, 1] == 0).all()  # dog missed in frame 1


def test_track_follows_weak_boxes_and_drops_the_track_after_misses():
    chosen = track([[DOG], [LYING]] + [[]] * REACQUIRE_AFTER + [[FAR]])
    assert chosen[:2] == [DOG, LYING]
    assert chosen[-1] == FAR  # dropped after the misses, so a confident box elsewhere starts a new track
    assert track([[DOG], [FAR]]) == [DOG, None]  # while following, a far box is not taken


def test_a_weak_box_where_the_target_was_last_seen_picks_the_track_up_after_misses():
    weak_far = (100, 100, 140, 130, 0.2, "dog")
    chosen = track([[DOG]] + [[]] * (REACQUIRE_AFTER + 2) + [[weak_far, LYING]])
    assert chosen[-1] == LYING  # the dog came back at 0.15 where it was; the weak box elsewhere is ignored
    assert track([[DOG]] + [[]] * REACQUIRE_AFTER + [[weak_far]])[-1] is None


def test_a_never_confident_target_starts_from_the_best_box_of_the_clip():
    a, b, c = ((10, 10, 50, 40, s, "cat") for s in (0.12, 0.2, 0.15))
    assert follow([[a], [b], [c]]) == [a, b, c]  # seeded at 0.2 in the middle, followed both ways
    assert follow([[(0, 0, 5, 5, 0.05, "dog")]]) == [None]  # nothing worth following


def test_frames_before_the_first_confident_box_are_followed_backwards():
    early = (10, 11, 50, 40, 0.2, "cat")  # the same dog before it scores 0.3
    assert follow([[early], [early], [DOG], [DOG]]) == [early, early, DOG, DOG]
    weak_elsewhere = (100, 100, 140, 130, 0.2, "dog")
    assert follow([[weak_elsewhere], [DOG]]) == [None, DOG]  # a weak box elsewhere before the start is not taken


def test_a_person_needs_a_strong_box_to_start_and_weak_boxes_never_carry_it():
    coat = (0, 0, 20, 40, 0.43, "person")  # clutter: confident for a dog, never 0.5
    _, boxes = run([{"person": [coat], "dog": [DOG]}] * 3)
    assert (boxes[0] == 0).all() and (boxes[1, :, 4] > 0).all()
    person, half_hidden, weak = ((0, 0, 20, 40, s, "person") for s in (0.9, 0.35, 0.2))
    _, boxes = run([{"person": [p], "dog": [DOG]} for p in (half_hidden, person, half_hidden, weak)])
    assert np.allclose(boxes[0, :, 4], [0.35, 0.9, 0.35, 0])  # 0.3 continues both ways, 0.2 does not


def test_dog_only_clips_leave_the_person_stream_empty_and_confidence_stays_above_zero():
    person = (0, 0, 20, 40, 0.9, "person")
    skeletons, boxes = run([{"person": [person], "dog": [DOG]}] * 2, streams=("dog",), confidence=-0.01)
    assert (skeletons[0] == 0).all() and (boxes[0] == 0).all()  # a detected person is not exported
    assert (skeletons[1, :, :, 2] > 0).all()  # a slightly negative peak still reads as "detected"


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
