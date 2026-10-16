"""Run directly: python tests/test_rtmdet_choose.py (needs the MMDetection stack installed)."""
from pet_recognition.models.rtmdet import choose, iou


def test_first_frame_takes_the_best_score():
    boxes = [(0, 0, 10, 10, 0.5, "dog"), (50, 50, 60, 60, 0.9, "cat")]
    assert choose(boxes) == boxes[1]
    assert choose([]) is None


def test_later_frames_follow_the_overlap_not_the_score():
    last = (0, 0, 10, 10, 0.8, "dog")
    here, other = (1, 1, 11, 11, 0.6, "dog"), (50, 50, 60, 60, 0.9, "dog")  # a second, more confident dog
    assert choose([here, other], last) == here
    assert iou(here, last) > 0.6
    closer, farther = (0, 0, 10, 10, 0.4, "dog"), (5, 5, 15, 15, 0.9, "dog")
    assert choose([farther, closer], last) == closer  # more overlap wins even with a lower score


def test_no_overlapping_confident_box_means_not_detected():
    last = (0, 0, 10, 10, 0.8, "dog")
    noise, other = (0, 0, 10, 10, 0.05, "dog"), (50, 50, 60, 60, 0.9, "dog")
    assert choose([noise, other], last) is None  # the far dog is not taken, a near-zero box is not a detection


def test_a_weak_box_continues_the_track_only_when_it_overlaps_a_lot():
    last = (0, 0, 10, 10, 0.8, "dog")
    lying = (0, 1, 10, 10, 0.15, "dog")  # same place, low score: a dog that lay down
    assert choose([lying], last) == lying
    shifted = (6, 6, 16, 16, 0.15, "dog")  # weak and only touching the last box
    assert iou(shifted, last) < 0.5 and choose([shifted], last) is None


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
