"""Draw person and dog boxes and skeletons on a BGR frame."""
import cv2

# 0-based keypoint pairs: COCO human order, then AP-10K animal order
EDGES = {
    "person": [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
               (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
    "dog": [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (6, 7), (3, 8), (8, 9), (9, 10), (4, 11),
            (11, 12), (12, 13), (4, 14), (14, 15), (15, 16)],
}
COLORS = {"person": (214, 120, 42), "dog": (52, 104, 235)}  # BGR: blue person, orange dog
LOW = (160, 160, 160)  # gray for points below the confidence threshold


def draw_skeletons(frame, skeletons, boxes, streams=("person", "dog"), min_score=0.3):
    """skeletons (2, 17, 3) and boxes (2, 5) for one frame, in the order of streams; draws in place."""
    thickness = max(2, round(frame.shape[0] / 360))
    for stream, points, box in zip(streams, skeletons, boxes):
        if box[4] == 0:
            continue
        color = COLORS[stream]
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)
        for a, b in EDGES[stream]:
            if min(points[a, 2], points[b, 2]) >= min_score:
                cv2.line(frame, (int(points[a, 0]), int(points[a, 1])), (int(points[b, 0]), int(points[b, 1])),
                         color, thickness)
        for x, y, score in points:
            cv2.circle(frame, (int(x), int(y)), thickness + 1, color if score >= min_score else LOW, -1)
    return frame
