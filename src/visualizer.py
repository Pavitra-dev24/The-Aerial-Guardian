import cv2
import numpy as np

_PALETTE = [
    (255,  56,  56), (255, 157, 151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), ( 72, 249,  10), (146, 204,  23), ( 61, 219, 134),
    ( 26, 147,  52), (  0, 212, 187), ( 44, 153, 168), (  0, 194, 255),
    ( 52,  69, 147), (100, 115, 255), (  0,  24, 236), (132,  56, 255),
    ( 82,   0, 133), (203,  56, 255), (255, 149, 200), (255,  55, 199),
]


def _color(track_id):
    return _PALETTE[track_id % len(_PALETTE)]


def draw_tracks(frame, tracks, tail_len=40):
    H, W = frame.shape[:2]
    result = frame.copy()

    for t in tracks:
        color = _color(t.track_id)
        tail = t.tail[-tail_len:]
        n = len(tail)
        for i in range(1, n):
            frac = i / n
            thickness = max(1, int(3 * frac))
            c = tuple(int(v * (0.25 + 0.75 * frac)) for v in color)
            p1 = (
                int(np.clip(tail[i - 1][0], 0, W - 1)),
                int(np.clip(tail[i - 1][1], 0, H - 1)),
            )
            p2 = (
                int(np.clip(tail[i][0], 0, W - 1)),
                int(np.clip(tail[i][1], 0, H - 1)),
            )
            cv2.line(result, p1, p2, c, thickness, cv2.LINE_AA)

    for t in tracks:
        color = _color(t.track_id)
        box = t.xyxy
        x1 = int(np.clip(box[0], 0, W - 1))
        y1 = int(np.clip(box[1], 0, H - 1))
        x2 = int(np.clip(box[2], 0, W - 1))
        y2 = int(np.clip(box[3], 0, H - 1))
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"P{t.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx1, ly1 = x1, max(y1 - th - 6, 0)
        cv2.rectangle(result, (lx1, ly1), (lx1 + tw + 4, ly1 + th + 6), color, -1)
        cv2.putText(
            result, label, (lx1 + 2, ly1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return result


def draw_hud(frame, fps, num_tracks):
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA,
    )
    H, W = frame.shape[:2]
    label = f"Persons: {num_tracks}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cx = W - tw - 14
    cv2.rectangle(frame, (cx - 2, 8), (W - 4, 8 + th + 10), (0, 0, 0), -1)
    cv2.putText(
        frame, label, (cx, 8 + th + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return frame
