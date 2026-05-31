import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(boxes_a, boxes_b):
    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def linear_assignment(cost, threshold):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    rows, cols = linear_sum_assignment(cost)

    matched, u_rows, u_cols = [], [], []
    matched_rows, matched_cols = set(), set()

    for r, c in zip(rows, cols):
        if cost[r, c] <= threshold:
            matched.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)
        else:
            u_rows.append(r)
            u_cols.append(c)

    for i in range(cost.shape[0]):
        if i not in matched_rows and i not in set(u_rows):
            u_rows.append(i)
    for j in range(cost.shape[1]):
        if j not in matched_cols and j not in set(u_cols):
            u_cols.append(j)

    return matched, u_rows, u_cols


def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_threshold]
    return np.array(keep, dtype=int)
