import numpy as np
from ultralytics import YOLO
from .matching import nms


class SAHIDetector:
    """
    Slicing Aided Hyper Inference (SAHI) wrapper around YOLOv8.

    Drone imagery at altitude compresses persons to 10-40 pixel heights.
    A standard 640x640 forward pass at 1920x1080 reduces them further to
    ~4-17 pixels, well below the model's effective detection range.

    SAHI fixes this by tiling the frame into overlapping patches at native
    resolution, running inference on each tile, re-projecting detections to
    full-frame coordinates, and merging with NMS. A full-frame pass is also
    retained for medium and large persons near the drone.
    """

    PERSON_CLASS = 0

    def __init__(self, model_path="yolov8n.pt", conf=0.25, nms_iou=0.45,
                 slice_size=640, overlap=0.25, full_image=True, device="cpu"):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf = conf
        self.nms_iou = nms_iou
        self.slice_size = slice_size
        self.overlap = overlap
        self.full_image = full_image

    def detect(self, frame):
        h, w = frame.shape[:2]
        all_boxes, all_scores = [], []

        if self.full_image:
            boxes, scores = self._infer(frame)
            if len(boxes):
                all_boxes.append(boxes)
                all_scores.append(scores)

        stride = max(1, int(self.slice_size * (1.0 - self.overlap)))

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x1 = max(0, min(x, max(0, w - self.slice_size)))
                y1 = max(0, min(y, max(0, h - self.slice_size)))
                x2 = min(x1 + self.slice_size, w)
                y2 = min(y1 + self.slice_size, h)

                if (x2 - x1) < 32 or (y2 - y1) < 32:
                    continue

                patch = frame[y1:y2, x1:x2]
                boxes, scores = self._infer(patch)

                if len(boxes):
                    boxes[:, [0, 2]] += x1
                    boxes[:, [1, 3]] += y1
                    all_boxes.append(boxes)
                    all_scores.append(scores)

        if not all_boxes:
            return np.empty((0, 4)), np.empty(0)

        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        keep = nms(boxes, scores, self.nms_iou)
        return boxes[keep], scores[keep]

    def _infer(self, image):
        results = self.model(
            image,
            classes=[self.PERSON_CLASS],
            conf=self.conf,
            verbose=False,
        )
        if len(results[0].boxes) == 0:
            return np.empty((0, 4)), np.empty(0)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        return boxes.copy(), scores.copy()
