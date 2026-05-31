import os
import glob
import time
import cv2
import numpy as np
import yaml

from .detector import SAHIDetector
from .tracker import ByteTracker
from .cmc import CMC
from .visualizer import draw_tracks, draw_hud


def _load_cfg(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


class Pipeline:
    def __init__(self, config_path=None):
        cfg = _load_cfg(config_path)

        det = cfg.get("detection", {})
        sahi = cfg.get("sahi", {})
        trk = cfg.get("tracking", {})
        cmc = cfg.get("cmc", {})
        vis = cfg.get("visualization", {})

        self.detector = SAHIDetector(
            model_path=det.get("model", "yolov8n.pt"),
            conf=det.get("conf_threshold", 0.25),
            nms_iou=det.get("nms_threshold", 0.45),
            slice_size=sahi.get("slice_size", 640),
            overlap=sahi.get("overlap_ratio", 0.25),
            full_image=sahi.get("full_image", True),
            device=det.get("device", "cpu"),
        )
        self.tracker = ByteTracker(
            track_thresh=trk.get("track_thresh", 0.45),
            match_thresh=trk.get("match_thresh", 0.5),
            second_thresh=trk.get("second_match_thresh", 0.5),
            max_age=trk.get("max_age", 30),
            n_init=trk.get("n_init", 3),
        )
        self.cmc = CMC(num_features=cmc.get("num_features", 500)) if cmc.get("enabled", True) else None
        self.tail_len = vis.get("tail_length", 40)

    def set_device(self, device):
        self.detector.model.to(device)

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (W, H))
        avg = self._run_frames(self._video_iter(cap), writer)
        cap.release()
        writer.release()
        return avg

    def process_sequence(self, seq_dir, output_path):
        img_dir = os.path.join(seq_dir, "img1") if os.path.isdir(os.path.join(seq_dir, "img1")) else seq_dir
        paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                       glob.glob(os.path.join(img_dir, "*.png")) +
                       glob.glob(os.path.join(img_dir, "*.jpeg")))
        if not paths:
            raise ValueError(f"No images found in {img_dir}")
        sample = cv2.imread(paths[0])
        H, W = sample.shape[:2]
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (W, H))
        avg = self._run_frames(self._seq_iter(paths), writer, total=len(paths))
        writer.release()
        return avg

    def _video_iter(self, cap):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

    def _seq_iter(self, paths):
        for p in paths:
            yield cv2.imread(p)

    def _run_frames(self, source, writer, total=None):
        self.tracker.reset()
        if self.cmc:
            self.cmc.reset()

        fps_history = []
        count = 0
        for frame in source:
            if frame is None:
                continue
            out, fps = self._process_frame(frame)
            writer.write(out)
            fps_history.append(fps)
            count += 1
            if count % 50 == 0:
                recent = np.mean(fps_history[-50:])
                suffix = f"/{total}" if total else ""
                print(f"  Frame {count}{suffix}  |  FPS (last 50): {recent:.1f}")

        avg = float(np.mean(fps_history)) if fps_history else 0.0
        print(f"\n  Done. {count} frames | Average pipeline FPS: {avg:.2f}")
        return avg

    def _process_frame(self, frame):
        t0 = time.perf_counter()
        warp = self.cmc.apply(frame) if self.cmc else None
        dets, scores = self.detector.detect(frame)
        tracks = self.tracker.update(dets, scores, warp)
        elapsed = time.perf_counter() - t0
        fps = 1.0 / (elapsed + 1e-9)
        result = draw_tracks(frame, tracks, self.tail_len)
        draw_hud(result, fps, len(tracks))
        return result, fps
