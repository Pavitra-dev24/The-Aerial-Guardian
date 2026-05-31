# Aerial Guardian

A lightweight drone-optimized pipeline for person detection and multi-object tracking (MOT), built for the VisDrone2019-MOT dataset. The pipeline combines SAHI-style sliced inference on top of YOLOv8n with a custom ByteTrack implementation augmented by Camera Motion Compensation (CMC) to address the specific challenges of aerial surveillance from a moving platform.

Model size: ~6 MB (YOLOv8n). Well within the 300 MB constraint.

---

## Architecture at a Glance

| Component | Choice |
|-----------|--------|
| Detector | YOLOv8n (COCO-pretrained) |
| Small object strategy | Custom SAHI - sliced inference |
| Tracker | ByteTrack (custom implementation) |
| Ego-motion handling | CMC via sparse optical flow |

---

## Project Structure

```
aerial_guardian/
    run.py                  entry point
    requirements.txt
    configs/
        default.yaml        all tunable parameters
    src/
        kalman.py           constant-velocity Kalman filter
        matching.py         IoU matrix, Hungarian assignment, NMS
        cmc.py              Camera Motion Compensation
        tracker.py          ByteTrack + CMC integration
        detector.py         SAHI detector wrapping YOLOv8
        visualizer.py       bounding boxes, tails, HUD
        pipeline.py         full orchestration
```

---

## Setup

Python 3.8 or higher is required.

```bash
git clone <your-repo-url>
cd aerial_guardian
pip install -r requirements.txt
```

YOLOv8n weights (~6 MB) are downloaded automatically on first run from the Ultralytics servers. No manual download needed.

### Dataset

Download the VisDrone2019-MOT validation set from the official repository:

```
https://github.com/VisDrone/VisDrone-Dataset
```

Extract so sequences appear at this path:

```
VisDrone2019-MOT-val/
    sequences/
        uav0000009_03358_v/
            img1/
                000001.jpg
                000002.jpg
                ...
```

---

## Usage

**Process a single image sequence (VisDrone format):**

```bash
python run.py VisDrone2019-MOT-val/sequences/uav0000009_03358_v output.mp4
```

The pipeline auto-detects the `img1/` subdirectory if present.

**Process a flat folder of images:**

```bash
python run.py path/to/frames/ output.mp4
```

**Process a video file:**

```bash
python run.py path/to/video.mp4 output.mp4
```

**Use GPU (recommended for speed):**

```bash
python run.py input/ output.mp4 --device cuda
```

**Custom config:**

```bash
python run.py input/ output.mp4 --config configs/default.yaml
```

---

## Configuration

All parameters live in `configs/default.yaml`. Key knobs:

```yaml
detection:
  conf_threshold: 0.25    # lower for more recall on tiny persons
  device: cpu             # cpu, cuda, or mps

sahi:
  slice_size: 640         # patch size fed to YOLOv8
  overlap_ratio: 0.25     # overlap between patches
  full_image: true        # also run a full-frame pass

tracking:
  track_thresh: 0.45      # separates high-conf from low-conf detections
  max_age: 30             # frames before a lost track is pruned
  n_init: 3               # detections needed to confirm a new track

cmc:
  enabled: true           # toggle Camera Motion Compensation
  num_features: 500       # Shi-Tomasi feature budget per frame
```

---

## Output

Each output video shows:
- Colored bounding boxes per tracked person, labeled `P<ID>`
- A trajectory tail that grows thicker and brighter toward the current position, showing recent movement history
- FPS counter (full pipeline, not just inference)
- Active person count

---

## Results

| Hardware | Device | Avg Pipeline FPS |
|----------|--------|-----------------|
| Fill in after running | | |

FPS covers the complete pipeline: CMC + SAHI detection (full-frame + patches) + tracking + visualization. Detection is the dominant cost. On CUDA the bottleneck shifts to SAHI patch scheduling overhead.

---

## Technical Summary

### Detector Choice and Small Object Detection

YOLOv8n uses a CSPDarknet backbone with C2f modules (cross-stage partial with 2 bottleneck blocks each). The neck is a PAN-FPN that fuses features from three scales. The head is decoupled - separate branches for classification and regression - with Distribution Focal Loss (DFL) for sub-pixel box accuracy. The nano variant preserves the full architecture but with reduced channel counts, resulting in ~6 MB and very fast inference.

The problem with running YOLOv8 directly on a 1920x1080 drone frame is resolution collapse. A person standing 40 m below the drone might occupy 30x70 pixels in the raw frame. After the model's internal resize to 640x640, that same person becomes ~10x23 pixels. At that scale, the receptive field of the detector's smallest anchor region (~8x8 in the output) covers the entire person, and positional precision collapses.

**SAHI (Slicing Aided Hyper Inference)** solves this without retraining. Instead of passing the full frame to the model, the implementation:

1. Divides the frame into overlapping 640x640 patches with 25% overlap (giving ~8 patches on a 1920x1080 frame)
2. Runs YOLOv8n on each patch at native resolution, where persons are at the scale the model was trained on
3. Re-projects each detection back to full-frame pixel coordinates by adding the patch offset
4. Merges all detections (from all patches plus an optional full-frame pass) with NMS

The 25% overlap ensures that an object straddling a patch boundary appears fully within at least one patch. The full-frame pass is retained at reduced effective resolution to catch medium and large persons (close-range targets) that SAHI might fragment across patch seams.

### Addressing ID Switching

ID switches in drone footage have two distinct causes: drone ego-motion and short occlusions.

**Ego-motion** is the bigger problem. The drone translates and rotates between frames, so ALL tracked objects shift globally in image space simultaneously. Without correction, the Kalman filter's constant-velocity prediction diverges from the true position by 10-30 pixels per frame during aggressive maneuvers. For a person bounding box of 30x70 pixels, a 15-pixel global shift drops the predicted-to-detected IoU from ~0.7 to ~0.4 - often below the matching threshold, causing a spurious ID switch.

The **CMC module** fixes this before matching:

1. Detect Shi-Tomasi corner features in the current frame (background features, not moving objects)
2. Track them to the next frame with pyramidal Lucas-Kanade optical flow
3. Estimate a partial affine transform (rotation + uniform scale + translation) from RANSAC-filtered correspondences
4. Apply the estimated warp to each Kalman filter state vector before the IoU matching step

The warp is applied to both the position component and the velocity component of each Kalman state, and the corresponding blocks of the covariance matrix are also rotated. This corrects the prediction so that the IoU between predicted track positions and new detections stays high even during aggressive camera motion.

**Short occlusions** are handled by ByteTrack's two-stage association. Standard SORT-based trackers only try to match high-confidence detections. When a person is briefly occluded (by another person, a vehicle, or a building corner), their detection score drops below the high-confidence threshold or disappears entirely. ByteTrack recovers by:

1. Stage 1: match high-confidence detections (score above track_thresh) against all active tracks using IoU and Hungarian assignment
2. Stage 2: match remaining low-confidence detections against the tracks left unmatched in stage 1

Low-confidence detections, which would normally be discarded, often represent partially-visible or occluded persons. The second stage gives active tracks a chance to survive brief occlusions without requiring a visible high-confidence detection. The `max_age` parameter (default 30 frames) provides a 1-second survival window at 30 FPS for tracks that receive no detections at all.

New tracks require `n_init=3` consecutive detections before being displayed, filtering one-frame false positives from SAHI patch boundary artifacts.

### Edge Hardware Adaptation (NVIDIA Jetson)

The full pipeline is designed with Jetson deployment in mind. ByteTrack, CMC, and the Kalman filter are pure NumPy and run efficiently on the Jetson ARM cores with no GPU memory. Only the YOLOv8n forward pass needs the GPU.

**Step 1: Export to TensorRT**

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="engine", device=0, half=True, imgsz=640)
```

Then in the config, change `model: yolov8n.engine`.

**Step 2: Reduce SAHI patch count**

On a Jetson Orin NX at 1920x1080, 8 patches per frame is too slow for real-time. Increase stride to reduce patch count:

```yaml
sahi:
  slice_size: 640
  overlap_ratio: 0.1    # 4 patches instead of 8 at 1920x1080
```

This trades some detection recall on edge-straddling persons for speed.

**Step 3: Scale down CMC**

Run the optical flow on a half-resolution grayscale copy:

```python
gray_small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))
```

Scale the resulting warp translation by 2 before applying to Kalman states.

**Step 4: INT8 quantization**

For maximum throughput on Jetson:

```python
model.export(format="engine", device=0, int8=True, data="coco.yaml", imgsz=640)
```

INT8 requires a calibration dataset (a few hundred VisDrone frames work well).

**Expected performance on Jetson Orin NX 16 GB with TensorRT FP16 and 4 SAHI patches: approximately 15-25 FPS.**

For a production pipeline, NVIDIA DeepStream handles camera input, preprocessing, and inference in a single GPU pipeline, eliminating the Python overhead entirely.

---

## Key Design Decisions

**Why YOLOv8n and not a larger model?** The 300 MB model size constraint and Jetson deployment target make the nano variant the right choice. SAHI compensates for the reduced capacity by ensuring objects are always at an appropriate scale for the model.

**Why not fine-tune on VisDrone?** The pipeline is intentionally kept as a demonstration of architectural choices rather than a benchmarked system. Fine-tuning YOLOv8n on VisDrone persons (or using a VisDrone-specific checkpoint) would significantly improve detection recall at very small scales and is the natural next step.

**Why partial affine for CMC instead of full homography?** Drone motion at altitude is well approximated by rotation + translation + small scale change. Full homography estimation is less stable with sparse feature sets and more sensitive to dynamic objects (moving persons) contaminating the feature set before RANSAC filtering.
