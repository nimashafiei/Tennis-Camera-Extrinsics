# Tennis Court Camera Extrinsics (4 Cameras) + Court Projection Overlay

This repository contains a **two-step pipeline** to:

1. **Estimate camera extrinsics** (3D position + orientation) for **four fixed cameras** using a standard tennis court model.
2. **Validate** the estimated poses by projecting a full **3D tennis court line model** back onto the video frames.

---

## 1) Project Goal

### Inputs
- `cam1.mp4 ... cam4.mp4` (one video per camera)
- `cam1.calib ... cam4.calib` (intrinsics + distortion per camera)

### Outputs (per camera)
- Rotation matrix `R` and translation vector `t` (**world → camera**)
- Camera center in world coordinates: `C = -Rᵀ t`
- Visual verification overlays:
  - clicked points vs reprojection
  - full court line projection on the video

---

## 2) World Coordinate System (Court Model)

- **World origin (0,0,0)**: at the **center of the tennis court** on the court plane  
- Court plane: **Z = 0**
- Court geometry uses **standard ITF dimensions**:
  - Court length: 23.77 m
  - Doubles width: 10.97 m
  - Singles width: 8.23 m
  - Service line distance from the net: 6.40 m

---

## 3) Repository Contents

### A) `estimate_extrinsics_pnp_ippe.py`
Estimates extrinsics from **4 manually clicked** court intersections:
- Loads intrinsics `K` and distortion `dist` from `.calib`
- User clicks 4 points in a fixed order (2D image points)
- Uses known 3D reference points on the court plane (Z=0)
- Solves pose using **IPPE** (planar PnP) with a fallback to iterative PnP
- Saves results:
  - `results/extrinsics_all.json`
  - `results/camX_overlay.png`

### B) `overlay_court_lines_undistort.py`
Projects a full tennis court model onto each video frame using the estimated pose:
- Loads `R, t, K, dist` from `results/extrinsics_all.json`
- Optionally **undistorts** frames for cleaner alignment
- Projects and draws all major court lines safely (depth checks, clipping, anti-jump)
- Saves overlay videos:
  - `results/camX_court_overlay_UNDIST.mp4`

---

## 4) How To Run (Step-by-Step)

### Step 1 — Estimate extrinsics (manual point selection)
```bash
python estimate_extrinsics_pnp_ippe.py
```

Outputs:
- `results/extrinsics_all.json`
- `results/cam1_overlay.png ... results/cam4_overlay.png`

### Step 2 — Project full court lines on the video
```bash
python overlay_court_lines_undistort.py
```

Outputs:
- `results/cam1_court_overlay_UNDIST.mp4 ... results/cam4_court_overlay_UNDIST.mp4`

---

## 7) Notes

- The court is modeled as a **planar surface (Z=0)**. Planar pose estimation can have mirror ambiguity, so IPPE may return multiple candidates.
- Final pose selection is based on:
  - **positive depths** (points in front of the camera)
  - **minimum reprojection RMSE**
- Accuracy depends on click precision. Using more reference points (8–12) improves robustness.

---

## 8) Requirements

- Python 3.x
- OpenCV
- NumPy

Install:
```bash
pip install opencv-python numpy
```
