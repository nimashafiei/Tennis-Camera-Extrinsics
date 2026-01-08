import os
import json
import cv2
import numpy as np

# ========= Project Settings =========
BASE_DIR = "/Users/nima/Documents/00Criont/Final"
OUT_DIR  = os.path.join(BASE_DIR, "results")
EXTR_JSON = os.path.join(OUT_DIR, "extrinsics_all.json")
CAMS = ["cam1", "cam2", "cam3", "cam4"]

# When projecting 3D points, we only draw points that are in front of the camera.
# A small epsilon avoids numerical issues around Z ≈ 0.
Z_EPS = 1e-3

# Number of samples along each court line segment.
# More samples => smoother projected curves (especially with distortion / strong perspective).
SAMPLES = 120

# Safety threshold: if two consecutive projected samples jump too far in pixel space,
# we assume projection became unstable (e.g., near camera plane or extreme distortion)
# and we avoid drawing that connection.
MAX_JUMP_PX = 80

# Alpha controls how much of the original image is preserved during undistortion:
# 0.0 -> minimal black borders, more cropping
# 1.0 -> keep more FOV, but may introduce black borders
ALPHA = 0.0
# ===========================


# ---------- Tennis Court Geometry (World Frame) ----------
# World origin is assumed at the court center, with the court lying on Z=0.
# X axis: along court length, Y axis: along court width, Z axis: upwards.
L  = 23.77 / 2.0   # Half court length (baseline to baseline / 2)
Wd = 10.97 / 2.0   # Half doubles width (outer sidelines)
Ws =  8.23 / 2.0   # Half singles width (inner sidelines)
S  =  6.40         # Service line distance from net (net at X=0)


def build_court_segments():
    """
    Build a list of 3D line segments representing the key tennis court lines.
    Each segment is defined by two 3D endpoints on the plane Z=0 in meters.

    Why segments (not infinite lines)?
    - We want to render only the visible finite court markings.
    - This is also easier to clip safely to image boundaries after projection.
    """
    segs = []

    # Doubles outer rectangle (full playable area for doubles)
    segs += [
        ([-L, -Wd, 0.0], [-L, +Wd, 0.0]),  # Near baseline
        ([+L, -Wd, 0.0], [+L, +Wd, 0.0]),  # Far baseline
        ([-L, -Wd, 0.0], [+L, -Wd, 0.0]),  # Left doubles sideline
        ([-L, +Wd, 0.0], [+L, +Wd, 0.0]),  # Right doubles sideline
    ]

    # Singles rectangle (inner sidelines define singles court)
    segs += [
        ([-L, -Ws, 0.0], [-L, +Ws, 0.0]),  # Near baseline (singles width)
        ([+L, -Ws, 0.0], [+L, +Ws, 0.0]),  # Far baseline (singles width)
        ([-L, -Ws, 0.0], [+L, -Ws, 0.0]),  # Left singles sideline
        ([-L, +Ws, 0.0], [+L, +Ws, 0.0]),  # Right singles sideline
    ]

    # Service lines (within singles width)
    segs += [
        ([-S, -Ws, 0.0], [-S, +Ws, 0.0]),  # Near service line
        ([+S, -Ws, 0.0], [+S, +Ws, 0.0]),  # Far service line
    ]

    # Center service line (splits service boxes)
    segs += [
        ([-S, 0.0, 0.0], [0.0, 0.0, 0.0]),  # From near service line to net
        ([0.0, 0.0, 0.0], [+S, 0.0, 0.0]),  # From net to far service line
    ]

    # Net line (at X = 0) across doubles width
    segs += [
        ([0.0, -Wd, 0.0], [0.0, +Wd, 0.0]),
    ]

    return [(np.array(a, np.float64), np.array(b, np.float64)) for a, b in segs]


def world_to_cam(Xw, R, t):
    """
    Transform 3D points from world coordinates to camera coordinates:
        X_cam = R * X_world + t

    - R: 3x3 rotation matrix
    - t: 3x1 translation vector
    - Xw: Nx3 array of world points
    """
    return (R @ Xw.T + t).T


def project_points(Xw, R, t, K, dist):
    """
    Project 3D world points onto the image plane using camera extrinsics and intrinsics.

    OpenCV handles:
    - Perspective projection with (R, t)
    - Pinhole intrinsics K
    - Lens distortion dist (k1,k2,p1,p2,k3)
    """
    rvec, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(Xw, rvec, t, K, dist)
    return uv.reshape(-1, 2)


def draw_segment_safe(frame, p0, p1, R, t, K, dist):
    """
    Render one 3D line segment onto the 2D image safely.

    Why "safe" drawing?
    - Some 3D samples may land behind the camera (Z<=0) -> invalid projection.
    - Distortion + perspective can create extreme jumps near the camera plane.
    - We clip lines to the image and skip unstable connections to avoid artifacts.
    """
    h, w = frame.shape[:2]
    rect = (0, 0, w, h)

    # Sample along the 3D segment to approximate it in image space.
    # This avoids missing curvature caused by distortion / perspective.
    a = np.linspace(0.0, 1.0, SAMPLES).reshape(-1, 1)
    Xw = (1.0 - a) * p0.reshape(1, 3) + a * p1.reshape(1, 3)

    # Check depth in camera coordinates to ensure points are in front of the camera.
    Xc = world_to_cam(Xw, R, t)
    z = Xc[:, 2]

    # Project sampled 3D points to image pixels.
    uv = project_points(Xw, R, t, K, dist)

    # Validity masks:
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(z)
    front  = z > Z_EPS  # keep only points with positive depth

    # Additional sanity bounds: allow some margin outside the image, but reject extreme outliers.
    # This prevents drawing long lines across the entire frame due to numerical blow-ups.
    sane = (uv[:, 0] > -2*w) & (uv[:, 0] < 3*w) & (uv[:, 1] > -2*h) & (uv[:, 1] < 3*h)

    ok = finite & front & sane

    # Draw short line pieces between consecutive valid samples.
    for i in range(SAMPLES - 1):
        if not (ok[i] and ok[i+1]):
            continue

        u0, v0 = uv[i]
        u1, v1 = uv[i+1]

        # Skip sudden discontinuities (typical when crossing behind the camera plane or near singularities).
        if np.hypot(u1 - u0, v1 - v0) > MAX_JUMP_PX:
            continue

        pt0 = (int(round(u0)), int(round(v0)))
        pt1 = (int(round(u1)), int(round(v1)))

        # Clip the line to the image rectangle so we never draw outside the frame.
        clipped = cv2.clipLine(rect, pt0, pt1)
        if clipped[0]:
            c0, c1 = clipped[1], clipped[2]
            cv2.line(frame, c0, c1, (0, 255, 0), 2, cv2.LINE_AA)


def draw_court(frame, R, t, K, dist):
    """
    Draw all court line segments by projecting the 3D court model to the image.
    """
    for p0, p1 in build_court_segments():
        draw_segment_safe(frame, p0, p1, R, t, K, dist)


def overlay_video(video_in, video_out, K, dist, R, t):
    """
    Generate a new video where the (undistorted) original frames are overlaid with
    the projected 3D tennis court lines.

    Key idea:
    - Undistort the frame once per image using an optimal new camera matrix newK.
    - After undistortion, projection should be done with dist=0 and intrinsics=newK.
      (because lens distortion has already been removed from the pixels)
    """
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_in}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    # Precompute undistortion maps for speed:
    # This is much faster than calling cv2.undistort() per frame.
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), ALPHA, (W, H))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (W, H), cv2.CV_16SC2)

    # After undistortion, we treat the image as distortion-free.
    dist0 = np.zeros_like(dist, dtype=np.float64)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_out, fourcc, fps, (W, H))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Undistort current frame to a "pinhole-like" image with intrinsics newK.
        und = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Project and draw court lines on the undistorted frame.
        draw_court(und, R, t, newK, dist0)
        out.write(und)

    cap.release()
    out.release()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load per-camera extrinsics & intrinsics saved from the pose estimation step.
    data = json.load(open(EXTR_JSON, "r", encoding="utf-8"))

    for cam in CAMS:
        if cam not in data:
            print("Skip:", cam)
            continue

        info = data[cam]
        video_in = os.path.join(BASE_DIR, f"{cam}.mp4")

        # Note: these K/dist correspond to the ORIGINAL distorted video frames.
        K = np.array(info["K"], dtype=np.float64)
        dist = np.array(info["dist"], dtype=np.float64).reshape(-1)

        # Extrinsics (R,t) map world -> camera.
        R = np.array(info["R"], dtype=np.float64)
        t = np.array(info["t"], dtype=np.float64).reshape(3, 1)

        video_out = os.path.join(OUT_DIR, f"{cam}_court_overlay_UNDIST.mp4")
        print(f"Processing: {cam} -> {video_out}")
        overlay_video(video_in, video_out, K, dist, R, t)

    print("Done.")


if __name__ == "__main__":
    main()
