import os
import json
import cv2
import numpy as np



CAMS = [1, 2, 3, 4]
FRAME_IDX = 300                      
OUT_DIR = "results"
POINT_SET_NAME = "FAR_DBL+SRV_SGL" 

# World geometry (origin at court center, Z=0 on court plane)
L  = 23.77 / 2.0   # 11.885
Wd = 10.97 / 2.0   # 5.485
Ws =  8.23 / 2.0   # 4.115
S  =  6.40         # service line x from center

# FAR baseline × doubles corners + FAR service line × singles sidelines
OBJECT_POINTS = np.array([
    [ +L, -Wd, 0.0],  # 1) FAR baseline × DOUBLES left corner
    [ +L, +Wd, 0.0],  # 2) FAR baseline × DOUBLES right corner
    [ +S, +Ws, 0.0],  # 3) FAR service line × SINGLES right
    [ +S, -Ws, 0.0],  # 4) FAR service line × SINGLES left
], dtype=np.float64)

CLICK_LABELS = [
    "1) FAR baseline × DOUBLES left corner",
    "2) FAR baseline × DOUBLES right corner",
    "3) FAR service line × SINGLES right",
    "4) FAR service line × SINGLES left",
]


# -------------------------
# calib parsing
# -------------------------
def parse_calib_file(calib_path: str):
    raw = {}
    with open(calib_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                raw[k] = float(v)
            except ValueError:
                pass

    fx = raw["f"] * raw["mx"]
    fy = raw["f"] * raw["my"]
    cx, cy = raw["cx"], raw["cy"]

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    dist = np.array([raw["k1"], raw["k2"], raw["p1"], raw["p2"], raw["k3"]], dtype=np.float64)
    return K, dist


# -------------------------
# frame read
# -------------------------
def read_frame(video_path: str, idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {idx} from {video_path}")
    return frame


# -------------------------
# click UI
# -------------------------
def collect_points(frame_bgr: np.ndarray, labels, window: str):
    pts = []
    base = frame_bgr.copy()
    vis = base.copy()

    def redraw():
        nonlocal vis
        vis = base.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(vis, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(vis, f"c{i+1}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        i = len(pts)
        msg = labels[i] if i < len(labels) else "Done. Press ENTER"
        cv2.putText(vis, msg, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, "ENTER=finish | u=undo | r=reset | q=quit",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) < len(labels):
                pts.append((x, y))
                redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u") and pts:
            pts.pop()
            redraw()
        elif key == ord("r"):
            pts.clear()
            redraw()
        elif key == ord("q") or key == 27:
            cv2.destroyWindow(window)
            raise SystemExit("Quit.")
        elif key in (10, 13):  
            if len(pts) == len(labels):
                cv2.destroyWindow(window)
                return np.array(pts, dtype=np.float64)


# -------------------------
# pose + metrics
# -------------------------
def project_points(obj_pts, R, t, K, dist):
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(obj_pts, rvec, t, K, dist)
    return proj.reshape(-1, 2)

def rmse_reproj(obj_pts, img_pts, R, t, K, dist):
    proj = project_points(obj_pts, R, t, K, dist)
    e = np.linalg.norm(proj - img_pts, axis=1)
    return float(np.sqrt(np.mean(e**2)))

def depths_z(obj_pts, R, t):
    Xcam = (R @ obj_pts.T + t.reshape(3, 1)).T
    return Xcam[:, 2]

def camera_center_world(R, t):
    return (-R.T @ t).reshape(3)

def solve_pose_ippe(obj_pts, img_pts, K, dist):
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok or len(rvecs) == 0:
        ok2, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok2:
            raise RuntimeError("solvePnP failed.")
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        z = depths_z(obj_pts, R, t)
        return R, t, rmse_reproj(obj_pts, img_pts, R, t, K, dist), z

    best = None
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        z = depths_z(obj_pts, R, t)
        rmse = rmse_reproj(obj_pts, img_pts, R, t, K, dist)

        score = (0 if np.all(z > 1e-6) else 1, rmse)
        if best is None or score < best[0]:
            best = (score, R, t, z, rmse)

    _, R, t, z, rmse = best
    return R, t, float(rmse), z


# -------------------------
# overlay save
# -------------------------
def save_overlay(frame_bgr, img_pts, proj_pts, out_path):
    vis = frame_bgr.copy()
    cv2.putText(vis, "Red=clicked | Green=reprojection", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    for i, (u, v) in enumerate(img_pts.astype(int)):
        cv2.circle(vis, (u, v), 7, (0, 0, 255), -1)
        cv2.putText(vis, f"c{i+1}", (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for i, (u, v) in enumerate(proj_pts.astype(int)):
        cv2.circle(vis, (u, v), 9, (0, 255, 0), 2)
        cv2.putText(vis, f"p{i+1}", (u + 10, v + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(out_path, vis)


# -------------------------
# main
# -------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    for c in CAMS:
        cam = f"cam{c}"
        video = f"{cam}.mp4"
        calib = f"{cam}.calib"

        if not os.path.exists(video) or not os.path.exists(calib):
            print(f"[{cam}] missing file(s). Skipping.")
            continue

        print(f"\n=== {cam} ===")
        K, dist = parse_calib_file(calib)
        frame = read_frame(video, FRAME_IDX)

        img_pts = collect_points(frame, CLICK_LABELS, window=f"{cam} - click 4 points ({POINT_SET_NAME})")
        R, t, rmse, z = solve_pose_ippe(OBJECT_POINTS, img_pts, K, dist)
        C = camera_center_world(R, t)

        proj = project_points(OBJECT_POINTS, R, t, K, dist)
        overlay_path = os.path.join(OUT_DIR, f"{cam}_overlay.png")
        save_overlay(frame, img_pts, proj, overlay_path)

        print(f"[{cam}] RMSE(px) = {rmse:.3f}")
        print(f"[{cam}] depths(z) = {z}")
        print(f"[{cam}] C(world) = {C}")

        e = np.linalg.norm(proj - img_pts, axis=1)
        print("per-point error:", e)


        results[cam] = {
            "cam": cam,
            "frame_idx": FRAME_IDX,
            "point_set": POINT_SET_NAME,
            "image_points_uv": img_pts.tolist(),
            "object_points_xyz_m": OBJECT_POINTS.tolist(),
            "K": K.tolist(),
            "dist": dist.reshape(-1).tolist(),
            "R": R.tolist(),
            "t": t.reshape(-1).tolist(),
            "camera_center_world_C": C.reshape(-1).tolist(),
            "reprojection_rmse_px": float(rmse),
            "depths_z_cam": z.tolist(),
            "overlay_path": os.path.abspath(overlay_path),
        }

    out_json = os.path.join(OUT_DIR, f"extrinsics_all_cam_{cam}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nDone.")
    print("Saved:", os.path.abspath(out_json))
    print("Overlays in:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
