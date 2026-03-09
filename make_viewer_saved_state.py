# rgbd_to_viser.py

from pathlib import Path
import argparse
import numpy as np
import cv2
import viser


parser = argparse.ArgumentParser()
parser.add_argument("scene", type=str)
parser.add_argument("--fps", type=float, default=10.0)
parser.add_argument("--point_size", type=float, default=0.003)
parser.add_argument("--depth_scale", type=float, default=1000.0)
parser.add_argument("--stride", type=int, default=6)
parser.add_argument("--frame_index", type=int, default=None)

args = parser.parse_args()


BASE = Path(r"D:\RESEARCH\data_gen\file_of_network\video2robo.github.io")

RGB_DIR = BASE / "assets/img" / args.scene
DEPTH_DIR = BASE / "assets/depth" / args.scene
INTRINSIC_PATH = RGB_DIR / "intrinsic.txt"

OUT_DIR = (
    BASE
    / "assets/viser-client/assets/fmb_fake_viser"
    / args.scene
    / "viewer"
)

OUT_FILE = OUT_DIR / "saved_state.viser"


def rgbd_to_points(rgb, depth, K):

    H, W = depth.shape

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(W)
    v = np.arange(H)

    uu, vv = np.meshgrid(u, v)

    Z = depth

    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)

    valid = Z.reshape(-1) > 0

    return points[valid], colors[valid]


# ----------------------------
# intrinsic
# ----------------------------

K = np.loadtxt(INTRINSIC_PATH).astype(np.float32)

if K.shape != (3, 3):
    raise ValueError("Intrinsic must be 3x3")


# ----------------------------
# frame list
# ----------------------------

rgb_files = sorted(RGB_DIR.glob("*.jpg"))
depth_files = sorted(DEPTH_DIR.glob("*.png"))

if len(rgb_files) == 0:
    raise ValueError("No RGB images")

if len(rgb_files) != len(depth_files):
    raise ValueError("RGB / Depth mismatch")


if args.frame_index is not None:

    rgb_files = [rgb_files[args.frame_index]]
    depth_files = [depth_files[args.frame_index]]


print("Total frames:", len(rgb_files))


# ----------------------------
# start viser
# ----------------------------

server = viser.ViserServer(host="127.0.0.1", port=0)

serializer = server.get_scene_serializer()


cloud = None

frame_dt = 1.0 / max(args.fps, 1e-6)


for i, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):

    print("Processing frame", i)

    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)

    depth = depth / args.depth_scale

    pts, cols = rgbd_to_points(rgb, depth, K)

    pts = pts[:: args.stride]
    cols = cols[:: args.stride]

    print("Points:", pts.shape[0])

    if cloud is None:

        cloud = server.scene.add_point_cloud(
            name="/rgb_cloud",
            points=pts.astype(np.float32),
            colors=cols.astype(np.uint8),
            point_size=args.point_size,
        )

    else:

        cloud.points = pts.astype(np.float32)
        cloud.colors = cols.astype(np.uint8)

    serializer.insert_sleep(frame_dt)


# ----------------------------
# save viewer state
# ----------------------------

data = serializer.serialize()

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE.write_bytes(data)

print("Saved:", OUT_FILE)

server.stop()