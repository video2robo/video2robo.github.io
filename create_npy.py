# make_pointcloud_npy_from_rgbd.py
from pathlib import Path
import argparse
import re
import numpy as np
from PIL import Image


def numeric_key(p: Path):
    # 按文件名里的数字排序，比如 1.png, 2.png, 10.png
    nums = re.findall(r"\d+", p.stem)
    return tuple(int(x) for x in nums) if nums else (p.stem,)


def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=numeric_key)


def load_intrinsics(txt_path: Path):
    K = np.loadtxt(txt_path, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics 必须是 3x3，当前是 {K.shape}")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return float(fx), float(fy), float(cx), float(cy)


def load_depth(depth_path: Path, depth_unit: str):
    if depth_path.suffix.lower() == ".npy":
        depth = np.load(depth_path).astype(np.float32)
    else:
        depth = np.array(Image.open(depth_path))
        depth = depth.astype(np.float32)

    # 深度单位转换到米
    if depth_unit == "mm":
        depth = depth / 1000.0
    elif depth_unit == "m":
        pass
    else:
        raise ValueError("depth_unit 只能是 mm 或 m")
    return depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rgb_dir",
        default=r"D:\RESEARCH\data_gen\file_of_network\video2robo.github.io\assets\img\attach\rgb",
        help="RGB 文件夹",
    )
    parser.add_argument(
        "--depth_dir",
        default=r"D:\RESEARCH\data_gen\file_of_network\video2robo.github.io\assets\img\attach\depth",        
        help="Depth 文件夹",
    )
    parser.add_argument(
        "--intrinsics",

default=r"D:\RESEARCH\data_gen\file_of_network\video2robo.github.io\assets\img\attach\intrinsics.txt",       
        help="3x3 相机内参 txt",
    )
    parser.add_argument(
        "--out_dir",
        default=r"D:\RESEARCH\data_gen\file_of_network\video2robo.github.io\assets\npy",
        help="输出目录",
    )
    parser.add_argument("--scene", default="step1", help="输出文件名前缀")
    parser.add_argument("--frame_idx", type=int, default=0, help="取第几帧（按排序后，0-based）")
    parser.add_argument("--depth_unit", choices=["mm", "m"], default="mm", help="depth 图单位")
    parser.add_argument("--depth_min", type=float, default=0.05, help="最小深度（米）")
    parser.add_argument("--depth_max", type=float, default=5.0, help="最大深度（米）")
    parser.add_argument("--stride", type=int, default=1, help="下采样步长，1=不下采样")
    args = parser.parse_args()

    rgb_dir = Path(args.rgb_dir)
    depth_dir = Path(args.depth_dir)
    intrinsics_path = Path(args.intrinsics)
    out_dir = Path(args.out_dir)

    rgb_files = list_images(rgb_dir)
    depth_files = list_images(depth_dir)

    if not rgb_files:
        raise FileNotFoundError(f"RGB 文件夹为空: {rgb_dir}")
    if not depth_files:
        raise FileNotFoundError(f"Depth 文件夹为空: {depth_dir}")

    if args.frame_idx < 0 or args.frame_idx >= min(len(rgb_files), len(depth_files)):
        raise IndexError(
            f"frame_idx={args.frame_idx} 越界。RGB={len(rgb_files)} 张，Depth={len(depth_files)} 张"
        )

    rgb_path = rgb_files[args.frame_idx]
    depth_path = depth_files[args.frame_idx]

    fx, fy, cx, cy = load_intrinsics(intrinsics_path)

    rgb = np.array(Image.open(rgb_path).convert("RGB"))  # H,W,3 uint8
    depth = load_depth(depth_path, args.depth_unit)      # H,W float32(m)

    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(
            f"RGB 和 Depth 分辨率不一致: RGB={rgb.shape[:2]}, Depth={depth.shape[:2]}"
        )

    H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    Z = depth
    valid = np.isfinite(Z) & (Z > args.depth_min) & (Z < args.depth_max)

    if args.stride > 1:
        keep = np.zeros_like(valid, dtype=bool)
        keep[::args.stride, ::args.stride] = True
        valid &= keep

    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    points = np.stack([X[valid], Y[valid], Z[valid]], axis=1).astype(np.float32)  # (N,3)
    colors = rgb[valid].astype(np.uint8)                                           # (N,3)

    out_dir.mkdir(parents=True, exist_ok=True)
    points_path = out_dir / f"{args.scene}.npy"
    colors_path = out_dir / f"{args.scene}_colors.npy"

    np.save(points_path, points)
    np.save(colors_path, colors)

    print(f"[OK] RGB:   {rgb_path.name}")
    print(f"[OK] Depth: {depth_path.name}")
    print(f"[OK] points -> {points_path}  shape={points.shape}, dtype={points.dtype}")
    print(f"[OK] colors -> {colors_path} shape={colors.shape}, dtype={colors.dtype}")


if __name__ == "__main__":
    main()