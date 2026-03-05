#下面给你一个“只生成 viewer 的最小脚本”（硬编码路径版）：

# make_step1_viewer_saved_state.py
from pathlib import Path
import time
import numpy as np
import viser
import argparse

parser = argparse.ArgumentParser(description="需要提供的参数")
parser.add_argument("scene", help="场景名称（必须）")
args = parser.parse_args()
    
# 1) 你的 npy 目录（请确认存在）
NPY_DIR = Path(r"D:\RESEARCH\data_gen\file_of_network\video2robo.github.io\assets\npy")

# 2) {scene} viewer 目标目录（原位置）
OUT_DIR = Path(r"D:\RESEARCH\data_gen\file_of_network\video2robo.github.io\assets\viser-client\assets\fmb_fake_viser") / args.scene / "viewer"

OUT_FILE = OUT_DIR / "saved_state.viser"

# 3) 读取数据（按你的文件名改这里）
points = np.load(NPY_DIR / f"{args.scene}.npy").astype(np.float32)        # (N,3)
flow = np.load(NPY_DIR / f"{args.scene}_flow.npy").astype(np.float32)     # (T,3)

colors_path = NPY_DIR / f"{args.scene}_colors.npy"
if colors_path.exists():
    colors = np.load(colors_path).astype(np.uint8)                        # (N,3)
else:
    colors = np.tile(np.array([[180, 180, 255]], dtype=np.uint8), (points.shape[0], 1))

OUT_DIR.mkdir(parents=True, exist_ok=True)

server = viser.ViserServer(host="127.0.0.1", port=0)

server.scene.add_point_cloud(
    name="/point_cloud",
    points=points,
    colors=colors,
    point_size=0.003,
)

server.scene.add_spline_catmull_rom(
    name="/object_flow",
    positions=flow,
    color=(255, 80, 80),
    line_width=3.0,
)

time.sleep(0.5)
server.scene.save(OUT_FILE)
server.stop()

print(f"Generated: {OUT_FILE}")