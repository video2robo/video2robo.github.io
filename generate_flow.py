import numpy as np
from pathlib import Path

NPY_DIR = Path(r"D:\RESEARCH\data_gen\file_of_network\video2robo.github.io\assets\npy")

# 生成 attach_flow.npy - 简单的轨迹（从原点到某个目标点的直线）
num_points = 50
flow = np.linspace([0, 0, 0], [1, 1, 1], num_points).astype(np.float32)
np.save(NPY_DIR / "attach_flow.npy", flow)
print(f"Generated attach_flow.npy: shape={flow.shape}, dtype={flow.dtype}")
