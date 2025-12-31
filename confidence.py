import os
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt


# =========================
# Config (edit here)
# =========================
DOP_PATH = r"C:\Users\HMT\Documents\GitHub\dop\001_DoP.png"  # 输入：你的DoP图（png/jpg/tif都行）
OUT_DIR = r"C:\Users\HMT\Documents\GitHub\dop"  # 输出目录
OUT_NAME = "Fig9_confidence_analysis.png"

# 你DoP图的数值语义：
# - 若是“已经是[0,1]浮点存的tif/exr”，设为 "float01"
# - 若是“8bit灰度(0~255)”，设为 "u8"
# - 若是“16bit灰度(0~65535)”，设为 "u16"
DOP_VALUE_MODE = "auto"  # "auto" / "float01" / "u8" / "u16"

# DoP -> reliability 映射（物理含义：DoP越大，角度越可靠）
# 推荐用 sigmoid：能产生“门控”效果，像论文里PUGM那种抑制/放行
USE_SIGMOID = True
SIGMOID_K = 12.0  # 越大越“硬门控”
SIGMOID_TAU = 0.08  # DoP阈值（低于此更不可靠），可按数据调整

# 多尺度设置：三层 reliability map
# Level-1: 细粒度（轻微平滑）
# Level-2: 中尺度（更强平滑）
# Level-3: 粗尺度（先降采样再上采样，模拟更粗的空间一致性）
GAUSS_SIGMA_L1 = 0.6
GAUSS_SIGMA_L2 = 2.0
DOWNSAMPLE_SCALE_L3 = 0.25  # 0.25表示缩到1/4再放大回去

# 出图
FIG_DPI = 300
SHOW_AXES = False
COLORMAP = "jet"  # 你也可以改成 "viridis" / "turbo" / "gray"


# =========================
# Utils
# =========================
def read_as_gray(path: str) -> np.ndarray:
    """Read image, return float32 gray array with original numeric range kept."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    # If color, convert to gray
    if img.ndim == 3:
        # handle BGRA/BGR/RGB-like
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img.astype(np.float32)


def normalize_dop(dop_raw: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Normalize DoP to [0,1]."""
    if mode == "float01":
        dop = dop_raw.copy()
    elif mode == "u8":
        dop = dop_raw / 255.0
    elif mode == "u16":
        dop = dop_raw / 65535.0
    elif mode == "auto":
        # heuristic: if max <= 1.5 -> assume float01, else decide by dtype-like range
        mx = float(np.nanmax(dop_raw))
        if mx <= 1.5:
            dop = dop_raw.copy()
        else:
            # If looks like 16-bit
            dop = dop_raw / (65535.0 if mx > 300.0 else 255.0)
    else:
        raise ValueError(f"Unknown DOP_VALUE_MODE: {mode}")

    dop = np.clip(dop, 0.0, 1.0)
    return dop.astype(np.float32)


def dop_to_reliability(dop01: np.ndarray) -> np.ndarray:
    """Map DoP in [0,1] to reliability/confidence in [0,1]."""
    if USE_SIGMOID:
        # sigmoid gate around tau
        x = SIGMOID_K * (dop01 - SIGMOID_TAU)
        rel = 1.0 / (1.0 + np.exp(-x))
    else:
        # simple monotonic mapping (gamma)
        gamma = 0.7
        rel = np.power(dop01, gamma)

    return rel.astype(np.float32)


def gaussian_blur(img01: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img01
    k = int(round(sigma * 6)) | 1  # odd kernel size
    return cv2.GaussianBlur(
        img01, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
    )


def down_up(img01: np.ndarray, scale: float) -> np.ndarray:
    h, w = img01.shape[:2]
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = cv2.resize(img01, (nw, nh), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back.astype(np.float32)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a1 = a.reshape(-1).astype(np.float64)
    b1 = b.reshape(-1).astype(np.float64)
    a1 -= a1.mean()
    b1 -= b1.mean()
    denom = np.sqrt((a1 * a1).sum()) * np.sqrt((b1 * b1).sum()) + 1e-12
    return float((a1 * b1).sum() / denom)


def save_figure(dop01, r1, r2, r3, c1, c2, c3, out_path: str):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), dpi=FIG_DPI)

    ims = []
    ims.append(axes[0].imshow(dop01, cmap=COLORMAP, vmin=0, vmax=1))
    axes[0].set_title("DoP")

    ims.append(axes[1].imshow(r1, cmap=COLORMAP, vmin=0, vmax=1))
    axes[1].set_title(f"Reliability (L1)\nr={c1:.2f}")

    ims.append(axes[2].imshow(r2, cmap=COLORMAP, vmin=0, vmax=1))
    axes[2].set_title(f"Reliability (L2)\nr={c2:.2f}")

    ims.append(axes[3].imshow(r3, cmap=COLORMAP, vmin=0, vmax=1))
    axes[3].set_title(f"Reliability (L3)\nr={c3:.2f}")

    if not SHOW_AXES:
        for ax in axes:
            ax.axis("off")

    # one shared colorbar
    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Normalized value")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    dop_raw = read_as_gray(DOP_PATH)
    dop01 = normalize_dop(dop_raw, DOP_VALUE_MODE)

    # base reliability
    rel0 = dop_to_reliability(dop01)

    # multi-scale (simulate different encoder levels)
    r1 = gaussian_blur(rel0, GAUSS_SIGMA_L1)
    r2 = gaussian_blur(rel0, GAUSS_SIGMA_L2)
    r3 = down_up(rel0, DOWNSAMPLE_SCALE_L3)

    # correlations
    c1 = pearson_corr(dop01, r1)
    c2 = pearson_corr(dop01, r2)
    c3 = pearson_corr(dop01, r3)

    out_path = str(Path(OUT_DIR) / OUT_NAME)
    save_figure(dop01, r1, r2, r3, c1, c2, c3, out_path)

    print("Saved:", out_path)
    print(f"Pearson r: L1={c1:.4f}, L2={c2:.4f}, L3={c3:.4f}")


if __name__ == "__main__":
    main()
