import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt


# =========================
# Config (edit here)
# =========================
DOP_PATH = r"C:\Users\HMT\Documents\GitHub\dop\231.png"  # input DoP image
OUT_DIR = r"C:\Users\HMT\Documents\GitHub\dop"  # output directory
OUT_NAME = "Fig9_confidence_analysis.png"

# Input format:
# - "auto": if single-channel, treat as grayscale; if color, assume COLORMAP_JET.
# - "gray": force grayscale conversion.
# - "jet": decode COLORMAP_JET pseudo-color to DoP values.
DOP_INPUT_FORMAT = "auto"  # "auto" / "gray" / "jet"
AUTO_GRAY_TOL = 2.0  # mean absolute channel-diff threshold for auto gray check

# DoP numeric range:
# - "float01": already in [0,1] float
# - "u8": 8-bit grayscale (0-255)
# - "u16": 16-bit grayscale (0-65535)
DOP_VALUE_MODE = "auto"  # "auto" / "float01" / "u8" / "u16"

# DoP -> reliability mapping (higher DoP -> higher reliability)
USE_SIGMOID = True
SIGMOID_K = 12.0  # higher = harder gate
SIGMOID_TAU = 0.08  # DoP threshold

# Multi-scale settings: three reliability maps
GAUSS_SIGMA_L1 = 0.6
GAUSS_SIGMA_L2 = 1.5
GAUSS_SIGMA_L3 = 2.8
DOWNSAMPLE_SCALE_L2 = 0.25  # 0.25 means downsample to 1/4 then upsample
DOWNSAMPLE_SCALE_L3 = 0.0625  # 0.0625 means downsample to 1/16 then upsample

# JET decode (approximate inverse colormap)
JET_DECODE_BATCH = 50000

# Figure output
FIG_DPI = 300
SHOW_AXES = False
COLORMAP = "jet"  # also try "viridis" / "turbo" / "gray"


# =========================
# Utils
# =========================
_JET_LUT_BGR: Optional[np.ndarray] = None


def build_jet_lut() -> np.ndarray:
    values = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    colors = cv2.applyColorMap(values, cv2.COLORMAP_JET)
    return colors.reshape(-1, 3).astype(np.int16)


def get_jet_lut() -> np.ndarray:
    global _JET_LUT_BGR
    if _JET_LUT_BGR is None:
        _JET_LUT_BGR = build_jet_lut()
    return _JET_LUT_BGR


def decode_jet_to_u8(
    img_bgr: np.ndarray, batch_size: int = JET_DECODE_BATCH
) -> np.ndarray:
    """Approximate inverse of COLORMAP_JET. Returns u8 (0-255)."""
    lut = get_jet_lut()
    h, w = img_bgr.shape[:2]
    pixels = img_bgr.reshape(-1, 3).astype(np.int16)
    out = np.empty((pixels.shape[0],), dtype=np.uint8)

    total = pixels.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        chunk = pixels[start:end]
        diff = chunk[:, None, :] - lut[None, :, :]
        dist2 = (diff.astype(np.int32) ** 2).sum(axis=2)
        out[start:end] = dist2.argmin(axis=1).astype(np.uint8)

    return out.reshape(h, w)


def is_near_gray(img_bgr: np.ndarray, tol: float = AUTO_GRAY_TOL) -> bool:
    b = img_bgr[:, :, 0].astype(np.int16)
    g = img_bgr[:, :, 1].astype(np.int16)
    r = img_bgr[:, :, 2].astype(np.int16)
    return (np.mean(np.abs(b - g)) < tol) and (np.mean(np.abs(g - r)) < tol)


def read_dop_image(path: str, input_format: str) -> Tuple[np.ndarray, Optional[str]]:
    """Read DoP image and return (raw_values, value_mode_override)."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if img.ndim == 2:
        return img.astype(np.float32), None

    if img.shape[2] == 4:
        img = img[:, :, :3]

    fmt = input_format.lower()
    if fmt == "gray":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float32), None
    if fmt == "jet":
        print("Decoding COLORMAP_JET pseudo-color...")
        return decode_jet_to_u8(img).astype(np.float32), "u8"
    if fmt == "auto":
        if is_near_gray(img, AUTO_GRAY_TOL):
            print("Auto mode: color image looks grayscale; using gray conversion.")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return gray.astype(np.float32), None
        print("Auto mode: color image detected; decoding COLORMAP_JET.")
        return decode_jet_to_u8(img).astype(np.float32), "u8"

    raise ValueError(f"Unknown DOP_INPUT_FORMAT: {input_format}")


def normalize_dop(dop_raw: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Normalize DoP to [0,1]."""
    if mode == "float01":
        dop = dop_raw.copy()
    elif mode == "u8":
        dop = dop_raw / 255.0
    elif mode == "u16":
        dop = dop_raw / 65535.0
    elif mode == "auto":
        # heuristic: if max <= 1.5 -> assume float01, else decide by range
        mx = float(np.nanmax(dop_raw))
        if mx <= 1.5:
            dop = dop_raw.copy()
        else:
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

    # one shared colorbar at the side
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.92, 0.14, 0.02, 0.72])
    cbar = fig.colorbar(ims[-1], cax=cax)
    cbar.set_label("Normalized value")

    fig.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    dop_raw, mode_override = read_dop_image(DOP_PATH, DOP_INPUT_FORMAT)
    value_mode = mode_override or DOP_VALUE_MODE
    dop01 = normalize_dop(dop_raw, value_mode)

    # base reliability
    rel0 = dop_to_reliability(dop01)

    # multi-scale (simulate different encoder levels)
    r1 = gaussian_blur(rel0, GAUSS_SIGMA_L1)
    r2 = down_up(rel0, DOWNSAMPLE_SCALE_L2)
    r2 = gaussian_blur(r2, GAUSS_SIGMA_L2)
    r3 = down_up(rel0, DOWNSAMPLE_SCALE_L3)
    r3 = gaussian_blur(r3, GAUSS_SIGMA_L3)

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
