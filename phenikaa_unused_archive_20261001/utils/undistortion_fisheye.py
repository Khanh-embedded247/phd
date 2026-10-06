#!/usr/bin/env python3
"""Test undistort fisheye bang dung calib VF6_02.

File nay chi dung de kiem tra truc quan anh fisheye sau khi nan.
Logic chinh duoc giu giong benchmark/common.py:

1. Doc K, D tu VF6_02_Intrinsics.json.
2. Scale K neu kich thuoc anh thuc te khac kich thuoc calib.
3. Neu la CAM_F_* thi dung cv2.fisheye.* voi 4 he so D.
4. Xuat anh theo nhieu gia tri balance de so sanh do meo/do crop.

balance nho  -> crop nhieu hon, anh phang va it bi keo mep hon.
balance lon  -> giu FOV nhieu hon, mep anh bi keo manh hon.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PHENIKAA_ROOT = Path(__file__).resolve().parents[1]
CALIB_ROOT = PHENIKAA_ROOT / "calib" / "vf_06_02"
INTRINSIC_JSON = CALIB_ROOT / "VF6_02_Intrinsics.json"
DEFAULT_IMAGE = (
    PHENIKAA_ROOT
    / "data"
    / "Normal"
    / "CAMERA"
    / "CAM_F_F"
    / "1781509259-100421069.jpg"
)
DEFAULT_OUT_DIR = PHENIKAA_ROOT / "outputs" / "debug_undistortion_fisheye"


def load_intrinsic(camera: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Doc K, D va kich thuoc anh calib cua mot camera."""
    with open(INTRINSIC_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    calib = data[camera]
    K = np.asarray(calib["camera_matrix"], dtype=np.float64)
    D = np.asarray(calib["distortion_coefficients"], dtype=np.float64).reshape(-1)
    return K, D, int(calib["image_width"]), int(calib["image_height"])


def scale_camera_matrix(
    K: np.ndarray,
    calib_width: int,
    calib_height: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Scale K khi anh thuc te khac kich thuoc anh dung luc calib."""
    K_scaled = K.copy()
    K_scaled[0, :] *= image_width / calib_width
    K_scaled[1, :] *= image_height / calib_height
    return K_scaled


def undistort_fisheye(
    image: np.ndarray,
    K_calib: np.ndarray,
    D: np.ndarray,
    calib_width: int,
    calib_height: int,
    balance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Undistort CAM_F_* bang OpenCV fisheye model."""
    h, w = image.shape[:2]
    K_scaled = scale_camera_matrix(K_calib, calib_width, calib_height, w, h)
    D_use = np.asarray(D, dtype=np.float64).reshape(-1)[:4].reshape(4, 1)

    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_scaled,
        D_use,
        (w, h),
        np.eye(3),
        balance=float(balance),
        new_size=(w, h),
        fov_scale=1.0,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_scaled,
        D_use,
        np.eye(3),
        K_new,
        (w, h),
        cv2.CV_16SC2,
    )
    image_new = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return image_new, K_new


def undistort_pinhole(
    image: np.ndarray,
    K_calib: np.ndarray,
    D: np.ndarray,
    calib_width: int,
    calib_height: int,
    balance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Undistort CAM_P_* bang OpenCV pinhole/radtan model."""
    h, w = image.shape[:2]
    K_scaled = scale_camera_matrix(K_calib, calib_width, calib_height, w, h)
    K_new, _ = cv2.getOptimalNewCameraMatrix(
        K_scaled,
        np.asarray(D, dtype=np.float64).reshape(-1),
        (w, h),
        float(balance),
        (w, h),
    )
    image_new = cv2.undistort(image, K_scaled, D, None, K_new)
    return image_new, K_new


def resize_width(image: np.ndarray, width: int) -> np.ndarray:
    """Resize anh theo chieu rong de ghep anh so sanh."""
    if image.shape[1] == width:
        return image
    scale = width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def put_label(image: np.ndarray, text: str) -> np.ndarray:
    """Ghi nhan balance len anh de xem nhanh."""
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (255, 255, 255), -1)
    cv2.putText(
        out,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def make_contact_sheet(images: list[np.ndarray], out_path: Path, tile_width: int = 640) -> None:
    """Ghep cac anh output thanh mot anh lon de so sanh balance."""
    tiles = [resize_width(img, tile_width) for img in images]
    max_h = max(tile.shape[0] for tile in tiles)
    padded = []
    for tile in tiles:
        if tile.shape[0] < max_h:
            pad = np.zeros((max_h - tile.shape[0], tile.shape[1], 3), dtype=np.uint8)
            tile = np.vstack([tile, pad])
        padded.append(tile)
    sheet = np.hstack(padded)
    cv2.imwrite(str(out_path), sheet)


def parse_balance_list(text: str) -> list[float]:
    """Parse chuoi kieu '0,0.2,0.5,1.0' thanh list float."""
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("Danh sach balance rong")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test undistort fisheye/pinhole bang calib VF6_02.")
    parser.add_argument("--input", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--camera", default="CAM_F_F")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--balances",
        default="0,0.1,0.2,0.4,0.7,1.0",
        help="Cac gia tri balance can test, vi du: 0,0.2,1.0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {args.input}")

    K, D, calib_w, calib_h = load_intrinsic(args.camera)
    is_fisheye = args.camera.startswith("CAM_F")
    balances = parse_balance_list(args.balances)

    print(f"Input image : {args.input}")
    print(f"Camera      : {args.camera}")
    print(f"Image shape : {image.shape[1]}x{image.shape[0]}")
    print(f"Calib shape : {calib_w}x{calib_h}")
    print(f"Model       : {'fisheye cv2.fisheye' if is_fisheye else 'pinhole cv2.undistort'}")
    print(f"D len       : {len(D)}")
    print(f"D           : {D.tolist()}")
    print("K:")
    print(K)

    labelled_outputs = []
    for balance in balances:
        if is_fisheye:
            undistorted, K_new = undistort_fisheye(image, K, D, calib_w, calib_h, balance)
        else:
            undistorted, K_new = undistort_pinhole(image, K, D, calib_w, calib_h, balance)

        out_path = args.out_dir / f"{args.camera}_balance_{balance:.2f}.jpg"
        cv2.imwrite(str(out_path), undistorted)
        labelled_outputs.append(put_label(undistorted, f"{args.camera} balance={balance:.2f}"))

        print(f"\nBalance {balance:.2f}")
        print(f"Saved   : {out_path}")
        print("K_new:")
        print(K_new)

    sheet_path = args.out_dir / f"{args.camera}_balance_compare.jpg"
    make_contact_sheet(labelled_outputs, sheet_path)
    print(f"\nCompare : {sheet_path}")


if __name__ == "__main__":
    main()
