#!/usr/bin/env python3
"""
extract_vf6_window.py

Cắt một đoạn thời gian đồng bộ từ raw dataset VF6.

Hỗ trợ:
- CAMERA: copy các file có timestamp trong tên file nằm trong khoảng thời gian.
- LIDAR: copy các file có timestamp trong tên file nằm trong khoảng thời gian.
- NAV / IMU:
    + Chọn các CSV có khả năng overlap cửa sổ thời gian.
    + Lọc các row bên trong CSV theo cột Timestamp nếu tìm thấy.
    + Nếu không tìm thấy cột timestamp phù hợp thì copy nguyên CSV.
- Giữ nguyên cấu trúc thư mục tương đối.

Cách dùng:
1. Sửa phần CONFIG bên dưới.
2. Chạy:
       python3 extract_vf6_window.py
"""

from __future__ import annotations

import csv
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Iterable

# ============================================================
# CONFIG - CHỈ CẦN SỬA PHẦN NÀY CHO MỖI LẦN CHẠY
# ============================================================

# Thư mục acquisition gốc
INPUT_DIR = Path(
    "/home/khanh247/Documents/Survey/phd/phenikaa/data/raw/big_road"
)

# Thư mục output
OUTPUT_DIR = Path(
    "/home/khanh247/Documents/Survey/phd/phenikaa/data/raw/big_road/cut_data"
)

# Timestamp bắt đầu theo Unix seconds.
# Ví dụ acquisition bắt đầu ở 1781509258.
START_TIMESTAMP = 1781512295

# Thời lượng cần lấy, giây.
# 300 = 5 phút.
DURATION_SECONDS = 90

# Các nhóm dữ liệu cần lấy.
# Benchmark hiện tại chỉ cần 4 nhóm này.
COPY_CAMERA = True
COPY_LIDAR = True
COPY_NAV = True
COPY_IMU = True

# Nếu sau này cần thì bật lên.
COPY_VEHICLE_INFO = False
COPY_VEHICLE_STEER = False
COPY_RADAR = False
COPY_ULTRASONIC = False

# Nếu True, output cũ sẽ bị xoá trước khi chạy.
# Cẩn thận với đường dẫn OUTPUT_DIR.
OVERWRITE_OUTPUT = False

# Với NAV/IMU:
# True = cố gắng lọc từng row theo Timestamp.
# False = chỉ copy các CSV liên quan.
FILTER_CSV_ROWS = True

# File log các file/frame bị lỗi. Script sẽ bỏ qua và tiếp tục.
ERROR_LOG_NAME = "bad_files.txt"

# Tên các cột timestamp thường gặp.
TIMESTAMP_COLUMN_CANDIDATES = [
    "Timestamp",
    "timestamp",
    "TimeStamp",
    "time_stamp",
    "time",
    "Time",
]

# ============================================================
# END CONFIG
# ============================================================


FILENAME_TS_PATTERN = re.compile(r"^(\d+)-(\d+)")


def parse_filename_timestamp(path: Path) -> Optional[float]:
    """
    Parse timestamp từ tên kiểu:
        1781509258-032816576.jpg
        1781509258-096009016.laz

    Trả về timestamp dạng float:
        1781509258.032816576
    """
    m = FILENAME_TS_PATTERN.match(path.name)
    if not m:
        return None

    sec = int(m.group(1))
    frac = m.group(2)

    try:
        return sec + int(frac) / (10 ** len(frac))
    except ValueError:
        return float(sec)


def timestamp_in_window(ts: float, start: float, end: float) -> bool:
    return start <= ts < end


def safe_prepare_output(output_dir: Path) -> None:
    """
    Chuẩn bị output theo kiểu RESUME:
    - Output chưa có: tạo mới.
    - OVERWRITE_OUTPUT=True: xoá toàn bộ output cũ rồi tạo lại.
    - OVERWRITE_OUTPUT=False: giữ output cũ, file đã tải đủ sẽ được bỏ qua.
    """
    if output_dir.exists():
        if OVERWRITE_OUTPUT:
            print(f"[INFO] Xoá output cũ: {output_dir}")
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"[INFO] Output đã tồn tại -> RESUME: {output_dir}")
            print("[INFO] File đã tải đủ sẽ bỏ qua, file còn thiếu sẽ tải tiếp.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def log_bad_file(output_root: Path, src_file: Path, error: Exception) -> None:
    """Ghi lỗi vào bad_files.txt nhưng không làm dừng chương trình."""
    try:
        with (output_root / ERROR_LOG_NAME).open("a", encoding="utf-8") as f:
            f.write(f"{src_file}\t{type(error).__name__}: {error}\n")
    except Exception as log_error:
        print(f"[WARN] Không ghi được error log: {log_error}")


def copy_file_preserve_relative(
    src_file: Path,
    input_root: Path,
    output_root: Path,
) -> str:
    """
    Copy 1 file với cơ chế resume.

    Return:
      "copied": copy thành công
      "exists": đích đã tồn tại và có cùng kích thước -> bỏ qua
      "error": lỗi I/O hoặc copy -> bỏ qua file/frame đó
    """
    rel = src_file.relative_to(input_root)
    dst_file = output_root / rel
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = dst_file.with_name(dst_file.name + ".part")

    try:
        if dst_file.exists():
            try:
                if dst_file.stat().st_size == src_file.stat().st_size:
                    return "exists"
            except OSError:
                pass

        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass

        shutil.copy2(src_file, tmp_file)
        os.replace(tmp_file, dst_file)
        return "copied"

    except (OSError, IOError) as e:
        print(f"[ERROR] Bỏ qua file lỗi: {src_file}")
        print(f"        {type(e).__name__}: {e}")
        log_bad_file(output_root, src_file, e)

        try:
            if tmp_file.exists():
                tmp_file.unlink()
        except OSError:
            pass

        return "error"

def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file())


def copy_timestamped_files(
    source_dir: Path,
    input_root: Path,
    output_root: Path,
    start: float,
    end: float,
) -> tuple[int, int, int, int]:
    """
    Copy các file timestamped trong cửa sổ thời gian.

    Return:
      copied, skipped_existing, bad_files, unparsed
    """
    copied = 0
    skipped_existing = 0
    bad_files = 0
    unparsed = 0

    if not source_dir.exists():
        print(f"[WARN] Không tồn tại: {source_dir}")
        return copied, skipped_existing, bad_files, unparsed

    for src in iter_files(source_dir):
        ts = parse_filename_timestamp(src)

        if ts is None:
            unparsed += 1
            continue

        if timestamp_in_window(ts, start, end):
            result = copy_file_preserve_relative(
                src, input_root, output_root
            )
            if result == "copied":
                copied += 1
            elif result == "exists":
                skipped_existing += 1
            else:
                bad_files += 1

    return copied, skipped_existing, bad_files, unparsed

def detect_timestamp_column(fieldnames: list[str]) -> Optional[str]:
    if not fieldnames:
        return None

    for candidate in TIMESTAMP_COLUMN_CANDIDATES:
        if candidate in fieldnames:
            return candidate

    lower_map = {name.lower(): name for name in fieldnames}
    for candidate in TIMESTAMP_COLUMN_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def parse_csv_timestamp(value: str) -> Optional[float]:
    """
    Hỗ trợ:
      1781509201-082899210
      1781509201.082899210
      1781509201
    """
    if value is None:
        return None

    value = str(value).strip()
    if not value:
        return None

    if "-" in value:
        left, right = value.split("-", 1)
        if left.isdigit() and right.isdigit():
            return int(left) + int(right) / (10 ** len(right))

    try:
        return float(value)
    except ValueError:
        return None


def csv_might_overlap_by_filename(
    csv_path: Path,
    start: float,
    end: float,
    nominal_chunk_seconds: int = 180,
) -> bool:
    """
    CSV NAV/IMU có vẻ được chia theo chunk lớn, tên file mang timestamp đầu chunk.
    Vì không biết chính xác chiều dài mọi chunk, ta lấy một margin để tránh bỏ sót.
    """
    ts = parse_filename_timestamp(csv_path)
    if ts is None:
        return True

    return (ts < end) and (ts + nominal_chunk_seconds >= start)


def filter_csv_file(
    src_csv: Path,
    input_root: Path,
    output_root: Path,
    start: float,
    end: float,
) -> tuple[int, bool, str]:
    """
    Lọc rows trong CSV theo timestamp.

    Return:
      rows_kept, actually_filtered, status

    status:
      "written" : đã tạo output
      "exists"  : output đã tồn tại -> bỏ qua
      "empty"   : không có row nào trong cửa sổ
      "error"   : lỗi đọc/xử lý -> bỏ qua
    """
    rel = src_csv.relative_to(input_root)
    dst_csv = output_root / rel
    dst_csv.parent.mkdir(parents=True, exist_ok=True)

    # Resume: nếu CSV output đã tồn tại thì bỏ qua.
    if dst_csv.exists():
        return 0, False, "exists"

    tmp_csv = dst_csv.with_name(dst_csv.name + ".part")

    try:
        with src_csv.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            ts_col = detect_timestamp_column(fieldnames)

            if not ts_col:
                result = copy_file_preserve_relative(
                    src_csv, input_root, output_root
                )
                if result == "copied":
                    return 0, False, "written"
                if result == "exists":
                    return 0, False, "exists"
                return 0, False, "error"

            kept_rows = []
            for row in reader:
                ts = parse_csv_timestamp(row.get(ts_col))
                if ts is not None and timestamp_in_window(ts, start, end):
                    kept_rows.append(row)

        if not kept_rows:
            return 0, True, "empty"

        if tmp_csv.exists():
            try:
                tmp_csv.unlink()
            except OSError:
                pass

        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)

        os.replace(tmp_csv, dst_csv)
        return len(kept_rows), True, "written"

    except Exception as e:
        print(f"[ERROR] Bỏ qua CSV lỗi: {src_csv}")
        print(f"        {type(e).__name__}: {e}")
        log_bad_file(output_root, src_csv, e)

        try:
            if tmp_csv.exists():
                tmp_csv.unlink()
        except OSError:
            pass

        return 0, False, "error"

def process_csv_group(
    source_dir: Path,
    input_root: Path,
    output_root: Path,
    start: float,
    end: float,
) -> tuple[int, int, int, int, int]:
    """
    Return:
      csv_written, csv_skipped_existing, csv_bad,
      csv_filtered, total_rows_kept
    """
    csv_written = 0
    csv_skipped_existing = 0
    csv_bad = 0
    csv_filtered = 0
    total_rows_kept = 0

    if not source_dir.exists():
        print(f"[WARN] Không tồn tại: {source_dir}")
        return (
            csv_written,
            csv_skipped_existing,
            csv_bad,
            csv_filtered,
            total_rows_kept,
        )

    csv_files = sorted(source_dir.rglob("*.csv"))

    for csv_path in csv_files:
        if not csv_might_overlap_by_filename(csv_path, start, end):
            continue

        if FILTER_CSV_ROWS:
            rows_kept, actually_filtered, status = filter_csv_file(
                csv_path, input_root, output_root, start, end
            )

            if status == "written":
                csv_written += 1
            elif status == "exists":
                csv_skipped_existing += 1
            elif status == "error":
                csv_bad += 1

            if actually_filtered and status == "written":
                csv_filtered += 1
                total_rows_kept += rows_kept

        else:
            result = copy_file_preserve_relative(
                csv_path, input_root, output_root
            )

            if result == "copied":
                csv_written += 1
            elif result == "exists":
                csv_skipped_existing += 1
            else:
                csv_bad += 1

    return (
        csv_written,
        csv_skipped_existing,
        csv_bad,
        csv_filtered,
        total_rows_kept,
    )

def count_files(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file())


def get_size_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def main() -> None:
    input_root = INPUT_DIR.expanduser().resolve()
    output_root = OUTPUT_DIR.expanduser()

    start = float(START_TIMESTAMP)
    end = start + float(DURATION_SECONDS)

    if not input_root.exists():
        raise FileNotFoundError(f"Không tìm thấy INPUT_DIR: {input_root}")

    print("=" * 72)
    print("VF6 synchronized dataset extractor")
    print("=" * 72)
    print(f"Input       : {input_root}")
    print(f"Output      : {output_root}")
    print(f"Start       : {start}")
    print(f"End         : {end}")
    print(f"Duration    : {DURATION_SECONDS} s "
          f"({DURATION_SECONDS / 60:.2f} min)")
    print("=" * 72)

    safe_prepare_output(output_root)

    stats = {}

    if COPY_CAMERA:
        print("\n[CAMERA] Đang xử lý...")
        copied, skipped, bad, unparsed = copy_timestamped_files(
            input_root / "CAMERA",
            input_root,
            output_root,
            start,
            end,
        )
        stats["CAMERA copied"] = copied
        stats["CAMERA existing"] = skipped
        stats["CAMERA bad"] = bad
        print(
            f"[CAMERA] Copied: {copied}, existing/skipped: {skipped}, "
            f"bad/skipped: {bad}, unparsed filename: {unparsed}"
        )

    if COPY_LIDAR:
        print("\n[LIDAR] Đang xử lý...")
        copied, skipped, bad, unparsed = copy_timestamped_files(
            input_root / "LIDAR",
            input_root,
            output_root,
            start,
            end,
        )
        stats["LIDAR copied"] = copied
        stats["LIDAR existing"] = skipped
        stats["LIDAR bad"] = bad
        print(
            f"[LIDAR] Copied: {copied}, existing/skipped: {skipped}, "
            f"bad/skipped: {bad}, unparsed filename: {unparsed}"
        )

    csv_groups = []

    if COPY_NAV:
        csv_groups.append(("NAV", input_root / "OTHERS" / "NAV"))

    if COPY_IMU:
        csv_groups.append(("IMU", input_root / "OTHERS" / "IMU"))

    if COPY_VEHICLE_INFO:
        csv_groups.append(("VEHICLE_INFO", input_root / "OTHERS" / "VEHICLE_INFO"))

    if COPY_VEHICLE_STEER:
        csv_groups.append(("VEHICLE_STEER", input_root / "OTHERS" / "VEHICLE_STEER"))

    for name, folder in csv_groups:
        print(f"\n[{name}] Đang xử lý CSV...")
        written, skipped, bad, filtered, rows = process_csv_group(
            folder,
            input_root,
            output_root,
            start,
            end,
        )
        stats[f"{name} written"] = written
        stats[f"{name} existing"] = skipped
        stats[f"{name} bad"] = bad
        print(
            f"[{name}] CSV written: {written}, existing/skipped: {skipped}, "
            f"bad/skipped: {bad}, filtered files: {filtered}, rows kept: {rows}"
        )

    if COPY_RADAR:
        print("\n[RADAR] Đang xử lý...")
        copied, skipped, bad, unparsed = copy_timestamped_files(
            input_root / "RADAR",
            input_root,
            output_root,
            start,
            end,
        )
        stats["RADAR copied"] = copied
        stats["RADAR existing"] = skipped
        stats["RADAR bad"] = bad
        print(
            f"[RADAR] Copied: {copied}, existing/skipped: {skipped}, "
            f"bad/skipped: {bad}, unparsed filename: {unparsed}"
        )

    if COPY_ULTRASONIC:
        print("\n[ULTRASONIC] Đang xử lý...")
        copied, skipped, bad, unparsed = copy_timestamped_files(
            input_root / "ULTRASONIC",
            input_root,
            output_root,
            start,
            end,
        )
        stats["ULTRASONIC copied"] = copied
        stats["ULTRASONIC existing"] = skipped
        stats["ULTRASONIC bad"] = bad
        print(
            f"[ULTRASONIC] Copied: {copied}, existing/skipped: {skipped}, "
            f"bad/skipped: {bad}, unparsed filename: {unparsed}"
        )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)

    for name, value in stats.items():
        print(f"{name:16s}: {value}")

    total_files = count_files(output_root)
    total_size = get_size_bytes(output_root)

    print("-" * 72)
    print(f"Total files : {total_files}")
    print(f"Total size  : {human_size(total_size)}")
    print(f"Output      : {output_root}")
    print("=" * 72)

    print("\nKiểm tra nhanh:")
    print(f'  find "{output_root}" -type f | wc -l')
    print(f'  du -sh "{output_root}"')


if __name__ == "__main__":
    main()
