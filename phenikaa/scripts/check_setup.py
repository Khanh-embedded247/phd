#!/usr/bin/env python3
"""Smoke-check Phenikaa project layout, symlinks, and basic data presence."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phenikaa_hdmap import paths as P  # noqa: E402


def status(ok: bool, label: str, detail: str = "") -> bool:
    mark = "OK" if ok else "MISSING"
    extra = f" — {detail}" if detail else ""
    print(f"[{mark:7}] {label}{extra}")
    return ok


def main() -> int:
    print(f"PROJECT_ROOT = {P.PROJECT_ROOT}\n")
    checks: list[bool] = []

    checks.append(status(P.THIRD_PARTY_MAPTR.is_dir(), "third_party/MapTR", str(P.THIRD_PARTY_MAPTR)))
    checks.append(status(P.TUTORIALS_ROOT.is_dir(), "tutorials", str(P.TUTORIALS_ROOT)))
    checks.append(status(P.INTRINSIC_JSON.is_file(), "Camera_Intrinsics.json", str(P.INTRINSIC_JSON)))
    checks.append(status(P.EXTRINSIC_JSON.is_file(), "Sensor_Extrinsics.json", str(P.EXTRINSIC_JSON)))
    checks.append(status(P.SEQUENCE_FOLDER.is_dir(), "RESIDENTIAL_AREA sequence", str(P.SEQUENCE_FOLDER)))

    for sub in ("Image", "Lidar", "Pose", "Label"):
        checks.append(status((P.SEQUENCE_FOLDER / sub).is_dir(), f"  sequence/{sub}"))

    try:
        stamps = P.collect_timestamps(P.SEQUENCE_FOLDER)
        checks.append(status(True, "timestamps", f"{len(stamps)} frames ({stamps[0]} → {stamps[-1]})"))
    except Exception as exc:  # noqa: BLE001
        checks.append(status(False, "timestamps", str(exc)))

    nusc = P.NUSCENES_DATA / "raw"
    checks.append(status(nusc.is_dir(), "nuScenes raw", str(nusc)))

    map_ok = any(p.is_file() for p in P.MAP_CANDIDATES)
    checks.append(status(map_ok, "map PCD candidates (optional/unused)", ", ".join(str(p.name) for p in P.MAP_CANDIDATES)))
    checks.append(
        status(
            P.RESIDENTIAL_MAP_PCD.is_file(),
            "residential.pcd (B6 only)",
            "optional until B6",
        )
    )

    print()
    # Candidates + residential.pcd are optional for B0–B5
    optional_missing = 0
    if not map_ok:
        optional_missing += 1
    if not P.RESIDENTIAL_MAP_PCD.is_file():
        optional_missing += 1
    failed = sum(1 for c in checks if not c)
    critical_failed = failed - optional_missing
    if critical_failed <= 0 and (P.INTRINSIC_JSON.is_file() and P.SEQUENCE_FOLDER.is_dir()):
        print("Setup usable. Next: tutorials calib check → B1 MapTR → build infos.")
        return 0

    print(f"Setup incomplete ({failed} issues). See docs/DATA.md.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
