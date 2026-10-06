"""
Build Phenikaa frame index (infos) for MapTR-style inference.

Output: data/phenikaa/infos_residential.pkl
Does NOT create lane GT (Phenikaa labels are object boxes only).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from phenikaa_hdmap.paths import (
    CAMERAS,
    OUTPUT_ROOT,
    PHENIKAA_DATA,
    SEQUENCE_FOLDER,
    SEQUENCE_NAME,
    collect_timestamps,
)
from phenikaa_hdmap.utils.pose import load_pose_3x4, translation


def build_infos(sequence_folder: Path = SEQUENCE_FOLDER) -> list[dict]:
    stamps = collect_timestamps(sequence_folder)
    infos: list[dict] = []
    for ts in stamps:
        pose_path = sequence_folder / "Pose" / f"{ts}.txt"
        pose = load_pose_3x4(pose_path)
        info = {
            "token": ts,
            "timestamp": ts,
            "sequence": SEQUENCE_NAME,
            "lidar_path": str(sequence_folder / "Lidar" / f"{ts}.laz"),
            "pose_path": str(pose_path),
            "ego_translation": translation(pose).tolist(),
            "cams": {
                cam: str(sequence_folder / "Image" / cam / f"{ts}.jpg")
                for cam in CAMERAS
            },
            "label_path": str(sequence_folder / "Label" / f"{ts}.txt"),
            # Map GT intentionally absent for Phenikaa residential dump.
            "has_map_gt": False,
        }
        infos.append(info)
    return infos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PHENIKAA_DATA / "infos_residential.pkl",
        help="Output pickle path",
    )
    args = parser.parse_args()

    infos = build_infos()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"infos": infos, "metadata": {"sequence": SEQUENCE_NAME}}, f)

    meta = OUTPUT_ROOT / "phenikaa_vis" / "infos_summary.txt"
    meta.parent.mkdir(parents=True, exist_ok=True)
    with open(meta, "w") as f:
        f.write(f"sequence={SEQUENCE_NAME}\nframes={len(infos)}\n")
        if infos:
            f.write(f"first={infos[0]['token']}\nlast={infos[-1]['token']}\n")

    print(f"Wrote {len(infos)} frames → {args.out}")
    print(f"Summary → {meta}")


if __name__ == "__main__":
    main()
