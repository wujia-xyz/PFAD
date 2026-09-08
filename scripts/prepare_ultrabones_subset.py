#!/usr/bin/env python3
"""Extract the minimum UltraBones100k inputs needed by the active method.

The official dataset is distributed as one large ZIP archive per specimen.
This utility reads those archives through Hugging Face's seekable filesystem
and materializes only:

* CT bone meshes;
* per-record tracking tables; and
* released ``Labels_pred`` masks.

Raw B-mode images, CT-derived 2-D labels, pretrained checkpoints, and released
3-D reconstructions are not inputs to the active refinement method and are not
downloaded.  Existing files with the expected uncompressed size are retained,
so an interrupted extraction can be resumed safely.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

from huggingface_hub import HfFileSystem


REPOSITORY = "datasets/luohwu/UltraBones100k"
SPECIMENS = tuple(f"specimen{index:02d}" for index in range(1, 15))
ANATOMIES = ("fibula", "foot", "tibia")


def selected_member(info: ZipInfo, anatomies: set[str]) -> bool:
    path = PurePosixPath(info.filename.replace("\\", "/"))
    parts = path.parts
    if info.is_dir() or not parts:
        return False
    if len(parts) == 3 and parts[1] == "CT_bone_segmentations":
        return path.suffix.lower() == ".stl"
    try:
        ultrasound_index = parts.index("ultrasound_records")
    except ValueError:
        return False
    remaining = parts[ultrasound_index + 1 :]
    if len(remaining) < 3 or remaining[0] not in anatomies:
        return False
    if remaining[-1] == "tracking.csv":
        return True
    return (
        "Labels_pred" in remaining
        and path.name.endswith("_label_pred.png")
    )


def safe_destination(root: Path, member: str) -> Path:
    relative = PurePosixPath(member.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe ZIP member: {member}")
    return root.joinpath(*relative.parts)


def extract_specimen(
    filesystem: HfFileSystem,
    specimen: str,
    anatomies: set[str],
    output_root: Path,
    block_size: int,
) -> None:
    remote = f"{REPOSITORY}/{specimen}.zip"
    copied_files = 0
    copied_bytes = 0
    skipped_files = 0
    with filesystem.open(remote, "rb", block_size=block_size) as source:
        with ZipFile(source) as archive:
            members = [
                info
                for info in archive.infolist()
                if selected_member(info, anatomies)
            ]
            for index, info in enumerate(members, start=1):
                destination = safe_destination(output_root, info.filename)
                if destination.is_file() and destination.stat().st_size == info.file_size:
                    skipped_files += 1
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as compressed, destination.open("wb") as target:
                    shutil.copyfileobj(compressed, target, length=1024 * 1024)
                if destination.stat().st_size != info.file_size:
                    raise IOError(f"incomplete extraction: {destination}")
                copied_files += 1
                copied_bytes += info.file_size
                if index % 500 == 0 or index == len(members):
                    print(
                        f"{specimen}: {index}/{len(members)} members; "
                        f"copied {copied_bytes / 1e9:.2f} GB",
                        flush=True,
                    )
    print(
        f"{specimen}: complete; copied={copied_files}, skipped={skipped_files}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/UltraBones100k"
        ),
    )
    parser.add_argument(
        "--specimen",
        action="append",
        choices=SPECIMENS,
        help="repeat to select specimens; default: all 14",
    )
    parser.add_argument(
        "--anatomy",
        action="append",
        choices=ANATOMIES,
        help="repeat to select anatomies; default: all three",
    )
    parser.add_argument("--block-size-mb", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    specimens = tuple(args.specimen or SPECIMENS)
    anatomies = set(args.anatomy or ANATOMIES)
    if args.block_size_mb <= 0:
        raise ValueError("block size must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    filesystem = HfFileSystem()
    for specimen in specimens:
        extract_specimen(
            filesystem,
            specimen,
            anatomies,
            args.output_root,
            args.block_size_mb * 1024 * 1024,
        )


if __name__ == "__main__":
    main()
