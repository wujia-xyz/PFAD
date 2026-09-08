#!/usr/bin/env python3
"""Download versioned study artifacts, verify SHA-256, and restore their paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=["models", "predictions", "results", "all"])
    parser.add_argument("--archive-dir", type=Path, help="Use previously downloaded release ZIPs.")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "paper_results/release_assets.json").read_text())
    selected = list(manifest["assets"]) if args.asset == "all" else [args.asset]
    folder = args.archive_dir or ROOT / "artifacts/downloads"
    folder.mkdir(parents=True, exist_ok=True)
    for name in selected:
        entry = manifest["assets"][name]
        archive = folder / entry["filename"]
        if not archive.exists():
            if args.archive_dir:
                raise FileNotFoundError(archive)
            url = manifest["release_url"] + "/" + entry["filename"]
            temporary = archive.with_suffix(".zip.partial")
            print(f"Downloading {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if sha256(temporary) != entry["sha256"]:
                raise RuntimeError(f"Checksum mismatch: {temporary}")
            temporary.replace(archive)
        if sha256(archive) != entry["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {archive}")
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                destination = (ROOT / member.filename).resolve()
                if not destination.is_relative_to(ROOT.resolve()) or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"Unsafe archive entry: {member.filename}")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                expected = entry["files"][member.filename]
                if destination.exists():
                    if sha256(destination) == expected:
                        continue
                    raise FileExistsError(f"Existing file differs; preserve or move it before restoring: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".partial")
                with bundle.open(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if sha256(temporary) != expected:
                    raise RuntimeError(f"Extracted checksum mismatch: {member.filename}")
                temporary.replace(destination)
        print(f"Verified and restored {name}: {len(entry['files'])} files", flush=True)


if __name__ == "__main__":
    main()
