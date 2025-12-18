#!/usr/bin/env python3
"""
Remove training samples whose saliency maps are missing.

This script scans the metadata file produced during data preparation
and filters out entries whose corresponding saliency maps are absent.
Optionally, the image files for those entries can be deleted as well.
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance-dir",
        type=Path,
        required=True,
        help="Directory containing the training images and metadata.jsonl.",
    )
    parser.add_argument(
        "--saliency-dir",
        type=Path,
        required=True,
        help="Directory containing saliency maps aligned with the training images.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata.jsonl (defaults to <instance-dir>/metadata.jsonl).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be removed without touching files.",
    )
    parser.add_argument(
        "--delete-images",
        action="store_true",
        help="Delete orphaned images when their saliency map is missing.",
    )
    return parser.parse_args()


def load_metadata(metadata_path: Path):
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield line, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON line in {metadata_path}: {exc}") from exc


def main() -> None:
    args = parse_args()
    instance_dir = args.instance_dir.resolve()
    saliency_dir = args.saliency_dir.resolve()
    metadata_path = (args.metadata or instance_dir / "metadata.jsonl").resolve()

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not saliency_dir.is_dir():
        raise FileNotFoundError(f"Saliency directory not found: {saliency_dir}")

    kept_lines = []
    removed = 0
    removed_saliency_missing = 0
    removed_image_missing = 0

    for raw_line, item in load_metadata(metadata_path):
        file_name = item.get("file_name")
        if not file_name:
            # Keep malformed entries for manual inspection.
            kept_lines.append(raw_line)
            continue

        image_path = instance_dir / file_name
        saliency_path = saliency_dir / file_name

        if not saliency_path.is_file():
            removed += 1
            removed_saliency_missing += 1
            if args.delete_images and image_path.is_file() and not args.dry_run:
                image_path.unlink()
            continue

        if not image_path.is_file():
            removed += 1
            removed_image_missing += 1
            continue

        kept_lines.append(raw_line)

    if not args.dry_run:
        backup_path = metadata_path.with_suffix(metadata_path.suffix + ".bak")
        if not backup_path.exists():
            metadata_path.replace(backup_path)
        else:
            backup_path.write_text(metadata_path.read_text(encoding="utf-8"), encoding="utf-8")
        with metadata_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(kept_lines))
            f.write("\n")

    print(f"Total entries processed: {removed + len(kept_lines)}")
    print(f"Removed entries: {removed}")
    print(f"  - Missing saliency: {removed_saliency_missing}")
    print(f"  - Missing image: {removed_image_missing}")
    print(f"Kept entries: {len(kept_lines)}")
    if args.dry_run:
        print("Dry run complete – no files were modified.")
    else:
        print(f"Updated metadata saved to {metadata_path}")
        print("Original metadata backed up next to the file with a .bak suffix.")


if __name__ == "__main__":
    main()
