from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a portable archive of a training run directory.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--format", choices=["zip", "gztar"], default="zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    if args.output_path:
        output_path = Path(args.output_path).resolve()
        archive_base = str(output_path.with_suffix(""))
    else:
        suffix = ".zip" if args.format == "zip" else ".tar.gz"
        output_path = source_dir.parent / f"{source_dir.name}{suffix}"
        archive_base = str(output_path.with_suffix(""))
        if args.format == "gztar":
            archive_base = str(output_path).removesuffix(".tar.gz")

    archive_file = shutil.make_archive(
        archive_base,
        args.format,
        root_dir=source_dir.parent,
        base_dir=source_dir.name,
    )
    print(archive_file)


if __name__ == "__main__":
    main()
