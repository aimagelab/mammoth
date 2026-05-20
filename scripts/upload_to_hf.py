#!/usr/bin/env python3
"""
Upload local files to a Hugging Face repository.

Generic uploader that can be used for checkpoints, Fisher caches, logs, or any
other artifacts.

Examples:
  # Upload all .pt files from a directory to a model repo folder
  uv run python scripts/upload_to_hf.py \
    --repo-id your-user/your-repo \
    --local-dir /path/to/files \
    --remote-dir artifacts/fisher \
    --pattern "*.pt"

  # Dry run
  uv run python scripts/upload_to_hf.py \
    --repo-id your-user/your-repo \
    --local-dir /path/to/files \
    --pattern "**/*" \
    --dry-run

  # Upload folder in a single commit (avoids per-file commit limits)
  uv run python scripts/upload_to_hf.py \
    --repo-id your-user/your-dataset \
    --repo-type dataset \
    --local-dir data/your_dataset \
    --pattern "**/*" \
    --upload-mode folder

  # Upload large folder with resilient uploader
  uv run python scripts/upload_to_hf.py \
    --repo-id your-user/your-dataset \
    --repo-type dataset \
    --local-dir data/your_dataset \
    --pattern "**/*" \
    --upload-mode large-folder

  # Upload folder in batches (multiple commits)
  uv run python scripts/upload_to_hf.py \
    --repo-id your-user/your-dataset \
    --repo-type dataset \
    --local-dir data/your_dataset \
    --pattern "images/**" \
    --upload-mode folder \
    --batch-size 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload local files to Hugging Face Hub"
    )
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. user/repo")
    parser.add_argument(
        "--local-dir", required=True, help="Local directory containing files to upload"
    )
    parser.add_argument(
        "--remote-dir", default="", help="Target folder inside the HF repo"
    )
    parser.add_argument(
        "--pattern", default="**/*", help="Glob pattern relative to local-dir"
    )
    parser.add_argument("--revision", default="main", help="Target branch/revision")
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="HF repo type",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private if it does not exist",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List files without uploading"
    )
    parser.add_argument(
        "--upload-mode",
        default="per-file",
        choices=["per-file", "folder", "large-folder"],
        help="Upload strategy: per-file (default), folder, or large-folder",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers for --upload-mode large-folder",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="When > 0 and --upload-mode=folder, split uploads into commit batches of this size",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude glob pattern(s), can be passed multiple times",
    )
    return parser.parse_args()


def should_exclude(
    file_path: Path, relative_posix: str, exclude_patterns: list[str]
) -> bool:
    for pattern in exclude_patterns:
        if file_path.match(pattern) or Path(relative_posix).match(pattern):
            return True
    return False


def main() -> None:
    args = parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError(
            "Please install huggingface_hub: pip install huggingface_hub"
        ) from e

    local_dir = Path(args.local_dir)
    if not local_dir.exists() or not local_dir.is_dir():
        raise ValueError(f"Local directory not found: {local_dir}")
    if args.batch_size < 0:
        raise ValueError("--batch-size must be >= 0")

    files = sorted([p for p in local_dir.glob(args.pattern) if p.is_file()])
    selected_files = []
    for file_path in files:
        rel = file_path.relative_to(local_dir).as_posix()
        if should_exclude(file_path, rel, args.exclude):
            continue
        selected_files.append((file_path, rel))

    if not selected_files:
        raise ValueError(
            f"No files matched pattern `{args.pattern}` in `{local_dir}` after exclusions"
        )

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    print(f"Found {len(selected_files)} files to upload")

    if args.upload_mode == "per-file":
        for file_path, rel in selected_files:
            path_in_repo = (
                rel if not args.remote_dir else f"{args.remote_dir.rstrip('/')}/{rel}"
            )
            if args.dry_run:
                print(f"[DRY-RUN] {file_path} -> {args.repo_id}:{path_in_repo}")
                continue

            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
            )
            print(f"Uploaded: {file_path} -> {path_in_repo}")
    elif args.upload_mode == "folder":
        if args.batch_size > 0:
            total_batches = (len(selected_files) + args.batch_size - 1) // args.batch_size
            if args.dry_run:
                print(
                    f"[DRY-RUN] upload folder in {total_batches} batches, "
                    f"batch_size={args.batch_size}, remote_dir={args.remote_dir or '<root>'}"
                )
            else:
                from huggingface_hub import CommitOperationAdd

            for batch_idx in range(total_batches):
                start = batch_idx * args.batch_size
                end = min((batch_idx + 1) * args.batch_size, len(selected_files))
                chunk = selected_files[start:end]

                if args.dry_run:
                    print(
                        f"[DRY-RUN] batch {batch_idx + 1}/{total_batches}: "
                        f"{len(chunk)} files"
                    )
                    continue

                operations = []
                for file_path, rel in chunk:
                    path_in_repo = (
                        rel if not args.remote_dir else f"{args.remote_dir.rstrip('/')}/{rel}"
                    )
                    operations.append(
                        CommitOperationAdd(
                            path_in_repo=path_in_repo,
                            path_or_fileobj=str(file_path),
                        )
                    )

                api.create_commit(
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    revision=args.revision,
                    operations=operations,
                    commit_message=(
                        f"Upload batch {batch_idx + 1}/{total_batches} "
                        f"({start + 1}-{end} of {len(selected_files)})"
                    ),
                )
                print(
                    f"Uploaded batch {batch_idx + 1}/{total_batches}: "
                    f"{len(chunk)} files"
                )
        else:
            if args.dry_run:
                print(
                    f"[DRY-RUN] upload_folder local_dir={local_dir} pattern={args.pattern} "
                    f"exclude={args.exclude} remote_dir={args.remote_dir or '<root>'}"
                )
            else:
                api.upload_folder(
                    repo_id=args.repo_id,
                    folder_path=str(local_dir),
                    path_in_repo=args.remote_dir or None,
                    repo_type=args.repo_type,
                    revision=args.revision,
                    allow_patterns=args.pattern,
                    ignore_patterns=args.exclude or None,
                )
                print(
                    f"Uploaded folder: {local_dir} -> {args.repo_id}:{args.remote_dir or '/'} "
                    f"(pattern={args.pattern})"
                )
    else:
        if args.dry_run:
            print(
                f"[DRY-RUN] upload_large_folder local_dir={local_dir} pattern={args.pattern} "
                f"exclude={args.exclude}"
            )
        else:
            api.upload_large_folder(
                repo_id=args.repo_id,
                folder_path=str(local_dir),
                repo_type=args.repo_type,
                revision=args.revision,
                allow_patterns=args.pattern,
                ignore_patterns=args.exclude or None,
                num_workers=args.num_workers,
                print_report=True,
            )
            print(
                f"Uploaded large folder: {local_dir} -> {args.repo_id}:{args.remote_dir or '/'} "
                f"(pattern={args.pattern})"
            )

    if args.dry_run:
        print("Dry run complete, no files uploaded.")
    else:
        if args.repo_type == "dataset":
            print(f"Upload complete: https://huggingface.co/datasets/{args.repo_id}")
        elif args.repo_type == "space":
            print(f"Upload complete: https://huggingface.co/spaces/{args.repo_id}")
        else:
            print(f"Upload complete: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
