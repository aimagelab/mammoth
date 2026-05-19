import os
import logging


def _missing_files(local_dir: str, required_relpaths: list[str]) -> list[str]:
    return [
        relpath for relpath in required_relpaths
        if not os.path.isfile(os.path.join(local_dir, relpath))
    ]


def download_dataset_snapshot(repo_id: str, local_dir: str, revision: str = "main") -> None:
    """Download a Hugging Face dataset repository snapshot into ``local_dir``."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download datasets automatically from HF. "
            "Install it with `pip install huggingface_hub`."
        ) from e

    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
    )


def download_dataset_snapshot_with_patterns(
    repo_id: str,
    local_dir: str,
    revision: str = "main",
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
) -> None:
    """Download a filtered HF dataset snapshot into ``local_dir``."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download datasets automatically from HF. "
            "Install it with `pip install huggingface_hub`."
        ) from e

    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


def download_dataset_file(
    repo_id: str,
    local_dir: str,
    filename: str,
    revision: str = "main",
) -> str:
    """Download a single file from a HF dataset repo into ``local_dir``."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download datasets automatically from HF. "
            "Install it with `pip install huggingface_hub`."
        ) from e

    os.makedirs(local_dir, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
    )
    return downloaded_path


def ensure_required_files_from_hf(
    local_dir: str,
    required_relpaths: list[str],
    repo_id: str,
    revision: str = "main",
) -> None:
    """Ensure required local files exist, downloading the HF snapshot if needed."""
    missing_before = _missing_files(local_dir, required_relpaths)
    if not missing_before:
        return

    logging.info(
        "Missing %d required files in `%s`. Downloading from `hf://%s@%s`.",
        len(missing_before),
        local_dir,
        repo_id,
        revision,
    )

    download_dataset_snapshot(repo_id=repo_id, local_dir=local_dir, revision=revision)

    missing_after = _missing_files(local_dir, required_relpaths)
    if missing_after:
        listed = ", ".join(missing_after[:10])
        if len(missing_after) > 10:
            listed += f" ... (+{len(missing_after) - 10} more)"
        raise FileNotFoundError(
            f"Could not find required files in `{local_dir}` after downloading `{repo_id}@{revision}`. "
            f"Missing: {listed}"
        )
