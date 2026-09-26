"""Download one pinned Coletti BF16 snapshot onto a non-removable volume."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from exvram.storage import StorageSafetyError, assert_storage_safe

REPO_ID = "JonathanColetti/Qwen3.8-27B-Uncensored"
REVISION = "5bb7aa90f0efef548e87005b1fb7658e522b6b7f"
# GetDriveTypeW reported F: as fixed on 2026-09-25. The operator identified it
# as removable, so DriveType alone must not admit it.
OPERATOR_REMOVABLE_DRIVES = frozenset({"F:"})
MINIMUM_FREE_BYTES = 140 * 1024**3
DEFAULT_LOCAL_DIR = "models/sources/JonathanColetti-Qwen3.8-27B-Uncensored-5bb7aa90"
EXPECTED_SIZES = {
    ".gitattributes": 1570,
    "README.md": 7854,
    "chat_template.jinja": 8952,
    "config.json": 3688,
    "generation_config.json": 214,
    "model-00001-of-00012.safetensors": 2542796928,
    "model-00002-of-00012.safetensors": 4842451920,
    "model-00003-of-00012.safetensors": 4965227944,
    "model-00004-of-00012.safetensors": 4912819264,
    "model-00005-of-00012.safetensors": 4986198544,
    "model-00006-of-00012.safetensors": 4912819320,
    "model-00007-of-00012.safetensors": 4932703272,
    "model-00008-of-00012.safetensors": 4966314576,
    "model-00009-of-00012.safetensors": 4964162248,
    "model-00010-of-00012.safetensors": 4933789824,
    "model-00011-of-00012.safetensors": 4965228032,
    "model-00012-of-00012.safetensors": 2789094896,
    "model-mtp.safetensors": 849400424,
    "model.safetensors.index.json": 112085,
    "processor_config.json": 1191,
    "tokenizer.json": 19989325,
    "tokenizer_config.json": 1165,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument(
        "--manifest",
        default="experiments/manifests/coletti_qwen38_uncensored_bf16.json",
    )
    return parser


def _drive(path: Path) -> str:
    anchor = path.anchor
    if len(anchor) < 2 or anchor[1] != ":":
        raise StorageSafetyError(f"refusing a path without a drive letter: {path}")
    return anchor[:2].upper()


def _reject_forbidden_volume(path: Path) -> str:
    drive = _drive(path)
    if drive in OPERATOR_REMOVABLE_DRIVES:
        raise StorageSafetyError(
            f"refusing {path}: {drive} is operator-removable even if Windows "
            "reports DriveType fixed"
        )
    assert_storage_safe([path])
    return drive


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    local_dir = Path(args.local_dir)
    if not local_dir.is_absolute():
        local_dir = Path.cwd() / local_dir
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    try:
        drive = _reject_forbidden_volume(local_dir)
        _reject_forbidden_volume(manifest_path)
    except StorageSafetyError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    usage = shutil.disk_usage(local_dir.anchor)
    if usage.free < MINIMUM_FREE_BYTES:
        print(
            json.dumps(
                {
                    "error": "free space below the GGUF planning floor",
                    "free_bytes": usage.free,
                    "required_bytes": MINIMUM_FREE_BYTES,
                    "drive": drive,
                }
            ),
            file=sys.stderr,
        )
        return 2
    try:
        import huggingface_hub
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print(json.dumps({"error": str(exc), "interpreter": sys.executable}), file=sys.stderr)
        return 2
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO_ID,
        revision=REVISION,
        local_dir=str(local_dir),
    )
    files = []
    missing = []
    for name, expected in EXPECTED_SIZES.items():
        path = local_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        size = path.stat().st_size
        if size != expected:
            missing.append(f"{name}:size:{size}!={expected}")
            continue
        files.append({"name": name, "bytes": size, "sha256": _sha256(path)})
    if missing:
        print(json.dumps({"error": "source check failed", "missing": missing}), file=sys.stderr)
        return 2
    payload = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": "Apache-2.0",
        "local_dir": str(local_dir),
        "drive": drive,
        "free_bytes_before_download": usage.free,
        "immutable": True,
        "huggingface_hub": huggingface_hub.__version__,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "manifest": str(manifest_path), "files": len(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
