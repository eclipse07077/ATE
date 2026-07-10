from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "shape": value.shape, "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot encode {type(value)!r}")


def canonical_bytes(value: Any) -> bytes:
    """Return stable bytes for hashes over arrays and JSON-like records."""
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        header = json.dumps(
            {"kind": "ndarray", "shape": arr.shape, "dtype": str(arr.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return header + b"\0" + arr.view(np.uint8).tobytes()
    if np.isscalar(value):
        arr = np.ascontiguousarray(np.asarray(value))
        header = json.dumps(
            {"kind": "scalar", "shape": arr.shape, "dtype": str(arr.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return header + b"\0" + arr.view(np.uint8).tobytes()
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_file_set(paths: list[str | Path], root: str | Path | None = None) -> str:
    root_path = Path(root).resolve() if root is not None else None
    records = []
    for item in sorted(Path(p).resolve() for p in paths):
        if not item.exists():
            records.append({"path": str(item), "sha256": "missing"})
            continue
        label = str(item.relative_to(root_path)) if root_path else str(item)
        records.append({"path": label.replace("\\", "/"), "sha256": file_sha256(item)})
    return sha256_hex(records)

