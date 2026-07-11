"""Persistence for `BasanosStream` — save/load of the full stream state.

The archive layout is versioned by `_SAVE_FORMAT_VERSION`.  `load_stream_archive`
validates the version and the presence of every required key before
deserialising anything, so a stale or hand-edited archive fails with a
descriptive error rather than a bare ``KeyError``.

These are free functions (rather than `BasanosStream` methods) so the
serialisation format depends only on the state layout in
:mod:`basanos.math._stream_state`, not on the stream façade.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import numpy as np

from ..exceptions import StreamStateCorruptError
from ._config import BasanosConfig
from ._stream_state import _REQUIRED_KEYS, _SAVE_FORMAT_VERSION, _StreamState


def save_stream_archive(
    cfg: BasanosConfig,
    assets: list[str],
    state: _StreamState,
    path: str | os.PathLike[str],
) -> None:
    """Serialise a stream's config, assets, and state to a ``.npz`` archive.

    All `_StreamState` arrays, the configuration, and the asset
    list are written in a single `np.savez` call.  A stream
    restored via `load_stream_archive` produces bit-for-bit identical
    `BasanosStream.step` output.

    Args:
        cfg: The stream configuration to serialise.
        assets: Ordered asset column names.
        state: The mutable stream state carrier to serialise.
        path: Destination file path.  `np.savez` appends
            ``.npz`` automatically when the suffix is absent.
    """
    # Build the per-field dict automatically from _StreamState so that any
    # new field added to the dataclass is included without manual updates.
    state_arrays: dict[str, Any] = {}
    for field in dataclasses.fields(_StreamState):
        value = getattr(state, field.name)
        if field.name in ("sw_ret_buf", "corr_ret_buf"):
            # Sentinel: use an empty (0, 0) array to represent None so the
            # key is always present in the archive and load() can detect it.
            state_arrays[field.name] = value if value is not None else np.empty((0, 0), dtype=float)
        elif field.name == "step_count":
            state_arrays[field.name] = np.array(value)
        else:
            state_arrays[field.name] = value
    np.savez(
        path,
        format_version=np.array(_SAVE_FORMAT_VERSION),
        cfg_json=np.array(cfg.model_dump_json()),
        assets=np.array(assets),
        **state_arrays,
    )


def load_stream_archive(
    path: str | os.PathLike[str],
) -> tuple[BasanosConfig, list[str], _StreamState]:
    """Restore a stream's ``(cfg, assets, state)`` from a saved ``.npz`` archive.

    Args:
        path: Path to a ``.npz`` archive written by `save_stream_archive`.

    Returns:
        A ``(cfg, assets, state)`` tuple whose reconstructed state reproduces
        the original stream bit-for-bit at the time it was saved.

    Raises:
        ValueError: If the archive is missing its format-version tag or was
            written with an incompatible format version.
        StreamStateCorruptError: If a required key is absent from the archive.
    """
    with np.load(path, allow_pickle=False) as data:
        if "format_version" not in data:
            raise ValueError(  # noqa: TRY003
                "Stream file is missing a format version tag. "
                "It was written with an incompatible version of BasanosStream. "
                "Re-generate it via BasanosStream.from_warmup()."
            )
        found = int(data["format_version"])
        if found != _SAVE_FORMAT_VERSION:
            raise ValueError(  # noqa: TRY003
                f"Stream file was written with format version {found}, "
                f"but the current version is {_SAVE_FORMAT_VERSION}. "
                "Re-generate it via BasanosStream.from_warmup()."
            )
        # Validate that every required key is present.  This catches archives
        # that were produced by an older codebase missing a newly added field,
        # or archives that have been manually edited, with a descriptive error
        # instead of a bare KeyError.
        archive_keys = frozenset(data.files)
        missing = _REQUIRED_KEYS - archive_keys
        if missing:
            raise StreamStateCorruptError(missing)
        cfg = BasanosConfig.model_validate_json(data["cfg_json"].item())
        assets: list[str] = list(data["assets"])
        state_kwargs: dict[str, Any] = {}
        for field in dataclasses.fields(_StreamState):
            raw = data[field.name]
            if field.name in ("sw_ret_buf", "corr_ret_buf"):
                state_kwargs[field.name] = raw if raw.size > 0 else None
            elif field.name == "step_count":
                state_kwargs[field.name] = int(raw)
            else:
                state_kwargs[field.name] = raw
    state = _StreamState(**state_kwargs)
    return cfg, assets, state
