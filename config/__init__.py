from pathlib import Path
from typing import Any

import yaml
from yacs.config import CfgNode as CN

from .defaults import _C as _CFG_TEMPLATE

cfg = _CFG_TEMPLATE.clone()


def _safe_load_yaml_dict(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file_handle:
        raw = yaml.safe_load(file_handle)

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping at root: {config_path}")
    return raw


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_config_dict(config_path: Path, chain: list[Path] | None = None) -> dict[str, Any]:
    resolved_path = config_path.resolve()
    chain = chain or []

    if resolved_path in chain:
        cycle = " -> ".join(str(path) for path in [*chain, resolved_path])
        raise ValueError(f"Detected _base_ cycle in config inheritance: {cycle}")

    raw = _safe_load_yaml_dict(resolved_path)
    base_entry = raw.pop("_base_", None)

    merged: dict[str, Any] = {}
    if base_entry:
        if isinstance(base_entry, str):
            base_entries = [base_entry]
        elif isinstance(base_entry, list):
            base_entries = base_entry
        else:
            raise ValueError(
                f"_base_ must be a string or list of strings: {resolved_path}"
            )

        for base_item in base_entries:
            base_path = (resolved_path.parent / base_item).resolve()
            base_cfg = _resolve_config_dict(base_path, chain=[*chain, resolved_path])
            merged = _deep_merge_dicts(merged, base_cfg)

    return _deep_merge_dicts(merged, raw)


def load_config(config_path: str):
    """Load configuration with recursive _base_ inheritance into global cfg."""
    config_file = Path(config_path).resolve()
    resolved = _resolve_config_dict(config_file)

    cfg.defrost()
    cfg.merge_from_other_cfg(_CFG_TEMPLATE.clone())
    cfg.merge_from_other_cfg(CN(resolved))
    return cfg


def save_resolved_config(active_cfg, source_config_path: str, output_dir: str | None = None):
    """Persist the fully resolved runtime config for reproducibility."""
    destination_dir = Path(output_dir or active_cfg.OUTPUT_DIR)
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination_path = destination_dir / "resolved_config.yml"
    with destination_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write(f"# source_config: {Path(source_config_path).resolve()}\n")
        file_handle.write(active_cfg.dump())

    return destination_path