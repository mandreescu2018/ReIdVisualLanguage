import os
import tempfile
import yaml
from pathlib import Path
from .defaults import _C as cfg


def load_config(config_path: str):
    """Load a YAML config with optional single-level _base_ inheritance.

    Resolves _base_ relative to the config file's directory, merges the base
    first, then applies the overrides in config_path on top.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    base_key = raw.pop('_base_', None)
    if base_key:
        base_file = Path(config_path).parent / base_key
        cfg.merge_from_file(str(base_file))

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as tf:
        yaml.dump(raw, tf)
        temp_path = tf.name

    try:
        cfg.merge_from_file(temp_path)
    finally:
        os.unlink(temp_path)

    return cfg