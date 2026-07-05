"""Configuration loading and management utilities."""

import os
import yaml
import copy
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).parent
_DEFAULT_CONFIG = _CONFIG_DIR / "default.yaml"
_EXPERIMENTS_DIR = _CONFIG_DIR / "experiments"
_PROJECT_ROOT = _CONFIG_DIR.parent


class Config:
    """Dot-accessible, nested configuration wrapper.

    Allows attribute-style access to configuration values:
        cfg = Config({"model": {"lr": 0.01}})
        cfg.model.lr  # -> 0.01

    Supports merging, deep-copy, and YAML serialization.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        data = data or {}
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, list):
                setattr(self, key, [
                    Config(item) if isinstance(item, dict) else item
                    for item in value
                ])
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a plain dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, Config) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Safe attribute access with default."""
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    experiment: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load configuration.

    1. Start from ``config/default.yaml``.
    2. If *experiment* is given, merge ``config/experiments/<experiment>.yaml``
       on top.
    3. Apply any *overrides* dict on top.

    Parameters
    ----------
    experiment : str, optional
        Name of experiment config (without ``.yaml`` extension).
    overrides : dict, optional
        Key-value overrides applied last.

    Returns
    -------
    Config
        Fully-merged configuration object.
    """
    # 1. Load defaults
    with open(_DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}

    # 2. Merge experiment config
    if experiment:
        exp_path = _EXPERIMENTS_DIR / f"{experiment}.yaml"
        if exp_path.exists():
            with open(exp_path, "r", encoding="utf-8") as fh:
                exp_data = yaml.safe_load(fh) or {}
            base = _deep_merge(base, exp_data)
        else:
            print(f"[Config] Warning: experiment config '{exp_path}' not found, using defaults.")

    # 3. Apply overrides
    if overrides:
        base = _deep_merge(base, overrides)

    # Inject project root
    base["project_root"] = str(_PROJECT_ROOT)

    return Config(base)


def save_config(config: Config, path: str) -> None:
    """Save a Config object to a YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(config.to_dict(), fh, default_flow_style=False, indent=2)


def get_project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT
