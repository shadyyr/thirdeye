"""Centralized configuration loading and validation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_config(path: str | Path = "config/default.json") -> dict[str, Any]:
    """Load the single JSON settings file used by all software adapters."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    validate_config(config)
    return config


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay a local configuration without mutating either input."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config_with_override(
    default_path: str | Path = "config/default.json", local_path: str | Path | None = None
) -> dict[str, Any]:
    base = load_config(default_path)
    if local_path is None:
        return base
    with Path(local_path).open(encoding="utf-8") as file:
        configured = merge_config(base, json.load(file))
    validate_config(configured)
    return configured


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "cameras",
        "detector",
        "tracking",
        "direction",
        "motion",
        "salience",
        "haptics",
        "simulation",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Configuration is missing required sections: {sorted(missing)}")

    direction = config["direction"]
    left, right = direction["left_boundary"], direction["right_boundary"]
    if not 0 <= left < right <= 1:
        raise ValueError("direction boundaries must satisfy 0 <= left < right <= 1")

    if not 0 <= config["detector"]["confidence_threshold"] <= 1:
        raise ValueError("detector confidence_threshold must be between 0 and 1")

    salience = config["salience"]
    if not 0 <= salience["alert_threshold"] <= 1:
        raise ValueError("salience alert_threshold must be between 0 and 1")
    if not 0 <= salience["minimum_intensity"] <= 100:
        raise ValueError("salience minimum_intensity must be between 0 and 100")
    if config["simulation"]["frame_interval_seconds"] < 0:
        raise ValueError("simulation frame_interval_seconds cannot be negative")
