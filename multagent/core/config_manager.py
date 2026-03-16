"""
Configuration management.
Loads and merges YAML configuration files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self._config_dir = Path(config_dir)
        self._config: Dict[str, Any] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all YAML files from config directory."""
        if not self._config_dir.exists():
            logger.warning(f"Config directory not found: {self._config_dir}")
            return

        for yaml_file in sorted(self._config_dir.glob("*.yaml")):
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f) or {}
                key = yaml_file.stem  # e.g. "default", "agents", "hardware"
                self._config[key] = data
                logger.info(f"Loaded config: {yaml_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")

    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """
        Get config value.
        Examples:
            config.get("agents", "perception.confidence_threshold", 0.5)
            config.get("default", "edge_bridge.port", 8765)
        """
        data = self._config.get(section, {})
        if key is None:
            return data

        keys = key.split(".")
        for k in keys:
            if isinstance(data, dict):
                data = data.get(k)
            else:
                return default
            if data is None:
                return default
        return data

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire config section."""
        return self._config.get(section, {})

    def reload(self) -> None:
        """Reload all configurations from disk."""
        self._config.clear()
        self._load_all()

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a config value at runtime (does not persist to disk)."""
        if section not in self._config:
            self._config[section] = {}

        keys = key.split(".")
        data = self._config[section]
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value
