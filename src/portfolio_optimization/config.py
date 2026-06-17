"""Typed configuration loaded from YAML.

Settings are plain frozen dataclasses so they are hashable, easy to construct in
tests, and serialize cleanly. ``load_settings`` reads ``config/settings.yaml`` by
default; the path can be overridden with the ``PFOPT_CONFIG`` environment variable
or an explicit argument.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "settings.yaml"
_DEFAULT_UNIVERSES = _REPO_ROOT / "config" / "universes.yaml"


@dataclass(frozen=True)
class DataSettings:
    source: str = "yfinance"
    cache_dir: str = "data_cache"
    offline: bool = False
    max_ffill_days: int = 3
    max_missing_pct: float = 0.05


@dataclass(frozen=True)
class MarketSettings:
    periods_per_year: int = 252
    risk_free_rate: float = 0.02


@dataclass(frozen=True)
class EstimationSettings:
    return_model: str = "mean_historical"
    risk_model: str = "ledoit_wolf"


@dataclass(frozen=True)
class OptimizationSettings:
    long_only: bool = True
    w_min: float = 0.0
    w_max: float = 1.0
    frontier_points: int = 50


@dataclass(frozen=True)
class RandomSettings:
    seed: int = 42
    n_portfolios: int = 20000


@dataclass(frozen=True)
class Settings:
    data: DataSettings = field(default_factory=DataSettings)
    market: MarketSettings = field(default_factory=MarketSettings)
    estimation: EstimationSettings = field(default_factory=EstimationSettings)
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    random: RandomSettings = field(default_factory=RandomSettings)

    @property
    def repo_root(self) -> Path:
        return _REPO_ROOT

    @property
    def cache_path(self) -> Path:
        p = Path(self.data.cache_dir)
        return p if p.is_absolute() else _REPO_ROOT / p


def _filter_kwargs(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys that are actual fields of ``cls`` (forgiving of extras)."""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


def load_settings(path: str | Path | None = None) -> Settings:
    """Load :class:`Settings` from YAML, falling back to dataclass defaults.

    Resolution order: explicit ``path`` arg → ``PFOPT_CONFIG`` env var →
    bundled ``config/settings.yaml``. Missing sections use their defaults.
    """
    cfg_path = Path(path or os.environ.get("PFOPT_CONFIG", _DEFAULT_CONFIG))
    if not cfg_path.exists():
        return Settings()

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return Settings(
        data=DataSettings(**_filter_kwargs(DataSettings, raw.get("data", {}))),
        market=MarketSettings(**_filter_kwargs(MarketSettings, raw.get("market", {}))),
        estimation=EstimationSettings(
            **_filter_kwargs(EstimationSettings, raw.get("estimation", {}))
        ),
        optimization=OptimizationSettings(
            **_filter_kwargs(OptimizationSettings, raw.get("optimization", {}))
        ),
        random=RandomSettings(**_filter_kwargs(RandomSettings, raw.get("random", {}))),
    )


def load_universe(name: str, path: str | Path | None = None) -> list[str]:
    """Return the ticker list for a named universe from ``universes.yaml``."""
    uni_path = Path(path or _DEFAULT_UNIVERSES)
    raw = yaml.safe_load(uni_path.read_text(encoding="utf-8")) or {}
    if name not in raw:
        available = ", ".join(k for k in raw if k != "benchmarks")
        raise KeyError(f"Unknown universe {name!r}. Available: {available}")
    return list(raw[name])
