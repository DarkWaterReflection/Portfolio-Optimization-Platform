"""Content-addressed Parquet cache for price panels.

A request is keyed by ``(source, tickers, start, end, field)`` hashed to a short
digest, so identical requests return instantly and offline runs are fully
reproducible. Market data is never committed to git (see ``.gitignore``).
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd

from portfolio_optimization.logging_setup import get_logger

logger = get_logger("data.cache")


class PriceCache:
    """Reads/writes wide price panels as Parquet, keyed by request content."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(source: str, tickers: list[str], start: date, end: date, field: str) -> str:
        payload = "|".join(
            [source, ",".join(sorted(tickers)), start.isoformat(), end.isoformat(), field]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def get(
        self, source: str, tickers: list[str], start: date, end: date, field: str
    ) -> pd.DataFrame | None:
        """Return a cached panel, or ``None`` on a miss."""
        path = self._path(self._key(source, tickers, start, end, field))
        if not path.exists():
            return None
        logger.info("Cache hit: %s", path.name)
        return pd.read_parquet(path)

    def put(
        self,
        source: str,
        tickers: list[str],
        start: date,
        end: date,
        field: str,
        prices: pd.DataFrame,
    ) -> None:
        """Persist a panel to the cache."""
        path = self._path(self._key(source, tickers, start, end, field))
        prices.to_parquet(path)
        logger.info("Cached %d rows x %d cols -> %s", prices.shape[0], prices.shape[1], path.name)
