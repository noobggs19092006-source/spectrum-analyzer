"""
service/anomaly.py
==================
Per-band power-baseline anomaly detector.

Contract (spec §3H):
  - Bands: 1 MHz buckets keyed by centre frequency (round to nearest MHz).
  - EWM update: alpha = 0.01  (applied to power_dbfs).
  - Flag anomaly when power_dbfs > mean + 6 * sigma for 3 consecutive bursts.
  - Ignore first 200 observations per bucket (warm-up).

Usage:
    from service.anomaly import AnomalyDetector

    detector = AnomalyDetector()
    flag = detector.update(center_hz=915_000_000, power_dbfs=-45.3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


_ALPHA: float = 0.01      # EWM smoothing factor
_SIGMA_THRESH: float = 6.0
_CONSEC_NEEDED: int = 3
_WARMUP: int = 200
_BUCKET_HZ: int = 1_000_000   # 1 MHz bucket width


def _bucket_key(center_hz: int) -> int:
    """Round centre frequency to nearest 1 MHz bucket."""
    return int(round(center_hz / _BUCKET_HZ)) * _BUCKET_HZ


@dataclass
class _BandState:
    count: int = 0           # observations received
    mean:  float = 0.0       # EWM mean of power_dbfs
    var:   float = 1.0       # EWM variance (initialise > 0 to avoid /0)
    consec_above: int = 0    # consecutive exceedances


class AnomalyDetector:
    """
    Thread-safe (per-object) EWM anomaly detector.

    Call ``update(center_hz, power_dbfs)`` for every incoming burst header.
    Returns ``True`` when an anomaly is confirmed.
    """

    def __init__(
        self,
        alpha: float = _ALPHA,
        sigma_thresh: float = _SIGMA_THRESH,
        consec_needed: int = _CONSEC_NEEDED,
        warmup: int = _WARMUP,
    ) -> None:
        self._alpha = alpha
        self._sigma_thresh = sigma_thresh
        self._consec_needed = consec_needed
        self._warmup = warmup
        self._bands: dict[int, _BandState] = {}

    def _get_or_create(self, key: int) -> _BandState:
        if key not in self._bands:
            self._bands[key] = _BandState()
        return self._bands[key]

    def update(self, center_hz: int, power_dbfs: float) -> bool:
        """
        Ingest one burst observation and return anomaly flag.

        Parameters
        ----------
        center_hz  : burst centre frequency (Hz)
        power_dbfs : burst power in dBFS

        Returns
        -------
        bool  True if an anomaly is active for this band.
        """
        key = _bucket_key(center_hz)
        st = self._get_or_create(key)
        st.count += 1

        # ── Warm-up: no anomaly detection during warm-up ──────────────────
        is_anomaly = False
        if st.count > self._warmup:
            sigma = math.sqrt(max(st.var, 0.0))
            threshold = st.mean + self._sigma_thresh * sigma
            if power_dbfs > threshold:
                st.consec_above += 1
            else:
                st.consec_above = 0
            is_anomaly = (st.consec_above >= self._consec_needed)

        # ── EWM baseline update (update clean background baseline) ────────
        if st.count == 1:
            st.mean = power_dbfs
            st.var = 1.0
        elif st.count <= self._warmup or st.consec_above == 0:
            diff = power_dbfs - st.mean
            st.mean += self._alpha * diff
            st.var = (1.0 - self._alpha) * (st.var + self._alpha * diff * diff)

        return is_anomaly

    def stats(self, center_hz: int) -> dict:
        """Return current EWM stats for a band (for debugging/logging)."""
        key = _bucket_key(center_hz)
        if key not in self._bands:
            return {}
        st = self._bands[key]
        return {
            "center_hz": key,
            "count": st.count,
            "ewm_mean": st.mean,
            "ewm_std": math.sqrt(max(st.var, 0.0)),
            "consec_above": st.consec_above,
        }

    def reset(self, center_hz: int | None = None) -> None:
        """Reset one band or all bands."""
        if center_hz is None:
            self._bands.clear()
        else:
            self._bands.pop(_bucket_key(center_hz), None)
