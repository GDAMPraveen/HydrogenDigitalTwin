"""Multi-day Wind + Solar PV renewable power profile generator."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_renewable_profile(
    start_date: str = "2026-11-01",
    days: int = 7,
    freq_minutes: int = 15,
    wind_rated_kw: float = 1200.0,
    solar_rated_kw: float = 800.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate timestamped Wind + Solar PV power (kW) with diurnal solar & gusty wind."""
    rng = np.random.default_rng(seed)
    steps_per_day = int((24 * 60) / freq_minutes)
    total_steps = days * steps_per_day

    timestamps = pd.date_range(start=start_date, periods=total_steps, freq=f"{freq_minutes}min")
    hours = timestamps.hour + timestamps.minute / 60.0

    # Smooth diurnal solar curve (6 AM to 6 PM) + random cloud transients
    solar_base = np.maximum(0.0, np.sin(np.pi * (hours - 6.0) / 12.0)) ** 1.4
    cloud_factor = np.clip(1.0 - rng.exponential(0.12, size=total_steps), 0.15, 1.0)
    solar_kw = solar_rated_kw * solar_base * cloud_factor

    # Correlated random-walk wind profile with gusts
    wind_raw = np.zeros(total_steps)
    wind_raw[0] = 0.45
    for i in range(1, total_steps):
        drift = 0.03 * (0.48 - wind_raw[i - 1])
        gust = rng.normal(0.0, 0.08)
        wind_raw[i] = np.clip(wind_raw[i - 1] + drift + gust, 0.02, 1.0)
    wind_kw = wind_rated_kw * wind_raw

    return pd.DataFrame({
        "timestamp": timestamps,
        "wind_kw": np.round(wind_kw, 2),
        "solar_kw": np.round(solar_kw, 2),
        "total_renewable_kw": np.round(wind_kw + solar_kw, 2),
    })