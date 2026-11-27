"""End-to-end Green Hydrogen Plant Simulator (Wind + Solar -> PEM -> H2 Storage)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.physics.faraday_law import calculate_hydrogen_rate
from src.plant.renewable_grid import generate_renewable_profile
from src.plant.ems_controller import EMSConfig, EMSController
from src.plant.storage_and_bop import BoPStorageConfig, BoPStorageSystem


class GreenHydrogenPlantSimulator:
    """Simulates a complete industrial Wind + Solar PEM Green Hydrogen facility."""

    def __init__(
        self,
        ems_config: EMSConfig | None = None,
        bop_config: BoPStorageConfig | None = None,
        number_of_cells: int = 200,
        faradaic_efficiency: float = 0.98,
    ):
        self.ems_cfg = ems_config or EMSConfig()
        self.bop_cfg = bop_config or BoPStorageConfig()
        self.number_of_cells = number_of_cells
        self.faradaic_efficiency = faradaic_efficiency

    def run(
        self,
        start_date: str = "2026-11-01",
        days: int = 7,
        freq_minutes: int = 15,
        seed: int = 42,
    ) -> pd.DataFrame:
        dt_hours = freq_minutes / 60.0
        ren_df = generate_renewable_profile(
            start_date=start_date, days=days, freq_minutes=freq_minutes, seed=seed
        )

        ems = EMSController(self.ems_cfg)
        bop = BoPStorageSystem(self.bop_cfg)
        rows = []

        for idx, r in ren_df.iterrows():
            dispatch = ems.step(
                renewable_kw=float(r["total_renewable_kw"]),
                tank_fill_pct=bop.tank_fill_pct,
                dt_hours=dt_hours,
            )

            stack_kw = float(dispatch["stack_power_kw"])
            if dispatch["operating_mode"] in ("HOT_STANDBY", "CURTAILED_TANK_FULL"):
                stack_voltage_v = 310.0
                stack_current_a = 0.0
                h2_rate_kg_hr = 0.0
                sec_kwh_per_kg = np.nan
            else:
                # Realistic PEM polarization approximation (1.55V to 2.05V per cell)
                load_frac = min(1.0, stack_kw / self.ems_cfg.stack_rated_kw)
                cell_v = 1.55 + 0.45 * (load_frac ** 0.75)
                stack_voltage_v = cell_v * self.number_of_cells
                stack_current_a = (stack_kw * 1000.0) / stack_voltage_v
                h2_rate_kg_hr = calculate_hydrogen_rate(
                    stack_current_a, self.number_of_cells, self.faradaic_efficiency
                )
                sec_kwh_per_kg = stack_kw / h2_rate_kg_hr if h2_rate_kg_hr > 0.01 else np.nan

            bop_state = bop.step(h2_rate_kg_hr, dt_hours)

            rows.append({
                "step": idx,
                "timestamp": r["timestamp"],
                "wind_kw": r["wind_kw"],
                "solar_kw": r["solar_kw"],
                "total_renewable_kw": r["total_renewable_kw"],
                **dispatch,
                "stack_voltage_v": round(stack_voltage_v, 1),
                "stack_current_a": round(stack_current_a, 1),
                "h2_rate_kg_hr": round(h2_rate_kg_hr, 3),
                "sec_kwh_per_kg": round(sec_kwh_per_kg, 2) if pd.notna(sec_kwh_per_kg) else np.nan,
                **bop_state,
            })

        return pd.DataFrame(rows)