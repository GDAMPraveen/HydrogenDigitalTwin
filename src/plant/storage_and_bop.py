"""Balance of Plant (BoP) water system, compression, and H2 storage tank."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoPStorageConfig:
    tank_capacity_kg: float = 500.0
    tank_initial_kg: float = 120.0
    max_pressure_bar: float = 350.0
    offtake_demand_kg_hr: float = 8.0      # Continuous industrial / tube-trailer offtake
    water_l_per_kg_h2: float = 9.5         # Stoichiometric (~9 L) + purification losses
    compressor_kwh_per_kg: float = 1.6     # Compression from 30 bar -> 350 bar
    co2_avoided_kg_per_kg_h2: float = 10.0 # Grey H2 (SMR) benchmark


class BoPStorageSystem:
    """Tracks H2 storage inventory, pressure, water usage, and CO2 avoided."""

    def __init__(self, config: BoPStorageConfig | None = None):
        self.cfg = config or BoPStorageConfig()
        self.tank_mass_kg = self.cfg.tank_initial_kg

    @property
    def tank_fill_pct(self) -> float:
        return (self.tank_mass_kg / self.cfg.tank_capacity_kg) * 100.0

    @property
    def tank_pressure_bar(self) -> float:
        return (self.tank_mass_kg / self.cfg.tank_capacity_kg) * self.cfg.max_pressure_bar

    def step(self, h2_rate_kg_hr: float, dt_hours: float) -> dict[str, float]:
        h2_produced_kg = max(0.0, h2_rate_kg_hr) * dt_hours
        offtake_kg = min(self.tank_mass_kg + h2_produced_kg, self.cfg.offtake_demand_kg_hr * dt_hours)

        self.tank_mass_kg = min(
            self.cfg.tank_capacity_kg,
            max(0.0, self.tank_mass_kg + h2_produced_kg - offtake_kg),
        )

        water_used_l = h2_produced_kg * self.cfg.water_l_per_kg_h2
        compressor_kw = max(0.0, h2_rate_kg_hr) * self.cfg.compressor_kwh_per_kg
        co2_avoided_kg = h2_produced_kg * self.cfg.co2_avoided_kg_per_kg_h2

        return {
            "h2_produced_kg": round(h2_produced_kg, 4),
            "h2_offtake_kg": round(offtake_kg, 4),
            "tank_mass_kg": round(self.tank_mass_kg, 2),
            "tank_fill_pct": round(self.tank_fill_pct, 1),
            "tank_pressure_bar": round(self.tank_pressure_bar, 1),
            "water_used_l": round(water_used_l, 2),
            "compressor_kw": round(compressor_kw, 2),
            "co2_avoided_kg": round(co2_avoided_kg, 2),
        }