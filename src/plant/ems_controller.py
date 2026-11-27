"""Energy Management System (EMS) & Battery Energy Storage (BESS) dispatch."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EMSConfig:
    stack_rated_kw: float = 1000.0
    stack_min_load_kw: float = 100.0       # 10% minimum load to avoid gas crossover
    hot_standby_kw: float = 25.0           # Trickle power to keep stack warm
    max_ramp_kw_per_step: float = 250.0    # Protects PEM membrane from violent gusts
    rectifier_efficiency: float = 0.96
    bess_capacity_kwh: float = 500.0
    bess_max_power_kw: float = 250.0
    bess_soc_min: float = 0.15
    bess_soc_max: float = 0.95
    bess_initial_soc: float = 0.50


class EMSController:
    """Dispatches Wind + Solar + BESS power to the PEM Electrolyzer Stack."""

    def __init__(self, config: EMSConfig | None = None):
        self.cfg = config or EMSConfig()
        self.bess_energy_kwh = self.cfg.bess_capacity_kwh * self.cfg.bess_initial_soc
        self.prev_stack_kw = 0.0

    @property
    def soc_pct(self) -> float:
        return (self.bess_energy_kwh / self.cfg.bess_capacity_kwh) * 100.0

    def step(self, renewable_kw: float, tank_fill_pct: float, dt_hours: float) -> dict[str, float | str]:
        avail_dc_kw = renewable_kw * self.cfg.rectifier_efficiency
        bess_kw = 0.0       # + = discharging to stack, - = charging from renewables
        grid_import_kw = 0.0
        curtailed_kw = 0.0

        # If H2 tank is nearly full (>= 96%), curtail electrolyzer to standby
        if tank_fill_pct >= 96.0:
            target_stack_kw = self.cfg.hot_standby_kw
            mode = "CURTAILED_TANK_FULL"
        elif avail_dc_kw >= self.cfg.stack_min_load_kw:
            target_stack_kw = min(avail_dc_kw, self.cfg.stack_rated_kw)
            mode = "NOMINAL"
        else:
            # Try discharging BESS to keep stack above minimum load
            needed_kw = self.cfg.stack_min_load_kw - avail_dc_kw
            max_discharge = min(
                self.cfg.bess_max_power_kw,
                max(0.0, (self.bess_energy_kwh - self.cfg.bess_capacity_kwh * self.cfg.bess_soc_min) / dt_hours),
            )
            if max_discharge >= needed_kw:
                bess_kw = needed_kw
                target_stack_kw = self.cfg.stack_min_load_kw
                mode = "BESS_ASSISTED"
            else:
                target_stack_kw = self.cfg.hot_standby_kw
                mode = "HOT_STANDBY"

        # Apply ramp-rate protection
        delta = target_stack_kw - self.prev_stack_kw
        if abs(delta) > self.cfg.max_ramp_kw_per_step:
            stack_kw = self.prev_stack_kw + (
                self.cfg.max_ramp_kw_per_step if delta > 0 else -self.cfg.max_ramp_kw_per_step
            )
            if mode == "NOMINAL":
                mode = "RAMPING"
        else:
            stack_kw = target_stack_kw

        # Reconcile surplus or deficit with BESS / Grid / Curtailment
        net_power = avail_dc_kw + bess_kw - stack_kw
        if net_power > 0:
            # Charge BESS with surplus
            room_kwh = max(0.0, self.cfg.bess_capacity_kwh * self.cfg.bess_soc_max - self.bess_energy_kwh)
            charge_kw = min(net_power, self.cfg.bess_max_power_kw, room_kwh / dt_hours)
            bess_kw -= charge_kw
            self.bess_energy_kwh += charge_kw * dt_hours
            curtailed_kw = max(0.0, net_power - charge_kw)
        elif net_power < 0:
            deficit_kw = -net_power
            avail_kwh = max(0.0, self.bess_energy_kwh - self.cfg.bess_capacity_kwh * self.cfg.bess_soc_min)
            discharge_kw = min(deficit_kw, self.cfg.bess_max_power_kw, avail_kwh / dt_hours)
            bess_kw += discharge_kw
            self.bess_energy_kwh -= discharge_kw * dt_hours
            grid_import_kw = max(0.0, deficit_kw - discharge_kw)

        self.prev_stack_kw = stack_kw
        return {
            "operating_mode": mode,
            "stack_power_kw": round(stack_kw, 2),
            "bess_power_kw": round(bess_kw, 2),
            "bess_soc_pct": round(self.soc_pct, 1),
            "grid_import_kw": round(grid_import_kw, 2),
            "curtailed_kw": round(curtailed_kw, 2),
        }