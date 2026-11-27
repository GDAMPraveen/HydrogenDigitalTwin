"""Date-range operational report generator for the Green Hydrogen Digital Twin."""
from __future__ import annotations

import pandas as pd

def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(str(val) for val in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])

def generate_period_report(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> tuple[str, pd.DataFrame, dict[str, float]]:
    """Filter plant telemetry by date range and build a Markdown report + daily summary table."""
    ts = pd.to_datetime(df["timestamp"])
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    period = df[(ts >= start_ts) & (ts <= end_ts)].copy()
    if period.empty:
        raise ValueError(f"No plant data found between {start_date} and {end_date}.")

    dt_hours = (
        (period["timestamp"].iloc[1] - period["timestamp"].iloc[0]).total_seconds() / 3600.0
        if len(period) > 1
        else 0.25
    )

    period["date"] = pd.to_datetime(period["timestamp"]).dt.date
    daily = (
        period.groupby("date")
        .agg(
            Wind_MWh=("wind_kw", lambda s: round(s.sum() * dt_hours / 1000.0, 2)),
            Solar_MWh=("solar_kw", lambda s: round(s.sum() * dt_hours / 1000.0, 2)),
            Stack_MWh=("stack_power_kw", lambda s: round(s.sum() * dt_hours / 1000.0, 2)),
            Curtailed_MWh=("curtailed_kw", lambda s: round(s.sum() * dt_hours / 1000.0, 2)),
            H2_Produced_kg=("h2_produced_kg", lambda s: round(s.sum(), 2)),
            Avg_SEC_kWh_kg=("sec_kwh_per_kg", lambda s: round(s.mean(), 2)),
            Water_Used_L=("water_used_l", lambda s: round(s.sum(), 1)),
            CO2_Avoided_kg=("co2_avoided_kg", lambda s: round(s.sum(), 1)),
            End_Tank_Pct=("tank_fill_pct", "last"),
        )
        .reset_index()
    )

    kpis = {
        "total_h2_kg": float(period["h2_produced_kg"].sum()),
        "total_wind_mwh": float(period["wind_kw"].sum() * dt_hours / 1000.0),
        "total_solar_mwh": float(period["solar_kw"].sum() * dt_hours / 1000.0),
        "total_stack_mwh": float(period["stack_power_kw"].sum() * dt_hours / 1000.0),
        "total_curtailed_mwh": float(period["curtailed_kw"].sum() * dt_hours / 1000.0),
        "total_grid_mwh": float(period["grid_import_kw"].sum() * dt_hours / 1000.0),
        "avg_sec_kwh_kg": float(period["sec_kwh_per_kg"].mean()),
        "total_water_l": float(period["water_used_l"].sum()),
        "total_co2_avoided_kg": float(period["co2_avoided_kg"].sum()),
        "standby_hours": float((period["operating_mode"] == "HOT_STANDBY").sum() * dt_hours),
        "curtailed_tank_hours": float((period["operating_mode"] == "CURTAILED_TANK_FULL").sum() * dt_hours),
    }

    mode_counts = period["operating_mode"].value_counts()
    mode_lines = "\n".join(
        f"- **{mode}:** {count * dt_hours:.2f} hours ({count / len(period) * 100:.1f}%)"
        for mode, count in mode_counts.items()
    )

    report_md = f"""# Green Hydrogen PEM Plant — Operational Report
**Reporting Period:** {start_ts.strftime('%Y-%m-%d')} to {end_ts.strftime('%Y-%m-%d')}  
**Total Steps Analyzed:** {len(period):,} ({len(period) * dt_hours:.1f} hours)

---

## 1. Executive KPI Summary
| Metric | Value |
| :--- | :--- |
| **Total Green H₂ Produced** | {kpis['total_h2_kg']:.2f} kg |
| **Wind Energy Supplied** | {kpis['total_wind_mwh']:.2f} MWh |
| **Solar PV Energy Supplied** | {kpis['total_solar_mwh']:.2f} MWh |
| **Electrolyzer Stack Consumption** | {kpis['total_stack_mwh']:.2f} MWh |
| **Renewable Curtailment** | {kpis['total_curtailed_mwh']:.2f} MWh |
| **Grid Trickle Import** | {kpis['total_grid_mwh']:.2f} MWh |
| **Average Specific Energy Consumption (SEC)** | {kpis['avg_sec_kwh_kg']:.2f} kWh/kg H₂ |
| **Deionized Water Consumed** | {kpis['total_water_l']:.1f} L |
| **CO₂ Emissions Avoided (vs. Grey H₂)** | {kpis['total_co2_avoided_kg']:.1f} kg CO₂ |

---

## 2. Operating Mode Breakdown
{mode_lines}

---

## 3. Day-by-Day Operational Breakdown
{_df_to_markdown(daily)}
"""
    return report_md, daily, kpis