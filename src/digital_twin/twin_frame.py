"""Shared, tagged data contract for synchronized Digital Twin execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import uuid


@dataclass
class PhysicsSnapshot:
    faraday_flow_kg_hr: float
    dc_power_kw: float
    live_flow_residual_kg_hr: float
    current_cmd_gap_a: Optional[float] = None


@dataclass
class ForecastSnapshot:
    horizon_steps: int
    predicted_flow_kg_hr: float
    faraday_baseline_kg_hr: float
    ai_residual_kg_hr: float


@dataclass
class UncertaintySnapshot:
    lower_bound_kg_hr: float
    upper_bound_kg_hr: float
    interval_width_kg_hr: float
    aleatoric_half_width_kg_hr: float
    epistemic_score: float
    confidence_level: float = 0.90


@dataclass
class ResilienceSnapshot:
    is_ood: bool
    anomaly_score: float
    sensor_faults_detected: list[str] = field(default_factory=list)
    status: str = "NORMAL"  # NORMAL | WARNING_OOD | FAULT_DETECTED


@dataclass
class XAISnapshot:
    top_drivers: dict[str, float] = field(default_factory=dict)
    explanation_summary: str = ""


@dataclass
class TwinFrame:
    """Single tagged state packet flowing sequentially through all Digital Twin layers."""
    frame_id: str
    step_index: int
    experiment: str
    raw_sensors: dict[str, float]
    timestamp: Optional[str] = None

    physics: Optional[PhysicsSnapshot] = None
    forecast: Optional[ForecastSnapshot] = None
    uncertainty: Optional[UncertaintySnapshot] = None
    resilience: Optional[ResilienceSnapshot] = None
    xai: Optional[XAISnapshot] = None

    pipeline_trace: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        step_index: int,
        experiment: str,
        raw_sensors: dict[str, float],
        timestamp: Optional[str] = None,
    ) -> "TwinFrame":
        tag = f"FRM-{step_index:06d}-{uuid.uuid4().hex[:6].upper()}"
        return cls(
            frame_id=tag,
            step_index=step_index,
            experiment=experiment,
            raw_sensors=dict(raw_sensors),
            timestamp=timestamp,
        )

    def mark_stage(self, stage_name: str) -> None:
        self.pipeline_trace.append(stage_name)

    def assert_synchronized(self) -> None:
        """Verify all 5 core layers processed this exact tagged frame."""
        required = ["physics", "forecast", "uncertainty", "resilience", "xai"]
        missing = [s for s in required if getattr(self, s) is None]
        if missing:
            raise RuntimeError(
                f"Frame {self.frame_id} is incomplete; missing layers: {missing}"
            )

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten the synchronized frame for logging, CSV export, or dashboard display."""
        return {
            "frame_id": self.frame_id,
            "step_index": self.step_index,
            "experiment": self.experiment,
            "faraday_flow": self.physics.faraday_flow_kg_hr if self.physics else None,
            "predicted_flow": self.forecast.predicted_flow_kg_hr if self.forecast else None,
            "lower_90": self.uncertainty.lower_bound_kg_hr if self.uncertainty else None,
            "upper_90": self.uncertainty.upper_bound_kg_hr if self.uncertainty else None,
            "epistemic_score": self.uncertainty.epistemic_score if self.uncertainty else None,
            "is_ood": self.resilience.is_ood if self.resilience else None,
            "status": self.resilience.status if self.resilience else None,
            "faults": ", ".join(self.resilience.sensor_faults_detected) if self.resilience else "",
            "explanation": self.xai.explanation_summary if self.xai else "",
        }