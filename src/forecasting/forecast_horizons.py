"""Row-based forecast horizon definitions; no time units are implied."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ForecastHorizon:
    name: str
    steps: int
    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Horizon name must be a non-empty string.")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise TypeError("Horizon steps must be an integer.")
        if self.steps < 1:
            raise ValueError("Horizon steps must be >= 1.")

DEFAULT_HORIZONS = {
    "one_step": ForecastHorizon("one_step", 1),
    "three_steps": ForecastHorizon("three_steps", 3),
    "five_steps": ForecastHorizon("five_steps", 5),
    "ten_steps": ForecastHorizon("ten_steps", 10),
}

def create_horizons(mapping):
    return {name: ForecastHorizon(name, steps) for name, steps in mapping.items()}

def get_horizon(name, horizons=None):
    collection = DEFAULT_HORIZONS if horizons is None else horizons
    if name not in collection:
        raise KeyError(f"Unknown forecast horizon: {name}")
    return collection[name]
