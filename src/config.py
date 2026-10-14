from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class SmtpConfig:
    host: str = "smtp.gmail.com"
    port: int = 465
    sender_name: str = "Consumption-drop analytics"


@dataclass
class Config:
    baseline_days: int = 90
    baseline_gap_days: int = 30
    current_window_days: int = 30
    min_baseline_days_covered: int = 60
    min_peer_group_size: int = 20
    min_comm_availability_pct: float = 85.0
    min_drop_kwh_per_day: float = 2.0
    excess_drop_threshold_pct: float = 20.0
    vacancy_kwh_per_day: float = 0.1
    onset_fraction: float = 0.7
    load_band_edges_kw: list[float] = field(default_factory=lambda: [0, 2, 5, 10, 25, 1000])
    model_version: str = "v1-mvp"
    smtp: SmtpConfig = field(default_factory=SmtpConfig)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    smtp = SmtpConfig(**raw.pop("smtp", {}))
    return Config(**raw, smtp=smtp)
