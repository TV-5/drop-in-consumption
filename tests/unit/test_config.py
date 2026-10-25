import pytest

from src.config import Config, load_config
from tests.helpers import ROOT


def test_project_config_loads_with_its_values():
    cfg = load_config(ROOT / "config.yaml")
    assert (cfg.baseline_days, cfg.baseline_gap_days, cfg.current_window_days) == (90, 30, 30)
    assert cfg.excess_drop_threshold_pct == 20.0 and cfg.min_drop_kwh_per_day == 2.0
    assert cfg.smtp.host == "smtp.gmail.com" and cfg.smtp.port == 465


def test_missing_keys_fall_back_to_defaults(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("excess_drop_threshold_pct: 25\n")
    cfg = load_config(f)
    assert cfg.excess_drop_threshold_pct == 25
    assert cfg.baseline_days == Config().baseline_days


def test_empty_file_gives_defaults(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("")
    assert load_config(f) == Config()


def test_unknown_key_is_rejected(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("threshold_typo: 5\n")
    with pytest.raises(ValueError, match="threshold_typo"):
        load_config(f)
