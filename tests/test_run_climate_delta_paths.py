from __future__ import annotations

from pathlib import Path


def test_resolve_city_dir_accepts_spaced_name(tmp_path):
    from samy_wgen.pipeline.run_climate_delta import _resolve_city_dir

    spaced = tmp_path / "Los Angeles"
    spaced.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_city_dir(tmp_path, "Los Angeles")

    assert resolved == spaced
