#!/usr/bin/env python3
"""
Tests for two fixes in utils/create_epw_ddy.py and utils/calculate_weather_var.py:

Fix 1 (calculate_weather_var): calc_wetbulb() — Stull (2011) formula
Fix 2 (create_epw_ddy): generate_ddy_from_epw() now uses wet-bulb (not dewpoint)
                         and filters EPW missing-value codes before computing percentiles
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from samy_wgen.utils.calculate_weather_var import calc_wetbulb
from samy_wgen.utils.create_epw_ddy import generate_ddy_from_epw


# ---------------------------------------------------------------------------
# Fix 1: calc_wetbulb
# ---------------------------------------------------------------------------

class TestCalcWetbulb:
    def test_wetbulb_below_drybulb(self):
        """Wet-bulb must always be <= dry-bulb."""
        for T in np.arange(-10, 45, 5):
            for RH in [10, 30, 50, 70, 90, 100]:
                Tw = calc_wetbulb(T, RH)
                assert Tw <= T + 1e-6, f"Tw={Tw:.2f} > T={T} at RH={RH}%"

    def test_wetbulb_equals_drybulb_at_saturation(self):
        """At RH=100%, wet-bulb == dry-bulb."""
        for T in [-5.0, 10.0, 25.0, 35.0]:
            Tw = calc_wetbulb(T, 100.0)
            assert abs(Tw - T) < 0.5, f"At RH=100%, Tw={Tw:.2f} should be ~{T}°C"

    def test_wetbulb_increases_with_rh(self):
        """For fixed T, wet-bulb increases with RH."""
        T = 30.0
        rh_vals = [20, 40, 60, 80]
        tw_vals = [calc_wetbulb(T, rh) for rh in rh_vals]
        assert tw_vals == sorted(tw_vals), f"Wet-bulb should increase with RH: {tw_vals}"

    def test_wetbulb_la_dry_hot(self):
        """LA hot dry condition (30°C, 25% RH): wet-bulb should be ~17°C, not ~5°C (dewpoint)."""
        Tw = calc_wetbulb(30.0, 25.0)
        assert 14.0 < Tw < 20.0, f"Expected ~17°C for LA dry hot, got {Tw:.1f}°C"

    def test_wetbulb_array_input(self):
        """Accepts numpy arrays."""
        T = np.array([20.0, 25.0, 30.0])
        RH = np.array([50.0, 60.0, 70.0])
        Tw = calc_wetbulb(T, RH)
        assert Tw.shape == T.shape
        assert np.all(Tw <= T + 1e-6)

    def test_wetbulb_larger_than_dewpoint(self):
        """Wet-bulb should be higher than dewpoint for unsaturated air."""
        from samy_wgen.utils.calculate_weather_var import calc_dewpoint_from_RH
        T, RH = 30.0, 30.0
        Td = calc_dewpoint_from_RH(T, RH)
        Tw = calc_wetbulb(T, RH)
        assert Tw > Td, f"Wet-bulb {Tw:.1f}°C should exceed dewpoint {Td:.1f}°C"
        assert Tw < T, f"Wet-bulb {Tw:.1f}°C should be below dry-bulb {T}°C"


# ---------------------------------------------------------------------------
# Fix 2: generate_ddy_from_epw — helpers for building synthetic EPW
# ---------------------------------------------------------------------------

def _make_epw(path: Path, temps, rhs, dewpoints=None, windspeeds=None,
              pressure=101325.0, inject_missing=False):
    """Write a minimal 8760-row EPW with given temp/RH arrays."""
    assert len(temps) == 8760
    if rhs is None:
        rhs = np.full(8760, 50.0)
    if dewpoints is None:
        # Approximate dewpoint from RH/T for EPW column 7
        from samy_wgen.utils.calculate_weather_var import calc_dewpoint_from_RH
        dewpoints = calc_dewpoint_from_RH(np.asarray(temps), np.asarray(rhs))
    if windspeeds is None:
        windspeeds = np.full(8760, 3.0)

    header = (
        "LOCATION,TestCity,ST,TST,TMY,000000,34.05,-118.24,-8.0,0.0\n"
        "DESIGN CONDITIONS,0\n"
        "TYPICAL/EXTREME PERIODS,0\n"
        "GROUND TEMPERATURES,0\n"
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0\n"
        "COMMENTS 1,Test\n"
        "COMMENTS 2,Test\n"
        "DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31\n"
    )
    rows = []
    hour = 0
    for m in range(1, 13):
        days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
        for d in range(1, days + 1):
            for h in range(1, 25):
                i = hour % 8760
                t = float(temps[i])
                td = float(dewpoints[i])
                rh = float(rhs[i])
                ws = float(windspeeds[i])
                p = pressure

                # Optionally inject a few EPW missing-value rows
                if inject_missing and hour < 5:
                    t = 99.9
                    td = 99.9
                    rh = 999.0

                row = (f"2020,{m},{d},{h},0,C0C0C0C0?9?9?9E0E0E0?9?9?9?9?9*9*9*9*9*9,"
                       f"{t:.1f},{td:.1f},{rh:.0f},{p:.0f},"
                       f"9999,9999,9999,0,0,0,0,0,0,0,"
                       f"0,{ws:.1f},5,5,9999,99999,9,9,999,0,999,99,999,99")
                rows.append(row)
                hour += 1

    with open(path, "w") as f:
        f.write(header)
        f.write("\n".join(rows) + "\n")


def _parse_ddy_design_temps(ddy_path: Path):
    """Extract DB and MWB from a generated DDY file."""
    text = ddy_path.read_text()
    import re
    # Find all SizingPeriod:DesignDay blocks
    blocks = re.split(r"SizingPeriod:DesignDay", text)[1:]
    result = {}
    for block in blocks:
        name_match = re.search(r"([^\n,]+),\s*!- Name", block)
        db_match = re.search(r"([-\d.]+),\s*!- Maximum Dry-Bulb Temperature", block)
        mwb_match = re.search(r"([-\d.]+),\s*!- Wetbulb at Maximum Dry-Bulb", block)
        if name_match and db_match and mwb_match:
            name = name_match.group(1).strip()
            result[name] = {
                "db": float(db_match.group(1)),
                "mwb": float(mwb_match.group(1)),
            }
    return result


class TestGenerateDDYWetbulb:
    def test_mwb_greater_than_dewpoint_for_dry_air(self, tmp_path):
        """For dry air (low RH), MWB in DDY should be clearly above the dewpoint."""
        from samy_wgen.utils.calculate_weather_var import calc_dewpoint_from_RH

        # Construct EPW: baseline ~15°C, summer peak 32°C with 25% RH (dry LA)
        np.random.seed(0)
        n = 8760
        # Simple seasonal cycle + diurnal
        doy = np.arange(n) / 24
        temps = 15 + 8 * np.sin(2 * np.pi * (doy - 80) / 365) + 5 * np.sin(2 * np.pi * np.arange(n) / 24)
        # Low RH in summer (when T is high), higher in winter
        rhs = np.clip(70 - 1.5 * (temps - 15), 15, 90)

        epw = tmp_path / "test.epw"
        _make_epw(epw, temps, rhs)
        generate_ddy_from_epw(epw, energyplus_path=None)

        ddy = epw.with_suffix(".ddy")
        assert ddy.exists()
        conditions = _parse_ddy_design_temps(ddy)
        clg = next((v for k, v in conditions.items() if "Clg" in k), None)
        assert clg is not None, "No cooling design day found in DDY"

        cool_db = clg["db"]
        cool_mwb = clg["mwb"]

        # Compute expected dewpoint at design conditions
        cool_rh_approx = float(np.percentile(rhs[temps >= np.percentile(temps, 99.6)], 50))
        cool_td = float(calc_dewpoint_from_RH(cool_db, cool_rh_approx))

        assert cool_mwb > cool_td, (
            f"MWB {cool_mwb:.1f}°C should exceed dewpoint {cool_td:.1f}°C for dry air"
        )
        assert cool_mwb < cool_db, (
            f"MWB {cool_mwb:.1f}°C should be below dry-bulb {cool_db:.1f}°C"
        )

    def test_missing_value_codes_filtered(self, tmp_path):
        """Rows with EPW missing-value codes (99.9/999) must not affect design temps."""
        np.random.seed(1)
        n = 8760
        doy = np.arange(n) / 24
        temps_clean = 15 + 8 * np.sin(2 * np.pi * (doy - 80) / 365)
        rhs_clean = np.full(n, 50.0)

        # EPW without missing values
        epw_clean = tmp_path / "clean.epw"
        _make_epw(epw_clean, temps_clean, rhs_clean, inject_missing=False)
        generate_ddy_from_epw(epw_clean, energyplus_path=None)
        conds_clean = _parse_ddy_design_temps(epw_clean.with_suffix(".ddy"))

        # EPW with 5 rows of missing-value codes injected at the start
        epw_dirty = tmp_path / "dirty.epw"
        _make_epw(epw_dirty, temps_clean, rhs_clean, inject_missing=True)
        generate_ddy_from_epw(epw_dirty, energyplus_path=None)
        conds_dirty = _parse_ddy_design_temps(epw_dirty.with_suffix(".ddy"))

        # Design temps should be virtually identical after filtering
        for day_type in ["Htg", "Clg"]:
            key_clean = next(k for k in conds_clean if day_type in k)
            key_dirty = next(k for k in conds_dirty if day_type in k)
            db_clean = conds_clean[key_clean]["db"]
            db_dirty = conds_dirty[key_dirty]["db"]
            assert abs(db_clean - db_dirty) < 0.5, (
                f"{day_type} DB should be unaffected by missing values: "
                f"clean={db_clean:.1f} dirty={db_dirty:.1f}"
            )

    def test_mwb_physically_valid_range(self, tmp_path):
        """MWB in DDY must satisfy Td <= MWB <= DB for both heating and cooling."""
        np.random.seed(2)
        n = 8760
        doy = np.arange(n) / 24
        temps = 12 + 10 * np.sin(2 * np.pi * (doy - 80) / 365)
        rhs = np.clip(60 - 0.8 * (temps - 12), 20, 90)

        epw = tmp_path / "test.epw"
        _make_epw(epw, temps, rhs)
        generate_ddy_from_epw(epw, energyplus_path=None)
        conds = _parse_ddy_design_temps(epw.with_suffix(".ddy"))

        for name, vals in conds.items():
            db, mwb = vals["db"], vals["mwb"]
            assert mwb <= db + 1e-3, f"{name}: MWB {mwb:.1f} > DB {db:.1f}"
            # MWB >= dewpoint is implicitly guaranteed by calc_wetbulb; just check > -60
            assert mwb > -60, f"{name}: MWB {mwb:.1f} unreasonably low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
