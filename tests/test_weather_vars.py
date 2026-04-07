"""Unit tests for calculate_weather_var.py humidity calculation functions."""
import numpy as np
import pytest

from samy_wgen.utils.calculate_weather_var import (
    calc_dewpoint_from_RH,
    calc_dewpoint_from_Q,
    calc_RH_from_dewpoint,
    calc_Q_from_dewpoint,
    calc_wetbulb,
)


class TestDewpointRoundtrip:
    """Roundtrip: T + RH → Td → RH should recover original RH."""

    def test_roundtrip_scalar(self):
        T, RH = 25.0, 60.0
        Td = calc_dewpoint_from_RH(T, RH)
        RH_recovered = calc_RH_from_dewpoint(T, Td)
        assert abs(RH_recovered - RH) < 0.5, f"RH roundtrip error: {RH} → {RH_recovered}"

    def test_roundtrip_array(self):
        T = np.array([0.0, 10.0, 20.0, 30.0, 35.0])
        RH = np.array([40.0, 55.0, 70.0, 80.0, 90.0])
        Td = calc_dewpoint_from_RH(T, RH)
        RH_back = calc_RH_from_dewpoint(T, Td)
        np.testing.assert_allclose(RH_back, RH, atol=0.5)

    def test_dewpoint_never_exceeds_temp(self):
        T = np.linspace(-10, 40, 50)
        RH = np.full(50, 80.0)
        Td = calc_dewpoint_from_RH(T, RH)
        assert np.all(Td <= T + 1e-6), "Dewpoint exceeded dry-bulb temperature"

    def test_high_rh_dewpoint_close_to_temp(self):
        T, RH = 20.0, 99.0
        Td = calc_dewpoint_from_RH(T, RH)
        assert abs(Td - T) < 1.0, f"At RH=99%, Td ({Td:.2f}) should be close to T ({T})"

    def test_low_rh_dewpoint_much_lower(self):
        T, RH = 30.0, 10.0
        Td = calc_dewpoint_from_RH(T, RH)
        assert Td < T - 10, f"At RH=10%, Td ({Td:.2f}) should be well below T ({T})"


class TestSpecificHumidityRoundtrip:
    """Roundtrip: Td + P → Q → Td should recover original dewpoint."""

    def test_roundtrip_typical_conditions(self):
        Td = np.array([10.0, 15.0, 20.0, 25.0])
        P = np.array([1013.25, 1013.25, 1013.25, 1013.25])
        T = Td + 5.0  # dry-bulb always above dewpoint
        Q = calc_Q_from_dewpoint(Td, P)
        Td_back = calc_dewpoint_from_Q(T, Q, P)
        np.testing.assert_allclose(Td_back, Td, atol=0.5)

    def test_q_physical_range(self):
        Td = np.linspace(-20, 35, 100)
        P = np.full(100, 1013.25)
        Q = calc_Q_from_dewpoint(Td, P)
        assert np.all(Q >= 0), "Specific humidity cannot be negative"
        assert np.all(Q <= 0.05), f"Specific humidity too high: max={Q.max():.4f}"


class TestWetbulb:
    def test_wetbulb_below_drybulb(self):
        T, RH = np.array([25.0, 30.0, 35.0]), np.array([50.0, 60.0, 70.0])
        Tw = calc_wetbulb(T, RH)
        assert np.all(Tw <= T + 1e-6)

    def test_wetbulb_saturated(self):
        # At RH=100%, wet-bulb ≈ dry-bulb
        T, RH = 25.0, 100.0
        Tw = calc_wetbulb(T, RH)
        assert abs(Tw - T) < 1.0

    def test_miami_typical(self):
        # Miami summer: ~30°C, ~80% RH → Tw ~26-27°C
        Tw = calc_wetbulb(30.0, 80.0)
        assert 24.0 < Tw < 29.0, f"Miami wetbulb {Tw:.1f}°C outside expected range"
