"""Unit tests for time_utils.py UTC↔LST conversion."""
import pandas as pd
import pytest

from samy_wgen.utils.time_utils import utc_to_lst, assert_is_lst, get_utc_offset
from samy_wgen.pipeline.download_data import _normalize_city_args


def make_utc_df(n_hours: int = 48) -> pd.DataFrame:
    """Return a simple DataFrame with UTC DatetimeIndex."""
    idx = pd.date_range("2050-01-01 00:00", periods=n_hours, freq="h")
    return pd.DataFrame({"temp": range(n_hours)}, index=idx)


class TestGetUtcOffset:
    def test_miami(self):
        assert get_utc_offset("Miami") == -5.0

    def test_los_angeles(self):
        assert get_utc_offset("Los_Angeles") == -8.0

    def test_montreal_ascii_code(self):
        assert get_utc_offset("Montreal") == -5.0

    def test_by_coordinates(self):
        assert get_utc_offset(latitude=25.7840, longitude=-80.2101) == -5.0

    def test_unknown_city_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_utc_offset("Atlantis")


class TestNormalizeCityArgs:
    def test_requires_canonical_ascii_codes(self):
        assert _normalize_city_args(["Montreal", "Toronto"]) == ["Montreal", "Toronto"]

    def test_unknown_city_suggests_ascii_code(self):
        with pytest.raises(ValueError, match="Did you mean: Montreal"):
            _normalize_city_args(["Montréal"])


class TestUtcToLst:
    def test_miami_shift(self):
        df = make_utc_df(48)
        lst = utc_to_lst(df, "Miami")
        # Miami is UTC-5; first timestamp should shift from 00:00 to 19:00 (previous day)
        assert lst.index[0] == pd.Timestamp("2049-12-31 19:00")

    def test_result_is_tz_naive(self):
        df = make_utc_df(24)
        lst = utc_to_lst(df, "Miami")
        assert lst.index.tz is None, "LST index must be timezone-naive"

    def test_length_preserved(self):
        df = make_utc_df(100)
        lst = utc_to_lst(df, "Miami")
        assert len(lst) == len(df)

    def test_data_unchanged(self):
        df = make_utc_df(24)
        lst = utc_to_lst(df, "Miami")
        pd.testing.assert_series_equal(lst["temp"].reset_index(drop=True),
                                        df["temp"].reset_index(drop=True))


class TestAssertIsLst:
    def test_valid_naive_hourly(self):
        idx = pd.date_range("2050-01-01", periods=100, freq="h")
        df = pd.DataFrame({"x": 1}, index=idx)
        assert_is_lst(df, "test")  # should not raise

    def test_tz_aware_raises(self):
        idx = pd.date_range("2050-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"x": 1}, index=idx)
        with pytest.raises(ValueError, match="timezone-aware"):
            assert_is_lst(df, "UTC input")

    def test_duplicate_timestamps_raises(self):
        idx = pd.DatetimeIndex(["2050-01-01 00:00"] * 2 + ["2050-01-01 02:00"] * 2)
        df = pd.DataFrame({"x": 1}, index=idx)
        with pytest.raises(ValueError, match="duplicate"):
            assert_is_lst(df, "dup test")
