"""Pytest fixtures for samy_wgen test suite."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite cached stage outputs and rerun all pipeline stages from scratch.",
    )


@pytest.fixture(scope="session")
def overwrite(request) -> bool:
    """Whether to overwrite cached pipeline stage outputs."""
    return request.config.getoption("--overwrite")


@pytest.fixture(scope="session")
def miami_data():
    """Paths to the Miami POC dataset."""
    return {
        "isd_dir": REPO_ROOT / "data" / "ISD" / "Miami" / "722020-12839_MIAMI_INTERNATIONAL_AIRPORT",
        "cmip6_dir": REPO_ROOT / "data" / "CMIP6" / "Miami",
        "era5_csv": REPO_ROOT / "data" / "ERA5" / "ERA5_Miami_full.csv",
        "nsrdb_csv": REPO_ROOT / "data" / "NSRDB" / "nsrdb_miami_1998-2024.csv",
    }


@pytest.fixture(scope="session")
def tmp_output():
    """Session-scoped persistent directory for pipeline output under tests/output."""
    out_dir = REPO_ROOT / "tests" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
