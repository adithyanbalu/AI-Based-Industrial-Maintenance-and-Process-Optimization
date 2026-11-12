
import subprocess
import sys
from pathlib import Path

import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "src" / "data_preprocessing.py"
DATA_PATH = ROOT / "data" / "Dataset.csv"
OUTPUT_DIR = ROOT / "outputs"

REPORT_PATH = OUTPUT_DIR / "preprocessing_report.txt"
IMAGE_PATH = OUTPUT_DIR / "first_five_rows.png"


@pytest.fixture(scope="module")
def preprocessing_run():
    """Execute the preprocessing script once for this test module."""

    assert SCRIPT_PATH.exists(), "Preprocessing script not found"
    assert DATA_PATH.exists(), "Dataset.csv not found"

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"Preprocessing script failed:\n{result.stderr}"
    )

    return result


def test_preprocessing_script_executes(preprocessing_run):
    """Check that the preprocessing script executes successfully."""

    assert "DATA PREPROCESSING INSPECTION REPORT" in preprocessing_run.stdout


def test_preprocessing_report_generated(preprocessing_run):
    """Check that the preprocessing report is generated."""

    assert REPORT_PATH.exists()
    assert REPORT_PATH.stat().st_size > 0

    report = REPORT_PATH.read_text(encoding="utf-8")

    assert "Dataset shape (rows, columns)" in report
    assert "MISSING VALUES BY COLUMN" in report
    assert "DUPLICATE ROWS" in report


def test_preprocessing_image_generated(preprocessing_run):
    """Check that the first-five-rows preview image is generated."""

    assert IMAGE_PATH.exists()
    assert IMAGE_PATH.stat().st_size > 0

    with IMAGE_PATH.open("rb") as image_file:
        assert image_file.read(8) == b"\x89PNG\r\n\x1a\n"
def test_dataset_expected_shape():
    df = pd.read_csv(DATA_PATH)
    assert df.shape == (10000, 14)


def test_dataset_has_no_missing_values():
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0


def test_dataset_has_no_duplicate_rows():
    df = pd.read_csv(DATA_PATH)
    assert df.duplicated().sum() == 0
