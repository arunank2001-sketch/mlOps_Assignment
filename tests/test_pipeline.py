import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import CATEGORICAL, NUMERIC, build_preprocessor
from train import load_data


@pytest.fixture
def sample_frame():
    return pd.DataFrame({
        "age": [40, 55, np.nan, 61],
        "trestbps": [120, 140, 130, np.nan],
        "chol": [200, 250, 230, 210],
        "thalach": [170, 150, np.nan, 160],
        "oldpeak": [0.0, 1.5, 2.0, np.nan],
        "sex": [1, 0, 1, 0],
        "cp": [0, 2, 1, 3],
        "fbs": [0, 1, 0, 0],
        "restecg": [0, 1, 0, 2],
        "exang": [0, 1, 0, 1],
        "slope": [1, 2, 2, 1],
        "ca": [0, np.nan, 1, 2],
        "thal": [3, 6, np.nan, 7],
    })


def test_preprocessor_handles_missing_values(sample_frame):
    out = build_preprocessor().fit_transform(sample_frame)
    assert not np.isnan(out).any()


def test_preprocessor_output_width(sample_frame):
    out = build_preprocessor().fit_transform(sample_frame)
    assert out.shape[0] == len(sample_frame)
    assert out.shape[1] > len(NUMERIC) + len(CATEGORICAL)  # one-hot expands categories


def test_preprocessor_ignores_unseen_categories(sample_frame):
    prep = build_preprocessor().fit(sample_frame)
    unseen = sample_frame.copy()
    unseen.loc[0, "cp"] = 99  # category never seen during fit
    assert prep.transform(unseen).shape[0] == len(unseen)


def test_load_data_returns_binary_target():
    X, y = load_data()
    assert set(y.unique()) <= {0, 1}
    assert "target" not in X.columns
    assert len(X) == len(y) == 303