import os
import sys

import pandas as pd
import pytest

from pylab3.app import compute_total_fare_by_class


def test_compute_total_fare_by_class_basic() -> None:
    df = pd.DataFrame(
        {
            "Sex": ["male", "female", "male", "female"],
            "Pclass": [1, 1, 2, 2],
            "Fare": [100.0, 200.0, 50.0, 75.0],
        }
    )
    result = compute_total_fare_by_class(df, "female")
    assert list(result["Класс обслуживания"]) == [1, 2]
    assert list(result["Суммарная стоимость билетов"]) == [200.0, 75.0]


def test_compute_total_fare_by_class_no_matching_sex() -> None:
    df = pd.DataFrame(
        {
            "Sex": ["male", "male"],
            "Pclass": [1, 2],
            "Fare": [30.0, 40.0],
        }
    )
    result = compute_total_fare_by_class(df, "female")
    assert result.empty
    assert list(result.columns) == ["Класс обслуживания", "Суммарная стоимость билетов"]


def test_compute_total_fare_by_class_missing_columns() -> None:
    df = pd.DataFrame(
        {
            "Sex": ["male", "female"],
            "Pclass": [1, 2],
        }
    )
    with pytest.raises(ValueError):
        compute_total_fare_by_class(df, "male")
