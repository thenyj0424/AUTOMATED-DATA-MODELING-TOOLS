import pandas as pd

from ai_agent.data_utils import build_summary


def test_build_summary_dtype_column_is_arrow_safe():
    df = pd.DataFrame(
        {
            "dtype": pd.Series([1, 2, 3], dtype="int64"),
            "category": ["a", "b", "c"],
        }
    )

    summary = build_summary(df)

    assert summary.dtypes["dtype"].map(type).eq(str).all()
    assert summary.dtypes["dtype"].tolist() == ["int64", "object"]
