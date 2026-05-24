import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


@dataclass
class DatasetSummary:
	rows: int
	cols: int
	numeric_cols: List[str]
	categorical_cols: List[str]
	datetime_cols: List[str]
	missing_by_col: pd.DataFrame
	dtypes: pd.DataFrame


def read_csv(uploaded_file: Any) -> pd.DataFrame:
	return pd.read_csv(uploaded_file)


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
	numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
	categorical_cols = [c for c in df.columns if c not in numeric_cols]
	return numeric_cols, categorical_cols


def find_datetime_columns(df: pd.DataFrame) -> List[str]:
	datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
	for col in df.columns:
		if col in datetime_cols:
			continue
		if df[col].dtype == "object":
			parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
			rate = parsed.notna().mean()
			if rate >= 0.6:
				datetime_cols.append(col)
	return datetime_cols


def build_summary(df: pd.DataFrame) -> DatasetSummary:
	numeric_cols, categorical_cols = split_columns(df)
	datetime_cols = find_datetime_columns(df)
	missing = df.isna().sum().reset_index()
	missing.columns = ["column", "missing_count"]
	dtypes = df.dtypes.reset_index()
	dtypes.columns = ["column", "dtype"]
	dtypes["dtype"] = dtypes["dtype"].astype(str)
	return DatasetSummary(
		rows=df.shape[0],
		cols=df.shape[1],
		numeric_cols=numeric_cols,
		categorical_cols=categorical_cols,
		datetime_cols=datetime_cols,
		missing_by_col=missing,
		dtypes=dtypes,
	)


def describe_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
	if not numeric_cols:
		return pd.DataFrame()
	return df[numeric_cols].describe().transpose()


def describe_categorical(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, pd.Series]:
	summaries: Dict[str, pd.Series] = {}
	for col in categorical_cols:
		summaries[col] = df[col].value_counts(dropna=False).head(20)
	return summaries


def count_iqr_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
	rows = []
	for col in numeric_cols:
		series = df[col].dropna()
		if series.empty:
			continue
		q1 = series.quantile(0.25)
		q3 = series.quantile(0.75)
		iqr = q3 - q1
		low = q1 - 1.5 * iqr
		high = q3 + 1.5 * iqr
		count = int(((series < low) | (series > high)).sum())
		rows.append({"column": col, "outlier_count": count})
	return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)


def remove_iqr_outliers(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, int]:
	if not cols:
		return df.copy(), 0
	mask = pd.Series(False, index=df.index)
	for col in cols:
		series = df[col].dropna()
		if series.empty:
			continue
		q1 = series.quantile(0.25)
		q3 = series.quantile(0.75)
		iqr = q3 - q1
		low = q1 - 1.5 * iqr
		high = q3 + 1.5 * iqr
		mask |= (df[col] < low) | (df[col] > high)
	removed = int(mask.sum())
	return df.loc[~mask].copy(), removed


def impute_missing_values(
	df: pd.DataFrame,
	numeric_strategy: str,
	impute_categorical: bool,
) -> pd.DataFrame:
	work_df = df.copy()
	numeric_cols, categorical_cols = split_columns(work_df)
	if numeric_strategy in ["mean", "median"]:
		for col in numeric_cols:
			if work_df[col].isna().any():
				fill_val = work_df[col].mean() if numeric_strategy == "mean" else work_df[col].median()
				work_df[col] = work_df[col].fillna(fill_val)
	if impute_categorical:
		for col in categorical_cols:
			if work_df[col].isna().any():
				mode = work_df[col].mode(dropna=True)
				if not mode.empty:
					work_df[col] = work_df[col].fillna(mode.iloc[0])
	return work_df
