from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


def render_correlation(df: pd.DataFrame, numeric_cols: List[str]) -> None:
	if len(numeric_cols) < 2:
		st.info("Need at least 2 numeric columns for correlation heatmap.")
		return
	corr = df[numeric_cols].corr()
	fig, ax = plt.subplots(figsize=(8, 6))
	sns.heatmap(corr, ax=ax, cmap="viridis", annot=False)
	ax.set_title("Correlation Heatmap")
	st.pyplot(fig, use_container_width=True)


def render_basic_plots(df: pd.DataFrame, numeric_cols: List[str]) -> None:
	if not numeric_cols:
		st.info("No numeric columns to plot.")
		return
	col = st.selectbox("Select numeric column", numeric_cols, key="eda_numeric_column")
	fig, axes = plt.subplots(1, 2, figsize=(10, 4))
	sns.histplot(df[col].dropna(), ax=axes[0], kde=True)
	axes[0].set_title("Histogram")
	sns.boxplot(x=df[col], ax=axes[1])
	axes[1].set_title("Box Plot")
	st.pyplot(fig, use_container_width=True)


def render_missingness_heatmap(df: pd.DataFrame, max_rows: int = 2000) -> None:
	if df.empty:
		st.info("No data available for missingness heatmap.")
		return
	plot_df = df
	if len(df) > max_rows:
		plot_df = df.sample(n=max_rows, random_state=42)
	fig, ax = plt.subplots(figsize=(10, 4))
	sns.heatmap(plot_df.isna(), cbar=False, ax=ax)
	ax.set_title("Missingness Heatmap (sampled)")
	st.pyplot(fig, use_container_width=True)


def render_target_distribution(df: pd.DataFrame, target_col: Optional[str]) -> None:
	if not target_col or target_col not in df.columns:
		st.info("Select a target column to see distribution.")
		return
	series = df[target_col].dropna()
	if series.empty:
		st.info("Target has no non-missing values to plot.")
		return
	fig, ax = plt.subplots(figsize=(8, 4))
	if pd.api.types.is_numeric_dtype(series):
		sns.histplot(series, kde=True, ax=ax)
		ax.set_title("Target Distribution")
	else:
		counts = series.value_counts().head(20)
		sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
		ax.set_title("Target Class Balance")
		ax.tick_params(axis="x", rotation=45)
		if len(counts) <= 6:
			fig2, ax2 = plt.subplots(figsize=(5, 5))
			ax2.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%")
			ax2.set_title("Target Share")
			st.pyplot(fig2, use_container_width=False)
	st.pyplot(fig, use_container_width=True)


def render_target_relationships(
	df: pd.DataFrame,
	target_col: Optional[str],
	numeric_cols: List[str],
	categorical_cols: List[str],
) -> None:
	if not target_col or target_col not in df.columns:
		st.info("Select a target column to see feature relationships.")
		return
	feature_cols = [c for c in df.columns if c != target_col]
	if not feature_cols:
		st.info("No feature columns available.")
		return
	feature = st.selectbox(
		"Feature for target relationship",
		feature_cols,
		key="eda_target_feature",
	)
	fig, ax = plt.subplots(figsize=(8, 4))
	if feature in numeric_cols and pd.api.types.is_numeric_dtype(df[target_col]):
		sns.scatterplot(x=df[feature], y=df[target_col], ax=ax)
		ax.set_title("Numeric vs Numeric")
	elif feature in numeric_cols:
		sns.boxplot(x=df[target_col].astype(str), y=df[feature], ax=ax)
		ax.set_title("Numeric by Target")
		ax.tick_params(axis="x", rotation=45)
	elif pd.api.types.is_numeric_dtype(df[target_col]):
		sns.boxplot(x=df[feature].astype(str), y=df[target_col], ax=ax)
		ax.set_title("Target by Category")
		ax.tick_params(axis="x", rotation=45)
	else:
		counts = df.groupby([feature, target_col]).size().reset_index(name="count")
		sns.barplot(data=counts, x=feature, y="count", hue=target_col, ax=ax)
		ax.set_title("Category vs Target")
		ax.tick_params(axis="x", rotation=45)
	st.pyplot(fig, use_container_width=True)


def render_pairplot(df: pd.DataFrame, numeric_cols: List[str], max_rows: int = 400) -> None:
	if len(numeric_cols) < 2:
		st.info("Need at least 2 numeric columns for pairplot.")
		return
	plot_cols = numeric_cols[:6]
	plot_df = df[plot_cols].dropna()
	if len(plot_df) > max_rows:
		plot_df = plot_df.sample(n=max_rows, random_state=42)
	grid = sns.pairplot(plot_df)
	st.pyplot(grid.fig, use_container_width=True)


def render_outlier_summary(df: pd.DataFrame, numeric_cols: List[str]) -> None:
	if not numeric_cols:
		st.info("No numeric columns for outlier summary.")
		return
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
	if not rows:
		st.info("No numeric columns with outlier counts.")
		return
	st.dataframe(pd.DataFrame(rows).sort_values("outlier_count", ascending=False))
