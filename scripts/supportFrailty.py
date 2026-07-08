#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for frailty classification with Digital Mobility Outcomes (DMOs).

This module centralizes DMO metadata, frailty-label normalization, simple
cleaning utilities, and optional descriptive/statistical helper functions used
by the orthogonal-feature and machine-learning scripts.

Authors
-------
Valerio Antonio Arcobelli
Jose Albites Sanabria
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, kruskal, mannwhitneyu, shapiro, ttest_ind
from statsmodels.stats.multitest import multipletests


# -----------------------------------------------------------------------------
# DMO metadata
# -----------------------------------------------------------------------------

DMO_INFO: dict[str, dict[str, str]] = {
    "walkdur_all_sum_w": {
        "label": "Walking duration",
        "unit": "h/day",
        "domain": "Amount",
    },
    "steps_all_sum_w": {
        "label": "Number of steps",
        "unit": "steps/day",
        "domain": "Amount",
    },
    "wb_all_sum_w": {
        "label": "Number of walking bouts",
        "unit": "WB/day",
        "domain": "Pattern",
    },
    "wb_10_sum_w": {
        "label": "Number of WB >10s",
        "unit": "WB/day",
        "domain": "Pattern",
    },
    "wb_30_sum_w": {
        "label": "Number of WB >30s",
        "unit": "WB/day",
        "domain": "Pattern",
    },
    "wb_60_sum_w": {
        "label": "Number of WB >60s",
        "unit": "WB/day",
        "domain": "Pattern",
    },
    "wbdur_all_avg_w": {
        "label": "WB duration",
        "unit": "s",
        "domain": "Pattern",
    },
    "wbdur_all_max_w": {
        "label": "P90 WB duration",
        "unit": "s",
        "domain": "Pattern",
    },
    "wbdur_all_var_w": {
        "label": "WB duration variability",
        "unit": "[-]",
        "domain": "Pattern",
    },
    "ws_1030_avg_w": {
        "label": "Walking speed 10–30s WB",
        "unit": "m/s",
        "domain": "Pace",
    },
    "ws_30_avg_w": {
        "label": "Walking speed >30s WB",
        "unit": "m/s",
        "domain": "Pace",
    },
    "ws_10_max_w": {
        "label": "P90 walking speed >10s WB",
        "unit": "m/s",
        "domain": "Pace",
    },
    "ws_30_max_w": {
        "label": "P90 walking speed >30s WB",
        "unit": "m/s",
        "domain": "Pace",
    },
    "strlen_1030_avg_w": {
        "label": "Stride length 10–30s WB",
        "unit": "cm",
        "domain": "Pace",
    },
    "strlen_30_avg_w": {
        "label": "Stride length >30s WB",
        "unit": "cm",
        "domain": "Pace",
    },
    "cadence_all_avg_w": {
        "label": "Cadence all WB",
        "unit": "steps/min",
        "domain": "Rhythm",
    },
    "cadence_30_avg_w": {
        "label": "Cadence >30s WB",
        "unit": "steps/min",
        "domain": "Rhythm",
    },
    "cadence_30_max_w": {
        "label": "P90 cadence >30s WB",
        "unit": "steps/min",
        "domain": "Rhythm",
    },
    "strdur_all_avg_w": {
        "label": "Stride duration all WB",
        "unit": "s",
        "domain": "Rhythm",
    },
    "strdur_30_avg_w": {
        "label": "Stride duration >30s WB",
        "unit": "s",
        "domain": "Rhythm",
    },
    "ws_30_var_w": {
        "label": "Walking speed variability >30s WB",
        "unit": "[-]",
        "domain": "Bout-to-bout variability",
    },
    "strlen_30_var_w": {
        "label": "Stride length variability >30s WB",
        "unit": "[-]",
        "domain": "Bout-to-bout variability",
    },
    "cadence_all_var_w": {
        "label": "Cadence variability all WB",
        "unit": "[-]",
        "domain": "Bout-to-bout variability",
    },
    "strdur_all_var_w": {
        "label": "Stride duration variability",
        "unit": "[-]",
        "domain": "Bout-to-bout variability",
    },
}

# Backward-compatible lower-case alias used in previous exploratory notebooks.
dmo_info = DMO_INFO

dmo_cols: list[str] = list(DMO_INFO.keys())
FRAILTY_ORDER: list[str] = ["robust", "pre-frail", "frail"]
frailty_order = FRAILTY_ORDER


# -----------------------------------------------------------------------------
# Metadata helpers
# -----------------------------------------------------------------------------

def get_dmo_metadata() -> pd.DataFrame:
    """Return DMO metadata as a tidy dataframe."""
    rows = []
    for feature, meta in DMO_INFO.items():
        rows.append(
            {
                "DMO": feature,
                "label": meta.get("label", feature),
                "unit": meta.get("unit", ""),
                "category": meta.get("domain", "Unknown"),
                "domain": meta.get("domain", "Unknown"),
            }
        )
    return pd.DataFrame(rows)


def get_domain_map() -> dict[str, str]:
    """Return mapping from DMO feature code to DMO domain."""
    return {feature: meta.get("domain", "Unknown") for feature, meta in DMO_INFO.items()}


def get_display_label_map(include_unit: bool = False) -> dict[str, str]:
    """Return mapping from DMO feature code to readable label."""
    labels = {}
    for feature, meta in DMO_INFO.items():
        label = meta.get("label", feature)
        unit = meta.get("unit", "")
        if include_unit and unit:
            label = f"{label} [{unit}]"
        labels[feature] = label
    return labels


def export_dmo_metadata(path: str | Path) -> None:
    """Save the DMO metadata table, useful as a GitHub-visible reference file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    get_dmo_metadata().to_csv(path, index=False)


def get_present_dmo_cols(df: pd.DataFrame, required_features: Sequence[str] | None = None) -> list[str]:
    """Return DMO columns present in a dataframe."""
    features = list(required_features) if required_features is not None else dmo_cols
    return [feature for feature in features if feature in df.columns]


# -----------------------------------------------------------------------------
# Frailty-label and dataframe cleaning
# -----------------------------------------------------------------------------

def normalize_frailty_status(values: pd.Series | Sequence[object]) -> pd.Series:
    """
    Normalize common Fried-status labels to lower-case labels used by the scripts.

    Accepted labels include Robust, Pre-frail / Prefrail / Pre frail, and Frail.
    Values outside robust/pre-frail/frail are returned as NaN.
    """
    series = pd.Series(values).astype(str).str.strip().str.lower()
    series = series.replace(
        {
            "pre frail": "pre-frail",
            "prefrail": "pre-frail",
            "pre_frail": "pre-frail",
            "pre-frailty": "pre-frail",
            "non robust": "non-robust",
            "not robust": "non-robust",
        }
    )
    series = series.where(series.isin(FRAILTY_ORDER), other=np.nan)
    return series


def clean_frailty_df(
    df: pd.DataFrame,
    outcome_col: str = "fried_status",
    age_col: str = "AGE",
    required_features: Sequence[str] | None = None,
    complete_case: bool = True,
) -> pd.DataFrame:
    """
    Standardize frailty labels, coerce age/DMOs to numeric, and optionally drop
    incomplete rows.

    Parameters
    ----------
    df:
        Input dataframe.
    outcome_col:
        Column containing Fried frailty status.
    age_col:
        Age column.
    required_features:
        DMO features required for complete-case cleaning. Defaults to all known
        DMOs that are present in the dataframe.
    complete_case:
        If True, drop rows missing outcome, age, or required features.
    """
    if outcome_col not in df.columns:
        raise KeyError(f"Missing outcome column: {outcome_col}")
    if age_col not in df.columns:
        raise KeyError(f"Missing age column: {age_col}")

    tmp = df.copy()
    tmp[outcome_col] = normalize_frailty_status(tmp[outcome_col])
    tmp[outcome_col] = pd.Categorical(tmp[outcome_col], categories=FRAILTY_ORDER, ordered=True)
    tmp[age_col] = pd.to_numeric(tmp[age_col], errors="coerce")

    if required_features is None:
        required_features = get_present_dmo_cols(tmp)
    required_features = list(required_features)

    for feature in required_features:
        if feature in tmp.columns:
            tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")

    if complete_case:
        subset = [outcome_col, age_col] + [feature for feature in required_features if feature in tmp.columns]
        tmp = tmp.dropna(subset=subset).copy()

    return tmp


def describe_by_group(
    df: pd.DataFrame,
    group_col: str = "fried_status",
    variables: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Descriptive statistics by frailty group."""
    if variables is None:
        variables = ["AGE", "steps_all_sum_w", "wb_60_sum_w", "ws_30_avg_w", "cadence_30_avg_w", "ws_30_var_w"]
    variables = [variable for variable in variables if variable in df.columns]
    return df.groupby(group_col, observed=False)[variables].agg(["mean", "std", "median", "min", "max", "count"])


# -----------------------------------------------------------------------------
# Optional H1/statistical helper functions
# -----------------------------------------------------------------------------

def kruskal_by_group(dataframe: pd.DataFrame, feature: str, group_col: str = "fried_status") -> tuple[float, float]:
    """Run Kruskal-Wallis test across frailty groups for one feature."""
    groups = []
    for group in FRAILTY_ORDER:
        values = dataframe.loc[dataframe[group_col].astype(str) == group, feature].dropna()
        if len(values) > 0:
            groups.append(values.values)
    if len(groups) < 2:
        return np.nan, np.nan
    statistic, p_value = kruskal(*groups)
    return float(statistic), float(p_value)


def pairwise_tests(dataframe: pd.DataFrame, feature: str, group_col: str = "fried_status") -> dict[str, float]:
    """Run pairwise Mann-Whitney tests and compute simple standardized mean differences."""
    pairs = [("robust", "pre-frail"), ("pre-frail", "frail"), ("robust", "frail")]
    results: dict[str, float] = {}

    for group_a, group_b in pairs:
        values_a = dataframe.loc[dataframe[group_col].astype(str) == group_a, feature].dropna()
        values_b = dataframe.loc[dataframe[group_col].astype(str) == group_b, feature].dropna()

        if len(values_a) > 0 and len(values_b) > 0:
            _, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
            pooled_sd = np.sqrt((values_a.var(ddof=1) + values_b.var(ddof=1)) / 2)
            effect_size = np.nan if pooled_sd == 0 else (values_b.mean() - values_a.mean()) / pooled_sd
        else:
            p_value = np.nan
            effect_size = np.nan

        results[f"{group_a}_vs_{group_b}_p"] = float(p_value) if not pd.isna(p_value) else np.nan
        results[f"{group_a}_vs_{group_b}_d"] = float(effect_size) if not pd.isna(effect_size) else np.nan

    return results


def build_h1_results(
    df: pd.DataFrame,
    features: Sequence[str] | None = None,
    group_col: str = "fried_status",
) -> pd.DataFrame:
    """Build a table of global and pairwise frailty-group comparisons for DMOs."""
    features = list(features) if features is not None else get_present_dmo_cols(df)
    rows = []

    for feature in features:
        if feature not in df.columns:
            continue
        _, p_global = kruskal_by_group(df, feature, group_col=group_col)
        pairwise = pairwise_tests(df, feature, group_col=group_col)
        grouped = df.groupby(group_col, observed=False)[feature].agg(["mean", "std", "median", "count"])

        row = {
            "feature": feature,
            "domain": get_domain_map().get(feature, "Unknown"),
            "label": get_display_label_map().get(feature, feature),
            "unit": DMO_INFO.get(feature, {}).get("unit", ""),
            "p_global": p_global,
            "robust_mean": grouped.loc["robust", "mean"] if "robust" in grouped.index else np.nan,
            "prefrail_mean": grouped.loc["pre-frail", "mean"] if "pre-frail" in grouped.index else np.nan,
            "frail_mean": grouped.loc["frail", "mean"] if "frail" in grouped.index else np.nan,
        }
        row.update(pairwise)
        rows.append(row)

    results = pd.DataFrame(rows)
    p_columns = [column for column in results.columns if column.endswith("_p") or column == "p_global"]
    for column in p_columns:
        mask = results[column].notna()
        results[column + "_FDR"] = np.nan
        if mask.any():
            results.loc[mask, column + "_FDR"] = multipletests(results.loc[mask, column], method="fdr_bh")[1]

    return results.sort_values(["domain", "p_global"], na_position="last")


def significance_stars(p_value: float) -> str:
    """Return conventional significance stars from a p-value."""
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def plot_fried_feature_pub(
    df: pd.DataFrame,
    feature: str,
    outdir: str | Path = ".",
    group_col: str = "fried_status",
) -> Path:
    """Save a publication-style box/swarm plot for one DMO by frailty group."""
    if feature not in df.columns:
        raise KeyError(f"Feature not found: {feature}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    label = get_display_label_map().get(feature, feature)
    unit = DMO_INFO.get(feature, {}).get("unit", "")
    ylabel = f"{label} [{unit}]" if unit else label

    df_plot = df[df[group_col].astype(str).isin(FRAILTY_ORDER)][[feature, group_col]].dropna().copy()
    df_plot[group_col] = pd.Categorical(df_plot[group_col].astype(str), categories=FRAILTY_ORDER, ordered=True)

    normal_groups = {}
    for group in FRAILTY_ORDER:
        values = df_plot.loc[df_plot[group_col] == group, feature]
        normal_groups[group] = False
        if len(values) >= 3:
            _, p_normal = shapiro(values)
            normal_groups[group] = bool(p_normal > 0.05)

    values_by_group = [df_plot.loc[df_plot[group_col] == group, feature] for group in FRAILTY_ORDER]
    if all(normal_groups.values()):
        test_name = "ANOVA"
        _, p_global = f_oneway(*values_by_group)
    else:
        test_name = "Kruskal-Wallis"
        _, p_global = kruskal(*values_by_group)

    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(
        data=df_plot,
        x=group_col,
        y=feature,
        order=FRAILTY_ORDER,
        color="lightgray",
        showcaps=True,
        showfliers=False,
        boxprops={"edgecolor": "gray"},
        medianprops={"color": "black"},
        whiskerprops={"color": "gray"},
        capprops={"color": "gray"},
    )
    sns.swarmplot(
        data=df_plot,
        x=group_col,
        y=feature,
        order=FRAILTY_ORDER,
        color="black",
        size=4,
        alpha=0.6,
        ax=ax,
    )
    ax.set_xlabel("Frailty status (Fried phenotype)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{label} by Fried status\n{test_name}, p={p_global:.3g}")
    plt.tight_layout()

    output_path = outdir / f"{feature}_by_fried_status.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
