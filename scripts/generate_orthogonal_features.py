#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate an orthogonal / low-redundancy Digital Mobility Outcome (DMO) feature set.

This script evaluates redundancy among candidate DMOs and proposes a compact
feature set for downstream frailty machine-learning models.

Authors
-------
Valerio Antonio Arcobelli
Jose Albites Sanabria

Example
-------
python scripts/generate_orthogonal_features.py \
    --input-csv FU4_FU5_merged.csv \
    --outdir orthogonality_report

Optional, when using an external metadata file:
python scripts/generate_orthogonal_features.py \
    --input-csv FU4_FU5_merged.csv \
    --labels-csv gait_labels.csv \
    --outdir orthogonality_report
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

# Non-interactive backend for servers and GitHub Actions.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

from supportFrailty import (
    clean_frailty_df,
    dmo_cols,
    export_dmo_metadata,
    get_display_label_map,
    get_domain_map,
)


DEFAULT_EXCLUDE_COLUMNS = {
    "CODE98",
    "ID",
    "Wave",
    "wave",
    "AGE",
    "SEX",
    "fried_status",
    "fried_status_all",
    "frailty_count",
    "frailty_status",
    "frailty_count_all",
    "frailty_status_all",
    "y",
    "target",
    "label",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a low-redundancy DMO feature set for frailty ML models."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV containing DMOs and Fried frailty status.")
    parser.add_argument("--outdir", default="orthogonality_report", help="Output folder.")
    parser.add_argument("--outcome-col", default="fried_status", help="Frailty-status column used for MI relevance.")
    parser.add_argument("--age-col", default="AGE", help="Age column used only for cleaning; age is not a candidate DMO.")
    parser.add_argument(
        "--labels-csv",
        default=None,
        help=(
            "Optional CSV with DMO metadata. If omitted, metadata from scripts/supportFrailty.py is used. "
            "The CSV should contain at least the feature-code and domain columns."
        ),
    )
    parser.add_argument("--feature-col", default="DMO", help="Feature-code column in labels CSV.")
    parser.add_argument("--domain-col", default="category", help="Domain/category column in labels CSV.")
    parser.add_argument(
        "--display-label-col",
        default=None,
        help="Optional readable-label column in labels CSV. If omitted, supportFrailty labels are used when available.",
    )
    parser.add_argument(
        "--rho-cap",
        type=float,
        default=0.50,
        help="Maximum allowed absolute Spearman correlation between selected features.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Optional maximum number of selected orthogonal features. Default: no fixed cap.",
    )
    parser.add_argument(
        "--mrmr-top-k",
        type=int,
        default=10,
        help="Number of features to include in the mRMR ranking output.",
    )
    parser.add_argument(
        "--mi-qbins",
        type=int,
        default=10,
        help="Quantile bins for pairwise MI redundancy in the mRMR step.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--strict-dmo-list",
        action="store_true",
        help=(
            "Use only DMOs listed in supportFrailty/labels. Without this flag, if no listed DMOs are "
            "present, numeric non-metadata columns are used as a fallback."
        ),
    )
    return parser.parse_args()


def ensure_outdir(path: str | Path) -> Path:
    """Create and return output directory."""
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def load_feature_metadata(
    labels_csv: str | None,
    feature_col: str,
    domain_col: str,
    display_label_col: str | None,
) -> tuple[list[str], dict[str, str], dict[str, str], pd.DataFrame]:
    """
    Load DMO metadata from an optional CSV, falling back to supportFrailty.py.

    Returns
    -------
    feature_codes, domain_map, display_map, metadata_dataframe
    """
    support_features = list(dmo_cols)
    domain_map = get_domain_map()
    display_map = get_display_label_map(include_unit=False)

    if labels_csv is None:
        metadata = pd.DataFrame(
            {
                "DMO": support_features,
                "label": [display_map.get(feature, feature) for feature in support_features],
                "category": [domain_map.get(feature, "Unknown") for feature in support_features],
            }
        )
        return support_features, domain_map, display_map, metadata

    labels = pd.read_csv(labels_csv)
    required = {feature_col, domain_col}
    missing = required.difference(labels.columns)
    if missing:
        raise KeyError(f"labels CSV is missing required columns: {sorted(missing)}")

    labels[feature_col] = labels[feature_col].astype(str)
    labels[domain_col] = labels[domain_col].astype(str)

    feature_codes = labels[feature_col].tolist()
    csv_domain_map = dict(zip(labels[feature_col], labels[domain_col]))
    domain_map.update(csv_domain_map)

    if display_label_col and display_label_col in labels.columns:
        csv_display_map = dict(zip(labels[feature_col], labels[display_label_col].astype(str)))
        display_map.update(csv_display_map)

    metadata = labels.copy()
    return feature_codes, domain_map, display_map, metadata


def infer_feature_columns(
    df: pd.DataFrame,
    listed_features: list[str],
    outcome_col: str,
    strict_dmo_list: bool,
) -> list[str]:
    """Return candidate DMO feature columns available in the input data."""
    present = [feature for feature in listed_features if feature in df.columns]
    missing = [feature for feature in listed_features if feature not in df.columns]

    if missing:
        print(f"[WARN] {len(missing)} listed DMO columns are not in the input file. First missing: {missing[:10]}")

    if present:
        return present

    if strict_dmo_list:
        raise ValueError("None of the listed DMO features are present in the input CSV.")

    excluded = set(DEFAULT_EXCLUDE_COLUMNS)
    excluded.add(outcome_col)
    numeric_cols = [
        column for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not numeric_cols:
        raise ValueError("No candidate DMO columns found. Check input data or provide --labels-csv.")

    print("[WARN] No listed DMO features were found; falling back to numeric columns.")
    return numeric_cols


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    outcome_col: str,
    age_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean frailty labels, coerce features to numeric, and return complete-case X/y."""
    # If age exists, use shared cleaning. If not, clean outcome/features manually.
    if age_col in df.columns:
        work = clean_frailty_df(
            df,
            outcome_col=outcome_col,
            age_col=age_col,
            required_features=feature_cols,
            complete_case=True,
        )
    else:
        work = df.copy()
        for feature in feature_cols:
            work[feature] = pd.to_numeric(work[feature], errors="coerce")
        work = work.dropna(subset=[outcome_col] + feature_cols).copy()

    if work.empty:
        raise RuntimeError("No complete rows remain after cleaning DMO/outcome values.")

    X = work[feature_cols].copy()
    y = work[outcome_col].astype(str).copy()
    return X, y


def save_correlation_outputs(X: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Save Spearman correlation matrix, heatmap, and hierarchical dendrogram."""
    corr = X.corr(method="spearman")
    corr = corr.loc[corr.columns, corr.columns]
    corr.to_csv(outdir / "spearman_corr.csv", float_format="%.6f")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="vlag", square=True, cbar=True)
    plt.title("Spearman correlation among DMOs")
    plt.tight_layout()
    plt.savefig(outdir / "spearman_corr_heatmap.png", dpi=300)
    plt.close()

    dist = 1 - corr.abs()
    np.fill_diagonal(dist.values, 0.0)
    linkage_matrix = linkage(squareform(dist.values, checks=False), method="ward")

    plt.figure(figsize=(12, 4))
    dendrogram(linkage_matrix, labels=corr.columns.tolist(), leaf_rotation=90)
    plt.title("Feature clustering: 1 - |Spearman rho|")
    plt.tight_layout()
    plt.savefig(outdir / "feature_dendrogram.png", dpi=300)
    plt.close()

    return corr


def save_vif_table(X: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Compute variance inflation factors for candidate features."""
    Xc = add_constant(X, has_constant="add")
    rows = []
    for index, feature in enumerate(Xc.columns):
        if feature == "const":
            continue
        try:
            value = variance_inflation_factor(Xc.values, index)
        except Exception:
            value = np.nan
        rows.append({"feature": feature, "VIF": value})

    vif = pd.DataFrame(rows).sort_values("VIF", ascending=False)
    vif.to_csv(outdir / "vif_table.csv", index=False)
    return vif


def save_belsley_diagnostics(X: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Compute Belsley condition indices from standardized features."""
    Xs = (X - X.mean()) / X.std(ddof=0)
    Xs = Xs.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    xtx = Xs.values.T @ Xs.values
    eigvals, _ = np.linalg.eigh(xtx)
    eigvals = np.maximum(eigvals, 1e-12)
    condition_index = np.sqrt(eigvals.max() / eigvals)

    belsley = pd.DataFrame({"eigenvalue": eigvals, "condition_index": condition_index})
    belsley = belsley.sort_values("eigenvalue", ascending=True)
    belsley.to_csv(outdir / "belsley_diagnostics.csv", index=False)
    return belsley


def save_unique_variance(X: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Compute unique variance: 1 - R²(feature predicted by all other features)."""
    rows = []
    model = LinearRegression()

    for feature in X.columns:
        if X.shape[1] == 1:
            r2 = 0.0
        else:
            y_feature = X[feature].values
            X_other = X.drop(columns=[feature]).values
            model.fit(X_other, y_feature)
            r2 = model.score(X_other, y_feature)

        rows.append(
            {
                "feature": feature,
                "unique_variance": float(max(0.0, 1.0 - r2)),
                "r2_explained_by_others": float(min(1.0, max(0.0, r2))),
            }
        )

    unique = pd.DataFrame(rows).sort_values("unique_variance", ascending=False)
    unique.to_csv(outdir / "unique_variance.csv", index=False)
    return unique


def pairwise_mutual_information(X: pd.DataFrame, feature_a: str, feature_b: str, qbins: int) -> float:
    """Estimate pairwise mutual information after quantile binning two continuous features."""
    a = pd.qcut(X[feature_a], q=qbins, duplicates="drop").cat.codes
    b = pd.qcut(X[feature_b], q=qbins, duplicates="drop").cat.codes
    return float(mutual_info_score(a, b))


def save_mrmr_rank(
    X: pd.DataFrame,
    y: pd.Series,
    outdir: Path,
    top_k: int,
    qbins: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Create a simple greedy mRMR ranking: MI relevance minus MI redundancy."""
    y_encoded = LabelEncoder().fit_transform(y.astype(str).values)
    mi_relevance = mutual_info_classif(X.values, y_encoded, discrete_features=False, random_state=random_state)
    relevance = dict(zip(X.columns.tolist(), mi_relevance))

    selected: list[str] = []
    remaining = X.columns.tolist()
    if not remaining:
        raise RuntimeError("No features available for mRMR ranking.")

    seed = max(remaining, key=lambda feature: relevance[feature])
    selected.append(seed)
    remaining.remove(seed)

    def mrmr_score(feature: str, selected_features: Iterable[str]) -> float:
        selected_features = list(selected_features)
        if not selected_features:
            return relevance[feature]
        redundancy = np.mean(
            [pairwise_mutual_information(X, feature, selected_feature, qbins) for selected_feature in selected_features]
        )
        return float(relevance[feature] - redundancy)

    while remaining and len(selected) < min(top_k, X.shape[1]):
        next_feature = max(remaining, key=lambda feature: mrmr_score(feature, selected))
        selected.append(next_feature)
        remaining.remove(next_feature)

    mrmr = pd.DataFrame(
        {
            "rank": np.arange(1, len(selected) + 1),
            "feature": selected,
            "mi_relevance": [relevance[feature] for feature in selected],
        }
    )
    mrmr.to_csv(outdir / "mrmr_rank.csv", index=False)
    return mrmr, relevance


def save_optional_distance_correlation(X: pd.DataFrame, outdir: Path) -> None:
    """Try to compute distance correlation if supported by the local SciPy version."""
    try:
        from scipy.stats import distance_correlation

        features = X.columns.tolist()
        matrix = np.ones((len(features), len(features)))
        for i, feature_a in enumerate(features):
            for j, feature_b in enumerate(features):
                if j < i:
                    continue
                value = distance_correlation(X[feature_a].values, X[feature_b].values)
                matrix[i, j] = matrix[j, i] = float(value)

        pd.DataFrame(matrix, index=features, columns=features).to_csv(
            outdir / "distance_corr_matrix.csv", float_format="%.6f"
        )
    except Exception as exc:
        (outdir / "distance_corr_matrix.SKIPPED.txt").write_text(str(exc), encoding="utf-8")


def select_orthogonal_features(
    corr: pd.DataFrame,
    unique: pd.DataFrame,
    relevance: dict[str, float],
    rho_cap: float,
    max_features: int | None,
) -> list[str]:
    """
    Greedily select features with high unique variance and outcome relevance.

    A candidate is accepted only if its absolute Spearman correlation with all
    already-selected features is <= rho_cap.
    """
    unique_score = unique.set_index("feature")["unique_variance"]
    unique_score = (unique_score - unique_score.min()) / (unique_score.max() - unique_score.min() + 1e-12)

    relevance_score = pd.Series(relevance, name="mi_relevance")
    relevance_score = (relevance_score - relevance_score.min()) / (
        relevance_score.max() - relevance_score.min() + 1e-12
    )

    total_score = unique_score.add(relevance_score, fill_value=0.0)
    total_score = total_score.replace([np.inf, -np.inf], np.nan).dropna()
    order = [feature for feature in total_score.sort_values(ascending=False).index if feature in corr.columns]

    selected: list[str] = []
    for feature in order:
        is_low_redundancy = all(abs(corr.loc[feature, selected_feature]) <= rho_cap for selected_feature in selected)
        if is_low_redundancy:
            selected.append(feature)
        if max_features is not None and len(selected) >= max_features:
            break

    return selected


def save_orthogonal_outputs(
    selected: list[str],
    corr: pd.DataFrame,
    domain_map: dict[str, str],
    display_map: dict[str, str],
    outdir: Path,
) -> None:
    """Save selected feature list, feature mapping, and selected-set heatmap."""
    (outdir / "orthogonal_feature_set.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")

    mapping = pd.DataFrame(
        {
            "feature": selected,
            "display_label": [display_map.get(feature, feature) for feature in selected],
            "domain": [domain_map.get(feature, "Unknown") for feature in selected],
        }
    )
    mapping.to_csv(outdir / "orthogonal_feature_set_mapping.csv", index=False)

    if len(selected) < 2:
        print("[INFO] Fewer than two features selected; selected-set heatmap skipped.")
        return

    corr_subset = corr.loc[selected, selected].copy()

    # Reorder selected features for a cleaner visual block structure.
    try:
        dist_subset = 1 - corr_subset.abs()
        np.fill_diagonal(dist_subset.values, 0.0)
        linkage_subset = linkage(squareform(dist_subset.values, checks=False), method="ward")
        order_idx = leaves_list(linkage_subset)
        corr_subset = corr_subset.iloc[order_idx, order_idx]
    except Exception:
        pass

    labels = [display_map.get(feature, feature) for feature in corr_subset.columns]

    fig_size = max(8, 0.75 * len(labels) + 4)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        corr_subset,
        vmin=-1,
        vmax=1,
        cmap="vlag",
        square=True,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Spearman rho"},
    )
    plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=35, ha="right")
    plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
    plt.title(f"Spearman correlation: orthogonal DMO set (n={len(selected)})")
    plt.tight_layout()
    plt.savefig(outdir / "spearman_heatmap_orthogonal_set.png", dpi=300)
    plt.savefig(outdir / "spearman_heatmap_orthogonal_set.svg")
    plt.close()


def save_run_config(args: argparse.Namespace, feature_cols: list[str], outdir: Path) -> None:
    """Save run settings for reproducibility."""
    config = vars(args).copy()
    config["n_candidate_features"] = len(feature_cols)
    config["candidate_features"] = feature_cols
    with open(outdir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def main() -> None:
    args = parse_args()
    outdir = ensure_outdir(args.outdir)

    df = pd.read_csv(args.input_csv)
    listed_features, domain_map, display_map, metadata = load_feature_metadata(
        args.labels_csv,
        feature_col=args.feature_col,
        domain_col=args.domain_col,
        display_label_col=args.display_label_col,
    )

    # Save a visible metadata snapshot used by this run.
    metadata.to_csv(outdir / "dmo_metadata_used.csv", index=False)
    export_dmo_metadata(outdir / "supportFrailty_dmo_metadata.csv")

    feature_cols = infer_feature_columns(
        df=df,
        listed_features=listed_features,
        outcome_col=args.outcome_col,
        strict_dmo_list=args.strict_dmo_list,
    )
    X, y = prepare_feature_matrix(df, feature_cols, args.outcome_col, args.age_col)

    print(f"[INFO] Complete-case sample: n={X.shape[0]}, features={X.shape[1]}")

    corr = save_correlation_outputs(X, outdir)
    save_vif_table(X, outdir)
    save_belsley_diagnostics(X, outdir)
    unique = save_unique_variance(X, outdir)
    _, relevance = save_mrmr_rank(
        X,
        y,
        outdir,
        top_k=args.mrmr_top_k,
        qbins=args.mi_qbins,
        random_state=args.random_state,
    )
    save_optional_distance_correlation(X, outdir)

    selected = select_orthogonal_features(
        corr=corr,
        unique=unique,
        relevance=relevance,
        rho_cap=args.rho_cap,
        max_features=args.max_features,
    )
    save_orthogonal_outputs(selected, corr, domain_map, display_map, outdir)
    save_run_config(args, feature_cols, outdir)

    print(f"[OK] Orthogonality report saved to: {outdir}")
    print(f"[OK] Selected orthogonal features (n={len(selected)}): {selected}")


if __name__ == "__main__":
    main()
