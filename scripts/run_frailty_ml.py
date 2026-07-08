#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine-learning pipeline for frailty classification from Digital Mobility Outcomes.

The pipeline evaluates two binary classification tasks:
1. Robust vs Pre-frail
2. Robust vs Non-robust, where Non-robust = Pre-frail + Frail

The default analysis uses two feature sets:
1. ORTHO: orthogonal/low-redundancy DMO features
2. ORTHO_AGE: orthogonal/low-redundancy DMO features plus age

Authors
-------
Valerio Antonio Arcobelli

Example
-------
python scripts/run_frailty_ml.py \
    --frailty-csv FU4_FU5_merged.csv \
    --dmo-csv dmos_python.csv \
    --orthogonal-features orthogonality_report/orthogonal_feature_set.txt \
    --outdir ml_results
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, shapiro
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from supportFrailty import (
    clean_frailty_df,
    dmo_cols,
    export_dmo_metadata,
    get_display_label_map,
    get_domain_map,
    normalize_frailty_status,
)


RANDOM_STATE = 123
VALID_STATUS = {"robust", "pre-frail", "frail"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run frailty ML models using orthogonal DMO features.")
    parser.add_argument("--frailty-csv", required=True, help="CSV containing CODE98, Wave, AGE, and fried_status.")
    parser.add_argument(
        "--dmo-csv",
        default=None,
        help=(
            "Optional DMO CSV to merge with frailty CSV on CODE98 and Wave. "
            "If omitted, --frailty-csv is assumed to already contain DMO columns."
        ),
    )
    parser.add_argument(
        "--orthogonal-features",
        required=True,
        help="Text file with one selected orthogonal DMO feature per line.",
    )
    parser.add_argument(
        "--labels-csv",
        default=None,
        help=(
            "Optional CSV with DMO metadata. If omitted, DMO metadata from scripts/supportFrailty.py is used. "
            "The CSV should contain columns such as DMO and category."
        ),
    )
    parser.add_argument("--outdir", default="ml_results", help="Output directory.")
    parser.add_argument("--id-col", default="CODE98")
    parser.add_argument("--wave-col", default="Wave")
    parser.add_argument("--age-col", default="AGE")
    parser.add_argument("--outcome-col", default="fried_status")
    parser.add_argument("--feature-col", default="DMO", help="Feature-code column in labels CSV.")
    parser.add_argument("--domain-col", default="category", help="Domain/category column in labels CSV.")
    parser.add_argument("--display-label-col", default=None, help="Optional readable-label column in labels CSV.")
    parser.add_argument("--n-outer", type=int, default=5, help="Outer CV folds.")
    parser.add_argument("--n-inner", type=int, default=5, help="Inner CV folds for hyperparameter tuning.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--permutation-repeats", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for grid search and tree models. Use -1 for all cores.")
    parser.add_argument(
        "--skip-permutation-importance",
        action="store_true",
        help="Skip permutation importance to speed up the run.",
    )
    parser.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="Save Shapiro and Mann-Whitney exploratory diagnostics for all listed DMOs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge_lr", "elasticnet_lr", "svm_linear", "random_forest", "gbm", "extra_trees"],
        choices=["ridge_lr", "elasticnet_lr", "svm_linear", "random_forest", "gbm", "extra_trees"],
        help="Models to run.",
    )
    return parser.parse_args()


def ensure_output_folders(outdir: str | Path) -> dict[str, Path]:
    """Create all output folders and return their paths."""
    root = Path(outdir)
    folders = {
        "root": root,
        "tables": root / "tables",
        "predictions": root / "predictions_oof",
        "plots": root / "plots",
        "config": root / "config",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def read_feature_metadata(
    labels_csv: str | None,
    feature_col: str,
    domain_col: str,
    display_label_col: str | None,
) -> tuple[list[str], dict[str, str], dict[str, str], pd.DataFrame]:
    """Read DMO metadata from labels CSV or from supportFrailty.py."""
    feature_codes = list(dmo_cols)
    domain_map = get_domain_map()
    display_map = get_display_label_map(include_unit=False)

    if labels_csv is None:
        metadata = pd.DataFrame(
            {
                "DMO": feature_codes,
                "label": [display_map.get(feature, feature) for feature in feature_codes],
                "category": [domain_map.get(feature, "Unknown") for feature in feature_codes],
            }
        )
        return feature_codes, domain_map, display_map, metadata

    labels = pd.read_csv(labels_csv)
    missing = {feature_col, domain_col}.difference(labels.columns)
    if missing:
        raise KeyError(f"labels CSV is missing required columns: {sorted(missing)}")

    labels[feature_col] = labels[feature_col].astype(str)
    labels[domain_col] = labels[domain_col].astype(str)

    feature_codes = labels[feature_col].tolist()
    domain_map.update(dict(zip(labels[feature_col], labels[domain_col])))

    if display_label_col and display_label_col in labels.columns:
        display_map.update(dict(zip(labels[feature_col], labels[display_label_col].astype(str))))

    return feature_codes, domain_map, display_map, labels.copy()


def read_orthogonal_features(path: str | Path) -> list[str]:
    """Read one feature name per line from the orthogonal-feature text file."""
    features = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    features = [feature for feature in features if feature]
    if not features:
        raise ValueError("The orthogonal feature file is empty.")
    return features


def standardize_merge_keys(df: pd.DataFrame, id_col: str, wave_col: str) -> pd.DataFrame:
    """Standardize ID and wave columns before merging."""
    df = df.copy()
    if id_col not in df.columns and "ID" in df.columns:
        df = df.rename(columns={"ID": id_col})
    if id_col not in df.columns:
        raise KeyError(f"Missing ID column: {id_col}")
    if wave_col not in df.columns:
        raise KeyError(f"Missing wave column: {wave_col}")

    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    df[wave_col] = df[wave_col].astype(str)
    return df.dropna(subset=[id_col, wave_col])


def load_analysis_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """Load frailty data and optionally merge DMO data."""
    frailty = pd.read_csv(args.frailty_csv)
    frailty = standardize_merge_keys(frailty, args.id_col, args.wave_col)

    if args.dmo_csv is None:
        return frailty

    dmo = pd.read_csv(args.dmo_csv)
    dmo = standardize_merge_keys(dmo, args.id_col, args.wave_col)

    duplicated_frailty = frailty.duplicated([args.id_col, args.wave_col]).sum()
    duplicated_dmo = dmo.duplicated([args.id_col, args.wave_col]).sum()
    if duplicated_frailty:
        print(f"[WARN] Frailty file has {duplicated_frailty} duplicated ID-wave rows.")
    if duplicated_dmo:
        print(f"[WARN] DMO file has {duplicated_dmo} duplicated ID-wave rows.")

    merged = pd.merge(frailty, dmo, on=[args.id_col, args.wave_col], how="inner")
    if merged.empty:
        raise RuntimeError("The frailty and DMO files did not merge to any rows. Check CODE98/Wave values.")
    return merged


def validate_inputs(
    df: pd.DataFrame,
    args: argparse.Namespace,
    listed_features: list[str],
    orthogonal_features: list[str],
) -> tuple[list[str], list[str]]:
    """Validate required columns and return present listed/orthogonal features."""
    required = [args.id_col, args.wave_col, args.age_col, args.outcome_col]
    missing_required = [column for column in required if column not in df.columns]
    if missing_required:
        raise KeyError(f"Missing required columns in analysis dataframe: {missing_required}")

    listed_present = [feature for feature in listed_features if feature in df.columns]
    if not listed_present:
        raise ValueError("No listed DMO features were found in the analysis dataframe.")

    orthogonal_present = [feature for feature in orthogonal_features if feature in df.columns]
    orthogonal_missing = [feature for feature in orthogonal_features if feature not in df.columns]
    if orthogonal_missing:
        print(f"[WARN] Orthogonal features missing from data: {orthogonal_missing}")
    if not orthogonal_present:
        raise ValueError("None of the orthogonal features are present in the analysis dataframe.")

    return listed_present, orthogonal_present


def make_outcome_dataframes(df: pd.DataFrame, outcome_col: str) -> dict[str, pd.DataFrame]:
    """Create the two binary classification outcomes."""
    work = df.copy()
    work[outcome_col] = normalize_frailty_status(work[outcome_col])
    work = work[work[outcome_col].isin(VALID_STATUS)].copy()

    robust_prefrail = work[work[outcome_col].isin(["robust", "pre-frail"])].copy()
    robust_prefrail["y"] = (robust_prefrail[outcome_col] == "pre-frail").astype(int)

    robust_nonrobust = work[work[outcome_col].isin(["robust", "pre-frail", "frail"])].copy()
    robust_nonrobust["y"] = (robust_nonrobust[outcome_col] != "robust").astype(int)

    return {
        "robust_vs_prefrail": robust_prefrail,
        "robust_vs_nonrobust": robust_nonrobust,
    }


def safe_n_splits(y: pd.Series, requested: int) -> int:
    """Return a feasible number of stratified folds."""
    class_counts = y.value_counts()
    min_class = int(class_counts.min())
    n_splits = min(requested, min_class)
    if n_splits < 2:
        raise ValueError(f"Not enough observations per class for CV. Class counts: {class_counts.to_dict()}")
    return n_splits


def get_model_grid(model_name: str, random_state: int, n_jobs: int = 1) -> tuple[Any, dict[str, list[Any]]]:
    """Return estimator and hyperparameter grid for a model."""
    if model_name == "ridge_lr":
        estimator = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000)),
            ]
        )
        grid = {"classifier__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}

    elif model_name == "elasticnet_lr":
        estimator = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000),
                ),
            ]
        )
        grid = {
            "classifier__C": [0.01, 0.03, 0.1, 0.3, 1.0],
            "classifier__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }

    elif model_name == "svm_linear":
        estimator = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="linear", probability=True, random_state=random_state)),
            ]
        )
        grid = {"classifier__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}

    elif model_name == "random_forest":
        estimator = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
        grid = {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 3, 6, 10],
            "min_samples_leaf": [1, 3, 5, 9],
        }

    elif model_name == "extra_trees":
        estimator = ExtraTreesClassifier(random_state=random_state, n_jobs=n_jobs)
        grid = {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 3, 6, 10],
            "min_samples_leaf": [1, 3, 5, 9],
        }

    elif model_name == "gbm":
        estimator = HistGradientBoostingClassifier(random_state=random_state)
        grid = {
            "learning_rate": [0.03, 0.1],
            "max_depth": [2, 3, None],
            "max_iter": [200, 500],
        }

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return estimator, grid


def run_nested_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    outcome_name: str,
    feature_set_name: str,
    args: argparse.Namespace,
    folders: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run nested CV for one outcome, feature set, and model."""
    model_df = df.copy()
    for feature in feature_cols:
        model_df[feature] = pd.to_numeric(model_df[feature], errors="coerce")
    model_df = model_df.dropna(subset=["y"] + feature_cols).reset_index(drop=True)

    y = model_df["y"].astype(int)
    X = model_df[feature_cols]

    outer_splits = safe_n_splits(y, args.n_outer)
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=args.random_state)

    estimator, grid = get_model_grid(model_name, args.random_state, args.n_jobs)

    fold_rows = []
    prediction_rows = []
    importance_rows = []
    mean_fpr = np.linspace(0, 1, 200)
    interpolated_tprs = []

    key_cols = [column for column in [args.id_col, args.wave_col] if column in model_df.columns]

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        inner_splits = safe_n_splits(y_train, args.n_inner)
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=args.random_state)

        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=args.n_jobs,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_proba = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        fold_rows.append(
            {
                "outcome": outcome_name,
                "feature_set": feature_set_name,
                "model": model_name,
                "fold": fold,
                "auc": float(auc),
                "best_inner_auc": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, default=str),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_features": int(len(feature_cols)),
            }
        )

        test_keys = model_df.iloc[test_idx][key_cols].reset_index(drop=True) if key_cols else pd.DataFrame()
        for row_idx, (truth, proba) in enumerate(zip(y_test.tolist(), y_proba.tolist())):
            row = {
                "outcome": outcome_name,
                "feature_set": feature_set_name,
                "model": model_name,
                "fold": fold,
                "y_true": int(truth),
                "predicted_probability": float(proba),
            }
            for key_col in key_cols:
                row[key_col] = test_keys.loc[row_idx, key_col]
            prediction_rows.append(row)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        interpolated = np.interp(mean_fpr, fpr, tpr)
        interpolated[0] = 0.0
        interpolated_tprs.append(interpolated)

        if not args.skip_permutation_importance:
            permutation = permutation_importance(
                best_model,
                X_test,
                y_test,
                scoring="roc_auc",
                n_repeats=args.permutation_repeats,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            )
            for feature, mean_drop, sd_drop in zip(
                feature_cols,
                permutation.importances_mean,
                permutation.importances_std,
            ):
                importance_rows.append(
                    {
                        "outcome": outcome_name,
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "fold": fold,
                        "feature": feature,
                        "permutation_auc_drop_mean": float(mean_drop),
                        "permutation_auc_drop_sd": float(sd_drop),
                    }
                )

    fold_results = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    importances = pd.DataFrame(importance_rows)

    pred_file = folders["predictions"] / f"{outcome_name}__{feature_set_name}__{model_name}__oof_predictions.csv"
    predictions.to_csv(pred_file, index=False)

    save_roc_plot(
        interpolated_tprs,
        mean_fpr,
        fold_results,
        outcome_name,
        feature_set_name,
        model_name,
        folders["plots"],
    )

    return fold_results, predictions, importances


def save_roc_plot(
    interpolated_tprs: list[np.ndarray],
    mean_fpr: np.ndarray,
    fold_results: pd.DataFrame,
    outcome_name: str,
    feature_set_name: str,
    model_name: str,
    plot_dir: Path,
) -> None:
    """Save mean ROC curve across outer folds."""
    if not interpolated_tprs:
        return

    tprs = np.vstack(interpolated_tprs)
    mean_tpr = tprs.mean(axis=0)
    sd_tpr = tprs.std(axis=0, ddof=1) if tprs.shape[0] > 1 else np.zeros_like(mean_tpr)
    mean_tpr[-1] = 1.0

    mean_auc = fold_results["auc"].mean()
    sd_auc = fold_results["auc"].std(ddof=1)

    upper = np.minimum(mean_tpr + sd_tpr, 1.0)
    lower = np.maximum(mean_tpr - sd_tpr, 0.0)

    plt.figure(figsize=(7.5, 6))
    plt.plot(mean_fpr, mean_tpr, linewidth=2, label=f"Mean ROC, AUC={mean_auc:.3f} ± {sd_auc:.3f}")
    plt.fill_between(mean_fpr, lower, upper, alpha=0.2, label="±1 SD")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{outcome_name} | {feature_set_name} | {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_dir / f"ROC__{outcome_name}__{feature_set_name}__{model_name}.png", dpi=300)
    plt.close()


def summarize_performance(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate outer-fold AUC by outcome, feature set, and model."""
    return (
        fold_results
        .groupby(["outcome", "feature_set", "model"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_sd=("auc", "std"),
            n_folds=("fold", "nunique"),
            n_features=("n_features", "first"),
        )
        .sort_values(["outcome", "feature_set", "auc_mean"], ascending=[True, True, False])
    )


def summarize_importance(importances: pd.DataFrame, domain_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate permutation importance at feature and domain level."""
    if importances.empty:
        return pd.DataFrame(), pd.DataFrame()

    feature_summary = (
        importances
        .groupby(["outcome", "feature_set", "model", "feature"], as_index=False)
        .agg(
            permutation_auc_drop_mean=("permutation_auc_drop_mean", "mean"),
            permutation_auc_drop_sd_across_folds=("permutation_auc_drop_mean", "std"),
        )
    )
    feature_summary["domain"] = feature_summary["feature"].map(domain_map).fillna("Age/Other")
    feature_summary = feature_summary.sort_values(
        ["outcome", "feature_set", "model", "permutation_auc_drop_mean"],
        ascending=[True, True, True, False],
    )

    domain_summary = (
        feature_summary
        .groupby(["outcome", "feature_set", "model", "domain"], as_index=False)
        .agg(
            domain_permutation_auc_drop_sum=("permutation_auc_drop_mean", "sum"),
            n_features=("feature", "nunique"),
        )
        .sort_values(
            ["outcome", "feature_set", "model", "domain_permutation_auc_drop_sum"],
            ascending=[True, True, True, False],
        )
    )
    return feature_summary, domain_summary


def save_exploratory_diagnostics(
    df: pd.DataFrame,
    feature_cols: list[str],
    outcome_col: str,
    folders: dict[str, Path],
) -> None:
    """Save simple Shapiro and Mann-Whitney diagnostics for listed DMOs."""
    diagnostic_df = df.copy()
    diagnostic_df[outcome_col] = normalize_frailty_status(diagnostic_df[outcome_col])

    rows = []
    for outcome_name, outcome_df in make_outcome_dataframes(diagnostic_df, outcome_col).items():
        for feature in feature_cols:
            if feature not in outcome_df.columns:
                continue
            working = outcome_df.copy()
            working["_feature"] = pd.to_numeric(working[feature], errors="coerce")
            working = working.dropna(subset=["_feature", "y"])
            if working.empty:
                continue

            sample = working["_feature"].sample(min(5000, working.shape[0]), random_state=1)
            shapiro_p = np.nan
            if len(sample) >= 3:
                _, shapiro_p = shapiro(sample)

            group0 = working.loc[working["y"] == 0, "_feature"]
            group1 = working.loc[working["y"] == 1, "_feature"]
            mannwhitney_p = np.nan
            if len(group0) > 0 and len(group1) > 0:
                _, mannwhitney_p = mannwhitneyu(group0, group1, alternative="two-sided")

            rows.append(
                {
                    "outcome": outcome_name,
                    "feature": feature,
                    "n": int(working.shape[0]),
                    "shapiro_p": shapiro_p,
                    "mannwhitney_p": mannwhitney_p,
                    "median_y0": float(group0.median()) if len(group0) else np.nan,
                    "median_y1": float(group1.median()) if len(group1) else np.nan,
                }
            )

    pd.DataFrame(rows).to_csv(folders["tables"] / "exploratory_diagnostics.csv", index=False)


def save_run_config(args: argparse.Namespace, feature_sets: dict[str, list[str]], folders: dict[str, Path]) -> None:
    """Save run settings and feature sets for reproducibility."""
    config = vars(args).copy()
    config["feature_sets"] = feature_sets
    with open(folders["config"] / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def save_metadata_tables(
    metadata: pd.DataFrame,
    display_map: dict[str, str],
    domain_map: dict[str, str],
    folders: dict[str, Path],
) -> None:
    """Save metadata snapshots used by the ML run."""
    metadata.to_csv(folders["config"] / "dmo_metadata_used.csv", index=False)
    export_dmo_metadata(folders["config"] / "supportFrailty_dmo_metadata.csv")

    mapping = pd.DataFrame(
        {
            "feature": list(domain_map.keys()),
            "display_label": [display_map.get(feature, feature) for feature in domain_map.keys()],
            "domain": [domain_map.get(feature, "Unknown") for feature in domain_map.keys()],
        }
    )
    mapping.to_csv(folders["config"] / "feature_domain_mapping.csv", index=False)


def main() -> None:
    args = parse_args()
    folders = ensure_output_folders(args.outdir)

    listed_features, domain_map, display_map, metadata = read_feature_metadata(
        args.labels_csv,
        feature_col=args.feature_col,
        domain_col=args.domain_col,
        display_label_col=args.display_label_col,
    )
    orthogonal_features_raw = read_orthogonal_features(args.orthogonal_features)
    analysis_df = load_analysis_dataframe(args)

    listed_present, orthogonal_features = validate_inputs(
        analysis_df,
        args,
        listed_features=listed_features,
        orthogonal_features=orthogonal_features_raw,
    )

    analysis_df = clean_frailty_df(
        analysis_df,
        outcome_col=args.outcome_col,
        age_col=args.age_col,
        required_features=orthogonal_features,
        complete_case=False,
    )

    feature_sets = {
        "ORTHO": orthogonal_features,
        "ORTHO_AGE": orthogonal_features + [args.age_col],
    }
    save_run_config(args, feature_sets, folders)
    save_metadata_tables(metadata, display_map, domain_map, folders)

    if args.save_diagnostics:
        save_exploratory_diagnostics(analysis_df, listed_present, args.outcome_col, folders)

    outcome_dataframes = make_outcome_dataframes(analysis_df, args.outcome_col)
    all_fold_results = []
    all_predictions = []
    all_importances = []

    for outcome_name, outcome_df in outcome_dataframes.items():
        print(f"\n[INFO] Outcome: {outcome_name}")
        print(outcome_df["y"].value_counts().to_dict())

        for feature_set_name, features in feature_sets.items():
            print(f"[INFO] Feature set: {feature_set_name} ({len(features)} features)")

            for model_name in args.models:
                print(f"[INFO] Running model: {model_name}")
                fold_results, predictions, importances = run_nested_cv(
                    df=outcome_df,
                    feature_cols=features,
                    model_name=model_name,
                    outcome_name=outcome_name,
                    feature_set_name=feature_set_name,
                    args=args,
                    folders=folders,
                )
                all_fold_results.append(fold_results)
                all_predictions.append(predictions)
                if not importances.empty:
                    all_importances.append(importances)

    fold_results = pd.concat(all_fold_results, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    importances = pd.concat(all_importances, ignore_index=True) if all_importances else pd.DataFrame()

    performance = summarize_performance(fold_results)
    feature_importance, domain_importance = summarize_importance(importances, domain_map)

    fold_results.to_csv(folders["tables"] / "fold_results.csv", index=False)
    predictions.to_csv(folders["tables"] / "all_oof_predictions.csv", index=False)
    performance.to_csv(folders["tables"] / "model_performance_summary.csv", index=False)

    if not feature_importance.empty:
        feature_importance.to_csv(folders["tables"] / "feature_permutation_importance.csv", index=False)
        domain_importance.to_csv(folders["tables"] / "domain_permutation_importance.csv", index=False)

    try:
        with pd.ExcelWriter(folders["root"] / "frailty_ml_results.xlsx", engine="openpyxl") as writer:
            performance.to_excel(writer, sheet_name="model_performance", index=False)
            fold_results.to_excel(writer, sheet_name="fold_results", index=False)
            if not feature_importance.empty:
                feature_importance.to_excel(writer, sheet_name="feature_importance", index=False)
                domain_importance.to_excel(writer, sheet_name="domain_importance", index=False)
    except Exception as exc:
        print(f"[WARN] Excel file not saved: {exc}")

    print("\n[OK] Machine-learning analysis completed.")
    print(f"[OK] Results saved to: {folders['root']}")
    print("\nBest models by outcome and feature set:")
    print(performance.groupby(["outcome", "feature_set"]).head(1).to_string(index=False))


if __name__ == "__main__":
    main()
