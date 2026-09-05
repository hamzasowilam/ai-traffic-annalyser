

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nids_pipeline")


LEAKY_COLUMNS = ["difficulty", "success_pred", "difficulty_level"]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
LABEL_COLUMN = "labels"
TARGET_COLUMN = "threat"


@dataclass
class PipelineConfig:
    train_path: Path
    test_path: Path
    output_dir: Path
    contamination: float = 0.01
    val_size: float = 0.15
    random_state: int = 42


def load_data(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading data from %s and %s", config.train_path, config.test_path)
    try:
        train_df = pd.read_csv(config.train_path)
        test_df = pd.read_csv(config.test_path)
    except FileNotFoundError as e:
        logger.error("Dataset file not found: %s", e)
        raise
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV: %s", e)
        raise

    if LABEL_COLUMN not in train_df.columns or LABEL_COLUMN not in test_df.columns:
        raise ValueError(f"Expected label column '{LABEL_COLUMN}' missing from input data.")

    return train_df, test_df


def clean_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop fully-empty columns, leakage-prone meta columns, and align schemas."""
    logger.info("Cleaning data...")

    for df in (train_df, test_df):
        df.dropna(axis=1, how="all", inplace=True)

   
    for df in (train_df, test_df):
        cols_to_drop = [c for c in LEAKY_COLUMNS if c in df.columns]
        if cols_to_drop:
            logger.warning("Dropping potential leakage columns: %s", cols_to_drop)
            df.drop(columns=cols_to_drop, inplace=True)

    before = len(train_df)
    train_df.drop_duplicates(inplace=True)
    if before != len(train_df):
        logger.info("Removed %d duplicate rows from training data", before - len(train_df))


    common_cols = [c for c in train_df.columns if c in test_df.columns]
    missing_in_test = set(train_df.columns) - set(common_cols)
    missing_in_train = set(test_df.columns) - set(common_cols)
    if missing_in_test or missing_in_train:
        logger.warning(
            "Schema mismatch detected. Dropped from train-only: %s | test-only: %s",
            missing_in_test, missing_in_train,
        )
    train_df = train_df[common_cols].copy()
    test_df = test_df[common_cols].copy()

    return train_df, test_df


def prepare_target(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Preparing binary target column '%s'...", TARGET_COLUMN)
    train_df[TARGET_COLUMN] = (train_df[LABEL_COLUMN] != "normal").astype(int)
    test_df[TARGET_COLUMN] = (test_df[LABEL_COLUMN] != "normal").astype(int)
    train_df.drop(columns=[LABEL_COLUMN], inplace=True)
    test_df.drop(columns=[LABEL_COLUMN], inplace=True)
    return train_df, test_df


def encode_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    logger.info("Encoding categorical columns: %s", CATEGORICAL_COLUMNS)
    present_cat_cols = [c for c in CATEGORICAL_COLUMNS if c in train_df.columns]
    if not present_cat_cols:
        logger.warning("No categorical columns found to encode.")
        return train_df, test_df, None

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_df[present_cat_cols] = encoder.fit_transform(train_df[present_cat_cols])
    test_df[present_cat_cols] = encoder.transform(test_df[present_cat_cols])
    return train_df, test_df, encoder


def train_supervised_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, random_state: int
) -> xgb.XGBClassifier:
    logger.info("Training supervised model (XGBoost)...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    logger.info("Supervised model validation accuracy: %.2f%%", val_acc * 100)
    return model


def train_unsupervised_model(
    X_train: pd.DataFrame, y_train: pd.Series, contamination: float, random_state: int
) -> IsolationForest:
    logger.info("Training unsupervised model (Isolation Forest) on normal traffic only...")
    X_train_normal = X_train[y_train == 0]
    if X_train_normal.empty:
        raise ValueError("No normal-traffic rows available to train Isolation Forest.")
    model = IsolationForest(
        n_estimators=100, contamination=contamination, random_state=random_state
    )
    model.fit(X_train_normal)
    return model


def evaluate_hybrid(
    supervised_model: xgb.XGBClassifier,
    iso_forest: IsolationForest,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> np.ndarray:
    logger.info("Evaluating hybrid system on test set...")
    sup_preds = supervised_model.predict(X_test)
    iso_preds = iso_forest.predict(X_test)
    unsup_preds = np.where(iso_preds == -1, 1, 0)

    hybrid_preds = np.logical_or(sup_preds, unsup_preds).astype(int)

    acc = accuracy_score(y_test, hybrid_preds)
    logger.info("Final Hybrid Accuracy: %.2f%%", acc * 100)
    logger.info("\n%s", classification_report(y_test, hybrid_preds))
    return hybrid_preds


def compute_baseline_stats(X: pd.DataFrame) -> dict:
    """Compute per-feature mean/std over the (encoded, numeric) training
    data. Saved alongside the model artifacts so the live API can detect
    drift between training-time and production feature distributions."""
    stats = {}
    for col in X.columns:
        series = pd.to_numeric(X[col], errors="coerce")
        stats[col] = {
            "mean": float(series.mean()) if not series.empty else 0.0,
            "std": float(series.std(ddof=0)) if not series.empty else 0.0,
        }
    return stats


def save_artifacts(
    supervised_model: xgb.XGBClassifier,
    iso_forest: IsolationForest,
    encoder: OrdinalEncoder,
    feature_columns: list[str],
    categorical_columns: list[str],
    output_dir: Path,
    baseline_stats: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving artifacts to %s ...", output_dir)

    try:
        joblib.dump(supervised_model, output_dir / "xgboost_traffic_model.pkl")
        joblib.dump(iso_forest, output_dir / "isolation_forest_model.pkl")
        joblib.dump(encoder, output_dir / "encoder.pkl")

        schema = {
            "feature_columns": feature_columns,
            "categorical_columns": categorical_columns,
            "baseline_stats": baseline_stats or {},
        }
        with open(output_dir / "feature_schema.json", "w") as f:
            json.dump(schema, f, indent=2)

    except (OSError, IOError) as e:
        logger.error("Failed to save model artifacts: %s", e)
        raise

    logger.info("✅ All artifacts saved successfully (incl. drift-monitoring baseline).")


def run_pipeline(config: PipelineConfig) -> None:
    train_df, test_df = load_data(config)
    train_df, test_df = clean_data(train_df, test_df)
    train_df, test_df = prepare_target(train_df, test_df)
    train_df, test_df, encoder = encode_features(train_df, test_df)

    X = train_df.drop(columns=[TARGET_COLUMN])
    y = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]


    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.val_size, random_state=config.random_state, stratify=y
    )

    supervised_model = train_supervised_model(
        X_train, y_train, X_val, y_val, config.random_state
    )
    iso_forest = train_unsupervised_model(
        X, y, config.contamination, config.random_state
    )

    evaluate_hybrid(supervised_model, iso_forest, X_test, y_test)

    present_cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    baseline_stats = compute_baseline_stats(X)
    save_artifacts(
        supervised_model,
        iso_forest,
        encoder,
        feature_columns=list(X.columns),
        categorical_columns=present_cat_cols,
        output_dir=config.output_dir,
        baseline_stats=baseline_stats,
    )


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Train NSL-KDD hybrid intrusion detection pipeline")
    parser.add_argument("--train", type=Path, default=Path("ml/data/kdd_train.csv"))
    parser.add_argument("--test", type=Path, default=Path("ml/data/kdd_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts"))
    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    return PipelineConfig(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output_dir,
        contamination=args.contamination,
        val_size=args.val_size,
        random_state=args.random_state,
    )


def main() -> int:
    try:
        config = parse_args()
        run_pipeline(config)
    except Exception as e:  # noqa: BLE001 - top-level guard for CLI usage
        logger.exception("Pipeline failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())