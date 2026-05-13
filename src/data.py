# ============================================================
# src/data.py — DATA PIPELINE (Leakage-Free)
# Goal: raw CSV → clean df → train/val/test → preprocess fit-on-train → matrices/tensors
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# -----------------------------
# :00_constants
# -----------------------------
# This block holds all of the fixed settings for the data pipeline in one place, 
# such as file paths, the target and ID column names, the random seed, and the 
# train/validation/test split ratios. Keeping these values together as a frozen 
# dataclass makes it easier to swap configurations without hunting through the 
# code, and the frozen flag prevents accidental changes during a run.
# -----------------------------

@dataclass(frozen=True)
class DataConfig:
    raw_path: str = "data/raw/telco.csv"
    target_col: str = "Churn"
    id_col: str = "customerID"
    random_state: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # :01_schema_definition (numeric list is explicit; categorical is derived)
    numeric_cols: Tuple[str, ...] = ("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen")


# -----------------------------
# helpers (assertions + checks)
# -----------------------------
# This block contains small helper functions used throughout the pipeline to 
# run safety checks, such as confirming that split ratios are valid, that 
# required columns are present, and that the different data splits do not 
# share any rows. Pulling these checks into reusable helpers keeps the main 
# pipeline functions cleaner and makes it easier to catch problems early.
# -----------------------------
    
def _assert_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-9, f"Split ratios must sum to 1.0, got {total}"
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1, "Ratios must be in (0, 1)."


def _assert_expected_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing expected columns: {missing}"


def _class_balance(y: pd.Series) -> Dict[int, float]:
    vc = y.value_counts(normalize=True).to_dict()
    return {int(k): float(v) for k, v in vc.items()}


def _assert_splits_disjoint(a: pd.core.generic.NDFrame, b: pd.core.generic.NDFrame, c: pd.core.generic.NDFrame) -> None:
    """
    Ensures three splits have no overlapping indices.
    NOTE: We should only compare across splits (train vs val vs test),
    NOT X_train vs y_train (those SHOULD share indices).
    """
    ia, ib, ic = set(a.index), set(b.index), set(c.index)
    ab = ia.intersection(ib)
    ac = ia.intersection(ic)
    bc = ib.intersection(ic)
    assert len(ab) == 0, f"Index overlap detected: train vs val ({len(ab)} rows)."
    assert len(ac) == 0, f"Index overlap detected: train vs test ({len(ac)} rows)."
    assert len(bc) == 0, f"Index overlap detected: val vs test ({len(bc)} rows)."


# -----------------------------
# :02_load_raw_data
# -----------------------------
# This block handles reading the raw CSV file from disk into a dataframe with 
# no modifications applied yet. A couple of basic sanity checks confirm that 
# the file actually exists and that the loaded data is not empty, which helps 
# surface issues like wrong paths or corrupted files right at the start.
# -----------------------------
    
def load_raw_data(path: str) -> pd.DataFrame:
    """
    read CSV
    basic sanity checks
    return df raw, unmodified
    """
    csv_path = Path(path)
    assert csv_path.exists(), f"RAW_PATH does not exist: {csv_path}"

    df = pd.read_csv(csv_path)
    assert df.shape[0] > 0 and df.shape[1] > 0, f"Loaded df is empty or malformed: shape={df.shape}"
    return df


# -----------------------------
# :03_clean_data
# -----------------------------
# This block does light cleanup on the raw dataframe before any splitting or 
# encoding happens. It converts the TotalCharges column to a numeric type, 
# fills in a known structural gap where customers with zero tenure have blank 
# charges, and strips extra whitespace from text columns. Encoding and 
# splitting are intentionally left out here so they can be handled later in 
# a leakage-free way.
# -----------------------------
     

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - TotalCharges: convert to numeric (coerce invalid to missing)
    - structural missing rule: missing TotalCharges corresponds to tenure==0 → set TotalCharges to 0
    - optional: strip whitespace in categorical/object columns
    - do NOT encode here, do NOT split here
    """
    df_clean = df.copy(deep=True)

    # Strip whitespace in object columns WITHOUT converting NaNs into "nan" strings.
    obj_cols = df_clean.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        # Pandas "string" dtype preserves missing values as <NA>
        df_clean[c] = df_clean[c].astype("string").str.strip()

    # TotalCharges -> numeric (coerce invalid/blank to NaN)
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

    # structural missingness: tenure==0 implies TotalCharges should be 0
    if "tenure" in df_clean.columns:
        mask_struct = (df_clean["tenure"] == 0) & (df_clean["TotalCharges"].isna())
        df_clean.loc[mask_struct, "TotalCharges"] = 0.0

    assert pd.api.types.is_numeric_dtype(df_clean["TotalCharges"]), "TotalCharges must be numeric after cleaning."
    return df_clean


# -----------------------------
# :04_build_target_and_features
# -----------------------------
# This block separates the cleaned dataframe into the feature matrix and the 
# target column, mapping the Yes/No labels into 1s and 0s along the way. The 
# customer ID column is also dropped from the features so it cannot be used 
# by the model by accident, and a check confirms that both classes are present 
# in the target before moving on.
# -----------------------------

def build_X_y(
    df_clean: pd.DataFrame,
    target_col: str,
    id_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    - map target: Yes→1, No→0
    - drop target + customerID from features
    - keep X as dataframe (mixed types)
    """
    _assert_expected_columns(df_clean, [target_col, id_col])

    y_raw = df_clean[target_col].astype("string").str.strip()
    y = y_raw.map({"Yes": 1, "No": 0})

    assert y.notna().all(), f"Target mapping produced NaNs. Unique raw target values: {sorted(pd.Series(y_raw).dropna().unique().tolist())}"
    y = y.astype(int)

    uniq = sorted(y.unique().tolist())
    assert uniq == [0, 1], f"y must contain both classes [0,1]. Got {uniq}."

    X = df_clean.drop(columns=[target_col, id_col])
    assert id_col not in X.columns, f"{id_col} leaked into features."
    return X, y


# -----------------------------
# :01_schema_definition (categorical derived from remaining feature cols)
# -----------------------------
# This block sorts out which columns in the dataset should be treated as 
# categorical, by taking whatever is left over after the numeric columns 
# have been set aside. It also runs a couple of safety checks to make sure 
# no column ends up in both groups and that none of them get accidentally 
# left out, which helps catch mistakes early before they cause problems later on.
# -----------------------------

def infer_categorical_cols(
    X_df: pd.DataFrame,
    numeric_cols: List[str],
) -> List[str]:
    """
    categorical_cols = all remaining feature columns except numeric
    with strict overlap + existence assertions
    """
    _assert_expected_columns(X_df, numeric_cols)

    num_set = set(numeric_cols)
    all_cols = list(X_df.columns)

    cat_cols = [c for c in all_cols if c not in num_set]

    overlap = set(cat_cols).intersection(num_set)
    assert len(overlap) == 0, f"Overlap between numeric and categorical cols: {sorted(overlap)}"

    covered = set(cat_cols).union(num_set)
    missing_from_coverage = set(all_cols) - covered
    assert len(missing_from_coverage) == 0, f"Some columns were not categorized: {sorted(missing_from_coverage)}"

    return cat_cols


# -----------------------------
# :05_split_train_val_test
# -----------------------------
# This block divides the data into training, validation, and test sets using a 
# two-step split that keeps the class balance roughly the same across all three 
# groups. After splitting, it runs checks to confirm the same row does not show 
# up in more than one set and that the class proportions stay close to the 
# original, which helps avoid surprises during model training and evaluation.
# -----------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> Dict[str, Any]:
    """
    - stratify by y
    - produce X_train, y_train, X_val, y_val, X_test, y_test
    - checks: class balance roughly consistent, no overlap indices
    """
    _assert_ratios(train_ratio, val_ratio, test_ratio)

    # Step 1: split off test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state,
    )

    # Step 2: split train vs val from trainval
    val_share_of_trainval = val_ratio / (train_ratio + val_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_share_of_trainval,
        stratify=y_trainval,
        random_state=random_state,
    )

    # Checks: no overlap indices across splits (X splits)
    _assert_splits_disjoint(X_train, X_val, X_test)
    # Checks: no overlap indices across splits (y splits)
    _assert_splits_disjoint(y_train, y_val, y_test)

    # Checks: class balance roughly consistent (simple heuristic)
    base = _class_balance(y)
    b_train = _class_balance(y_train)
    b_val = _class_balance(y_val)
    b_test = _class_balance(y_test)

    def _assert_close(name: str, dist: Dict[int, float], tol: float = 0.03) -> None:
        for k in [0, 1]:
            assert abs(dist.get(k, 0.0) - base.get(k, 0.0)) <= tol, (
                f"Class balance drift too large in {name}. base={base}, {name}={dist}"
            )

    _assert_close("train", b_train)
    _assert_close("val", b_val)
    _assert_close("test", b_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


# -----------------------------
# :06_preprocessor_definition (Fit-on-train only)
# -----------------------------
# This block sets up the preprocessing steps that will be applied to the feature 
# columns, using one pipeline for numeric values and another for categorical ones. 
# Numeric columns get missing values filled in with the median and are then put 
# on the same scale, while categorical columns are filled with the most common 
# value and turned into one-hot encoded columns. A small helper handles the 
# difference between older and newer versions of scikit-learn so the code works 
# across both.
# -----------------------------

def _make_onehot() -> OneHotEncoder:
    """
    sklearn version compatibility: OneHotEncoder changed sparse -> sparse_output.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """
    numeric: median impute + standardize
    categorical: most_frequent impute + one-hot (handle_unknown safe)
    """
    overlap = set(numeric_cols).intersection(set(categorical_cols))
    assert len(overlap) == 0, f"numeric/categorical overlap: {sorted(overlap)}"

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_onehot()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# -----------------------------
# :07_transform_splits
# -----------------------------
# This block applies the preprocessor to the three data splits, but only learns 
# the transformation rules from the training set. The same rules are then used 
# to transform the validation and test sets, which keeps information from those 
# sets from leaking into the training process. The function also tries to grab 
# the resulting feature names if scikit-learn supports it, since those are 
# useful for later inspection.
# -----------------------------
    
def fit_transform_preprocess(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[Any, Any, Any, Optional[np.ndarray]]:
    """
    - fit on X_train ONLY
    - transform train/val/test
    - return matrices (+ feature_names if available)
    """
    X_train_mat = preprocessor.fit_transform(X_train)
    X_val_mat = preprocessor.transform(X_val)
    X_test_mat = preprocessor.transform(X_test)

    feature_names = None
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = None

    return X_train_mat, X_val_mat, X_test_mat, feature_names


# -----------------------------
# :08_pytorch_dataloaders
# -----------------------------
# This block takes the processed feature matrices and labels and wraps them in 
# PyTorch data loaders that can be used for training a neural network. The 
# feature matrices are converted into dense float tensors, and the training 
# loader is set to shuffle each epoch while the validation loader keeps its 
# order fixed. A small helper handles the case where the matrices come in as 
# sparse arrays and need to be converted before being turned into tensors.
# -----------------------------

def make_dataloaders(
    X_train_mat: Any,
    y_train: pd.Series,
    X_val_mat: Any,
    y_val: pd.Series,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    - converts matrices to torch tensors
    - wraps in TensorDataset
    - returns DataLoaders
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def _to_dense_float32(mat: Any) -> np.ndarray:
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        return np.asarray(mat, dtype=np.float32)

    Xtr = torch.tensor(_to_dense_float32(X_train_mat), dtype=torch.float32)
    Xva = torch.tensor(_to_dense_float32(X_val_mat), dtype=torch.float32)

    # Note: for BCEWithLogitsLoss later you may want float labels, but int is fine for now.
    ytr = torch.tensor(np.asarray(y_train, dtype=np.int64))
    yva = torch.tensor(np.asarray(y_val, dtype=np.int64))

    train_ds = TensorDataset(Xtr, ytr)
    val_ds = TensorDataset(Xva, yva)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {"train": train_loader, "val": val_loader}


# -----------------------------
# :09_single_entrypoint
# -----------------------------
# This block ties the whole pipeline together into one function that runs each 
# step in the right order, starting from the raw CSV and ending with ready-to-use 
# matrices and the fitted preprocessor. Returning everything as a single bundle 
# makes it easy to pass the prepared data into a training script without having 
# to manage all of the intermediate objects separately. The numeric and 
# categorical column lists are also included in the output so they can be checked 
# during debugging.
# -----------------------------

def prepare_data(config: DataConfig = DataConfig()) -> Dict[str, Any]:
    """
    Returns a bundle:
    - X_train_mat, y_train
    - X_val_mat, y_val
    - X_test_mat, y_test
    - preprocessor
    - feature_names (optional)
    - numeric_cols, categorical_cols (for debugging)
    """
    df_raw = load_raw_data(config.raw_path)

    required = [config.target_col, config.id_col, "TotalCharges"]
    _assert_expected_columns(df_raw, required)

    df_clean = clean_data(df_raw)
    X_df, y = build_X_y(df_clean, config.target_col, config.id_col)

    numeric_cols = list(config.numeric_cols)
    categorical_cols = infer_categorical_cols(X_df, numeric_cols)

    splits = split_data(
        X_df,
        y,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_state=config.random_state,
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train_mat, X_val_mat, X_test_mat, feature_names = fit_transform_preprocess(
        preprocessor,
        splits["X_train"],
        splits["X_val"],
        splits["X_test"],
    )

    return {
        "X_train_mat": X_train_mat,
        "y_train": splits["y_train"],
        "X_val_mat": X_val_mat,
        "y_val": splits["y_val"],
        "X_test_mat": X_test_mat,
        "y_test": splits["y_test"],
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


# ============================================================
# END
# ============================================================
