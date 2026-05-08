import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score,
)
from sklearn.utils.multiclass import type_of_target
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def _detect_columns(df: pd.DataFrame, target: str | None):
    """Split columns into numeric and categorical lists (excluding target)."""
    feature_cols = [c for c in df.columns if c != target]
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(exclude="number").columns.tolist()
    return feature_cols, numeric_cols, categorical_cols


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]):
    """Build a fresh ColumnTransformer for imputation + scaling/encoding."""
    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            categorical_cols,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _get_feature_importances(fitted_pipeline):
    """Extract feature names and importances from a fitted pipeline."""
    # Get feature names from the preprocessor
    try:
        feature_names = fitted_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    except Exception:
        feature_names = []

    # Get the fitted model from the pipeline
    fitted_model = fitted_pipeline.named_steps["model"]

    importances = []
    if hasattr(fitted_model, "feature_importances_"):
        importances = fitted_model.feature_importances_.tolist()
    elif hasattr(fitted_model, "coef_"):
        coef = fitted_model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        importances = np.abs(coef).tolist()

    return feature_names, importances


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def run_classification(df: pd.DataFrame, target: str) -> dict:
    """Run classification with Random Forest and Logistic Regression."""
    df = df.dropna(subset=[target])

    feature_cols, numeric_cols, categorical_cols = _detect_columns(df, target)
    X = df[feature_cols]
    y = df[target]

    if type_of_target(y) == 'continuous':
        raise ValueError(
            f"Target column '{target}' contains continuous values. "
            "Please select Regression instead."
        )

    # Encode string labels to integers
    target_mapping = None
    if y.dtype == object or y.dtype.name == "category":
        target_mapping = {label: idx for idx, label in enumerate(y.unique())}
        y = y.map(target_mapping)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Check if SMOTE is needed (minority class < 40%)
    class_ratios = y_train.value_counts(normalize=True)
    use_smote = bool(class_ratios.min() < 0.4)
    smote_k = 5

    if use_smote:
        min_class_count = y_train.value_counts().min()
        if min_class_count < 2:
            use_smote = False
        else:
            smote_k = min(min_class_count - 1, 5)

    algorithms = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    best_name, best_pipeline, best_f1 = None, None, -1
    all_results = {}

    for name, model in algorithms.items():
        # Build a FRESH preprocessor for each model
        preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

        if use_smote:
            pipe = ImbPipeline([
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42, k_neighbors=smote_k)),
                ("model", model),
            ])
        else:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model),
            ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        feature_names, importances = _get_feature_importances(pipe)

        all_results[name] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "feature_names": feature_names,
            "feature_importances": importances,
        }

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipeline = pipe

    # Reverse mapping for label names on confusion matrix
    label_names = None
    if target_mapping:
        label_names = {v: k for k, v in target_mapping.items()}

    return {
        "task": "classification",
        "best_algorithm": best_name,
        "results": all_results,
        "label_names": label_names,
        "used_smote": use_smote,
        "pipeline": best_pipeline,
    }


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def run_regression(df: pd.DataFrame, target: str) -> dict:
    """Run regression with Random Forest and Linear Regression."""
    df = df.dropna(subset=[target])

    feature_cols, numeric_cols, categorical_cols = _detect_columns(df, target)
    X = df[feature_cols]
    y = df[target]

    if y.dtype == object or y.dtype.name == "category":
        raise ValueError(
            f"Target column '{target}' contains categorical values. "
            "Please select Classification instead."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    algorithms = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
    }

    best_name, best_pipeline, best_r2 = None, None, -np.inf
    all_results = {}

    for name, model in algorithms.items():
        # Build a FRESH preprocessor for each model
        preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        feature_names, importances = _get_feature_importances(pipe)

        # Sample actual vs predicted for the scatter plot (max 150 points)
        sample_size = min(150, len(y_test))
        indices = np.random.choice(len(y_test), sample_size, replace=False)
        actual = y_test.iloc[indices].tolist()
        predicted = y_pred[indices].tolist()

        all_results[name] = {
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "mse": round(float(mean_squared_error(y_test, y_pred)), 4),
            "r2_score": round(float(r2), 4),
            "feature_names": feature_names,
            "feature_importances": importances,
            "actual": actual,
            "predicted": predicted,
        }

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipeline = pipe

    return {
        "task": "regression",
        "best_algorithm": best_name,
        "results": all_results,
        "pipeline": best_pipeline,
    }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def run_clustering(df: pd.DataFrame) -> dict:
    """Run clustering with KMeans (k=3 and k=5)."""
    _, numeric_cols, categorical_cols = _detect_columns(df, target=None)
    X = df.copy()

    algorithms = {
        "KMeans (k=3)": KMeans(n_clusters=3, random_state=42, n_init=10),
        "KMeans (k=5)": KMeans(n_clusters=5, random_state=42, n_init=10),
    }

    best_name, best_pipeline, best_silhouette = None, None, -1
    all_results = {}

    for name, model in algorithms.items():
        # Build a FRESH preprocessor for each model
        preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipe.fit(X)
        labels = pipe.predict(X)

        X_transformed = pipe.named_steps["preprocessor"].transform(X)
        sil = silhouette_score(X_transformed, labels)

        unique, counts = np.unique(labels, return_counts=True)

        # PCA projection for visualization
        if hasattr(X_transformed, "toarray"):
            X_dense = X_transformed.toarray()
        else:
            X_dense = np.array(X_transformed)

        if X_dense.shape[1] >= 2:
            X_pca = PCA(n_components=2).fit_transform(X_dense)
        else:
            X_pca = np.hstack((X_dense, np.zeros((X_dense.shape[0], 1))))

        sample_size = min(300, len(X_pca))
        indices = np.random.choice(len(X_pca), sample_size, replace=False)

        all_results[name] = {
            "silhouette_score": round(float(sil), 4),
            "n_clusters": int(len(unique)),
            "cluster_distribution": {int(k): int(v) for k, v in zip(unique, counts)},
            "pca_x": X_pca[indices, 0].tolist(),
            "pca_y": X_pca[indices, 1].tolist(),
            "pca_labels": labels[indices].tolist(),
        }

        if sil > best_silhouette:
            best_silhouette = sil
            best_name = name
            best_pipeline = pipe

    return {
        "task": "clustering",
        "best_algorithm": best_name,
        "results": all_results,
        "pipeline": best_pipeline,
    }
