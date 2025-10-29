# model_training.py
"""
Improved transformer-enhanced Customer Risk Model Training Script

This updated script includes:
- Diagnostics prints for label balance and feature summaries.
- Option to use class_weight='balanced' for RandomForest.
- Optional oversampling (RandomOverSampler) to balance classes (requires imblearn).
- Stratified K-Fold cross-validation and PR AUC computation for robust evaluation.

Usage:
    python model_training.py [--oversample] [--use_class_weight] [--cv]

Data expected in project root:
    data/orders.csv
    data/customer_feedback.csv
    data/delivery_performance.csv

Model saved to:
    data/customer_risk_model.joblib

Dependencies:
    pip install -r requirements.txt
    (if using --oversample: pip install imbalanced-learn)
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from imblearn.over_sampling import RandomOverSampler
except Exception:
    RandomOverSampler = None


# ------------------------- CONFIG -------------------------
DATA_DIR = "data"
ORDERS_PATH = os.path.join(DATA_DIR, "orders.csv")
FEEDBACK_PATH = os.path.join(DATA_DIR, "customer_feedback.csv")
DELIVERY_PATH = os.path.join(DATA_DIR, "delivery_performance.csv")
MODEL_PATH = os.path.join(DATA_DIR, "customer_risk_model.joblib")
EMBEDDER_NAME = "all-MiniLM-L6-v2"
# -----------------------------------------------------------


def load_csv_safe(path):
    """Safely load a CSV file if it exists."""
    if os.path.exists(path):
        print(f"Loading {path} ...")
        return pd.read_csv(path)
    else:
        print(f"Warning: {path} not found.")
        return None


def build_tabular_features(orders, delivery):
    """Construct basic RFM-like features from orders and delivery data."""
    if orders is None:
        # Create synthetic fallback dataset
        n = 500
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            'customer_id': np.arange(1, n + 1),
            'num_orders_90d': rng.poisson(3, n),
            'avg_order_value': rng.uniform(50, 500, n),
            'days_since_last_order': rng.randint(1, 200, n),
        })
    else:
        df = orders.copy()
        if 'customer_id' in df.columns:
            # Aggregate by customer
            agg = df.groupby('customer_id').agg(
                num_orders_90d=('order_id', 'count') if 'order_id' in df.columns else ('order_date', 'count'),
                avg_order_value=('order_value', 'mean') if 'order_value' in df.columns else ('order_id', 'count'),
                days_since_last_order=('order_date', lambda x: (pd.Timestamp.now() - pd.to_datetime(x).max()).days if 'order_date' in df.columns else np.nan)
            ).reset_index()
            df = agg
        else:
            df = df.head(500)
            df['customer_id'] = np.arange(1, len(df) + 1)
            df['num_orders_90d'] = 1
            df['avg_order_value'] = df.get('order_value', 100)
            df['days_since_last_order'] = 30

    # Merge delivery data
    if delivery is not None and 'customer_id' in delivery.columns:
        dp = delivery.groupby('customer_id').agg(
            on_time_rate=('on_time', 'mean') if 'on_time' in delivery.columns else ('delay_hours', lambda x: (x == 0).mean()),
            avg_delay_hours=('delay_hours', 'mean') if 'delay_hours' in delivery.columns else ('on_time', lambda x: 0)
        ).reset_index()
        df = df.merge(dp, on='customer_id', how='left')
    else:
        df['on_time_rate'] = np.nan
        df['avg_delay_hours'] = np.nan

    # Derived features
    df['recency'] = df['days_since_last_order'].fillna(999)
    df['frequency'] = df['num_orders_90d'].fillna(0)
    df['monetary'] = df['avg_order_value'].fillna(df['avg_order_value'].median() if 'avg_order_value' in df else 100)

    return df[['customer_id', 'recency', 'frequency', 'monetary', 'on_time_rate', 'avg_delay_hours']]


def build_text_embeddings(feedback_df, embedder_name=EMBEDDER_NAME):
    """Convert customer feedback text to embeddings using SentenceTransformer."""
    if feedback_df is None or 'customer_id' not in feedback_df.columns:
        print("No feedback file found or 'customer_id' column missing.")
        return None

    if 'feedback_text' not in feedback_df.columns:
        # Try fallback column names
        if 'text' in feedback_df.columns:
            feedback_df = feedback_df.rename(columns={'text': 'feedback_text'})
        else:
            print("No 'feedback_text' column found in feedback file.")
            return None

    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed. Install with `pip install sentence-transformers` to use text embeddings.")

    model = SentenceTransformer(embedder_name)
    grouped = feedback_df.groupby('customer_id').agg({'feedback_text': lambda x: ' '.join(x.dropna().astype(str))}).reset_index()

    embeddings = []
    for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Encoding feedback"):
        emb = model.encode(row['feedback_text'], show_progress_bar=False)
        embeddings.append(emb)

    if len(embeddings) == 0:
        return None

    emb_arr = np.vstack(embeddings)
    emb_df = pd.DataFrame(emb_arr)
    emb_df['customer_id'] = grouped['customer_id'].values
    emb_df = emb_df[['customer_id'] + [c for c in emb_df.columns if c != 'customer_id']]
    emb_df.columns = ['customer_id'] + [f'emb_{i}' for i in range(emb_df.shape[1] - 1)]

    return emb_df


def create_target(df, feedback):
    """Create a simple heuristic label 'is_at_risk'."""
    df = df.copy()
    df['avg_rating'] = np.nan

    if feedback is not None and 'customer_id' in feedback.columns:
        if 'rating' in feedback.columns:
            fbagg = feedback.groupby('customer_id').agg({'rating': 'mean'}).reset_index()
            df = df.merge(fbagg, on='customer_id', how='left')
            df['avg_rating'] = df['rating'].fillna(np.nan)

    # Avoid labeling everything due to missing avg_rating — rely primarily on RFM
    df['is_at_risk'] = (((df['recency'] > 60) & (df['frequency'] <= 1)) | (df['avg_rating'] <= 2)).astype(int)
    return df


def train_model(X, y, n_estimators=200, max_depth=None, test_size=0.2, use_class_weight=False, oversample=False, cv=False):
    """Train a RandomForest classifier with diagnostics and optional balancing/CV."""
    # Diagnostics
    print("\n--- Diagnostics before training ---")
    print("Total samples:", X.shape[0])
    print("Feature shape:", X.shape)
    print("Target distribution:\n", pd.Series(y).value_counts(dropna=False))

    # Handle single-class case
    unique_classes = np.unique(y)
    if len(unique_classes) == 1:
        print("\nError: Only one class present in the target. Training a classifier is not meaningful.")
        print("Classes present:", unique_classes)
        print("Please check your label generation. Exiting.")
        raise ValueError("Only one class present in target y. Aborting training.")

    # Optionally oversample
    if oversample:
        if RandomOverSampler is None:
            raise ImportError("imblearn not installed. Install with `pip install imbalanced-learn` to use oversampling.")
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        print("After oversampling, target distribution:\n", pd.Series(y_res).value_counts())
        X_train_full, y_train_full = X_res, y_res
    else:
        X_train_full, y_train_full = X, y

    clf_kwargs = {'n_estimators': n_estimators, 'max_depth': max_depth, 'random_state': 42}
    if use_class_weight:
        clf_kwargs['class_weight'] = 'balanced'

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(**clf_kwargs))
    ])

    if cv:
        print('\nRunning stratified cross-validation (5-fold) ...')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # cross_val_predict with predict_proba
        y_pred_proba = cross_val_predict(pipeline, X_train_full, y_train_full, cv=skf, method='predict_proba')
        # y_pred_proba has shape (n_samples, n_classes)
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            y_scores = y_pred_proba[:, 1]
        else:
            # fallback
            y_scores = y_pred_proba.ravel()

        # PR AUC
        precision, recall, _ = precision_recall_curve(y_train_full, y_scores)
        pr_auc = auc(recall, precision)
        print('Cross-validated PR AUC:', round(pr_auc, 4))

        # Fit final model on full training data
        pipeline.fit(X_train_full, y_train_full)
        return pipeline

    # Otherwise, do a normal train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=test_size, random_state=42, stratify=y_train_full)

    print('\nTraining final model...')
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # Robust predict_proba handling
    try:
        proba = pipeline.predict_proba(X_test)
    except Exception as e:
        print("Warning: predict_proba failed:", e)
        proba = np.vstack([1 - y_pred, y_pred]).T

    if proba.ndim == 2 and proba.shape[1] == 2:
        y_prob = proba[:, 1]
    elif proba.ndim == 2 and proba.shape[1] == 1:
        classes_seen = getattr(pipeline.named_steps['clf'], 'classes_', None)
        if classes_seen is not None and len(classes_seen) == 1:
            only_class = classes_seen[0]
            if only_class == 1:
                y_prob = np.ones(len(y_test))
            else:
                y_prob = np.zeros(len(y_test))
        else:
            y_prob = 1.0 - proba[:, 0]
    else:
        y_prob = np.zeros(len(y_test))

    print('\n--- Model Evaluation (hold-out) ---')
    print(classification_report(y_test, y_pred))

    if len(np.unique(y_test)) > 1:
        try:
            print('ROC AUC:', round(roc_auc_score(y_test, y_prob), 4))
        except Exception as e:
            print('Could not compute ROC AUC:', e)
    else:
        print('ROC AUC: not computed because only one class present in the test set.')

    return pipeline


def main(args):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Load data
    orders = load_csv_safe(ORDERS_PATH)
    feedback = load_csv_safe(FEEDBACK_PATH)
    delivery = load_csv_safe(DELIVERY_PATH)

    # Build features
    tab = build_tabular_features(orders, delivery)
    emb_df = None
    try:
        emb_df = build_text_embeddings(feedback, embedder_name=EMBEDDER_NAME)
    except Exception as e:
        print('Warning: could not build embeddings:', e)
        emb_df = None

    # Merge embeddings + tabular features
    df = tab.copy()
    if emb_df is not None:
        # ensure consistent dtypes for merge
        df['customer_id'] = df['customer_id'].astype(str)
        emb_df['customer_id'] = emb_df['customer_id'].astype(str)
        df = df.merge(emb_df, on='customer_id', how='left')

    # Create labels
    df = create_target(df, feedback)

    # Define feature columns
    feature_cols = [col for col in df.columns if col.startswith(('recency', 'frequency', 'monetary', 'on_time_rate', 'avg_delay_hours', 'emb_'))]

    X = df[feature_cols].fillna(0)
    y = df['is_at_risk'].values

    # Extra diagnostic prints
    print('\nSample of feature matrix:')
    print(X.head())
    print('\nLabel distribution:')
    print(pd.Series(y).value_counts())

    # Train
    model_pipeline = train_model(X, y, n_estimators=args.n_estimators, max_depth=args.max_depth, use_class_weight=args.use_class_weight, oversample=args.oversample, cv=args.cv)

    # Save model
    joblib.dump({'pipeline': model_pipeline, 'feature_columns': feature_cols, 'embedder_name': EMBEDDER_NAME}, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--oversample', action='store_true', help='Apply RandomOverSampler to balance classes (requires imbalanced-learn)')
    parser.add_argument('--use_class_weight', action='store_true', help='Use class_weight="balanced" in RandomForest')
    parser.add_argument('--cv', action='store_true', help='Run stratified 5-fold CV and return model trained on full data')
    args = parser.parse_args()
    main(args)