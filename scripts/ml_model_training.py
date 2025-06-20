# scripts/ml_model_training.py

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np 
import shap
import matplotlib.pyplot as plt

from logger import get_logger

logger = get_logger(__name__)

# --- Configuration (Paths for this script - Inlined) ---
FEATURES_FOR_ML_CSV = os.path.join('data', 'features_for_ml.csv')
TRAINED_MODEL_PATH = os.path.join('models', 'delivery_delay_model.pkl') 
MODELS_DIR = 'models' 
TARGET_VARIABLE = 'delivery_status'


def train_and_evaluate_model():
    logger.info("Starting ML model training and evaluation...")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load the engineered features
    try:
        df_features = pd.read_csv(FEATURES_FOR_ML_CSV)
        logger.info(f"Loaded {len(df_features)} records from {FEATURES_FOR_ML_CSV} for training.")
    except FileNotFoundError:
        logger.error(f"Error: {FEATURES_FOR_ML_CSV} not found. Ensure feature engineering ran successfully.")
        return None
    except Exception as e:
        logger.error(f"Error loading {FEATURES_FOR_ML_CSV}: {e}")
        return None

    # Separate features (X) and target (y)
    X = df_features.drop(columns=[TARGET_VARIABLE])
    y = df_features[TARGET_VARIABLE]

    logger.info(f"Feature matrix shape: {X.shape}, Target vector shape: {y.shape}")

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Data split: Training {len(X_train)} records, Testing {len(X_test)} records.")
    logger.info(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
    logger.info(f"Testing target distribution:\n{y_test.value_counts(normalize=True)}")

    # 3. Choose and Train Models
    logger.info("Training Logistic Regression model...")
    lr_model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000) 
    lr_model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete.")

    logger.info("Training Random Forest Classifier model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) 
    rf_model.fit(X_train, y_train) 
    logger.info("Random Forest Classifier training complete.")

    # 4. Model Evaluation
    models = {'Logistic Regression': lr_model, 'Random Forest': rf_model}
    best_model = None
    best_roc_auc = -1 

    logger.info("\n--- Model Evaluation Results ---")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"\nModel: {name}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        logger.info(f"  Confusion Matrix:\n{cm}")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = model
            logger.info(f"  >>> {name} is currently the best model based on ROC AUC.")

    if best_model is None:
        logger.error("No model trained or evaluated successfully.")
        return None

    os.makedirs(os.path.dirname(TRAINED_MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, TRAINED_MODEL_PATH)
    logger.info(f"\nBest model ({best_model.__class__.__name__}) saved to '{TRAINED_MODEL_PATH}'")

    # --- SHAP Explainability ---
    logger.info("Computing SHAP values for model explainability...")
    try:
        # Use TreeExplainer for RandomForest, LinearExplainer for LogisticRegression
        if hasattr(best_model, 'feature_importances_'):
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.LinearExplainer(best_model, X_train, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_train)
        # Save SHAP values and feature names for later use
        np.save(os.path.join(MODELS_DIR, 'shap_values.npy'), shap_values)
        np.save(os.path.join(MODELS_DIR, 'shap_feature_names.npy'), X_train.columns.values)
        # SHAP summary plot
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X_train, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, 'shap_summary_plot.png'))
        plt.close()
        logger.info("SHAP values and summary plot saved in 'models/' directory.")
    except Exception as e:
        logger.error(f"Error during SHAP analysis: {e}")

    return best_model 

if __name__ == "__main__":
    train_and_evaluate_model()