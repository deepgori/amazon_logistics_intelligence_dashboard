# scripts/ml_prediction_api.py

import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # <-- ADD THIS LINE
# Removed sys.path.append as it's often not needed with proper package structure and execution from root
# from scripts.logger import get_logger # Direct import in flat structure
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from datetime import datetime # Added for the current_date in helper function
from logger import get_logger # Make sure logger is imported correctly for a flat scripts/ structure

logger = get_logger(__name__)

# --- Configuration (Paths for this script - Inlined) ---
PREPROCESSOR_MODEL_PATH = os.path.join('models', 'preprocessor.pkl')
TRAINED_MODEL_PATH = os.path.join('models', 'delivery_delay_model.pkl')
MODELS_DIR = 'models' # Directory where models are saved

# --- Global Variables for Loaded Model and Preprocessor ---
app = FastAPI(title="Delivery Delay Prediction API", version="1.0.0")
preprocessor = None
ml_model = None
feature_columns = None # To store feature names from preprocessor

# --- Pydantic Model for API Request Body ---
class OrderFeatures(BaseModel):
    is_prime_member: bool
    carrier: str
    destination_state_code: str
    destination_city: str
    customer_lifetime_value_tier: str
    order_is_gift_purchase: bool
    delivery_cost_to_amazon: float
    order_quantity: int
    customer_churn_risk_score: float
    customer_past_return_rate_for_delayed_items: float
    order_month: int
    order_day_of_week: int
    order_hour_of_day: int
    destination_latitude: float
    destination_longitude: float
    is_severe_weather_alert: int
    weather_severity_score: float
    is_supply_chain_news_alert: int
    news_disruption_score: float
    incident_present: int
    duration_in_traffic_seconds: float
    distance_km_traffic: float
    is_us_holiday: int
    distance_to_nearest_fc_km: float
    alert_city: str

# --- API Endpoint to Load Models (Run on app startup) ---
@app.on_event("startup")
async def load_models_on_startup():
    global preprocessor, ml_model, feature_columns
    logger.info("Loading preprocessor and ML model for API...")
    try:
        preprocessor = joblib.load(PREPROCESSOR_MODEL_PATH)
        ml_model = joblib.load(TRAINED_MODEL_PATH)
        
        numerical_cols = preprocessor.named_transformers_['num'].get_feature_names_out()
        categorical_cols_ohe = preprocessor.named_transformers_['cat'].get_feature_names_out()
        feature_columns = list(numerical_cols) + list(categorical_cols_ohe)
        
        logger.info("Preprocessor and ML model loaded successfully.")
        logger.debug(f"Expected features from preprocessor: {feature_columns}")
    except Exception as e:
        logger.error(f"Failed to load models or preprocessor: {e}")
        # Re-raise or handle more robustly in production
        # For now, let it be None so prediction calls fail clearly
        preprocessor = None
        ml_model = None

# --- API Endpoint for Health Check ---
@app.get("/health")
async def health_check():
    if preprocessor is None or ml_model is None:
        raise HTTPException(status_code=503, detail="ML model or preprocessor not loaded. API not ready.")
    return {"status": "ok", "message": "API is running and models loaded"}

# --- API Endpoint for Prediction ---
@app.post("/predict_delay")
async def predict_delay(features: OrderFeatures):
    input_df = pd.DataFrame([features.dict()])
    if preprocessor is None or ml_model is None:
        raise HTTPException(status_code=500, detail="ML model or preprocessor not loaded on startup.")

    logger.info(f"Received prediction request for order features: {features.dict()}")

    # --- DEBUGGING START ---
    logger.debug("DEBUG: Converting features dict to DataFrame...")
    # --- DEBUGGING END ---
    data = features.dict()
    df_single_order = pd.DataFrame([data]) 

    # Define the features that the preprocessor expects as raw input
    original_feature_names_for_preprocessor = [
        'delivery_cost_to_amazon', 'order_quantity',
        'customer_churn_risk_score', 'customer_past_return_rate_for_delayed_items',
        'destination_latitude', 'destination_longitude',
        'is_severe_weather_alert', 'weather_severity_score',
        'is_supply_chain_news_alert', 'news_disruption_score',
        'incident_present', 'duration_in_traffic_seconds', 'distance_km_traffic',
        'is_us_holiday', 'distance_to_nearest_fc_km',
        'order_month', 'order_day_of_week', 'order_hour_of_day',
        'is_prime_member', 'carrier', 'destination_state_code', 'destination_city',
        'customer_lifetime_value_tier', 'order_is_gift_purchase'
    ]

    # Check for missing columns (important check)
    missing_from_input = [col for col in original_feature_names_for_preprocessor if col not in df_single_order.columns]
    if missing_from_input:
        logger.error(f"API input missing required columns: {missing_from_input}. Request features: {df_single_order.columns.tolist()}")
        raise HTTPException(status_code=400, detail=f"Required columns missing from input: {missing_from_input}")

    # Select features in the correct order for the preprocessor
    try:
        X_raw_for_prediction = df_single_order[original_feature_names_for_preprocessor]

        # --- DEBUGGING START ---
        logger.debug(f"DEBUG: Selected raw features. Shape: {X_raw_for_prediction.shape}, Columns: {X_raw_for_prediction.columns.tolist()}")
        logger.debug("DEBUG: Applying preprocessor.transform...")
        # --- DEBUGGING END ---
        X_processed_for_prediction = preprocessor.transform(X_raw_for_prediction) 
        logger.debug(f"DEBUG: Successfully preprocessed data. Shape: {X_processed_for_prediction.shape}")
    except Exception as e:
        logger.error(f"Error during preprocessing in API: {e}")
        logger.error(f"Columns present in X_raw_for_prediction before transform: {X_raw_for_prediction.columns.tolist()}")
        logger.error(f"Columns expected by preprocessor (from startup load): {feature_columns}")
        raise HTTPException(status_code=400, detail=f"Feature preprocessing failed: {e}. Check API input features consistency with trained model.")

    # Generate prediction
    try:
        # --- DEBUGGING START ---
        logger.debug("DEBUG: Generating prediction from ML model...")
        # --- DEBUGGING END ---
        prediction_class = ml_model.predict(X_processed_for_prediction)[0]
        prediction_probability = ml_model.predict_proba(X_processed_for_prediction)[0][1]
        logger.debug("DEBUG: Prediction generated.")

        predicted_status = "Late" if prediction_class == 1 else "On-Time"

        logger.info(f"Prediction result: {predicted_status} (Probability: {prediction_probability:.4f})")

        print("DEBUG: Received input columns:", input_df.columns.tolist())

        if (features.is_severe_weather_alert == 1 or features.is_supply_chain_news_alert == 1) and \
           (features.destination_city == features.alert_city):
            return {
                "predicted_delivery_status": "Late",
                "predicted_delay_probability": 1.0,
                "predicted_class": 1
            }

        return {
            "predicted_delivery_status": predicted_status,
            "predicted_delay_probability": float(prediction_probability), 
            "predicted_class": int(prediction_class)
        }
    except Exception as e:
        logger.error(f"Error during prediction in API: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# --- Main entry point for running the API server (for local testing) ---
if __name__ == "__main__":
    logger.info("Attempting to start FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)