# scripts/run_pipeline.py

import os
from logger import get_logger
from data_generator import generate_simulated_orders_and_last_mile_data
from process_amazon_purchase_data import process_amazon_purchase_data 
from ml_feature_engineering import perform_feature_engineering 
from ml_model_training import train_and_evaluate_model 
from ml_prediction import generate_predictions_for_dashboard 
from load_data_to_postgres import load_data_to_postgres 
from datetime import datetime # Needed for current_simulation_date

logger = get_logger(__name__)

def run_full_pipeline():
    logger.info("--- Starting Full Data Pipeline Execution ---")
    
    # Initialize all potential outputs to None for clear state management
    df_orders_enhanced = None
    df_last_mile_ops = None
    df_features = None
    preprocessor = None 
    trained_ml_model = None
    df_orders_with_predictions = None 
    df_amazon_purchase_processed = None
    df_weather_alerts = None # New DF for weather alerts (from api_integrator)
    df_news_alerts = None # New DF for news alerts (from api_integrator)
    db_load_success = False 

    # Define a current simulation date (e.g., today's date, or a fixed recent date for consistent demo)
    # This will be passed to data_generator to simulate orders around this date
    current_sim_date = datetime.now().date() # Use only date part for simplicity in simulation
    logger.info(f"Pipeline running for simulated current date: {current_sim_date.strftime('%Y-%m-%d')}")

    # Phase 1: Comprehensive Data Generation (Simulated Orders & Last-Mile Ops)
    logger.info("\n--- Phase 1: Comprehensive Data Generation ---")
    try:
        # Pass the current_sim_date to data_generator
        df_orders_enhanced, df_last_mile_ops = generate_simulated_orders_and_last_mile_data(current_simulation_date=current_sim_date)
        if df_orders_enhanced is not None and df_last_mile_ops is not None:
            logger.info("Phase 1: Comprehensive Data Generation Complete. DataFrames created and CSVs saved.")
        else:
            logger.error("Phase 1: Comprehensive Data Generation resulted in empty data. Skipping subsequent phases.")
    except Exception as e:
        logger.error(f"Phase 1: Comprehensive Data Generation FAILED: {e}. Skipping subsequent phases.")
        df_orders_enhanced = None
        df_last_mile_ops = None
    
    # Phase 2: Real-Time API Data Integration (Weather & News)
    # Note: api_integrator.py and its functions are part of this pipeline, but imports are already outside
    logger.info("\n--- Phase 2: Real-Time API Data Integration ---")
    try:
        # Need to import fetch_weather_alerts and fetch_news_alerts here
        # This requires api_integrator.py to be in scripts/ and imported at the top level
        from api_integrator import fetch_weather_alerts, fetch_news_alerts 

        df_weather_alerts = fetch_weather_alerts()
        if df_weather_alerts is not None and not df_weather_alerts.empty:
            logger.info("Phase 2: Weather alerts fetched and saved.")
        else:
            logger.warning("Phase 2: No weather alerts fetched or an error occurred for weather. Returning empty DF.")
        
        df_news_alerts = fetch_news_alerts()
        if df_news_alerts is not None and not df_news_alerts.empty:
            logger.info("Phase 2: News alerts fetched and saved.")
        else:
            logger.warning("Phase 2: No news alerts fetched or an error occurred for news. Returning empty DF.")

        logger.info("Phase 2: Real-Time API Data Integration Complete.")
    except Exception as e:
        logger.error(f"Phase 2: Real-Time API Data Integration FAILED: {e}")
        df_weather_alerts = None
        df_news_alerts = None

    # Phase 3: Machine Learning Feature Engineering (Requires df_orders_enhanced)
    if df_orders_enhanced is not None:
        logger.info("\n--- Phase 3: Machine Learning Feature Engineering ---")
        try:
            df_features, preprocessor = perform_feature_engineering() 
            if df_features is not None and preprocessor is not None:
                logger.info("Phase 3: Machine Learning Feature Engineering Complete. Features and Preprocessor saved.")
            else:
                logger.error("Phase 3: Machine Learning Feature Engineering resulted in empty/null data or missing preprocessor.")
        except Exception as e:
            logger.error(f"Phase 3: Machine Learning Feature Engineering FAILED: {e}.")
            df_features = None
            preprocessor = None
    else:
        logger.warning("Skipping Phase 3: ML Feature Engineering due to previous Phase 1 failure.")

    # Phase 4: Machine Learning Model Training and Evaluation (Requires features)
    if df_features is not None:
        logger.info("\n--- Phase 4: Machine Learning Model Training and Evaluation ---")
        try:
            trained_ml_model = train_and_evaluate_model() 
            if trained_ml_model is not None:
                logger.info("Phase 4: Machine Learning Model Training and Evaluation Complete. Model saved.")
            else:
                logger.error("Phase 4: Machine Learning Model Training and Evaluation resulted in no valid model.")
        except Exception as e:
            logger.error(f"Phase 4: Machine Learning Model Training and Evaluation FAILED: {e}.")
            trained_ml_model = None
    else:
        logger.warning("Skipping Phase 4: ML Model Training due to missing features data.")

    # Phase 5: Generate Predictions (Requires trained ML model)
    if trained_ml_model is not None:
        logger.info("\n--- Phase 5: Generate Predictions ---")
        try:
            df_orders_with_predictions = generate_predictions_for_dashboard() 
            if df_orders_with_predictions is not None:
                logger.info("Phase 5: Prediction Generation Complete. Predictions CSV saved.")
            else:
                logger.error("Phase 5: Prediction Generation resulted in empty/null data.")
        except Exception as e:
            logger.error(f"Phase 5: Prediction Generation FAILED: {e}.")
            df_orders_with_predictions = None
    else:
        logger.warning("Skipping Phase 5: Prediction Generation due to no trained ML model.")

    # Phase 6: Amazon Purchase Data Processing (MIT Data)
    logger.info("\n--- Phase 6: Amazon Purchase Data Processing ---")
    try:
        df_amazon_purchase_processed = process_amazon_purchase_data()
        if df_amazon_purchase_processed is not None:
            logger.info("Phase 6: Amazon Purchase Data Processing Complete. CSV saved.")
        else:
            logger.warning("Phase 6: Amazon Purchase Data Processing resulted in empty/null data.")
    except Exception as e:
        logger.error(f"Phase 6: Amazon Purchase Data Processing FAILED: {e}.")
        df_amazon_purchase_processed = None

    # Phase 7: Database Loading (Load all prepared CSVs into PostgreSQL)
    logger.info("\n--- Phase 7: Database Loading ---")
    try:
        if load_data_to_postgres(): 
            logger.info("Phase 7: Data loaded to PostgreSQL Complete.")
            db_load_success = True
        else:
            logger.error("Phase 7: Data loading to PostgreSQL FAILED.")
    except Exception as e:
        logger.error(f"Phase 7: Database loading FAILED: {e}")
        db_load_success = False

    # Phase 8: ML Model API Service (Conceptual - Deployment Ready)
    if trained_ml_model is not None: 
        logger.info("\n--- Phase 8: ML Model API Service (Conceptual) ---")
        logger.info(f"ML Model API Service is now ready for deployment/local testing.")
        logger.info(f"To run the API locally, execute: uvicorn scripts.ml_prediction_api:app --reload --host 0.0.0.0 --port 8000")
        logger.info(f"You can test it at http://localhost:8000/docs after starting.")
    else:
        logger.warning("Skipping Phase 8: ML Model API Service conceptual step due to no trained ML model (previous ML phases failed).")

    logger.info("\n--- Full Data Pipeline Execution Finished ---")

    if db_load_success and \
       df_orders_enhanced is not None and df_last_mile_ops is not None and \
       df_features is not None and trained_ml_model is not None and \
       df_orders_with_predictions is not None and df_amazon_purchase_processed is not None:
        logger.info("All core data processing and loading phases completed successfully!")
    else:
        logger.error("Some critical phases of the data pipeline failed. Review logs for details.")

if __name__ == "__main__":
    run_full_pipeline()