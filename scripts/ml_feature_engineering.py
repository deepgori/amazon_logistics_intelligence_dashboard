# scripts/ml_feature_engineering.py

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np 
from datetime import datetime, timedelta, timezone # ADDED timezone import
from typing import Optional

from logger import get_logger

logger = get_logger(__name__)

# --- Configuration (Paths for this script - Inlined) ---
SIMULATED_ORDERS_ENHANCED_CSV = os.path.join('data', 'simulated_orders_enhanced.csv')
FEATURES_FOR_ML_CSV = os.path.join('data', 'features_for_ml.csv') 
PREPROCESSOR_MODEL_PATH = os.path.join('models', 'preprocessor.pkl') 
MODELS_DIR = 'models' 
TARGET_VARIABLE = 'delivery_status' 

WEATHER_ALERTS_CSV = os.path.join('data', 'weather_alerts.csv')
NEWS_ALERTS_CSV = os.path.join('data', 'news_alerts.csv')

# --- Additional Feature Engineering: Traffic, Holiday, Distance ---
TRAFFIC_DATA_CSV = os.path.join('data', 'traffic_data.csv')

try:
    if os.path.exists(TRAFFIC_DATA_CSV):
        df_traffic = pd.read_csv(TRAFFIC_DATA_CSV)
        logger.info(f"Loaded {len(df_traffic)} traffic records from {TRAFFIC_DATA_CSV}.")
    else:
        logger.warning(f"Traffic data CSV not found at {TRAFFIC_DATA_CSV}. Traffic features will be zero.")
        df_traffic = pd.DataFrame()
except Exception as e:
    logger.error(f"Error loading traffic data: {e}")
    df_traffic = pd.DataFrame()

# --- Helper Functions for External Data Impact (Ensuring column existence and UTC) ---

def _get_weather_impact_features(df_orders, df_weather_alerts):
    logger.info("Generating conceptual weather impact features...")
    # Ensure columns exist and are initialized to 0
    df_orders['is_severe_weather_alert'] = 0 
    df_orders['weather_severity_score'] = 0 

    if df_weather_alerts.empty:
        logger.info("No weather alerts loaded. Skipping weather impact feature generation.")
        return df_orders

    # Convert alert start/end to UTC timezone-aware datetimes
    df_weather_alerts['start'] = pd.to_datetime(df_weather_alerts['start'], errors='coerce', utc=True)
    df_weather_alerts['end'] = pd.to_datetime(df_weather_alerts['end'], errors='coerce', utc=True)
    
    severity_map = {'Minor': 1, 'Moderate': 2, 'Severe': 3, 'Extreme': 4}
    df_weather_alerts['severity_score'] = df_weather_alerts['severity'].map(severity_map).fillna(0).astype(int)

    # Compare with current UTC time
    current_utc_time = datetime.now(timezone.utc)
    active_alerts = df_weather_alerts[
        (df_weather_alerts['start'] <= current_utc_time) & (df_weather_alerts['end'] >= current_utc_time)
    ]
    
    if active_alerts.empty:
        logger.info("No active weather alerts for current simulation. Skipping detailed weather impact.")
        return df_orders

    alerted_city_severity = active_alerts.groupby('city')['severity_score'].max().to_dict()
    
    def assign_weather_impact(row):
        city = row['destination_city']
        if city in alerted_city_severity:
            row['is_severe_weather_alert'] = 1
            row['weather_severity_score'] = alerted_city_severity[city]
        return row
    
    df_orders = df_orders.apply(assign_weather_impact, axis=1)
    logger.info("Conceptual weather impact features generated.")
    return df_orders

def _get_news_impact_features(df_orders, df_news_alerts):
    logger.info("Generating conceptual news impact features...")
    # Ensure columns exist and are initialized to 0
    df_orders['is_supply_chain_news_alert'] = 0 
    df_orders['news_disruption_score'] = 0 

    if df_news_alerts.empty:
        logger.info("No news alerts loaded. Skipping news impact feature generation.")
        return df_orders

    # Convert publishedAt to UTC timezone-aware datetime
    df_news_alerts['publishedAt'] = pd.to_datetime(df_news_alerts['publishedAt'], errors='coerce', utc=True)

    disruption_keywords = ['accident', 'fire', 'strike', 'closure', 'disruption', 'outage', 'collapse', 'flood', 'tornado', 'blizzard']
    df_news_alerts['is_disruption_news'] = df_news_alerts['title'].astype(str).str.contains('|'.join(disruption_keywords), case=False, na=False)

    # Compare with current UTC time
    current_utc_time = datetime.now(timezone.utc)
    recent_disruption_news = df_news_alerts[
        (df_news_alerts['is_disruption_news'] == True) & 
        (df_news_alerts['publishedAt'] >= current_utc_time - timedelta(days=2))
    ]
    
    if recent_disruption_news.empty:
        logger.info("No recent disruption news found. Skipping detailed news impact.")
        return df_orders

    def assign_news_impact(row):
        city = row['destination_city']
        relevant_news = recent_disruption_news[recent_disruption_news['city'].str.lower() == city.lower()]
        if not relevant_news.empty:
            row['is_supply_chain_news_alert'] = 1
            row['news_disruption_score'] = relevant_news['score'].max() if 'score' in relevant_news.columns else 1
        return row

    df_orders = df_orders.apply(assign_news_impact, axis=1)
    logger.info("Conceptual news impact features generated.")
    return df_orders

def perform_feature_engineering():
    logger.info("Starting feature engineering for ML model...")

    os.makedirs(MODELS_DIR, exist_ok=True) 

    # 1. Load the core enhanced orders data
    try:
        df = pd.read_csv(SIMULATED_ORDERS_ENHANCED_CSV)
        logger.info(f"Loaded {len(df)} records from {SIMULATED_ORDERS_ENHANCED_CSV} for training.")
    except FileNotFoundError:
        logger.error(f"Error: {SIMULATED_ORDERS_ENHANCED_CSV} not found. Ensure data pipeline ran successfully.")
        return None, None 
    except Exception as e:
        logger.error(f"Error loading {SIMULATED_ORDERS_ENHANCED_CSV}: {e}")
        return None, None

    # --- Load External Real-Time Data (Weather & News) and Generate Impact Features ---
    logger.info("Loading external API data (weather and news alerts) for feature engineering...")
    df_weather_alerts = pd.DataFrame()
    df_news_alerts = pd.DataFrame()
    
    try:
        if os.path.exists(WEATHER_ALERTS_CSV):
            df_weather_alerts = pd.read_csv(WEATHER_ALERTS_CSV)
            logger.info(f"Loaded {len(df_weather_alerts)} weather alerts from {WEATHER_ALERTS_CSV}.")
        else:
            logger.warning(f"Weather alerts CSV not found at {WEATHER_ALERTS_CSV}. Weather impact features will be zero.")

        if os.path.exists(NEWS_ALERTS_CSV):
            df_news_alerts = pd.read_csv(NEWS_ALERTS_CSV)
            logger.info(f"Loaded {len(df_news_alerts)} news alerts from {NEWS_ALERTS_CSV}.")
        else:
            logger.warning(f"News alerts CSV not found at {NEWS_ALERTS_CSV}. News impact features will be zero.")

    except Exception as e:
        logger.error(f"Error loading external alert CSVs for feature engineering: {e}")
        df_weather_alerts = pd.DataFrame()
        df_news_alerts = pd.DataFrame()

    df = _get_weather_impact_features(df, df_weather_alerts)
    df = _get_news_impact_features(df, df_news_alerts)
    logger.info("External impact features generated.")

    # Traffic disruption feature: join on destination_city
    if not df_traffic.empty:
        df = df.merge(df_traffic[['destination_city', 'incident_present', 'duration_in_traffic_seconds', 'distance_km']],
                     how='left', on='destination_city', suffixes=('', '_traffic'))
        df['incident_present'] = df['incident_present'].fillna(0)
        df['duration_in_traffic_seconds'] = df['duration_in_traffic_seconds'].fillna(0)
        df['distance_km_traffic'] = df['distance_km'].fillna(0)
    else:
        df['incident_present'] = 0
        df['duration_in_traffic_seconds'] = 0
        df['distance_km_traffic'] = 0

    # Ensure order_date is datetime before using .dt
    df['order_date'] = pd.to_datetime(df['order_date'])

    # Holiday/seasonality feature: US federal holidays (simple hardcoded list for demo)
    US_HOLIDAYS_2024 = [
        '2024-01-01', '2024-01-15', '2024-02-19', '2024-05-27', '2024-06-19',
        '2024-07-04', '2024-09-02', '2024-10-14', '2024-11-11', '2024-11-28', '2024-12-25'
    ]
    df['is_us_holiday'] = df['order_date'].dt.strftime('%Y-%m-%d').isin(US_HOLIDAYS_2024).astype(int)

    # Distance to destination (from nearest FC, using haversine)
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c
        return distance

    SIMULATED_FC_COORDS = [
        (40.8000, -74.1000), # FC-NJ1
        (33.9000, -117.5000), # FC-CA1
        (41.6000, -88.0000), # FC-IL1
        (29.9000, -95.1000), # FC-TX1
        (33.5000, -84.7000), # FC-GA1
        (33.3000, -111.9000), # FC-AZ1
        (40.1000, -75.5000), # FC-PA1
        (29.6000, -98.0000), # FC-TX2
        (33.0000, -117.0000), # FC-CA2
        (32.9000, -96.5000)  # FC-TX3
    ]
    def min_distance_to_fc(dest_lat, dest_lon):
        return min([haversine_distance(dest_lat, dest_lon, fc_lat, fc_lon) for fc_lat, fc_lon in SIMULATED_FC_COORDS])
    df['distance_to_nearest_fc_km'] = df.apply(lambda row: min_distance_to_fc(row['destination_latitude'], row['destination_longitude']), axis=1)

    # --- 1. Prepare Target Variable ---
    df[TARGET_VARIABLE] = df[TARGET_VARIABLE].apply(lambda x: 1 if x == 'Late' else 0)
    logger.info(f"Target variable '{TARGET_VARIABLE}' converted to numerical (0/1).")
    logger.info(f"Distribution of target variable:\n{df[TARGET_VARIABLE].value_counts(normalize=True)}")

    # --- 2. Define Features (delivery_days_actual REMOVED) ---
    numerical_features = [
        'delivery_cost_to_amazon',
        'order_quantity',
        'customer_churn_risk_score',
        'customer_past_return_rate_for_delayed_items',
        'destination_latitude',
        'destination_longitude',
        'is_severe_weather_alert',
        'weather_severity_score',
        'is_supply_chain_news_alert',
        'news_disruption_score',
        'incident_present',
        'duration_in_traffic_seconds',
        'distance_km_traffic',
        'is_us_holiday',
        'distance_to_nearest_fc_km'
    ]

    categorical_features = [
        'is_prime_member', 
        'carrier',
        'destination_state_code', 
        'destination_city',
        'customer_lifetime_value_tier',
        'order_is_gift_purchase' 
    ]
    
    # --- 3. Create Time-Based Features (from order_date) ---
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['order_month'] = df['order_date'].dt.month
    df['order_day_of_week'] = df['order_date'].dt.dayofweek 
    df['order_hour_of_day'] = df['order_date'].dt.hour 

    numerical_features.extend(['order_month', 'order_day_of_week', 'order_hour_of_day'])
    
    # --- 4. Preprocessing Pipelines for Different Feature Types ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) 
        ],
        remainder='drop' 
    )

    # --- 5. Create the full feature matrix ---
    X = df[numerical_features + categorical_features] 
    y = df[TARGET_VARIABLE]

    X_processed = preprocessor.fit_transform(X)

    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_feature_names)
    
    df_features = pd.DataFrame(X_processed, columns=all_feature_names)

    # --- 6. Add Target Variable back to DataFrame ---
    df_features[TARGET_VARIABLE] = y.values

    # --- 7. Save Features & Preprocessor ---
    os.makedirs(MODELS_DIR, exist_ok=True) 
    os.makedirs(os.path.dirname(FEATURES_FOR_ML_CSV), exist_ok=True) 
    
    df_features.to_csv(FEATURES_FOR_ML_CSV, index=False)
    joblib.dump(preprocessor, PREPROCESSOR_MODEL_PATH)

    logger.info(f"Feature engineering complete. Features saved to '{FEATURES_FOR_ML_CSV}'")
    logger.info(f"Preprocessor saved to '{PREPROCESSOR_MODEL_PATH}'")
    logger.info("First 5 rows of feature data:\n" + str(df_features.head()))
    logger.info(f"Shape of feature matrix: {df_features.shape}")
    logger.info(f"Number of features created: {len(all_feature_names)}")

    return df_features, preprocessor 

if __name__ == "__main__":
    perform_feature_engineering()