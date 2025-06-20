# config/settings.py

import os
from datetime import datetime

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Root of the project
DATA_DIR = os.path.join(BASE_DIR, 'data')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts') # Where main execution scripts live
CONFIG_DIR = os.path.join(BASE_DIR, 'config')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# --- Data File Paths ---
SIMULATED_ORDERS_CSV = os.path.join(DATA_DIR, 'simulated_orders.csv')
US_ZIP_CODES_CSV = os.path.join(DATA_DIR, 'us_zip_codes.csv') # No longer used by data_generator
SIMULATED_ORDERS_ENHANCED_CSV = os.path.join(DATA_DIR, 'simulated_orders_enhanced.csv')

# ALMRRC Meta Data Paths (from your local downloads)
RAW_ALMRRC_META_TRAINING_DIR = os.path.join(DATA_DIR, 'last_mile_raw', 'almrrc2021-data-training')
RAW_ALMRRC_META_EVALUATION_DIR = os.path.join(DATA_DIR, 'last_mile_raw', 'almrrc2021-data-evaluation')
PROCESSED_ALMRRC_META_DIR = os.path.join(DATA_DIR, 'processed_last_mile_meta')
PROCESSED_ENHANCED_META_ROUTES_CSV = os.path.join(PROCESSED_ALMRRC_META_DIR, 'processed_enhanced_meta_routes.csv')

# MIT Purchase Data Paths
RAW_PURCHASE_CSV = os.path.join(DATA_DIR, 'amazon_purchase_history.csv')
PROCESSED_PURCHASE_CSV = os.path.join(DATA_DIR, 'processed_amazon_purchase.csv')

# --- Database Configuration (for PostgreSQL) ---
# It's best practice to load sensitive info from environment variables in production
DB_NAME = os.getenv('DB_NAME', 'amazon_delivery_db')
DB_USER = os.getenv('DB_USER', 'postgres') # Default postgres user
DB_PASSWORD = os.getenv('DB_PASSWORD', 'your_postgres_password') # <--- IMPORTANT: REPLACE THIS DEFAULT
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')

# --- Simulation Parameters (from data_generation.py) ---
NUM_ORDERS = 100000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)
PRIME_MEMBER_RATIO = 0.70

PRIME_DELIVERY_AVG_DAYS = 1.5
PRIME_DELIVERY_STD_DEV = 0.5
STANDARD_DELIVERY_AVG_DAYS = 6.0
STANDARD_DELIVERY_STD_DEV = 1.5

PRIME_DELAY_PROBABILITY = 0.05
STANDARD_DELAY_PROBABILITY = 0.20
AVERAGE_DELAY_DAYS = 2

PRIME_CARRIER_DIST = {
    'AMZL': 0.85, 'UPS': 0.07, 'USPS': 0.05, 'FedEx': 0.03
}
STANDARD_CARRIER_DIST = {
    'AMZL': 0.20, 'UPS': 0.40, 'USPS': 0.30, 'FedEx': 0.10
}

BASE_AMZL_COST_PER_PACKAGE = 5.00
BASE_3PC_COST_PER_PACKAGE = 4.00
PRIME_EXPEDITED_COST_PREMIUM = 1.2

# Customer Profile Simulation
CLV_TIER_DIST = {'High': 0.2, 'Medium': 0.5, 'Low': 0.3}
CHURN_RISK_DIST = {'High': 0.15, 'Medium': 0.45, 'Low': 0.4}
GIFT_PURCHASE_PROBABILITY = 0.05

# --- API Keys (Conceptual for real-time integration) ---
OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', 'YOUR_OPENWEATHERMAP_API_KEY')
NEWSAPI_API_KEY = os.getenv('NEWSAPI_API_KEY', 'YOUR_NEWSAPI_API_KEY') # <--- REPLACE WITH YOUR ACTUAL NEWSAPI KEY
GOOGLE_ROUTES_API_KEY = os.getenv('GOOGLE_ROUTES_API_KEY', '')  # <--- ADD YOUR GOOGLE ROUTES API KEY HERE
# Add other API keys here (e.g., Traffic API)

# --- Logging Configuration ---
LOG_FILE = os.path.join(BASE_DIR, 'app.log')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()