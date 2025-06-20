# scripts/load_data_to_postgres.py

import pandas as pd
from sqlalchemy import create_engine
import os
from urllib.parse import quote_plus # For safely encoding password

from logger import get_logger 

logger = get_logger(__name__)

# --- Configuration Parameters (MOVED HERE from config/settings.py and adjusted) ---
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'amazon_delivery_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')  # Must be set via environment variable

# File paths (relative to your project root)
SIMULATED_ORDERS_ENHANCED_CSV = os.path.join('data', 'simulated_orders_enhanced.csv')
PREDICTIONS_CSV = os.path.join('data', 'simulated_orders_with_predictions.csv') # Path to predictions CSV
PROCESSED_META_LAST_MILE_CSV = os.path.join('data', 'processed_last_mile_meta', 'processed_enhanced_meta_routes.csv')
PROCESSED_PURCHASE_CSV = os.path.join('data', 'processed_amazon_purchase.csv')
SIMULATED_LAST_MILE_OPERATIONS_CSV = os.path.join('data', 'simulated_last_mile_operations.csv')


def load_data_to_postgres():
    logger.info("Starting data loading to PostgreSQL...")

    # Construct the connection string using quote_plus for the password
    # This correctly encodes special characters like @ in the password
    encoded_password = quote_plus(DB_PASSWORD)
    db_connection_str = f'postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    db_connection = create_engine(db_connection_str)

    logger.info(f"Attempting to connect to database: '{DB_NAME}' as user '{DB_USER}'...")

    try:
        # Test connection
        with db_connection.connect() as connection:
            logger.info("Successfully connected to PostgreSQL!")

        # --- Load Simulated Orders Data (NOW FROM PREDICTIONS CSV) ---
        logger.info(f"\nLoading '{PREDICTIONS_CSV}' into 'simulated_orders_enhanced' table...") 
        df_simulated_orders = pd.read_csv(PREDICTIONS_CSV)

        # Ensure correct data types for PostgreSQL before loading
        df_simulated_orders['order_date'] = pd.to_datetime(df_simulated_orders['order_date'])
        df_simulated_orders['expected_delivery_date'] = pd.to_datetime(df_simulated_orders['expected_delivery_date'])
        df_simulated_orders['actual_delivery_date'] = pd.to_datetime(df_simulated_orders['actual_delivery_date'])
        df_simulated_orders['is_prime_member'] = df_simulated_orders['is_prime_member'].astype(bool)
        df_simulated_orders['order_is_gift_purchase'] = df_simulated_orders['order_is_gift_purchase'].astype(bool)
        # Add conversions for new prediction columns too:
        df_simulated_orders['predicted_delivery_status_class'] = df_simulated_orders['predicted_delivery_status_class'].astype(int)
        df_simulated_orders['predicted_delay_probability'] = df_simulated_orders['predicted_delay_probability'].astype(float)

        df_simulated_orders.to_sql('simulated_orders_enhanced', db_connection, if_exists='replace', index=False, schema='public')
        logger.info("Simulated orders data (with predictions) loaded successfully.")

        # --- Load Simulated Last Mile Operations Data ---
        logger.info(f"\nLoading '{SIMULATED_LAST_MILE_OPERATIONS_CSV}' into 'simulated_last_mile_operations' table...")
        df_simulated_last_mile_ops = pd.read_csv(SIMULATED_LAST_MILE_OPERATIONS_CSV)
        df_simulated_last_mile_ops['order_date'] = pd.to_datetime(df_simulated_last_mile_ops['order_date'], errors='coerce')
        df_simulated_last_mile_ops.to_sql('simulated_last_mile_operations', db_connection, if_exists='replace', index=False, schema='public')
        logger.info("Simulated last mile operations data loaded successfully.")


        # --- Load ALMRRC Meta Last Mile Routes Data ---
        # Note: This data is sparse as determined earlier. Loading for completeness.
        logger.info(f"\nLoading '{PROCESSED_META_LAST_MILE_CSV}' into 'almrrc_meta_routes' table...")
        df_almrrc_meta = pd.read_csv(PROCESSED_META_LAST_MILE_CSV)
        df_almrrc_meta['route_date'] = pd.to_datetime(df_almrrc_meta['route_date'], errors='coerce') 
        df_almrrc_meta.to_sql('almrrc_meta_routes', db_connection, if_exists='replace', index=False, schema='public')
        logger.info("ALMRRC meta routes data loaded successfully.")

        # --- Load Processed Amazon Purchase Data ---
        logger.info(f"\nLoading '{PROCESSED_PURCHASE_CSV}' into 'amazon_purchase_records' table...")
        df_amazon_purchase = pd.read_csv(PROCESSED_PURCHASE_CSV)
        df_amazon_purchase['order_date'] = pd.to_datetime(df_amazon_purchase['order_date'], errors='coerce')
        df_amazon_purchase.to_sql('amazon_purchase_records', db_connection, if_exists='replace', index=False, schema='public')
        logger.info("Amazon purchase data loaded successfully.")


        logger.info("\nAll data loaded into PostgreSQL.")
        return True 

    except Exception as e:
        logger.error(f"An error occurred during PostgreSQL loading: {e}")
        logger.error("\n--- Troubleshooting Tips for PostgreSQL Loading ---")
        logger.error("1. Ensure your PostgreSQL server is running.")
        logger.error("2. Verify the database name ('amazon_delivery_db') and user credentials are correct (especially password).")
        logger.error("3. Check if the CSV files exist at the specified paths and are not empty.")
        logger.error("4. Ensure 'psycopg2-binary' and 'SQLAlchemy' are installed in your virtual environment.")
        logger.error("5. Confirm the 'postgres' user has CREATE privileges on the 'public' schema of 'amazon_delivery_db'.")
        return False

if __name__ == "__main__":
    load_data_to_postgres()