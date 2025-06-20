# scripts/data_enricher.py

import pandas as pd
import os

# --- No 'from config import settings' here ---
from logger import get_logger # Import logger utility

logger = get_logger(__name__)

# --- Configuration Parameters (MOVED HERE and adjusted to be directly accessible) ---
US_ZIP_CODES_CSV = os.path.join('data', 'us_zip_codes.csv') # Path adjusted
SIMULATED_ORDERS_CSV = os.path.join('data', 'simulated_orders.csv')
SIMULATED_ORDERS_ENHANCED_CSV = os.path.join('data', 'simulated_orders_enhanced.csv')


def enrich_simulated_orders_data():
    logger.info("Starting geographical enrichment of simulated_orders.csv...")

    try:
        # Use inlined SIMULATED_ORDERS_CSV
        df_orders = pd.read_csv(SIMULATED_ORDERS_CSV)
        logger.info(f"Loaded {len(df_orders)} records from {SIMULATED_ORDERS_CSV}")
    except FileNotFoundError:
        logger.error(f"Error: {SIMULATED_ORDERS_CSV} not found. Ensure data generation ran successfully.")
        return None
    except Exception as e:
        logger.error(f"Error loading {SIMULATED_ORDERS_CSV}: {e}")
        return None

    try:
        # Use inlined US_ZIP_CODES_CSV
        # This part of the code implicitly expects US_ZIP_CODES_CSV to exist and be readable.
        # Its error handling will catch if it's not found.
        df_zip_codes = pd.read_csv(US_ZIP_CODES_CSV)
        logger.info(f"Loaded {len(df_zip_codes)} records from {US_ZIP_CODES_CSV}")
    except FileNotFoundError:
        logger.error(f"Error: Zip code file not found at {US_ZIP_CODES_CSV}. Please ensure you've downloaded it.")
        return None
    except Exception as e:
        logger.error(f"Error loading {US_ZIP_CODES_CSV}: {e}")
        return None

    df_orders['destination_zip_code'] = df_orders['destination_zip_code'].astype(str).str.zfill(5)
    df_zip_codes['zip'] = df_zip_codes['zip'].astype(str).str.zfill(5)

    if 'zip' in df_zip_codes.columns and 'city' in df_zip_codes.columns and 'state_id' in df_zip_codes.columns:
        df_zip_codes_selected = df_zip_codes[['zip', 'city', 'state_id']].rename(columns={'zip': 'destination_zip_code', 'city': 'destination_city', 'state_id': 'destination_state_code'})
    elif 'ZIP' in df_zip_codes.columns and 'CITY' in df_zip_codes.columns and 'STATE' in df_zip_codes.columns:
        df_zip_codes_selected = df_zip_codes[['ZIP', 'CITY', 'STATE']].rename(columns={'ZIP': 'destination_zip_code', 'CITY': 'destination_city', 'STATE': 'destination_state_code'})
    else:
        logger.error(f"Common zip code columns (zip/city/state) not found in {US_ZIP_CODES_CSV}. Available columns: {df_zip_codes.columns.tolist()}")
        return None

    df_zip_codes_selected = df_zip_codes_selected.drop_duplicates(subset=['destination_zip_code'])

    df_orders_enhanced = pd.merge(
        df_orders,
        df_zip_codes_selected,
        on='destination_zip_code',
        how='left'
    )

    unmapped_zips_count = df_orders_enhanced['destination_city'].isnull().sum()
    logger.info(f"Orders with unmapped zip codes: {unmapped_zips_count} out of {len(df_orders_enhanced)}")
    
    df_orders_enhanced['destination_city'] = df_orders_enhanced['destination_city'].fillna('Unknown City')
    df_orders_enhanced['destination_state_code'] = df_orders_enhanced['destination_state_code'].fillna('XX')

    output_file = SIMULATED_ORDERS_ENHANCED_CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_orders_enhanced.to_csv(output_file, index=False)
    
    logger.info(f"Geographical enrichment complete. Enhanced data saved to '{output_file}'")
    logger.info("First 5 rows of enhanced data:\n" + str(df_orders_enhanced.head()))
    logger.info("Value counts for top 10 destination cities:\n" + str(df_orders_enhanced['destination_city'].value_counts().head(10)))
    logger.info("Value counts for top 10 destination states:\n" + str(df_orders_enhanced['destination_state_code'].value_counts().head(10)))
    
    return df_orders_enhanced

if __name__ == "__main__":
    enrich_simulated_orders_data()