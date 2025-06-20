# scripts/process_amazon_purchase_data.py

import pandas as pd
import os

# --- No 'from config import settings' here ---
from logger import get_logger

logger = get_logger(__name__)

# --- Configuration Parameters (MOVED HERE and adjusted to be directly accessible) ---
RAW_PURCHASE_CSV = os.path.join('data', 'amazon_purchase_history.csv')
PROCESSED_PURCHASE_CSV = os.path.join('data', 'processed_amazon_purchase.csv')

def process_amazon_purchase_data():
    logger.info(f"Starting processing of {RAW_PURCHASE_CSV} (full dataset)...")

    try:
        df_purchase = pd.read_csv(RAW_PURCHASE_CSV)
        logger.info(f"Loaded {len(df_purchase)} records from {RAW_PURCHASE_CSV}")
    except FileNotFoundError:
        logger.error(f"Error: File not found at {RAW_PURCHASE_CSV}. Please ensure it exists.")
        return None
    except Exception as e:
        logger.error(f"Error loading {RAW_PURCHASE_CSV}: {e}")
        return None

    # --- Data Cleaning and Transformation (ADJUSTED FOR NEW HEADERS) ---
    # Headers confirmed: Order Date,Purchase Price Per Unit,Quantity,Shipping Address State,Title,ASIN/ISBN (Product Code),Category,Survey ResponseID
    df_purchase.rename(columns={
        'Order Date': 'order_date',
        'Purchase Price Per Unit': 'price_usd', 
        'Quantity': 'quantity_ordered',
        'Shipping Address State': 'shipping_state_code', 
        'Title': 'product_title',
        'ASIN/ISBN (Product Code)': 'product_code', 
        'Category': 'product_category' 
    }, inplace=True)

    # Convert 'order_date' to datetime
    df_purchase['order_date'] = pd.to_datetime(df_purchase['order_date'], errors='coerce')
    
    # Ensure numeric types and handle missing values (fill with 0 for safety if original data has NaNs)
    df_purchase['price_usd'] = pd.to_numeric(df_purchase['price_usd'], errors='coerce').fillna(0)
    df_purchase['quantity_ordered'] = pd.to_numeric(df_purchase['quantity_ordered'], errors='coerce').fillna(0).astype(int)

    # Calculate total sales value per order line
    df_purchase['line_item_total_usd'] = df_purchase['price_usd'] * df_purchase['quantity_ordered']

    # --- Handle missing string data for columns that should now be present ---
    # Only fill NaN for these if they are genuinely missing in the downloaded data, not to overwrite existing
    # Use .fillna('UNKNOWN') only where actual data might be missing, not to override valid data
    # Ensure state codes are consistent, fill with 'UNKNOWN' only if actual NaN
    df_purchase['shipping_state_code'] = df_purchase['shipping_state_code'].astype(str).str.strip().str.upper().fillna('UNKNOWN')
    # Filter out common non-US states if any crept in (optional, but good for cleanliness)
    US_STATE_ABBREVIATIONS_LIST = ["AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"]
    df_purchase = df_purchase[df_purchase['shipping_state_code'].isin(US_STATE_ABBREVIATIONS_LIST)].copy()
    
    # No longer use fillna for product_title/category if they are expected to be populated
    # The .fillna() calls will only act if the column truly has NaN. If empty string is present, use .replace('')
    df_purchase['product_title'] = df_purchase['product_title'].astype(str).replace('', 'UNKNOWN_PRODUCT_TITLE').fillna('UNKNOWN_PRODUCT_TITLE')
    df_purchase['product_category'] = df_purchase['product_category'].astype(str).replace('', 'UNKNOWN_CATEGORY_TYPE').fillna('UNKNOWN_CATEGORY_TYPE')


    # Select final columns 
    df_processed = df_purchase[[
        'order_date', 'product_code', 'product_title', 'product_category',
        'price_usd', 'quantity_ordered', 'line_item_total_usd', 'shipping_state_code'
    ]].copy()

    # --- Save Processed Data ---
    output_file = PROCESSED_PURCHASE_CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_processed.to_csv(output_file, index=False)
    
    logger.info(f"Processing complete. Processed {len(df_processed)} records.")
    logger.info(f"Data saved to '{output_file}'")
    logger.info("First 5 rows of processed data:\n" + str(df_processed.head()))
    logger.info("Descriptive statistics for key numerical columns:\n" + str(df_processed[['price_usd', 'quantity_ordered', 'line_item_total_usd']].describe()))
    logger.info("Top 10 shipping states:\n" + str(df_processed['shipping_state_code'].value_counts().head(10)))
    logger.info("Top 10 product categories:\n" + str(df_processed['product_category'].value_counts().head(10)))
    
    return df_processed

if __name__ == "__main__":
    process_amazon_purchase_data()