import pandas as pd
import os

# --- Configuration ---
SIMULATED_ORDERS_CSV = 'data/simulated_orders.csv'
US_ZIP_CODES_CSV = 'data/us_zip_codes.csv' # Make sure you've downloaded this file!
OUTPUT_ENHANCED_SIMULATED_ORDERS_CSV = 'data/simulated_orders_enhanced.csv'

# --- Main Logic ---
if __name__ == "__main__":
    print("Starting geographical enrichment of simulated_orders.csv...")

    # Load simulated orders data
    try:
        df_orders = pd.read_csv(SIMULATED_ORDERS_CSV)
        print(f"Loaded {len(df_orders)} records from {SIMULATED_ORDERS_CSV}")
    except FileNotFoundError:
        print(f"Error: {SIMULATED_ORDERS_CSV} not found. Please ensure it exists.")
        exit()
    except Exception as e:
        print(f"Error loading {SIMULATED_ORDERS_CSV}: {e}")
        exit()

    # Load US Zip Codes data
    try:
        df_zip_codes = pd.read_csv(US_ZIP_CODES_CSV)
        print(f"Loaded {len(df_zip_codes)} records from {US_ZIP_CODES_CSV}")
    except FileNotFoundError:
        print(f"Error: {US_ZIP_CODES_CSV} not found. Please ensure you've downloaded it.")
        exit()
    except Exception as e:
        print(f"Error loading {US_ZIP_CODES_CSV}: {e}")
        exit()

    # --- Data Cleaning and Preparation for Merge ---

    # Ensure zip codes are consistent types (e.g., string) for merging
    df_orders['destination_zip_code'] = df_orders['destination_zip_code'].astype(str).str.zfill(5) # Pad with zeros to 5 digits
    df_zip_codes['zip'] = df_zip_codes['zip'].astype(str).str.zfill(5) # Ensure zip column is 5 digits

    # Select relevant columns from zip codes data to avoid duplicates or unnecessary info
    # Common column names are 'zip', 'city', 'state_id', 'state_name' (or similar)
    # Adjust these column names based on the actual CSV you download!
    # Let's assume you have 'zip', 'city', 'state_id', 'state_name'
    if 'zip' in df_zip_codes.columns and 'city' in df_zip_codes.columns and 'state_id' in df_zip_codes.columns:
        df_zip_codes_selected = df_zip_codes[['zip', 'city', 'state_id']].rename(columns={'zip': 'destination_zip_code', 'city': 'destination_city', 'state_id': 'destination_state_code'})
    elif 'ZIP' in df_zip_codes.columns and 'CITY' in df_zip_codes.columns and 'STATE' in df_zip_codes.columns:
        # Common alternative casing/naming
        df_zip_codes_selected = df_zip_codes[['ZIP', 'CITY', 'STATE']].rename(columns={'ZIP': 'destination_zip_code', 'CITY': 'destination_city', 'STATE': 'destination_state_code'})
    else:
        print("Warning: Common zip code columns (zip/city/state) not found. Please check your US_ZIP_CODES_CSV file's column names and update the script accordingly.")
        print("Available columns in zip file:", df_zip_codes.columns.tolist())
        exit()

    # Drop duplicates in zip code mapping (some zips might map to multiple cities, pick one arbitrarily)
    df_zip_codes_selected = df_zip_codes_selected.drop_duplicates(subset=['destination_zip_code'])

    # --- Perform Merge ---
    # Merge df_orders with the zip code data
    df_orders_enhanced = pd.merge(
        df_orders,
        df_zip_codes_selected,
        on='destination_zip_code',
        how='left' # Use left join to keep all original orders
    )

    # Handle cases where a zip code might not be found in the mapping
    print(f"Orders with unmapped zip codes: {df_orders_enhanced['destination_city'].isnull().sum()} out of {len(df_orders_enhanced)}")
    df_orders_enhanced['destination_city'] = df_orders_enhanced['destination_city'].fillna('Unknown City')
    df_orders_enhanced['destination_state_code'] = df_orders_enhanced['destination_state_code'].fillna('XX') # Placeholder for unknown state

    # --- Save Enhanced Data ---
    os.makedirs(os.path.dirname(OUTPUT_ENHANCED_SIMULATED_ORDERS_CSV), exist_ok=True)
    df_orders_enhanced.to_csv(OUTPUT_ENHANCED_SIMULATED_ORDERS_CSV, index=False)
    
    print(f"\nGeographical enrichment complete. Enhanced data saved to '{OUTPUT_ENHANCED_SIMULATED_ORDERS_CSV}'")
    print("\nFirst 5 rows of enhanced data:")
    print(df_orders_enhanced.head())
    print("\nValue counts for top 10 destination cities:")
    print(df_orders_enhanced['destination_city'].value_counts().head(10))
    print("\nValue counts for top 10 destination states:")
    print(df_orders_enhanced['destination_state_code'].value_counts().head(10))