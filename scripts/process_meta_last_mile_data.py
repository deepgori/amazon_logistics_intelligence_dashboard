# scripts/process_meta_last_mile_data.py

import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta

# --- REMOVED: from config import settings ---
from logger import get_logger # Import logger utility

logger = get_logger(__name__)

# --- Configuration Parameters (MOVED HERE and adjusted to be directly accessible) ---
RAW_ALMRRC_META_TRAINING_DIR = os.path.join('data', 'last_mile_raw', 'almrrc2021-data-training')
RAW_ALMRRC_META_EVALUATION_DIR = os.path.join('data', 'last_mile_raw', 'almrrc2021-data-evaluation')
PROCESSED_META_DATA_DIR = os.path.join('data', 'processed_last_mile_meta')
PROCESSED_ENHANCED_META_ROUTES_CSV = os.path.join(PROCESSED_META_DATA_DIR, 'processed_enhanced_meta_routes.csv')


# --- Helper Functions ---

def load_json_file(filepath):
    """Loads a JSON file and handles potential errors."""
    if not os.path.exists(filepath):
        return {} # Return empty dict if file not found
    try:
        with open(filepath, 'r') as f:
            content = json.load(f)
            if not content: # Check for genuinely empty JSONs
                return {} # Return empty dict if JSON is empty
            return content
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        return {} # Return empty dict on decode error
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {filepath}: {e}")
        return {} # Return empty dict on other errors

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth using the Haversine formula.
    This function remains, but its output will likely be 0.0 due to data limitations.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0 

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

# --- MODIFIED process_single_route_data Function ---
def process_single_route_data(route_id, route_details, package_details_for_route, actual_sequences_for_route):
    """
    Processes details for a single route_id from combined JSON data.
    This version correctly extracts city, origin, packages, calculates volume, and sums duration from 'stops'.
    Distance is still 0.0 due to data limitation.
    """
    # Initialize all values to None or 0.0 for safety
    city = 'Unknown'
    route_date = None
    station_code = None
    route_score = None
    origin_lat, origin_lon = None, None
    vehicle_capacity = None
    num_deliveries = 0
    total_calculated_volume_cm3 = 0.0
    actual_route_duration_hours = 0.0
    actual_route_distance_km = 0.0

    # --- Extract basic route info from route_details (from route_data.json) ---
    if route_details: # Ensure route_details is not empty
        route_date = route_details.get('date_YYYY_MM_DD') or route_details.get('date')
        city = route_details.get('city', 'Unknown') 
        station_code = route_details.get('station_code')
        route_score = route_details.get('route_score')
        vehicle_capacity = route_details.get('executor_capacity_cm3') or route_details.get('vehicleCapacity')

        if 'origin' in route_details and isinstance(route_details.get('origin'), dict):
            origin_lat = route_details['origin'].get('latitude')
            origin_lon = route_details['origin'].get('longitude')
        
        # --- Calculate Actual Duration (using 'stops' from route_data.json) ---
        stops_data_from_route_json = route_details.get('stops', {}) 
        
        if stops_data_from_route_json and isinstance(stops_data_from_route_json, dict):
            total_travel_time_seconds = 0
            total_planned_service_time_seconds = 0
            
            for stop_id, stop_detail in stops_data_from_route_json.items():
                if isinstance(stop_detail, dict): # Ensure stop_detail is a dictionary
                    total_travel_time_seconds += stop_detail.get('travel_time_to_next_stop_in_seconds', 0)
                    total_planned_service_time_seconds += stop_detail.get('planned_service_time_seconds', 0)
            
            actual_route_duration_hours = (total_travel_time_seconds + total_planned_service_time_seconds) / 3600 
            if actual_route_duration_hours < 0: actual_route_duration_hours = 0.0 # Ensure non-negative

    # --- Get Package Details for this route ---
    if package_details_for_route: # Ensure package_details is not empty
        inner_packages_dict = package_details_for_route.get('AD', {}) 
        if not inner_packages_dict and 'packages' in package_details_for_route and isinstance(package_details_for_route['packages'], dict):
            inner_packages_dict = package_details_for_route['packages']
        
        packages_list_values = list(inner_packages_dict.values()) 
        num_deliveries = len(packages_list_values) 
        
        for pkg in packages_list_values:
            if 'dimensions' in pkg and isinstance(pkg.get('dimensions'), dict):
                depth = pkg['dimensions'].get('depth_cm', 0)
                height = pkg['dimensions'].get('height_cm', 0)
                width = pkg['dimensions'].get('width_cm', 0)
                total_calculated_volume_cm3 += (depth * height * width)
    
    # --- Actual Sequences Analysis (Acknowledged Limitation) ---
    # The actual_sequences.json structure you provided does NOT contain geographical sequences.
    # Therefore, actual_route_distance_km cannot be calculated from it.
    actual_route_distance_km = 0.0    

    return {
        'route_id': route_id,
        'city': city,
        'route_date': route_date,
        'station_code': station_code,
        'route_score': route_score,
        'origin_latitude': origin_lat,
        'origin_longitude': origin_lon,
        'vehicle_capacity_cm3': vehicle_capacity,
        'num_deliveries': num_deliveries, 
        'total_calculated_volume_cm3': total_calculated_volume_cm3, 
        'actual_route_duration_hours': actual_route_duration_hours,
        'actual_route_distance_km': actual_route_distance_km        # Remains 0.0
    }

# --- Main Processing Logic ---
def process_almrrc_meta_data():
    logger.info("Starting processing of ALMRRC meta-data JSONs...")
    
    os.makedirs(PROCESSED_META_DATA_DIR, exist_ok=True)
    all_processed_routes = []

    # --- Load all relevant JSONs from model_build_inputs ---
    logger.info(f"\n--- Loading from {RAW_ALMRRC_META_TRAINING_DIR}/model_build_inputs/ ---")
    route_data_build = load_json_file(os.path.join(RAW_ALMRRC_META_TRAINING_DIR, 'model_build_inputs', 'route_data.json'))
    package_data_build = load_json_file(os.path.join(RAW_ALMRRC_META_TRAINING_DIR, 'model_build_inputs', 'package_data.json'))
    actual_sequences_build = load_json_file(os.path.join(RAW_ALMRRC_META_TRAINING_DIR, 'model_build_inputs', 'actual_sequences.json'))
    logger.debug(f"DEBUG: route_data_build has {len(route_data_build)} entries." )
    logger.debug(f"DEBUG: package_data_build has {len(package_data_build)} entries.")
    logger.debug(f"DEBUG: actual_sequences_build has {len(actual_sequences_build)} entries.")


    # --- Load relevant JSONs from model_apply_inputs (if they exist) ---
    logger.info(f"\n--- Loading from {RAW_ALMRRC_META_TRAINING_DIR}/model_apply_inputs/ ---")
    route_data_apply = load_json_file(os.path.join(RAW_ALMRRC_META_TRAINING_DIR, 'model_apply_inputs', 'new_route_data.json'))
    package_data_apply = load_json_file(os.path.join(RAW_ALMRRC_META_TRAINING_DIR, 'model_apply_inputs', 'new_package_data.json'))
    actual_sequences_apply = load_json_file(os.path.join(RAW_ALMRRC_META_TRAINING_DIR, 'model_score_inputs', 'new_actual_sequences.json')) # Corrected path
    logger.debug(f"DEBUG: route_data_apply entries: {len(route_data_apply)}.")
    logger.debug(f"DEBUG: package_data_apply entries: {len(package_data_apply)}.")
    logger.debug(f"DEBUG: actual_sequences_apply entries: {len(actual_sequences_apply)}.")
    
    # --- Load relevant JSONs from evaluation data (if they exist) ---
    logger.info(f"\n--- Loading from {RAW_EVAL_DATA_DIR}/model_apply_inputs/ and other eval folders ---")
    route_data_eval = load_json_file(os.path.join(RAW_EVAL_DATA_DIR, 'model_apply_inputs', 'eval_route_data.json'))
    package_data_eval = load_json_file(os.path.join(RAW_EVAL_DATA_DIR, 'model_apply_inputs', 'eval_package_data.json'))
    actual_sequences_eval = load_json_file(os.path.join(RAW_EVAL_DATA_DIR, 'model_score_inputs', 'eval_actual_sequences.json')) # Assuming eval_actual_sequences.json is here
    logger.debug(f"DEBUG: route_data_eval entries: {len(route_data_eval)}.")
    logger.debug(f"DEBUG: package_data_eval entries: {len(package_data_eval)}.")
    logger.debug(f"DEBUG: actual_sequences_eval entries: {len(actual_sequences_eval)}.")
    
    # Consolidate all unique route IDs from all loaded route_data files
    all_route_ids_to_process = set()
    if route_data_build: all_route_ids_to_process.update(route_data_build.keys())
    if route_data_apply: all_route_ids_to_process.update(route_data_apply.keys())
    if route_data_eval: all_route_ids_to_process.update(route_data_eval.keys())

    if not all_route_ids_to_process:
        logger.warning("\nNo route IDs found in any of the specified meta-data JSONs. Ensure files are correctly downloaded and not empty.")
        return None # Return None if no routes to process

    processed_count = 0
    logger.info(f"\n--- Starting processing of {len(all_route_ids_to_process)} unique route IDs ---")
    processed_first_few_debug = 0 
    for i, route_id in enumerate(all_route_ids_to_process):
        # Get specific route details from their respective consolidated dictionaries
        route_details = route_data_build.get(route_id) or route_data_apply.get(route_id) or route_data_eval.get(route_id)
        package_details = package_data_build.get(route_id) or package_data_apply.get(route_id) or package_data_eval.get(route_id)
        actual_sequences = actual_sequences_build.get(route_id) or actual_sequences_apply.get(route_id) or actual_sequences_eval.get(route_id)
        
        # --- DEBUGGING MISSING CORE DATA ---
        if processed_first_few_debug < 5:
            logger.debug(f"\nDEBUG Route ID: {route_id}")
            logger.debug(f"  route_details found: {bool(route_details)} (len: {len(route_details) if route_details else 0})")
            logger.debug(f"  package_details found: {bool(package_details)} (len: {len(package_details) if package_details else 0})")
            logger.debug(f"  actual_sequences found: {bool(actual_sequences)} (len: {len(actual_sequences) if actual_sequences else 0})")
            if not route_details: logger.debug(f"  --> Missing route_details for {route_id}")
            if not package_details: logger.debug(f"  --> Missing package_details for {route_id}")
            if not actual_sequences: logger.debug(f"  --> Missing actual_sequences for {route_id}")
            
        if not all([route_details, package_details]): # Removed actual_sequences from this check
            # logger.debug(f"Skipping route {route_id}: Missing core data from meta files (route_details or package_details).")
            continue 

        processed_metrics = process_single_route_data(route_id, route_details, package_details, actual_sequences)
        
        if processed_metrics:
            all_processed_routes.append(processed_metrics)
            processed_count += 1
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} routes...")
            if processed_first_few_debug < 5: # Use separate counter for these prints
                pass # This print will be outside this block as it's repetitive


    if not all_processed_routes:
        logger.warning("\nNo valid routes processed after filtering for complete data. Ensure JSON files contain matching RouteIDs and relevant data.")
        return None # Return None if no valid routes processed

    df_enhanced_meta_routes = pd.DataFrame(all_processed_routes)
    
    df_enhanced_meta_routes['route_date'] = df_enhanced_meta_routes['route_date'].astype(str).str.strip()
    df_enhanced_meta_routes['route_date'] = pd.to_datetime(df_enhanced_meta_routes['route_date'], errors='coerce')

    output_file = PROCESSED_META_DATA_DIR 
    os.makedirs(os.path.dirname(output_file), exist_ok=True) 
    output_file = os.path.join(output_file, 'processed_enhanced_meta_routes.csv')
    df_enhanced_meta_routes.to_csv(output_file, index=False)
    
    logger.info(f"\nEnhanced meta-data processing complete. Processed {len(df_enhanced_meta_routes)} routes.")
    logger.info(f"Data saved to '{output_file}'")
    logger.info("First 5 rows of processed data:\n" + str(df_enhanced_meta_routes.head()))
    logger.info("Descriptive statistics for processed data:\n" + str(df_enhanced_meta_routes[['num_deliveries', 'total_calculated_volume_cm3', 'actual_route_duration_hours', 'actual_route_distance_km']].describe()))
    
    logger.info("\nRoute counts by city (from enhanced meta-data):\n" + str(df_enhanced_meta_routes['city'].value_counts()))
    return df_enhanced_meta_routes

if __name__ == "__main__":
    process_almrrc_meta_data()