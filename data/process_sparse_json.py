import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta

# --- Configuration ---
RAW_META_DATA_DIR = 'data/last_mile_raw/almrrc2021-data-training'
RAW_EVAL_DATA_DIR = 'data/last_mile_raw/almrrc2021-data-evaluation'
PROCESSED_META_DATA_DIR = 'data/processed_last_mile_meta'

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)

def safe_float(value) -> float:
    """Safely convert value to float, handling various formats."""
    if pd.isna(value) or value is None:
        return 0.0
    try:
        float_val = float(value)
        return float_val if np.isfinite(float_val) else 0.0
    except (ValueError, TypeError):
        return 0.0

def safe_str(value) -> str:
    """Safely convert value to string, handling various formats."""
    if pd.isna(value) or value is None:
        return ''
    return str(value)

class DataProcessor:
    def __init__(self):
        self.stats = {
            'total_records': 0,
            'processed_records': 0,
            'error_records': 0,
            'city_counts': defaultdict(int)
        }

    def load_json_with_pandas(self, filepath: str) -> pd.DataFrame:
        """Load JSON file using pandas with chunking."""
        try:
            # First try reading as lines
            try:
                df = pd.read_json(filepath, lines=True)
            except ValueError:
                # If lines=True fails, try regular JSON reading
                df = pd.read_json(filepath)

            # Clean the DataFrame
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)  # Fill numeric NaN with 0

            # Ensure route_id exists and is properly formatted
            if 'route_id' not in df.columns and 'RouteID' in df.columns:
                df['route_id'] = df['RouteID']
            elif 'route_id' not in df.columns:
                # Try to extract route_id from index if it's not in columns
                df['route_id'] = df.index

            # Convert route_id to string
            df['route_id'] = df['route_id'].astype(str)

            logging.info(f"Successfully loaded {len(df)} records from {filepath}")
            logging.info(f"Columns found: {df.columns.tolist()}")
            return df

        except Exception as e:
            logging.error(f"Error loading {filepath}: {str(e)}")
            return pd.DataFrame({'route_id': []})  # Return empty DataFrame with route_id column

    def process_route_data(self, route_df: pd.DataFrame) -> pd.DataFrame:
        """Process route data from DataFrame."""
        if route_df.empty:
            logging.warning("Empty route DataFrame received")
            return pd.DataFrame({'route_id': []})

        try:
            processed_data = []
            
            for _, row in route_df.iterrows():
                try:
                    route_id = safe_str(row.get('route_id', ''))
                    if not route_id:
                        continue

                    # Extract city and update stats
                    city = safe_str(row.get('city', 'Unknown'))
                    self.stats['city_counts'][city] += 1

                    # Extract coordinates from origin dictionary
                    origin = row.get('origin', {}) if isinstance(row.get('origin'), dict) else {}
                    lat = safe_float(origin.get('latitude', row.get('latitude', 0)))
                    lon = safe_float(origin.get('longitude', row.get('longitude', 0)))

                    # Extract vehicle capacity
                    vehicle_capacity = safe_float(row.get('executor_capacity_cm3', row.get('vehicleCapacity', 0)))

                    # Calculate duration from stops
                    stops = row.get('stops', {}) if isinstance(row.get('stops'), dict) else {}
                    total_duration = 0.0
                    for stop in stops.values():
                        if isinstance(stop, dict):
                            travel_time = safe_float(stop.get('travel_time_to_next_stop_in_seconds', 0))
                            service_time = safe_float(stop.get('planned_service_time_seconds', 0))
                            total_duration += travel_time + service_time

                    processed_data.append({
                        'route_id': route_id,
                        'city': city,
                        'route_date': row.get('date_YYYY_MM_DD', row.get('date')),
                        'station_code': safe_str(row.get('station_code')),
                        'route_score': safe_str(row.get('route_score')),
                        'vehicle_capacity_cm3': vehicle_capacity,
                        'origin_latitude': lat,
                        'origin_longitude': lon,
                        'actual_route_duration_hours': total_duration / 3600.0
                    })
                    
                    self.stats['processed_records'] += 1
                except Exception as e:
                    logging.error(f"Error processing route {route_id if 'route_id' in locals() else 'unknown'}: {str(e)}")
                    self.stats['error_records'] += 1

            result_df = pd.DataFrame(processed_data)
            if result_df.empty:
                logging.warning("No routes were processed successfully")
                return pd.DataFrame({'route_id': []})
                
            logging.info(f"Processed {len(result_df)} routes successfully")
            return result_df

        except Exception as e:
            logging.error(f"Error in process_route_data: {str(e)}")
            return pd.DataFrame({'route_id': []})

    def process_package_data(self, package_df: pd.DataFrame) -> pd.DataFrame:
        """Process package data from DataFrame."""
        if package_df.empty:
            logging.warning("Empty package DataFrame received")
            return pd.DataFrame({'route_id': []})

        try:
            processed_data = []
            
            for _, row in package_df.iterrows():
                try:
                    route_id = safe_str(row.get('route_id', ''))
                    if not route_id:
                        continue

                    total_volume = 0.0
                    num_deliveries = 0

                    # Process packages
                    packages = row.get('packages', {}) if isinstance(row.get('packages'), dict) else {}
                    for pkg in packages.values():
                        if isinstance(pkg, dict):
                            # Try to get dimensions
                            dims = pkg.get('dimensions', {}) if isinstance(pkg.get('dimensions'), dict) else {}
                            depth = safe_float(dims.get('depth_cm', 0))
                            height = safe_float(dims.get('height_cm', 0))
                            width = safe_float(dims.get('width_cm', 0))
                            
                            if all([depth, height, width]):
                                total_volume += depth * height * width
                                num_deliveries += 1

                    # Try AD structure if no packages found
                    if total_volume == 0:
                        ad_data = row.get('AD', {}) if isinstance(row.get('AD'), dict) else {}
                        for pkg in ad_data.values():
                            if isinstance(pkg, dict):
                                length = safe_float(pkg.get('length_cm', 0))
                                width = safe_float(pkg.get('width_cm', 0))
                                height = safe_float(pkg.get('height_cm', 0))
                                
                                if all([length, width, height]):
                                    total_volume += length * width * height
                                    num_deliveries += 1

                    processed_data.append({
                        'route_id': route_id,
                        'num_deliveries': num_deliveries,
                        'total_calculated_volume_cm3': total_volume
                    })
                    
                except Exception as e:
                    logging.error(f"Error processing packages for route {route_id if 'route_id' in locals() else 'unknown'}: {str(e)}")
                    processed_data.append({
                        'route_id': route_id if 'route_id' in locals() else 'unknown',
                        'num_deliveries': 0,
                        'total_calculated_volume_cm3': 0.0
                    })

            result_df = pd.DataFrame(processed_data)
            if result_df.empty:
                logging.warning("No packages were processed successfully")
                return pd.DataFrame({'route_id': []})
                
            logging.info(f"Processed {len(result_df)} package records successfully")
            return result_df

        except Exception as e:
            logging.error(f"Error in process_package_data: {str(e)}")
            return pd.DataFrame({'route_id': []})

def main():
    processor = DataProcessor()
    
    # Create output directory
    os.makedirs(PROCESSED_META_DATA_DIR, exist_ok=True)
    
    # Process route data
    logging.info("Loading and processing route data...")
    route_data_path = os.path.join(RAW_META_DATA_DIR, 'model_build_inputs', 'route_data.json')
    route_df = processor.load_json_with_pandas(route_data_path)
    
    if route_df.empty:
        logging.error("Failed to load route data")
        return
        
    processed_routes = processor.process_route_data(route_df)
    
    # Process package data
    logging.info("Loading and processing package data...")
    package_data_path = os.path.join(RAW_META_DATA_DIR, 'model_build_inputs', 'package_data.json')
    package_df = processor.load_json_with_pandas(package_data_path)
    
    if package_df.empty:
        logging.error("Failed to load package data")
        return
        
    processed_packages = processor.process_package_data(package_df)
    
    # Check if we have valid data to merge
    if processed_routes.empty or processed_packages.empty:
        logging.error("No data to merge - one or both DataFrames are empty")
        return
        
    if 'route_id' not in processed_routes.columns or 'route_id' not in processed_packages.columns:
        logging.error("Missing route_id column in one or both DataFrames")
        logging.info(f"Route columns: {processed_routes.columns.tolist()}")
        logging.info(f"Package columns: {processed_packages.columns.tolist()}")
        return
    
    # Merge route and package data
    logging.info("Merging processed data...")
    final_df = pd.merge(processed_routes, processed_packages, on='route_id', how='left')
    
    # Fill NaN values
    final_df = final_df.fillna({
        'num_deliveries': 0,
        'total_calculated_volume_cm3': 0.0,
        'city': 'Unknown',
        'route_score': '',
        'station_code': ''
    })
    
    # Convert date column
    final_df['route_date'] = pd.to_datetime(final_df['route_date'], errors='coerce')
    
    # Save results
    output_path = os.path.join(PROCESSED_META_DATA_DIR, 'processed_enhanced_meta_routes.csv')
    final_df.to_csv(output_path, index=False)
    
    # Print statistics
    logging.info("\nProcessing Statistics:")
    logging.info(f"Total records processed: {processor.stats['processed_records']}")
    logging.info(f"Records with errors: {processor.stats['error_records']}")
    logging.info("\nCity Distribution:")
    for city, count in processor.stats['city_counts'].items():
        logging.info(f"{city}: {count}")
    
    logging.info(f"\nProcessing complete. Results saved to {output_path}")
    logging.info("\nFirst 5 rows of processed data:")
    logging.info(final_df.head())
    logging.info("\nDescriptive statistics:")
    logging.info(final_df.describe())

if __name__ == "__main__":
    main() 