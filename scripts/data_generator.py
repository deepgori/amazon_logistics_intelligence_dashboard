# scripts/data_generator.py

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import os

from logger import get_logger 

logger = get_logger(__name__)
fake = Faker('en_US') 

# --- Configuration Parameters (Inlined) ---
NUM_ORDERS = 100000 
START_DATE = datetime(2024, 1, 1) # Retained for consistency, though generate_simulated_orders_and_last_mile_data uses current_simulation_date
END_DATE = datetime(2024, 12, 31) # Retained for consistency
PRIME_MEMBER_RATIO = 0.70 

PRIME_DELIVERY_AVG_DAYS = 1.5
PRIME_DELIVERY_STD_DEV = 0.5
STANDARD_DELIVERY_AVG_DAYS = 6.0
STANDARD_DELIVERY_STD_DEV = 1.5

PRIME_DELAY_PROBABILITY = 0.05
STANDARD_DELIVERY_PROBABILITY = 0.20 
AVERAGE_DELAY_DAYS = 2

# Carrier Distribution
PRIME_CARRIER_DIST = {
    'AMZL': 0.85, 'UPS': 0.07, 'USPS': 0.05, 'FedEx': 0.03
}
STANDARD_CARRIER_DIST = {
    'AMZL': 0.20, 'UPS': 0.40, 'USPS': 0.30, 'FedEx': 0.10
}

BASE_AMZL_COST_PER_PACKAGE = 5.00
BASE_3PC_COST_PER_PACKAGE = 4.00
PRIME_EXPEDITED_COST_PREMIUM = 1.2

# --- Customer Profile Simulation Parameters ---
CLV_TIER_DIST = {'High': 0.2, 'Medium': 0.5, 'Low': 0.3} 
CHURN_RISK_DIST = {'High': 0.15, 'Medium': 0.45, 'Low': 0.4} 
GIFT_PURCHASE_PROBABILITY = 0.05 

# --- Output File Paths ---
SIMULATED_ORDERS_CSV = os.path.join('data', 'simulated_orders.csv') 
SIMULATED_ORDERS_ENHANCED_CSV = os.path.join('data', 'simulated_orders_enhanced.csv')
SIMULATED_LAST_MILE_OPERATIONS_CSV = os.path.join('data', 'simulated_last_mile_operations.csv')
PROCESSED_AMAZON_PURCHASE_CSV = os.path.join('data', 'processed_amazon_purchase.csv') # Input for learning patterns

# --- Simulated Geographical Data for Consistency ---
US_CITIES = {
    'New York': {'state': 'NY', 'lat': 40.7128, 'lon': -74.0060},
    'Los Angeles': {'state': 'CA', 'lat': 34.0522, 'lon': -118.2437},
    'Chicago': {'state': 'IL', 'lat': 41.8781, 'lon': -87.6298},
    'Houston': {'state': 'TX', 'lat': 29.7604, 'lon': -95.3698},
    'Phoenix': {'state': 'AZ', 'lat': 33.4484, 'lon': -112.0740},
    'Philadelphia': {'state': 'PA', 'lat': 39.9526, 'lon': -75.1652},
    'San Antonio': {'state': 'TX', 'lat': 29.4241, 'lon': -98.4936},
    'San Diego': {'state': 'CA', 'lat': 32.7157, 'lon': -117.1611},
    'Dallas': {'state': 'TX', 'lat': 32.7767, 'lon': -96.7970},
    'San Jose': {'state': 'CA', 'lat': 37.3382, 'lon': -121.8863},
    'Austin': {'state': 'TX', 'lat': 30.2672, 'lon': -97.7431},
    'Boston': {'state': 'MA', 'lat': 42.3601, 'lon': -71.0589},
    'Seattle': {'state': 'WA', 'lat': 47.6062, 'lon': -122.3321},
    'Atlanta': {'state': 'GA', 'lat': 33.7488, 'lon': -84.3877},
    'Miami': {'state': 'FL', 'lat': 25.7617, 'lon': -80.1918}
}

# --- NEW: Simulated Fulfillment Center Locations (Distinct from City Destinations) ---
SIMULATED_FULFILLMENT_CENTERS = {
    'FC-NJ1': {'lat': 40.8000, 'lon': -74.1000}, # Near NYC
    'FC-CA1': {'lat': 33.9000, 'lon': -117.5000}, # Near LA
    'FC-IL1': {'lat': 41.6000, 'lon': -88.0000}, # Near Chicago
    'FC-TX1': {'lat': 29.9000, 'lon': -95.1000}, # Near Houston
    'FC-GA1': {'lat': 33.5000, 'lon': -84.7000}, # Near Atlanta
    'FC-AZ1': {'lat': 33.3000, 'lon': -111.9000}, # Near Phoenix
    'FC-PA1': {'lat': 40.1000, 'lon': -75.5000}, # Near Philadelphia
    'FC-TX2': {'lat': 29.6000, 'lon': -98.0000}, # Near San Antonio
    'FC-CA2': {'lat': 33.0000, 'lon': -117.0000}, # Near San Diego
    'FC-TX3': {'lat': 32.9000, 'lon': -96.5000}  # Near Dallas
}


# --- Helper Functions ---
def _get_random_date_around_current(current_date, date_range_days=1):
    """Generates a random datetime within a small window around a current_date."""
    offset = random.randint(-date_range_days, date_range_days)
    return current_date + timedelta(days=offset)

def _get_carrier(is_prime):
    return random.choices(list(PRIME_CARRIER_DIST.keys()), weights=list(PRIME_CARRIER_DIST.values()), k=1)[0]

def _get_delivery_time(is_prime):
    return max(1, round(np.random.normal(PRIME_DELIVERY_AVG_DAYS, PRIME_DELIVERY_STD_DEV)))

# Haversine distance function (needed for FC assignment)
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

# --- NEW Helper: Find the nearest simulated Fulfillment Center ---
def _find_nearest_fc(dest_lat, dest_lon):
    """Finds the simulated FC closest to a given destination city's coordinates."""
    min_dist = float('inf')
    nearest_fc_id = None
    nearest_fc_coords = {}

    for fc_id, coords in SIMULATED_FULFILLMENT_CENTERS.items():
        dist = haversine_distance(dest_lat, dest_lon, coords['lat'], coords['lon'])
        if dist < min_dist:
            min_dist = dist
            nearest_fc_id = fc_id
            nearest_fc_coords = coords
    return nearest_fc_id, nearest_fc_coords.get('lat'), nearest_fc_coords.get('lon')


# --- Function to learn patterns from Amazon Purchase Data ---
def _learn_from_purchase_data(df_purchase_data):
    logger.info("Learning patterns from Amazon Purchase Data...")
    
    if df_purchase_data.empty:
        logger.warning("Amazon Purchase Data is empty. Cannot learn patterns. Using defaults.")
        return [], [], [], [], {}, {}

    # Get top products by sales
    top_products_df = df_purchase_data.groupby('product_code')['line_item_total_usd'].sum().nlargest(50).reset_index()
    top_product_codes = top_products_df['product_code'].tolist()
    product_sales_weights = (top_products_df['line_item_total_usd'] / top_products_df['line_item_total_usd'].sum()).tolist()
    
    # Get top shipping states by sales
    top_states_df = df_purchase_data.groupby('shipping_state_code')['line_item_total_usd'].sum().nlargest(15).reset_index()
    top_shipping_states = top_states_df['shipping_state_code'].tolist()
    state_sales_weights = (top_states_df['line_item_total_usd'] / top_states_df['line_item_total_usd'].sum()).tolist()

    avg_price_per_product = df_purchase_data.groupby('product_code')['price_usd'].mean().to_dict()
    avg_qty_per_product = df_purchase_data.groupby('product_code')['quantity_ordered'].mean().to_dict()

    logger.info(f"Learned patterns: Top {len(top_product_codes)} products, top {len(top_shipping_states)} states.")
    return top_product_codes, product_sales_weights, top_shipping_states, state_sales_weights, avg_price_per_product, avg_qty_per_product


# --- Main Data Generation Function (Modified for FC origins & learning) ---
def generate_simulated_orders_and_last_mile_data(current_simulation_date=None, df_amazon_purchase_data=None):
    """
    Generates simulated orders and last-mile operations data.
    If current_simulation_date is provided, order_dates will cluster around it.
    Learns product/location patterns from provided Amazon purchase data.
    """
    logger.info("Starting comprehensive data generation process (simulated orders & last-mile operations)...")

    if current_simulation_date is None:
        current_simulation_date = datetime.now().date() 
        logger.warning(f"No current_simulation_date provided. Defaulting to {current_simulation_date.strftime('%Y-%m-%d')}.")
    else:
        logger.info(f"Generating orders around simulated date: {current_simulation_date.strftime('%Y-%m-%d')}.")

    # --- Learn patterns from real purchase data (if provided) ---
    top_product_codes = []
    product_sales_weights = []
    top_shipping_states = []
    state_sales_weights = []
    avg_price_per_product = {}
    avg_qty_per_product = {}

    if df_amazon_purchase_data is not None and not df_amazon_purchase_data.empty:
        top_product_codes, product_sales_weights, top_shipping_states, state_sales_weights, \
        avg_price_per_product, avg_qty_per_product = _learn_from_purchase_data(df_amazon_purchase_data)
        
    if not top_product_codes: # Fallback if purchase data is empty or not provided
        logger.warning("No real purchase data patterns learned. Using generic simulation for products/locations.")
        top_product_codes = [f"PROD-{i:03d}" for i in range(100, 200)] 
        top_shipping_states = list(US_CITIES.keys()) 
        product_sales_weights = [1/len(top_product_codes)] * len(top_product_codes)
        state_sales_weights = [1/len(top_shipping_states)] * len(top_shipping_states)

    orders_data = []
    last_mile_ops_data = []

    for i in range(NUM_ORDERS):
        order_id = f"ORD-{i:07d}"
        customer_id = f"CUST-{random.randint(10000, 99999):05d}"
        
        order_date = _get_random_date_around_current(current_simulation_date, date_range_days=7) 
        
        is_prime_member = random.random() < PRIME_MEMBER_RATIO

        delivery_days = _get_delivery_time(is_prime_member)
        expected_delivery_date = order_date + timedelta(days=delivery_days)
        actual_delivery_date = expected_delivery_date

        if (is_prime_member and random.random() < PRIME_DELAY_PROBABILITY) or \
           (not is_prime_member and random.random() < STANDARD_DELIVERY_PROBABILITY): 
            actual_delivery_date += timedelta(days=random.randint(1, AVERAGE_DELAY_DAYS))

        if actual_delivery_date < expected_delivery_date:
            delivery_status = 'Early'
        elif actual_delivery_date == expected_delivery_date:
            delivery_status = 'On-Time'
        else:
            delivery_status = 'Late'

        carrier = _get_carrier(is_prime_member)

        base_cost = BASE_AMZL_COST_PER_PACKAGE if carrier == 'AMZL' else BASE_3PC_COST_PER_PACKAGE
        delivery_cost_to_amazon = base_cost
        if is_prime_member:
            delivery_cost_to_amazon *= PRIME_EXPEDITED_COST_PREMIUM
        delivery_cost_to_amazon = round(delivery_cost_to_amazon * random.uniform(0.9, 1.1), 2)

        # --- Generate Destination City/State/Lat/Lon based on learned states ---
        shipping_state_code_for_order = random.choices(top_shipping_states, weights=state_sales_weights, k=1)[0]
        
        matching_cities = [city_name for city_name, data in US_CITIES.items() if data['state'] == shipping_state_code_for_order]
        if not matching_cities:
            matching_cities = list(US_CITIES.keys())
        
        destination_city_name = random.choice(matching_cities)
        destination_state_code = US_CITIES[destination_city_name]['state'] 
        destination_latitude = US_CITIES[destination_city_name]['lat']
        destination_longitude = US_CITIES[destination_city_name]['lon']

        destination_zip_code = destination_city_name 

        # --- Simulate Product ID, Quantity, Price based on learned patterns ---
        product_code_for_order = random.choices(top_product_codes, weights=product_sales_weights, k=1)[0]
        
        product_price_usd = avg_price_per_product.get(product_code_for_order, random.uniform(5.0, 50.0))
        product_quantity = avg_qty_per_product.get(product_code_for_order, random.randint(1, 3))
        
        line_item_total_usd = product_price_usd * product_quantity
        
        # --- Simulate Customer Profile ---
        customer_seed = sum(ord(c) for c in customer_id)
        customer_rng = random.Random(customer_seed) 

        clv_val = customer_seed % 100 
        if clv_val < (CLV_TIER_DIST['High'] * 100): 
            customer_lifetime_value_tier = 'High'
        elif clv_val < ((CLV_TIER_DIST['High'] + CLV_TIER_DIST['Medium']) * 100):
            customer_lifetime_value_tier = 'Medium'
        else:
            customer_lifetime_value_tier = 'Low'

        if customer_lifetime_value_tier == 'High':
            customer_churn_risk_score = customer_rng.randint(10, 30)
        elif customer_lifetime_value_tier == 'Low':
            customer_churn_risk_score = customer_rng.randint(70, 95)
        else: 
            customer_churn_risk_score = customer_rng.randint(30, 70)
        
        order_is_gift_purchase = customer_rng.random() < GIFT_PURCHASE_PROBABILITY

        if customer_churn_risk_score > 60:
            customer_past_return_rate_for_delayed_items = customer_rng.uniform(0.3, 0.7)
        else:
            customer_past_return_rate_for_delayed_items = customer_rng.uniform(0.01, 0.1)
        

        orders_data.append({
            'order_id': order_id,
            'customer_id': customer_id,
            'order_date': order_date.strftime('%Y-%m-%d'),
            'is_prime_member': is_prime_member,
            'expected_delivery_date': expected_delivery_date.strftime('%Y-%m-%d'),
            'actual_delivery_date': actual_delivery_date.strftime('%Y-%m-%d'),
            'delivery_status': delivery_status,
            'carrier': carrier,
            'delivery_cost_to_amazon': delivery_cost_to_amazon,
            'product_id': product_code_for_order, 
            'order_quantity': round(product_quantity), 
            'price_usd': round(product_price_usd, 2), 
            'line_item_total_usd': round(line_item_total_usd, 2), 
            'destination_zip_code': destination_zip_code, 
            'destination_city': destination_city_name,
            'destination_state_code': destination_state_code,
            'destination_latitude': destination_latitude,
            'destination_longitude': destination_longitude,
            'customer_lifetime_value_tier': customer_lifetime_value_tier,
            'customer_churn_risk_score': customer_churn_risk_score,
            'order_is_gift_purchase': order_is_gift_purchase,
            'customer_past_return_rate_for_delayed_items': round(customer_past_return_rate_for_delayed_items, 2)
        })

        # --- Simulate Last-Mile Operational Data (with new FC origin) ---
        fc_id, fc_lat, fc_lon = _find_nearest_fc(destination_latitude, destination_longitude) # Find nearest FC

        station_code_name = fc_id # Use FC ID as station code
        route_score_val = random.choices(['High', 'Medium', 'Low'], weights=[0.6, 0.3, 0.1], k=1)[0]
        
        simulated_num_deliveries = random.randint(15, 30)
        simulated_total_volume = simulated_num_deliveries * random.uniform(1000, 3000)
        
        simulated_duration_hours = np.random.normal(5.0, 1.5) 
        simulated_duration_hours = max(1.5, min(simulated_duration_hours, 10.0))
        simulated_distance_km = simulated_duration_hours * np.random.normal(25, 5) 
        simulated_distance_km = max(20.0, simulated_distance_km) 

        if route_score_val == 'High':
            simulated_duration_hours *= random.uniform(0.8, 1.0)
            simulated_distance_km *= random.uniform(0.8, 1.0)
        elif route_score_val == 'Low':
            simulated_duration_hours *= random.uniform(1.2, 1.5)
            simulated_distance_km *= random.uniform(1.2, 1.5)

        last_mile_ops_data.append({
            'route_id': f"ROUTE-{order_id}",
            'order_date': order_date.strftime('%Y-%m-%d'),
            'city': destination_city_name,
            'station_code': station_code_name, # Use FC ID
            'route_score': route_score_val,
            'origin_latitude': fc_lat, # Use FC's latitude
            'origin_longitude': fc_lon, # Use FC's longitude
            'vehicle_capacity_cm3': random.choice([3313071, 4247527, 4672279]), 
            'num_deliveries': round(simulated_num_deliveries),
            'total_calculated_volume_cm3': round(simulated_total_volume),
            'actual_route_duration_hours': round(simulated_duration_hours, 2),
            'actual_route_distance_km': round(simulated_distance_km, 2)
        })

    # Create DataFrames
    df_orders = pd.DataFrame(orders_data)
    df_orders['delivery_days_actual'] = (pd.to_datetime(df_orders['actual_delivery_date']) - pd.to_datetime(df_orders['order_date'])).dt.days

    df_last_mile_ops = pd.DataFrame(last_mile_ops_data)
    df_last_mile_ops.drop_duplicates(subset=['route_id'], inplace=True)


    # --- Save to CSVs ---
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)

    df_orders.to_csv(SIMULATED_ORDERS_CSV, index=False)
    logger.info(f"Generated {NUM_ORDERS} orders to '{SIMULATED_ORDERS_CSV}'")

    df_orders_enhanced = df_orders.copy()
    df_orders_enhanced['destination_zip_code'] = df_orders_enhanced['destination_zip_code'].astype(str).str.zfill(5)
    df_orders_enhanced.to_csv(SIMULATED_ORDERS_ENHANCED_CSV, index=False)
    logger.info(f"Generated {len(df_orders_enhanced)} enhanced orders to '{SIMULATED_ORDERS_ENHANCED_CSV}'")

    df_last_mile_ops.to_csv(SIMULATED_LAST_MILE_OPERATIONS_CSV, index=False)
    logger.info(f"Generated {len(df_last_mile_ops)} simulated last mile operations to '{SIMULATED_LAST_MILE_OPERATIONS_CSV}'")

    # --- Debugging Logs for Validation ---
    logger.info("\n--- Simulated Orders (Enhanced) Info ---")
    logger.info("First 5 rows:\n" + str(df_orders_enhanced.head()))
    logger.info("Descriptive statistics for Prime vs Standard delivery days:\n" + str(df_orders_enhanced.groupby('is_prime_member')['delivery_days_actual'].describe()))
    logger.info("Carrier distribution for Prime members:\n" + str(df_orders_enhanced[df_orders_enhanced['is_prime_member'] == True]['carrier'].value_counts(normalize=True)))
    logger.info("Carrier distribution for Standard members:\n" + str(df_orders_enhanced[df_orders['is_prime_member'] == False]['carrier'].value_counts(normalize=True)))
    logger.info("Top 10 destination cities:\n" + str(df_orders_enhanced['destination_city'].value_counts().head(10)))

    logger.info("\n--- Simulated Last-Mile Operations Info ---")
    logger.info("First 5 rows:\n" + str(df_last_mile_ops.head()))
    logger.info("Descriptive statistics:\n" + str(df_last_mile_ops.describe()))
    logger.info("Route counts by city:\n" + str(df_last_mile_ops['city'].value_counts().head(10)))
    logger.info("Route score counts:\n" + str(df_last_mile_ops['route_score'].value_counts()))

    return df_orders_enhanced, df_last_mile_ops 

if __name__ == "__main__":
    generate_simulated_orders_and_last_mile_data()