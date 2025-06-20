# scripts/web_app.py

"""
Amazon Logistics Intelligence Dashboard

A comprehensive Streamlit application for analyzing Amazon delivery operations,
cost efficiency, and predicting delivery delays using machine learning.

This dashboard provides real-time insights into logistics performance,
carrier analysis, and optimization opportunities.

Author: [Your Name]
Date: [Current Date]
"""

import streamlit as st
import pandas as pd
import requests 
import json 
import os
import altair as alt 
from datetime import datetime, timedelta 
import numpy as np
import sys

# Import page modules for better code organization
from pages.prime_performance import render_page as render_prime_performance
from pages.last_mile_operations import render_page as render_last_mile_operations
from pages.cost_efficiency_analysis import render_page as render_cost_efficiency_analysis

# Helper functions for Natural Language Query processing
def get_time_filtered_data(df, query_lower):
    """
    Filter data based on time-related keywords in natural language queries.
    
    This function helps users filter data by common time periods like
    'yesterday', 'this week', 'last month', etc.
    
    Args:
        df (pd.DataFrame): The dataframe to filter
        query_lower (str): Lowercase query string to analyze
        
    Returns:
        tuple: (filtered_dataframe, message_string)
    """
    try:
        if 'order_date' not in df.columns:
            return df, "Warning: No date information available for filtering."
        
        message = ""
        
        # Handle different time period requests
        if any(term in query_lower for term in ['yesterday', 'last day']):
            filtered = df[df['order_date'].dt.date == (datetime.now().date() - timedelta(days=1))]
            message = "Showing data for yesterday"
        elif any(term in query_lower for term in ['this week', 'past week', 'last 7 days']):
            filtered = df[df['order_date'].dt.date >= (datetime.now().date() - timedelta(days=7))]
            message = "Showing data for the last 7 days"
        elif any(term in query_lower for term in ['this month', 'past month', 'last 30 days', 'last month']):
            filtered = df[df['order_date'].dt.date >= (datetime.now().date() - timedelta(days=30))]
            message = "Showing data for the last 30 days"
        else:
            return df, "Showing all available data"
        
        # Check if we found any data for the specified period
        if len(filtered) == 0:
            return df, "No data found for the specified time period. Showing all available data instead."
        
        return filtered, message
    except Exception as e:
        return df, f"Error in time filtering: {str(e)}. Showing all available data."

def get_location_filtered_data(df, query_lower):
    """
    Filter data based on city names mentioned in natural language queries.
    
    This function searches for city names in the query and filters the data
    to show only records for that specific city.
    
    Args:
        df (pd.DataFrame): The dataframe to filter
        query_lower (str): Lowercase query string to analyze
        
    Returns:
        tuple: (filtered_dataframe, city_name, message_string)
    """
    try:
        if 'destination_city' not in df.columns:
            return df, None, "Warning: No city information available for filtering."
        
        # Get unique cities from the data
        cities = df['destination_city'].unique()
        
        # Search for city names in the query
        for city in cities:
            if city.lower() in query_lower:
                filtered = df[df['destination_city'] == city]
                if len(filtered) > 0:
                    return filtered, city, f"Filtered for {city}"
        
        return df, None, "No specific city mentioned or found in data"
    except Exception as e:
        return df, None, f"Error in location filtering: {str(e)}"

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Configuration (Paths for this script - Inlined) ---
from config.settings import (
    SIMULATED_ORDERS_ENHANCED_CSV,
    PROCESSED_ENHANCED_META_ROUTES_CSV,  # Fixed variable name
    PROCESSED_PURCHASE_CSV,
    GOOGLE_ROUTES_API_KEY,
    BASE_DIR
)

# --- File Paths ---
PREDICTIONS_CSV = os.path.join('data', 'simulated_orders_with_predictions.csv') 
WEATHER_ALERTS_CSV = os.path.join('data', 'weather_alerts.csv') 
NEWS_ALERTS_CSV = os.path.join('data', 'news_alerts.csv')     
TRAFFIC_DATA_CSV = os.path.join('data', 'traffic_data.csv')

# Data file paths
SIMULATED_LAST_MILE_OPERATIONS_CSV = os.path.join('data', 'simulated_last_mile_operations.csv')
PROCESSED_AMAZON_PURCHASE_CSV = os.path.join('data', 'processed_amazon_purchase.csv')

PREDICTION_API_URL = "http://localhost:8000/predict_delay"
HEALTH_API_URL = "http://localhost:8000/health"

# Google Routes API Configuration
# Use environment variable or Streamlit secrets for API key
GOOGLE_ROUTES_API_KEY = os.getenv('GOOGLE_ROUTES_API_KEY', '')  # Environment variable approach
if not GOOGLE_ROUTES_API_KEY:
    # Fallback to Streamlit secrets if environment variable not set
    try:
        GOOGLE_ROUTES_API_KEY = st.secrets.get("GOOGLE_ROUTES_API_KEY", "")
    except:
        GOOGLE_ROUTES_API_KEY = ""
        
GOOGLE_ROUTES_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"

# Simulated FC locations (matching the backend)
SIMULATED_FC_COORDS = {
    'FC-NJ1': {'lat': 40.8000, 'lon': -74.1000, 'name': 'New Jersey FC (Near NYC)'},
    'FC-CA1': {'lat': 33.9000, 'lon': -117.5000, 'name': 'California FC (Near LA)'},
    'FC-IL1': {'lat': 41.6000, 'lon': -88.0000, 'name': 'Illinois FC (Near Chicago)'},
    'FC-TX1': {'lat': 29.9000, 'lon': -95.1000, 'name': 'Texas FC (Near Houston)'},
    'FC-GA1': {'lat': 33.5000, 'lon': -84.7000, 'name': 'Georgia FC (Near Atlanta)'},
    'FC-AZ1': {'lat': 33.3000, 'lon': -111.9000, 'name': 'Arizona FC (Near Phoenix)'},
    'FC-PA1': {'lat': 40.1000, 'lon': -75.5000, 'name': 'Pennsylvania FC (Near Philadelphia)'},
    'FC-TX2': {'lat': 29.6000, 'lon': -98.0000, 'name': 'Texas FC2 (Near San Antonio)'},
    'FC-CA2': {'lat': 33.0000, 'lon': -117.0000, 'name': 'California FC2 (Near San Diego)'},
    'FC-TX3': {'lat': 32.9000, 'lon': -96.5000, 'name': 'Texas FC3 (Near Dallas)'}
}

# --- Simulated Geographical Data for Consistency (Used in ML Prediction Demo default values) ---
US_CITIES_WEBAPP = { 
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

# Add caching for AI Assistant computations
@st.cache_data
def get_filtered_data(df, date_range=None, selected_city=None):
    """Cache filtered data to avoid recomputation"""
    filtered_df = df.copy()
    if not filtered_df.empty:
        if date_range and len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['order_date'] >= pd.Timestamp(date_range[0])) &
                (filtered_df['order_date'] <= pd.Timestamp(date_range[1]))
            ]
        if selected_city and selected_city != 'All Cities':
            filtered_df = filtered_df[filtered_df['destination_city'] == selected_city]
    return filtered_df

@st.cache_data
def calculate_trend_metrics(filtered_df):
    """Cache trend calculations"""
    if filtered_df.empty:
        return None
    
    # Calculate recent trends
    recent_data = filtered_df.sort_values('order_date').tail(30)
    historical_data = filtered_df.sort_values('order_date').iloc[:-30]
    
    if len(recent_data) == 0 or len(historical_data) == 0:
        return None
    
    metrics = {}
    
    # Performance metrics
    if 'delivery_status' in recent_data.columns:
        current_on_time = (recent_data['delivery_status'] == 'On-Time').mean() * 100
        historical_on_time = (historical_data['delivery_status'] == 'On-Time').mean() * 100
        trend_direction = 'improving' if current_on_time > historical_on_time else 'declining'
        trend_magnitude = abs(current_on_time - historical_on_time)
        
        metrics['performance'] = {
            'current_on_time': current_on_time,
            'historical_on_time': historical_on_time,
            'trend_direction': trend_direction,
            'trend_magnitude': trend_magnitude
        }
    
    # Volume metrics
    if 'order_date' in filtered_df.columns:
        daily_orders = filtered_df.groupby(filtered_df['order_date'].dt.date)['order_id'].count()
        avg_daily_orders = daily_orders.mean()
        recent_daily_orders = daily_orders.tail(7).mean()
        volume_trend = (recent_daily_orders - avg_daily_orders) / avg_daily_orders * 100
        
        metrics['volume'] = {
            'avg_daily_orders': avg_daily_orders,
            'recent_daily_orders': recent_daily_orders,
            'volume_trend': volume_trend
        }
    
    # Cost metrics
    if 'delivery_cost_to_amazon' in filtered_df.columns:
        recent_avg_cost = recent_data['delivery_cost_to_amazon'].mean()
        historical_avg_cost = historical_data['delivery_cost_to_amazon'].mean()
        cost_trend = (recent_avg_cost - historical_avg_cost) / historical_avg_cost * 100
        
        metrics['cost'] = {
            'recent_avg_cost': recent_avg_cost,
            'historical_avg_cost': historical_avg_cost,
            'cost_trend': cost_trend
        }
    
    return metrics

@st.cache_data
def calculate_carrier_metrics(filtered_df):
    """Cache carrier performance calculations"""
    if filtered_df.empty or 'carrier' not in filtered_df.columns:
        return None
    
    carrier_metrics = filtered_df.groupby('carrier').agg({
        'delivery_status': lambda x: (x == 'On-Time').mean() * 100,
        'delivery_cost_to_amazon': 'mean',
        'order_id': 'count'
    }).round(2)
    
    # Calculate efficiency score safely
    carrier_metrics['efficiency_score'] = (
        carrier_metrics['delivery_status'] / 
        carrier_metrics['delivery_cost_to_amazon'].replace(0, float('inf'))
    ).round(2)
    
    return carrier_metrics.sort_values('efficiency_score', ascending=False)

# --- Helper function to load data (from CSVs for simplicity in Streamlit) ---
@st.cache_data # Cache data to avoid reloading on every rerun
def load_data(filepath):
    """
    Load and preprocess CSV data files with proper type conversion.
    
    This function handles the loading of various CSV files used in the dashboard,
    including automatic type conversion for common columns like dates and numeric fields.
    
    Args:
        filepath (str): Path to the CSV file to load
        
    Returns:
        pd.DataFrame: Loaded and preprocessed dataframe, or empty dataframe if error
        
    Note:
        Uses Streamlit caching to improve performance by avoiding repeated file reads.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Convert common date columns to datetime
        date_columns = ['order_date', 'delivery_datetime', 'pickup_datetime']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Ensure prediction columns are correctly typed
        if 'predicted_delivery_status_class' in df.columns:
            df['predicted_delivery_status_class'] = df['predicted_delivery_status_class'].astype(int)
        if 'predicted_delay_probability' in df.columns:
            df['predicted_delay_probability'] = df['predicted_delay_probability'].astype(float)

        return df
    except FileNotFoundError:
        st.error(f"Data file not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return pd.DataFrame()


# --- Helper function to make API call for prediction ---
def get_prediction_from_api(features_dict):
    try:
        response = requests.post(PREDICTION_API_URL, json=features_dict)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("API connection failed. Please ensure the prediction API server is running (uvicorn scripts.ml_prediction_api:app).")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API request failed with status {response.status_code}: {response.text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during API call: {e}")
        return None


# --- Google Routes API Functions ---
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth.
    
    Uses the Haversine formula to compute the distance between two geographic
    coordinates, accounting for the Earth's curvature.
    
    Args:
        lat1, lon1 (float): Latitude and longitude of first point
        lat2, lon2 (float): Latitude and longitude of second point
        
    Returns:
        float: Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Calculate differences
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    
    return distance

def find_nearest_fc(dest_lat, dest_lon):
    """
    Find the nearest fulfillment center to a given destination.
    
    Compares the destination coordinates with all known FC locations
    and returns the closest one along with the distance.
    
    Args:
        dest_lat, dest_lon (float): Destination coordinates
        
    Returns:
        tuple: (fc_id, distance_km)
    """
    min_dist = float('inf')
    nearest_fc = None
    
    for fc_id, coords in SIMULATED_FC_COORDS.items():
        dist = calculate_haversine_distance(dest_lat, dest_lon, coords['lat'], coords['lon'])
        if dist < min_dist:
            min_dist = dist
            nearest_fc = fc_id
    
    return nearest_fc, min_dist

def get_google_route_info(origin_lat, origin_lon, dest_lat, dest_lon):
    """
    Get real-time route information from Google Routes API.
    
    Fetches current traffic conditions, estimated travel time, and route
    distance between two points using Google's Routes API.
    
    Args:
        origin_lat, origin_lon (float): Origin coordinates
        dest_lat, dest_lon (float): Destination coordinates
        
    Returns:
        tuple: (route_info_dict, error_message) or (None, error_message)
    """
    if not GOOGLE_ROUTES_API_KEY:
        return None, "Google Routes API key not configured"
    
    try:
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': GOOGLE_ROUTES_API_KEY,
            'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters,routes.legs.staticDuration,routes.legs.distanceMeters'
        }
        
        payload = {
            "origin": {
                "location": {
                    "latLng": {
                        "latitude": origin_lat,
                        "longitude": origin_lon
                    }
                }
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude": dest_lat,
                        "longitude": dest_lon
                    }
                }
            },
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE",
            "departureTime": "now",
            "computeAlternativeRoutes": False,
            "routeModifiers": {
                "avoidTolls": False,
                "avoidHighways": False
            }
        }
        
        response = requests.post(GOOGLE_ROUTES_API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'routes' in data and len(data['routes']) > 0:
                route = data['routes'][0]
                duration_seconds = int(route.get('duration', '0s').replace('s', ''))
                distance_meters = route.get('distanceMeters', 0)
                distance_km = distance_meters / 1000
                
                return {
                    'duration_seconds': duration_seconds,
                    'distance_km': distance_km,
                    'duration_minutes': duration_seconds / 60,
                    'estimated_hours': duration_seconds / 3600
                }, None
            else:
                return None, "No route found"
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return None, "Request timeout"
    except requests.exceptions.RequestException as e:
        return None, f"Request error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def analyze_routing_delays(origin_lat, origin_lon, dest_lat, dest_lon):
    """
    Analyze potential routing delays using Google Routes API.
    
    Evaluates route characteristics to identify potential delay factors
    and provides recommendations for optimization.
    
    Args:
        origin_lat, origin_lon (float): Origin coordinates
        dest_lat, dest_lon (float): Destination coordinates
        
    Returns:
        dict: Analysis results with delay factors and recommendations
    """
    route_info, error = get_google_route_info(origin_lat, origin_lon, dest_lat, dest_lon)
    
    if error:
        return {
            'has_routing_data': False,
            'error': error,
            'recommendations': ['Unable to analyze routing due to API error']
        }
    
    # Initialize analysis structure
    analysis = {
        'has_routing_data': True,
        'route_info': route_info,
        'delay_factors': [],
        'recommendations': []
    }
    
    # Check for long distances that might cause delays
    if route_info['distance_km'] > 500:
        analysis['delay_factors'].append(f"Long distance route ({route_info['distance_km']:.1f} km)")
        analysis['recommendations'].append("Consider using regional FC or expedited shipping")
    
    # Check for long travel times
    if route_info['estimated_hours'] > 8:
        analysis['delay_factors'].append(f"Long travel time ({route_info['estimated_hours']:.1f} hours)")
        analysis['recommendations'].append("Route may require overnight delivery or alternative carrier")
    
    # Check for inefficient routes (distance vs time ratio)
    speed_kmh = route_info['distance_km'] / route_info['estimated_hours'] if route_info['estimated_hours'] > 0 else 0
    if speed_kmh < 40 and speed_kmh > 0:  # Average speed below 40 km/h suggests traffic or inefficient route
        analysis['delay_factors'].append(f"Slow average speed ({speed_kmh:.1f} km/h)")
        analysis['recommendations'].append("Route may have traffic congestion or inefficient path")
    
    # If no specific issues found
    if not analysis['delay_factors']:
        analysis['delay_factors'].append("Route appears optimal")
        analysis['recommendations'].append("Standard delivery should be sufficient")
    
    return analysis

# Add caching for Google Routes API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour since route data doesn't change frequently
def get_cached_route_info(origin_lat, origin_lon, dest_lat, dest_lon):
    """Cache route information to avoid repeated API calls"""
    return get_google_route_info(origin_lat, origin_lon, dest_lat, dest_lon)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_routing_analysis(origin_lat, origin_lon, dest_lat, dest_lon):
    """Cache routing analysis to avoid repeated computations"""
    return analyze_routing_delays(origin_lat, origin_lon, dest_lat, dest_lon)

# --- Helper: Prepare features for ML Prediction Demo API call ---
# This function will prepare the full feature dictionary required by the API
def _prepare_features_for_api(df_orders_enhanced, df_weather_alerts, df_news_alerts, 
                              is_prime_member_in, destination_city_in, order_quantity_in, is_severe_weather_alert, weather_severity_score, is_supply_chain_news_alert, news_disruption_score):
    
    # Initialize features with common defaults from overall data
    # These match the expected inputs for ml_prediction_api.py (OrderFeatures Pydantic model)
    features = {
        "is_prime_member": is_prime_member_in,
        "carrier": 'AMZL', # Default carrier for Prime
        "destination_state_code": "NY", # Default, will be updated by US_CITIES_WEBAPP
        "destination_city": destination_city_in,
        "customer_lifetime_value_tier": 'Medium', # Default
        "order_is_gift_purchase": False, # Default
        "delivery_cost_to_amazon": 5.0, # Default
        "order_quantity": order_quantity_in, # This comes from user input now
        "customer_churn_risk_score": 50.0, # Default
        "customer_past_return_rate_for_delayed_items": 0.05, # Default
        "order_month": datetime.now().month,
        "order_day_of_week": datetime.now().weekday(),
        "order_hour_of_day": datetime.now().hour,
        "destination_latitude": 40.7128, # Default to NY
        "destination_longitude": -74.0060, # Default to NY
        "is_severe_weather_alert": 0,
        "weather_severity_score": 0,
        "is_supply_chain_news_alert": 0,
        "news_disruption_score": 0
    }

    # --- Infer / Override based on df_orders_enhanced (historical data) ---
    if not df_orders_enhanced.empty:
        filtered_df_by_city_prime = df_orders_enhanced[
            (df_orders_enhanced['destination_city'] == destination_city_in) &
            (df_orders_enhanced['is_prime_member'] == is_prime_member_in)
        ]
        
        if not filtered_df_by_city_prime.empty:
            features["carrier"] = filtered_df_by_city_prime['carrier'].mode()[0] if not filtered_df_by_city_prime['carrier'].mode().empty else features["carrier"]
            features["customer_lifetime_value_tier"] = filtered_df_by_city_prime['customer_lifetime_value_tier'].mode()[0] if not filtered_df_by_city_prime['customer_lifetime_value_tier'].mode().empty else features["customer_lifetime_value_tier"]
            features["order_is_gift_purchase"] = filtered_df_by_city_prime['order_is_gift_purchase'].mode()[0] if not filtered_df_by_city_prime['order_is_gift_purchase'].mode().empty else features["order_is_gift_purchase"]
            features["delivery_cost_to_amazon"] = float(filtered_df_by_city_prime['delivery_cost_to_amazon'].mean())
            features["customer_churn_risk_score"] = float(filtered_df_by_city_prime['customer_churn_risk_score'].mean())
            features["customer_past_return_rate_for_delayed_items"] = float(filtered_df_by_city_prime['customer_past_return_rate_for_delayed_items'].mean())

        selected_city_coords = US_CITIES_WEBAPP.get(destination_city_in, {'lat': 40.7128, 'lon': -74.0060}) # Fallback to NY
        features["destination_state_code"] = selected_city_coords['state']
        features["destination_latitude"] = selected_city_coords['lat']
        features["destination_longitude"] = selected_city_coords['lon']

    # --- Integrate Real-Time (Actual) Weather/News Features ---
    # Weather impact
    if not df_weather_alerts.empty:
        # Filter for active alerts in the selected city today
        active_weather_alerts_for_city = df_weather_alerts[
            (pd.to_datetime(df_weather_alerts['start']) <= datetime.now()) & 
            (pd.to_datetime(df_weather_alerts['end']) >= datetime.now()) &
            (df_weather_alerts['city'] == destination_city_in)
        ]
        if not active_weather_alerts_for_city.empty:
            features['is_severe_weather_alert'] = 1
            features['weather_severity_score'] = active_weather_alerts_for_city['severity_score'].max() if 'severity_score' in active_weather_alerts_for_city.columns else 1 # Max severity
            st.info(f"**Live Alert (Weather):** Active {active_weather_alerts_for_city.iloc[0]['event']} in {destination_city_in}.")

    # News impact
    if not df_news_alerts.empty:
        # Check for recent news alerts related to the destination city OR general supply chain/logistics disruptions
        recent_news_alerts_for_city = df_news_alerts[
            (pd.to_datetime(df_news_alerts['publishedAt']) >= datetime.now() - timedelta(days=2)) &
            (df_news_alerts['city'].str.lower() == destination_city_in.lower())
        ]
        
        if not recent_news_alerts_for_city.empty:
            features['is_supply_chain_news_alert'] = 1
            features['news_disruption_score'] = recent_news_alerts_for_city['score'].max() if 'score' in recent_news_alerts_for_city.columns else 1
            
            # Show the actual news alerts to the user
            st.info(f"**Live Alert (News):** Recent disruption news found that may impact deliveries to {destination_city_in} or general logistics.")
            st.write("**Recent Relevant News:**")
            for idx, row in recent_news_alerts_for_city.head(3).iterrows():
                st.markdown(f"- **{row['title']}** (Source: {row['source']}, Published: {pd.to_datetime(row['publishedAt']).strftime('%Y-%m-%d %H:%M')})")
        else:
            st.info(f"No recent disruption news found for {destination_city_in} or general logistics.")


    return features


# --- Streamlit App Layout & Theming ---
st.set_page_config(
    layout="wide", 
    page_title="Amazon Logistics Intelligence Dashboard",
    page_icon="📦",
    initial_sidebar_state="expanded"
)

# Improved readability theme with better contrast
st.markdown("""
<style>
    /* Base theme colors with improved contrast */
    :root {
        --bg-primary: #1A1F24;
        --bg-secondary: #242A30;
        --text-primary: #FFFFFF;
        --text-secondary: rgba(255, 255, 255, 0.85);
        --accent-primary: #FF9900;
        --accent-secondary: #FFB84D;
        --border-color: rgba(255, 255, 255, 0.1);
    }

    /* Main app background */
    .stApp {
        background-color: var(--bg-primary) !important;
    }

    /* Main container background */
    .main > .block-container {
        background-color: var(--bg-primary) !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Consistent text styling for all content */
    .stMarkdown, .stText, p, span, div {
        color: var(--text-primary) !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Headers with improved contrast */
    h1 {
        color: var(--text-primary) !important;
        font-size: 32px !important;
        font-weight: 700 !important;
        margin: 24px 0 16px 0 !important;
    }

    h2 {
        color: var(--text-primary) !important;
        font-size: 24px !important;
        font-weight: 600 !important;
        margin: 20px 0 12px 0 !important;
    }

    h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        margin: 16px 0 8px 0 !important;
    }

    /* Input fields with better contrast */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTimeInput > div > div > input,
    .stTextArea > div > textarea,
    .stSelectbox > div,
    .stDateInput > div, /* Applied to container */
    .stRadio > div {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
        font-size: 16px !important;
    }

    /* Clean up inner style of date input now that container is styled */
    .stDateInput > div > div > input {
        background-color: transparent !important;
        border: none !important;
        color: var(--text-primary) !important;
        padding: 0 !important;
    }

    /* Tables with improved readability */
    .stTable th {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        padding: 12px !important;
        border-bottom: 1px solid var(--border-color) !important;
    }

    .stTable td {
        background-color: var(--bg-primary) !important;
        color: var(--text-secondary) !important;
        padding: 8px 12px !important;
    }

    /* Buttons with better contrast */
    .stButton > button {
        background: var(--accent-primary) !important;
        color: #000000 !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        border-radius: 4px !important;
        border: none !important;
        font-size: 16px !important;
    }

    .stButton > button:hover {
        background: var(--accent-secondary) !important;
    }

    /* Metrics with improved readability */
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricDelta"] {
        color: var(--text-secondary) !important;
        font-size: 14px !important;
    }

    /* Alerts and messages */
    .stAlert {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        padding: 16px !important;
        border-radius: 4px !important;
        margin: 8px 0 !important;
    }

    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background-color: var(--bg-primary) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary) !important;
    }

    /* Charts and visualizations */
    .stPlot {
        background-color: var(--bg-secondary) !important;
        padding: 16px !important;
        border-radius: 4px !important;
        border: 1px solid var(--border-color) !important;
    }

    /* Remove any white backgrounds and ensure dark theme */
    .stMarkdown > div,
    .element-container,
    .block-container {
        background-color: transparent !important;
    }

    /* Ensure all text has sufficient contrast */
    * {
        color: var(--text-primary) !important;
    }

    /* Override any white text on dark backgrounds */
    .stMarkdown p,
    .stMarkdown span,
    .stMarkdown div,
    .stText p,
    .stText span,
    .stText div {
        color: var(--text-primary) !important;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header with dark theme styling
st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #242A30 0%, #1A1F24 100%); border-radius: 12px; margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.1);">
        <h1 style="margin: 0; padding: 0; color: #FFFFFF;">📦 Amazon Logistics Intelligence Dashboard</h1>
        <p style="color: rgba(255, 255, 255, 0.85); font-size: 1.1rem; margin: 0.5rem 0 0 0; font-weight: 400;">
            Real-time insights and predictive analytics for optimal logistics operations
        </p>
    </div>
""", unsafe_allow_html=True)

# Enhanced sidebar with dark theme styling
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 2rem;">
            <h3 style="color: #FFFFFF; margin: 0;">🚀 Navigation</h3>
        </div>
    """, unsafe_allow_html=True)
    
    page_selection = st.radio(
        "Select Dashboard Section",
        ["Prime Performance", "Simulated Last-Mile Operations", "Cost Efficiency Analysis", "Amazon Purchase Trends", "ML Prediction Demo", "AI Dispatcher Assistant"],
        format_func=lambda x: {
            "Prime Performance": "📊 Prime Performance",
            "Simulated Last-Mile Operations": "🚚 Last-Mile Operations", 
            "Cost Efficiency Analysis": "💰 Cost Efficiency Analysis",
            "Amazon Purchase Trends": "📈 Purchase Trends",
            "ML Prediction Demo": "🤖 ML Prediction Demo",
            "AI Dispatcher Assistant": "💬 AI Assistant"
        }[x]
    )
    
    # Add helpful tips section with dark theme
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 8px; margin-top: 2rem; border: 1px solid rgba(255, 255, 255, 0.1);">
            <h4 style="color: #FFFFFF; margin: 0 0 0.5rem 0;">💡 Quick Tips</h4>
            <ul style="color: rgba(255, 255, 255, 0.85); font-size: 0.9rem; margin: 0; padding-left: 1rem;">
                <li>Use filters to focus on specific data</li>
                <li>Hover over charts for detailed info</li>
                <li>Try the AI Assistant for insights</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# --- Load Data for Dashboard (from CSVs as generated by pipeline) ---
df_orders_enhanced = load_data(PREDICTIONS_CSV) # Load predictions CSV for main orders data
df_last_mile_ops = load_data(SIMULATED_LAST_MILE_OPERATIONS_CSV)
df_amazon_purchase = load_data(PROCESSED_AMAZON_PURCHASE_CSV)
df_almrrc_meta = load_data(PROCESSED_ENHANCED_META_ROUTES_CSV)
df_weather_alerts = load_data(WEATHER_ALERTS_CSV) 
df_news_alerts = load_data(NEWS_ALERTS_CSV) 
df_traffic_alerts = load_data(TRAFFIC_DATA_CSV)

# --- Data Transformation: Add severity_score to weather alerts ---
if not df_weather_alerts.empty and 'severity' in df_weather_alerts.columns:
    severity_mapping = {
        'Extreme': 5,
        'Severe': 4,
        'Moderate': 3,
        'Minor': 2,
        'Unknown': 1
    }
    df_weather_alerts['severity_score'] = df_weather_alerts['severity'].map(severity_mapping).fillna(1).astype(int)

# --- Data Transformation: Add severity to traffic data ---
if not df_traffic_alerts.empty and 'incident_present' in df_traffic_alerts.columns:
    # Map incident_present (0/1) to severity levels
    # 0 = no incident (severity 1), 1 = incident (severity 4)
    df_traffic_alerts['severity'] = df_traffic_alerts['incident_present'].map({0: 1, 1: 4}).fillna(1).astype(int)
    
    # Add a 'city' column for consistency with other data sources
    # Combine origin and destination cities for broader coverage
    df_traffic_alerts['city'] = df_traffic_alerts['origin_city']
    
    # Add a 'description' column based on available data
    df_traffic_alerts['description'] = df_traffic_alerts.apply(
        lambda row: f"Traffic incident on route from {row['origin_city']} to {row['destination_city']} ({row['distance_km']}km)" 
        if row['incident_present'] == 1 
        else f"Normal traffic on route from {row['origin_city']} to {row['destination_city']} ({row['distance_km']}km)",
        axis=1
    )


# --- Page Content Logic ---
if page_selection == "Prime Performance":
    render_prime_performance(df_orders_enhanced)

elif page_selection == "Simulated Last-Mile Operations":
    render_last_mile_operations(df_last_mile_ops)

elif page_selection == "Cost Efficiency Analysis":
    render_cost_efficiency_analysis(df_orders_enhanced)

elif page_selection == "Amazon Purchase Trends":
    st.header("Real-World Amazon Purchase Trends (MIT Data)")
    st.write("*(Analysis of product sales, categories, and spending from research data)*")
    
    if not df_amazon_purchase.empty:
        st.subheader("Top Products by Sales")
        top_products = df_amazon_purchase.groupby('product_title')['line_item_total_usd'].sum().nlargest(10).reset_index(name='Total Sales')
        top_products_chart = alt.Chart(top_products).mark_bar().encode(
            x=alt.X('Total Sales:Q', title='Total Sales (USD)'),
            y=alt.Y('product_title:N', sort='-x', title='Product Title'),
            color=alt.value('#232F3E'), 
            tooltip=['product_title', alt.Tooltip('Total Sales', format='$.2f')]
        ).properties(
            title='Top 10 Products by Sales'
        ).interactive()
        st.altair_chart(top_products_chart, use_container_width=True)

        st.subheader("Sales Trend by Date")
        sales_by_date = df_amazon_purchase.groupby(pd.Grouper(key='order_date', freq='M'))['line_item_total_usd'].sum().reset_index()
        sales_by_date.columns = ['order_date', 'Total Sales']
        
        sales_trend_chart = alt.Chart(sales_by_date).mark_line(point=True).encode(
            x=alt.X('yearmonth(order_date):T', title='Month-Year', 
                    axis=alt.Axis(format="%Y-%m", labelAngle=-45, labelOverlap="greedy")), 
            y=alt.Y('Total Sales:Q', title='Total Sales (USD)'),
            color=alt.value('#FF9900'), 
            tooltip=[alt.Tooltip('yearmonth(order_date)', title='Month-Year'), alt.Tooltip('Total Sales', format='$.2f')]
        ).properties(
            title='Monthly Sales Trend'
        ).interactive()
        st.altair_chart(sales_trend_chart, use_container_width=True)

    else:
        st.warning("Amazon Purchase Trends data is not available. Please ensure data is downloaded and processed.")

elif page_selection == "ML Prediction Demo":
    # Enhanced header
    st.markdown("""
        <div style="background: #242A30; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.1);">
            <h2 style="margin: 0 0 0.5rem 0; color: #FFFFFF;">🤖 Delivery Delay Prediction Demo</h2>
            <p style="margin: 0; color: rgba(255, 255, 255, 0.85);">Enter order features to get real-time delay predictions from our ML model</p>
        </div>
    """, unsafe_allow_html=True)

    # API Status Check
    col_status1, col_status2 = st.columns([1, 3])
    with col_status1:
        try:
            response = requests.get(HEALTH_API_URL, timeout=5)
            if response.status_code == 200:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 0.75rem; border-radius: 8px; text-align: center;">
                        <div style="font-size: 0.9rem; font-weight: bold;">✅ API Online</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); color: white; padding: 0.75rem; border-radius: 8px; text-align: center;">
                        <div style="font-size: 0.9rem; font-weight: bold;">❌ API Offline</div>
                    </div>
                """, unsafe_allow_html=True)
        except:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%); color: white; padding: 0.75rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.9rem; font-weight: bold;">⚠️ API Unreachable</div>
                </div>
            """, unsafe_allow_html=True)
    
    with col_status2:
        st.markdown("""
            <div style="background: #1A1F24; padding: 1rem; border-radius: 8px; border-left: 4px solid #17a2b8; border: 1px solid rgba(255, 255, 255, 0.1);">
                <strong style="color: #FFFFFF;">💡 How it works:</strong> <span style="color: rgba(255, 255, 255, 0.85);">Our ML model analyzes order characteristics, customer data, and real-time factors to predict delivery delays with high accuracy.</span>
            </div>
        """, unsafe_allow_html=True)

    # Enhanced form with better styling
    st.markdown("""
        <div style="background: #242A30; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.1);">
            <h3 style="margin: 0 0 1rem 0; color: #FFFFFF;">📝 Enter Order Features</h3>
        </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form", clear_on_submit=False):
        # Main features in a card layout
        st.markdown("""
            <div style="background: #1A1F24; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(255, 255, 255, 0.1);">
                <h4 style="margin: 0 0 0.5rem 0; color: #FFFFFF;">🎯 Core Order Details</h4>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            is_prime_member_in = st.selectbox(
                "Prime Member?", 
                options=[True, False], 
                key="prime_member_in_form",
                help="Prime members typically get priority handling"
            )
            destination_city_in = st.selectbox(
                "Destination City", 
                options=list(US_CITIES_WEBAPP.keys()), 
                key="dest_city_in_form",
                help="Select the delivery destination"
            )
        
        with col2:
            order_quantity_in = st.selectbox(
                "Order Quantity",
                options=list(range(1, 11)),
                index=0,
                key="order_qty_in_form",
                help="Number of items in the order"
            )
            customer_lifetime_value_tier_in = st.selectbox(
                "Customer Value Tier", 
                options=['High', 'Medium', 'Low'], 
                key="clv_tier_in_form",
                help="Customer lifetime value classification"
            )
        
        with col3:
            order_is_gift_purchase_in = st.selectbox(
                "Gift Purchase?", 
                options=[True, False], 
                key="gift_purchase_in_form",
                help="Gift purchases may have different handling requirements"
            )
            
            # Advanced Options in an expander
            with st.expander("⚙️ Advanced Features (Optional)", expanded=False):
                st.markdown("""
                    <div style="background: #242A30; padding: 0.5rem; border-radius: 6px; margin-bottom: 0.5rem; border: 1px solid rgba(255, 255, 255, 0.1);">
                        <small style="color: rgba(255, 255, 255, 0.85);">These features will be auto-populated based on historical data if not specified</small>
                    </div>
                """, unsafe_allow_html=True)
                
                col_adv1, col_adv2 = st.columns(2)
                with col_adv1:
                    delivery_cost_to_amazon_in = st.number_input(
                        "Delivery Cost (USD)", 
                        min_value=1.0, 
                        max_value=20.0, 
                        value=float(df_orders_enhanced['delivery_cost_to_amazon'].mean()) if not df_orders_enhanced.empty else 5.0, 
                        step=0.1, 
                        key="delivery_cost_in_form"
                    )
                    customer_churn_risk_score_in = st.number_input(
                        "Churn Risk (0-100)", 
                        min_value=0, 
                        max_value=100, 
                        value=int(df_orders_enhanced['customer_churn_risk_score'].mean()) if not df_orders_enhanced.empty else 50, 
                        step=1, 
                        key="churn_risk_in_form"
                    )
                    customer_past_return_rate_for_delayed_items_in = st.number_input(
                        "Past Return Rate (0-1)", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=float(df_orders_enhanced['customer_past_return_rate_for_delayed_items'].mean()) if not df_orders_enhanced.empty else 0.05, 
                        format="%.2f", 
                        step=0.01, 
                        key="past_return_rate_in_form"
                    )
                
                with col_adv2:
                    current_date = datetime.now()
                    order_month_in = st.number_input(
                        "Order Month (1-12)", 
                        min_value=1, 
                        max_value=12, 
                        value=current_date.month, 
                        step=1, 
                        key="order_month_in_form"
                    )
                    order_day_of_week_in = st.number_input(
                        "Order Day of Week (0=Mon, 6=Sun)", 
                        min_value=0, 
                        max_value=6, 
                        value=current_date.weekday(), 
                        step=1, 
                        key="order_day_of_week_in_form"
                    )
                    order_hour_of_day_in = st.number_input(
                        "Order Hour of Day (0-23)", 
                        min_value=0, 
                        max_value=23, 
                        value=current_date.hour, 
                        step=1, 
                        key="order_hour_of_day_in_form"
                    )
                    
                    selected_city_coords = US_CITIES_WEBAPP.get(destination_city_in, {'lat': 40.7128, 'lon': -74.0060})
                    destination_latitude_in = st.number_input(
                        "Destination Latitude", 
                        value=selected_city_coords['lat'], 
                        format="%.4f", 
                        key="lat_in_form"
                    )
                    destination_longitude_in = st.number_input(
                        "Destination Longitude", 
                        value=selected_city_coords['lon'], 
                        format="%.4f", 
                        key="lon_in_form"
                    )
        
        # Submit button with enhanced styling
        col_submit1, col_submit2, col_submit3 = st.columns([1, 2, 1])
        with col_submit2:
            submitted = st.form_submit_button(
                "🚀 Get Prediction", 
                use_container_width=True,
                help="Click to analyze the order and get delay prediction"
            )

    if submitted:
        with st.spinner("🤖 Analyzing order data and generating prediction..."):
            features_dict = {
                "is_prime_member": is_prime_member_in,
                "carrier": None,
                "destination_state_code": None,
                "destination_city": destination_city_in,
                "customer_lifetime_value_tier": customer_lifetime_value_tier_in,
                "order_is_gift_purchase": order_is_gift_purchase_in,
                "delivery_cost_to_amazon": delivery_cost_to_amazon_in,
                "order_quantity": order_quantity_in,
                "customer_churn_risk_score": customer_churn_risk_score_in,
                "customer_past_return_rate_for_delayed_items": customer_past_return_rate_for_delayed_items_in,
                "order_month": order_month_in,
                "order_day_of_week": order_day_of_week_in,
                "order_hour_of_day": order_hour_of_day_in,
                "destination_latitude": destination_latitude_in,
                "destination_longitude": destination_longitude_in,
                "is_severe_weather_alert": 0,
                "weather_severity_score": 0,
                "is_supply_chain_news_alert": 0,
                "news_disruption_score": 0,
                "incident_present": 0,
                "duration_in_traffic_seconds": 0,
                "distance_km_traffic": 0,
                "is_us_holiday": 0,
                "distance_to_nearest_fc_km": 0,
                "alert_city": destination_city_in
            }

            inferred_df_context = df_orders_enhanced[
                (df_orders_enhanced['destination_city'] == destination_city_in) &
                (df_orders_enhanced['is_prime_member'] == is_prime_member_in)
            ]
            if not inferred_df_context.empty:
                features_dict["carrier"] = inferred_df_context['carrier'].mode()[0] if not inferred_df_context['carrier'].mode().empty else 'AMZL'
                features_dict["destination_state_code"] = inferred_df_context['destination_state_code'].mode()[0] if not inferred_df_context['destination_state_code'].mode().empty else US_CITIES_WEBAPP.get(destination_city_in, {}).get('state', 'NY')
            else:
                features_dict["carrier"] = 'AMZL'
                features_dict["destination_state_code"] = US_CITIES_WEBAPP.get(destination_city_in, {}).get('state', 'NY')
            
            nearest_fc_id, fc_distance_km = find_nearest_fc(destination_latitude_in, destination_longitude_in)
            features_dict["distance_to_nearest_fc_km"] = fc_distance_km
            fc_coords = SIMULATED_FC_COORDS.get(nearest_fc_id, {'lat': 40.8000, 'lon': -74.1000})
            routing_analysis = get_cached_routing_analysis(fc_coords['lat'], fc_coords['lon'], destination_latitude_in, destination_longitude_in)
            
            if routing_analysis['has_routing_data']:
                route_info = routing_analysis['route_info']
                features_dict["duration_in_traffic_seconds"] = route_info['duration_seconds']
                features_dict["distance_km_traffic"] = route_info['distance_km']
                speed_kmh = (route_info['distance_km'] / route_info['estimated_hours']) if route_info['estimated_hours'] > 0 else 0
                features_dict["incident_present"] = 1 if speed_kmh < 40 and speed_kmh > 0 else 0
            
            if not df_weather_alerts.empty:
                active_weather = df_weather_alerts[(df_weather_alerts['city'] == destination_city_in) & (pd.to_datetime(df_weather_alerts['start']) <= datetime.now()) & (pd.to_datetime(df_weather_alerts['end']) >= datetime.now())]
                if not active_weather.empty:
                    features_dict['is_severe_weather_alert'] = 1
                    features_dict['weather_severity_score'] = float(active_weather['severity_score'].max())
            
            if not df_news_alerts.empty:
                active_news = df_news_alerts[(df_news_alerts['city'].str.lower() == destination_city_in.lower()) & (pd.to_datetime(df_news_alerts['publishedAt']) >= datetime.now() - timedelta(days=2))]
                if not active_news.empty:
                    features_dict['is_supply_chain_news_alert'] = 1
                    features_dict['news_disruption_score'] = float(active_news['score'].max())

            for feature, value in features_dict.items():
                if isinstance(value, (np.int64, np.int32)): value = int(value)
                if isinstance(value, (np.float64, np.float32)): value = float(value)
                features_dict[feature] = value

            prediction_result = get_prediction_from_api(features_dict)

        if prediction_result:
            st.markdown("""
                <div style="background: #242A30; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-top: 2rem; border: 1px solid rgba(255, 255, 255, 0.1);">
                    <h3 style="margin: 0 0 1rem 0; color: #FFFFFF;">🎯 Prediction Result</h3>
                </div>
            """, unsafe_allow_html=True)

            if prediction_result['predicted_delivery_status'] == 'Late':
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #a11d33, #8a172b); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; border: 1px solid rgba(255, 255, 255, 0.2); box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
                        <h4 style="margin: 0; font-weight: 700; display: flex; align-items: center;">
                            <span style="font-size: 1.5rem; margin-right: 0.75rem;">🚨</span>
                            This order is predicted to be LATE.
                        </h4>
                    </div>
                """, unsafe_allow_html=True)
                
                factors = []
                if features_dict.get('is_severe_weather_alert') == 1: factors.append("🌩️ Active weather alert in destination area")
                if features_dict.get('is_supply_chain_news_alert') == 1: factors.append("📰 Recent supply chain disruption news")
                if features_dict.get('customer_churn_risk_score', 0) > 70: factors.append("⚠️ High customer churn risk")
                if not factors: factors.append("🧐 Historical patterns and order characteristics")
                
                factors_html = "".join([f'<div style="display: flex; align-items: center; padding: 0.75rem; margin-bottom: 0.5rem; background: #242A30; border-radius: 8px; border-left: 4px solid #dc3545;"><span style="font-size: 1.5rem; margin-right: 0.75rem;">{f.split(" ")[0]}</span><span style="color: #FFFFFF;">{" ".join(f.split(" ")[1:])}</span></div>' for f in factors])
                st.markdown(f'<div style="background: #1A1F24; padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1);"><h5 style="margin: 0 0 1rem 0; color: #FFFFFF;">🔍 Key Factors Contributing to Delay Risk</h5>{factors_html}</div>', unsafe_allow_html=True)

            else:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #166534, #14532d); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; border: 1px solid rgba(255, 255, 255, 0.2); box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
                        <h4 style="margin: 0; font-weight: 700; display: flex; align-items: center;">
                            <span style="font-size: 1.5rem; margin-right: 0.75rem;">✅</span>
                            This order is predicted to be ON-TIME.
                        </h4>
                    </div>
                """, unsafe_allow_html=True)
                
                factors = []
                if features_dict.get('is_prime_member'): factors.append("✅ Prime member (priority handling)")
                if features_dict.get('carrier') == 'AMZL': factors.append("🚚 Amazon Logistics (direct control)")
                if features_dict.get('is_severe_weather_alert') == 0 and features_dict.get('is_supply_chain_news_alert') == 0: factors.append("🌤️ No active disruptions")
                if not factors: factors.append("👍 Favorable delivery conditions")
                
                factors_html = "".join([f'<div style="display: flex; align-items: center; padding: 0.75rem; margin-bottom: 0.5rem; background: #242A30; border-radius: 8px; border-left: 4px solid #28a745;"><span style="font-size: 1.5rem; margin-right: 0.75rem;">{f.split(" ")[0]}</span><span style="color: #FFFFFF;">{" ".join(f.split(" ")[1:])}</span></div>' for f in factors])
                st.markdown(f'<div style="background: #1A1F24; padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1);"><h5 style="margin: 0 0 1rem 0; color: #FFFFFF;">Factors Supporting On-Time Delivery</h5>{factors_html}</div>', unsafe_allow_html=True)
        else:
            st.error("Could not get prediction from API.")

        # Real-time data context section with dark theme
        st.markdown("""
            <div style="background: #1A1F24; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 2rem; border: 1px solid rgba(255, 255, 255, 0.1);">
                <h4 style="margin: 0 0 1rem 0; color: #FFFFFF;">📈 Real-Time Context</h4>
        """, unsafe_allow_html=True)
        
        # Weather alerts
        if not df_weather_alerts.empty:
            city_weather = df_weather_alerts[df_weather_alerts['city'] == destination_city_in]
            if not city_weather.empty:
                st.markdown("<h5 style='color: #FFFFFF; margin-top: 1rem;'>🌩️ Active Weather Alerts:</h5>", unsafe_allow_html=True)
                for _, alert in city_weather.iterrows():
                    st.info(f"**{alert['event']}** - {alert['severity']} severity until {pd.to_datetime(alert['end']).strftime('%Y-%m-%d %H:%M')}")
            else:
                st.success("✅ No active weather alerts for {}".format(destination_city_in))
        
        # News alerts
        if not df_news_alerts.empty:
            relevant_news = df_news_alerts[df_news_alerts['city'].str.lower() == destination_city_in.lower()]
            if not relevant_news.empty:
                st.markdown("<h5 style='color: #FFFFFF; margin-top: 1rem;'>📰 Recent Relevant News:</h5>", unsafe_allow_html=True)
                for _, news in relevant_news.head(2).iterrows():
                    st.markdown(f"• **{news['title']}** (Source: {news['source']})")
            else:
                st.success("✅ No recent disruption news affecting deliveries")
        
        # Historical performance
        if not df_orders_enhanced.empty:
            city_performance = df_orders_enhanced[df_orders_enhanced['destination_city'] == destination_city_in]
            if not city_performance.empty:
                st.markdown("<h5 style='color: #FFFFFF;  margin-top: 1rem;'>📊 Historical Performance for this City:</h5>", unsafe_allow_html=True)
                col_hist1, col_hist2, col_hist3 = st.columns(3)
                on_time_pct = (city_performance['delivery_status'] == 'On-Time').mean() * 100
                col_hist1.metric("On-Time Rate", f"{on_time_pct:.1f}%")
                avg_days = city_performance['delivery_days_actual'].mean()
                col_hist2.metric("Avg Delivery Days", f"{avg_days:.1f}")
                prime_perf = city_performance[city_performance['is_prime_member'] == is_prime_member_in]
                if not prime_perf.empty:
                    prime_on_time = (prime_perf['delivery_status'] == 'On-Time').mean() * 100
                    col_hist3.metric(f"{'Prime' if is_prime_member_in else 'Standard'} On-Time", f"{prime_on_time:.1f}%")

        st.markdown("</div>", unsafe_allow_html=True)


elif page_selection == "AI Dispatcher Assistant":
    # Enhanced header
    st.markdown("""
        <div style="background: #242A30; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.1);">
            <h2 style="margin: 0 0 0.5rem 0; color: #FFFFFF;">💬 AI Dispatcher Assistant</h2>
            <p style="margin: 0; color: rgba(255, 255, 255, 0.85);">Your intelligent logistics advisor for actionable insights and recommendations</p>
        </div>
    """, unsafe_allow_html=True)

    # Enhanced info card
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1A1F24 0%, #242A30 100%); padding: 1rem; border-radius: 8px; border-left: 4px solid #17a2b8; margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 1.5rem; margin-right: 0.5rem;">🤖</div>
                <div>
                    <strong style="color: #FFFFFF;">AI Assistant Ready!</strong> <span style="color: rgba(255, 255, 255, 0.85);">I'm your intelligent logistics advisor focused on providing actionable insights and recommendations for logistics operations.</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Enhanced filters section
    st.markdown("""
        <div style="background: #242A30; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.1);">
            <h3 style="margin: 0 0 1rem 0; color: #FFFFFF;">🎯 Focus Area</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Date range filter
        if not df_orders_enhanced.empty and 'order_date' in df_orders_enhanced.columns:
            min_date = df_orders_enhanced['order_date'].min()
            max_date = df_orders_enhanced['order_date'].max()
            date_range = st.date_input(
                "📅 Time Period",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
                help="Select the time period for analysis"
            )
        else:
            date_range = None
    
    with col2:
        # City filter
        if not df_orders_enhanced.empty:
            all_cities = ['All Cities'] + sorted(df_orders_enhanced['destination_city'].unique().tolist())
            selected_city = st.selectbox(
                "📍 Location", 
                options=all_cities,
                help="Filter by specific city or view all cities"
            )
        else:
            selected_city = 'All Cities'

    with col3:
        # Analysis mode
        analysis_mode = st.selectbox(
            "🔍 Analysis Mode",
            ["Guided Questions", "Natural Language Query", "Proactive Alerts"],
            help="Choose how you want to interact with the AI assistant"
        )

    # Filter data based on selections
    with st.spinner("🔄 Loading and filtering data..."):
        filtered_df = get_filtered_data(df_orders_enhanced, date_range, selected_city)
        
        # Add debugging information
        if not filtered_df.empty:
            st.info(f"📊 **Data Summary:** {len(filtered_df):,} orders loaded")
            if 'destination_latitude' in filtered_df.columns and 'destination_longitude' in filtered_df.columns:
                st.success("✅ Location data available for routing analysis")
            else:
                st.warning("⚠️ Location data missing from filtered dataset")
        else:
            st.warning("⚠️ No data available after filtering. Please check your date range and city selection.")

    # Helper function to find and display proactive alerts
    def display_proactive_alerts(df, weather_df, news_df, traffic_df, city_selection):
        """Finds and displays proactive alerts from internal and external data sources."""
        
        # Enhanced header with status indicator
        st.markdown("""
            <div style="background: linear-gradient(135deg, #242A30 0%, #1A1F24 100%); padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.1);">
                <h3 style="margin: 0 0 0.5rem 0; color: #FFFFFF; display: flex; align-items: center;">
                    <span style="font-size: 1.5rem; margin-right: 0.75rem;">🚨</span>
                    Proactive Logistics Alerts
                </h3>
                <p style="margin: 0; color: rgba(255, 255, 255, 0.85);">Real-time monitoring of internal performance and external factors affecting delivery operations</p>
            </div>
        """, unsafe_allow_html=True)
        
        if not df.empty:
            # Enhanced metrics dashboard
            st.markdown("""
                <div style="background: #242A30; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.1);">
                    <h4 style="margin: 0 0 1rem 0; color: #FFFFFF; display: flex; align-items: center;">
                        <span style="font-size: 1.2rem; margin-right: 0.5rem;">🔍</span>
                        Key Metrics Under Review
                    </h4>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)

            # --- Calculate all metrics first ---
            on_time_rate = (df['delivery_status'] == 'On-Time').mean() if 'delivery_status' in df.columns else None
            cost_by_carrier = df.groupby('carrier')['delivery_cost_to_amazon'].mean() if 'carrier' in df.columns and 'delivery_cost_to_amazon' in df.columns else None
            max_cost = cost_by_carrier.max() if cost_by_carrier is not None and not cost_by_carrier.empty else 0
            on_time_by_city = df.groupby('destination_city')['delivery_status'].apply(lambda x: (x == 'On-Time').mean()) if 'destination_city' in df.columns and 'delivery_status' in df.columns else None
            min_perf = on_time_by_city.min() if on_time_by_city is not None and not on_time_by_city.empty else 1.0

            # --- Display enhanced metrics with color coding ---
            with col1:
                status_color = "#dc3545" if on_time_rate is not None and on_time_rate < 0.85 else "#28a745"
                status_icon = "🔴" if on_time_rate is not None and on_time_rate < 0.85 else "🟢"
                st.markdown(f"""
                    <div style="background: {status_color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {status_color}; margin-bottom: 1rem;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{status_icon}</span>
                            <strong style="color: #FFFFFF;">On-Time Rate</strong>
                        </div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: {status_color};">
                            {f"{on_time_rate:.1%}" if on_time_rate is not None else "N/A"}
                        </div>
                        <div style="font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">
                            {f"Critical" if on_time_rate is not None and on_time_rate < 0.85 else "Healthy"}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                cost_color = "#ffc107" if max_cost > 12 else "#28a745"
                cost_icon = "🟡" if max_cost > 12 else "🟢"
                st.markdown(f"""
                    <div style="background: {cost_color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {cost_color}; margin-bottom: 1rem;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{cost_icon}</span>
                            <strong style="color: #FFFFFF;">Max Carrier Cost</strong>
                        </div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: {cost_color};">
                            ${f"{max_cost:.2f}" if cost_by_carrier is not None else "N/A"}
                        </div>
                        <div style="font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">
                            {f"High" if max_cost > 12 else "Acceptable"}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                perf_color = "#dc3545" if min_perf < 0.8 else "#28a745"
                perf_icon = "🔴" if min_perf < 0.8 else "🟢"
                st.markdown(f"""
                    <div style="background: {perf_color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {perf_color}; margin-bottom: 1rem;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{perf_icon}</span>
                            <strong style="color: #FFFFFF;">Lowest City Rate</strong>
                        </div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: {perf_color};">
                            {f"{min_perf:.1%}" if on_time_by_city is not None else "N/A"}
                        </div>
                        <div style="font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">
                            {f"Critical" if min_perf < 0.8 else "Healthy"}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Enhanced alerts section
        st.markdown("""
            <div style="background: #242A30; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin: 2rem 0; border: 1px solid rgba(255, 255, 255, 0.1);">
                <h4 style="margin: 0 0 1rem 0; color: #FFFFFF; display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; margin-right: 0.5rem;">🚨</span>
                    Active Alerts & Recommendations
                </h4>
            </div>
        """, unsafe_allow_html=True)
        
        alerts_found = False
        alert_cards = []
        
        # --- Trigger Internal Alerts ---
        if not df.empty:
            if on_time_rate is not None and on_time_rate < 0.85:
                alert_cards.append({
                    'type': 'error',
                    'icon': '🔴',
                    'title': 'Performance Alert',
                    'message': f'Overall on-time delivery rate is critically low at {on_time_rate:.1%}',
                    'recommendation': 'Immediate action required: Review carrier performance, optimize routes, and consider capacity adjustments.'
                })
                alerts_found = True
            
            if max_cost > 12:
                high_cost_carrier = cost_by_carrier.idxmax()
                alert_cards.append({
                    'type': 'warning',
                    'icon': '🟡',
                    'title': 'Cost Alert',
                    'message': f'Carrier {high_cost_carrier} has a high average cost of ${max_cost:.2f} per delivery',
                    'recommendation': 'Consider renegotiating rates or shifting volume to more cost-effective carriers.'
                })
                alerts_found = True
            
            if city_selection == 'All Cities' and min_perf < 0.8:
                low_perf_city = on_time_by_city.idxmin()
                alert_cards.append({
                    'type': 'error',
                    'icon': '🔴',
                    'title': 'City Performance Alert',
                    'message': f'Delivery performance in {low_perf_city} is critically low at {min_perf:.1%}',
                    'recommendation': 'Investigate local infrastructure, carrier coverage, and consider regional optimization.'
                })
                alerts_found = True

        # --- Trigger External Alerts ---
        # Weather Alerts
        if not weather_df.empty:
            active_weather = weather_df[(pd.to_datetime(weather_df['start']) <= datetime.now()) & (pd.to_datetime(weather_df['end']) >= datetime.now()) & (weather_df['severity_score'] >= 3)]
            if city_selection != 'All Cities':
                active_weather = active_weather[active_weather['city'] == city_selection]
            if not active_weather.empty:
                alerts_found = True
                for _, alert in active_weather.iterrows():
                    severity_icon = "🔴" if alert['severity_score'] >= 4 else "🟡"
                    alert_cards.append({
                        'type': 'error' if alert['severity_score'] >= 4 else 'warning',
                        'icon': severity_icon,
                        'title': f'Weather Alert - {alert["city"]}',
                        'message': f'Active {alert["event"]} with {alert["severity"]} severity',
                        'recommendation': 'Consider delaying non-urgent deliveries and communicate with customers about potential delays.'
                    })

        # News Alerts
        if not news_df.empty:
            recent_news = news_df[(pd.to_datetime(news_df['publishedAt']) >= datetime.now() - timedelta(days=2)) & (news_df['score'] >= 4)]
            if city_selection != 'All Cities':
                recent_news = recent_news[recent_news['city'].str.lower() == city_selection.lower()]
            if not recent_news.empty:
                alerts_found = True
                alert_cards.append({
                    'type': 'warning',
                    'icon': '📰',
                    'title': 'Supply Chain News Alert',
                    'message': f'Recent high-impact news may affect logistics operations',
                    'recommendation': 'Monitor supply chain disruptions and adjust delivery expectations accordingly.'
                })
                for _, alert in recent_news.head(2).iterrows():
                    alert_cards.append({
                        'type': 'info',
                        'icon': '📋',
                        'title': f'News Update - {alert["city"]}',
                        'message': alert['title'],
                        'recommendation': 'Review impact on local operations and customer communications.'
                    })

        # Traffic Alerts
        if not traffic_df.empty:
            active_traffic = traffic_df[traffic_df['severity'] >= 4]
            if city_selection != 'All Cities':
                active_traffic = active_traffic[active_traffic['city'] == city_selection]
            if not active_traffic.empty:
                alerts_found = True
                alert_cards.append({
                    'type': 'warning',
                    'icon': '🚗',
                    'title': 'Traffic Alert',
                    'message': f'Significant traffic incidents reported affecting {len(active_traffic)} routes',
                    'recommendation': 'Consider alternative routes and adjust delivery time estimates.'
                })
                for _, alert in active_traffic.head(2).iterrows():
                    alert_cards.append({
                        'type': 'info',
                        'icon': '🛣️',
                        'title': f'Traffic Incident - {alert["city"]}',
                        'message': alert['description'],
                        'recommendation': 'Route optimization recommended for affected areas.'
                    })
        
        # Display all alert cards
        if alert_cards:
            for i, alert in enumerate(alert_cards):
                color_map = {
                    'error': '#dc3545',
                    'warning': '#ffc107', 
                    'info': '#17a2b8',
                    'success': '#28a745'
                }
                bg_color = color_map.get(alert['type'], '#17a2b8')
                
                st.markdown(f"""
                    <div style="background: {bg_color}15; padding: 1.5rem; border-radius: 12px; border-left: 6px solid {bg_color}; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <div style="display: flex; align-items: flex-start; margin-bottom: 1rem;">
                            <span style="font-size: 1.5rem; margin-right: 0.75rem; margin-top: 0.25rem;">{alert['icon']}</span>
                            <div style="flex: 1;">
                                <h5 style="margin: 0 0 0.5rem 0; color: {bg_color}; font-weight: 600;">{alert['title']}</h5>
                                <p style="margin: 0; color: #FFFFFF; font-size: 1rem;">{alert['message']}</p>
                            </div>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 0.75rem; border-radius: 6px; border-left: 3px solid {bg_color};">
                            <strong style="color: #FFFFFF; font-size: 0.9rem;">💡 Recommendation:</strong>
                            <p style="margin: 0.25rem 0 0 0; color: rgba(255, 255, 255, 0.9); font-size: 0.9rem;">{alert['recommendation']}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background: #28a74520; padding: 2rem; border-radius: 12px; border: 2px solid #28a745; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">✅</div>
                    <h4 style="margin: 0 0 0.5rem 0; color: #28a745;">All Systems Operational</h4>
                    <p style="margin: 0; color: rgba(255, 255, 255, 0.9);">No critical alerts detected from internal or external data sources.</p>
                </div>
            """, unsafe_allow_html=True)

    # Helper function to safely check column existence
    def safe_column_check(df, required_columns):
        missing_columns = [col for col in required_columns if col not in df.columns]
        return len(missing_columns) == 0, missing_columns

    # Guided Questions Mode
    if analysis_mode == "Guided Questions":
        # Enhanced question selection with better categorization
        question_options = [
            "How is our delivery performance trending?",
            "What are the current operational bottlenecks?", 
            "Which carriers are performing best/worst?",
            "What are the cost optimization opportunities?",
            "How can we improve customer satisfaction?",
            "What are the risk factors for delays?",
            "🗺️ How can we optimize routing and FC distances?"
        ]

        selected_question = st.selectbox("What would you like to know?", options=question_options)
        
        if selected_question != "Select a question...":
            st.subheader("🤖 AI Assistant Response:")
            
            # Process the selected question with loading indicator
            with st.spinner("🤖 Analyzing data and generating insights..."):
                # Process the selected question
                if selected_question == "How is our delivery performance trending?":
                    if not filtered_df.empty:
                        st.write("📈 **Delivery Performance Trend Analysis**")
                        
                        # Use cached trend metrics
                        trend_metrics = calculate_trend_metrics(filtered_df)
                        
                        if trend_metrics:
                            # Performance trend analysis
                            if 'performance' in trend_metrics:
                                perf = trend_metrics['performance']
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Current On-Time Rate", f"{perf['current_on_time']:.1f}%")
                                with col2:
                                    st.metric("Historical Average", f"{perf['historical_on_time']:.1f}%")
                                with col3:
                                    st.metric("Trend Direction", perf['trend_direction'].title())
                                with col4:
                                    st.metric("Trend Magnitude", f"{perf['trend_magnitude']:.1f}%")
                            
                            # Volume trend analysis
                            if 'volume' in trend_metrics:
                                vol = trend_metrics['volume']
                                st.write("\n**📦 Volume Trend Analysis:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Recent Daily Orders", f"{vol['recent_daily_orders']:.1f}")
                                with col2:
                                    st.metric("Historical Avg Daily Orders", f"{vol['avg_daily_orders']:.1f}")
                                with col3:
                                    st.metric("Volume Trend", f"{vol['volume_trend']:.1f}%")
                        else:
                            st.warning("Not enough data to analyze trends.")
                    else:
                        st.warning("No data available to analyze performance trends.")

                elif selected_question == "What are the current operational bottlenecks?":
                    if not filtered_df.empty:
                        st.write("🔍 **Operational Bottleneck Analysis**")
                        
                        # Identify potential bottlenecks
                        bottlenecks = []
                        
                        # Late deliveries by carrier
                        if 'carrier' in filtered_df.columns:
                            late_by_carrier = filtered_df[filtered_df['delivery_status'] == 'Late']['carrier'].value_counts()
                            if not late_by_carrier.empty:
                                worst_carrier = late_by_carrier.index[0]
                                bottlenecks.append(f"Carrier **{worst_carrier}** has the highest number of late deliveries.")
                        
                        # Low on-time rates by city
                        if 'destination_city' in filtered_df.columns:
                            on_time_by_city = filtered_df.groupby('destination_city')['delivery_status'].apply(lambda x: (x == 'On-Time').mean()).sort_values()
                            if not on_time_by_city.empty and on_time_by_city.iloc[0] < 0.8:
                                worst_city = on_time_by_city.index[0]
                                bottlenecks.append(f"City **{worst_city}** has a low on-time rate of {on_time_by_city.iloc[0]:.1%}.")
                        
                        if bottlenecks:
                            st.markdown("\n".join([f"- {b}" for b in bottlenecks]))
                        else:
                            st.success("✅ No significant bottlenecks identified in the current data.")
                    else:
                        st.warning("No data available to identify bottlenecks.")

                elif selected_question == "Which carriers are performing best/worst?":
                    if not filtered_df.empty and 'carrier' in filtered_df.columns:
                        st.write("🚚 **Carrier Performance Analysis**")
                        
                        carrier_metrics = calculate_carrier_metrics(filtered_df)
                        
                        if carrier_metrics is not None:
                            st.dataframe(carrier_metrics)
                            st.markdown(f"**Best performing carrier (by efficiency):** {carrier_metrics.index[0]}")
                            st.markdown(f"**Worst performing carrier (by efficiency):** {carrier_metrics.index[-1]}")
                        else:
                            st.warning("Could not calculate carrier metrics.")
                    else:
                        st.warning("No carrier data available.")

                elif selected_question == "What are the cost optimization opportunities?":
                    if not filtered_df.empty:
                        st.write("💰 **Cost Optimization Analysis**")
                        
                        # Enhanced cost analysis with more detailed insights
                        st.markdown("""
                            <div style="background: #242A30; padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.1);">
                                <h4 style="color: #FFFFFF; margin-bottom: 0.5rem;">💡 For comprehensive cost analysis, visit the dedicated <strong>Cost Efficiency Analysis</strong> page!</h4>
                                <p style="color: rgba(255, 255, 255, 0.85); margin: 0;">Get detailed breakdowns, optimization opportunities, and actionable recommendations.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Identify cost optimization opportunities
                        opportunities = []
                        
                        # High cost carriers
                        if 'carrier' in filtered_df.columns and 'delivery_cost_to_amazon' in filtered_df.columns:
                            cost_by_carrier = filtered_df.groupby('carrier')['delivery_cost_to_amazon'].mean().sort_values(ascending=False)
                            if not cost_by_carrier.empty and cost_by_carrier.iloc[0] > 10:
                                expensive_carrier = cost_by_carrier.index[0]
                                cheap_carrier = cost_by_carrier.index[-1]
                                cost_diff = cost_by_carrier.iloc[0] - cost_by_carrier.iloc[-1]
                                potential_savings = cost_diff * len(filtered_df[filtered_df['carrier'] == expensive_carrier])
                                opportunities.append(f"🚚 **Carrier Optimization:** {expensive_carrier} costs ${cost_diff:.2f} more per order than {cheap_carrier}. Potential savings: ${potential_savings:,.2f}")
                        
                        # High cost cities
                        if 'destination_city' in filtered_df.columns and 'delivery_cost_to_amazon' in filtered_df.columns:
                            cost_by_city = filtered_df.groupby('destination_city')['delivery_cost_to_amazon'].mean().sort_values(ascending=False)
                            if not cost_by_city.empty and cost_by_city.iloc[0] > 8:
                                high_cost_city = cost_by_city.index[0]
                                opportunities.append(f"🏙️ **Geographic Optimization:** {high_cost_city} has high delivery costs (${cost_by_city.iloc[0]:.2f} per order)")
                        
                        # Service type optimization
                        if 'is_prime_member' in filtered_df.columns:
                            prime_costs = filtered_df[filtered_df['is_prime_member']]['delivery_cost_to_amazon'].mean()
                            standard_costs = filtered_df[~filtered_df['is_prime_member']]['delivery_cost_to_amazon'].mean()
                            if abs(prime_costs - standard_costs) > 1.0:
                                if prime_costs > standard_costs:
                                    opportunities.append(f"⚡ **Service Optimization:** Prime delivery costs ${prime_costs - standard_costs:.2f} more than standard delivery")
                                else:
                                    opportunities.append(f"📈 **Service Expansion:** Prime delivery is ${standard_costs - prime_costs:.2f} cheaper - consider expanding Prime service")
                        
                        # Cost trends
                        if 'order_date' in filtered_df.columns:
                            daily_costs = filtered_df.groupby(filtered_df['order_date'].dt.date)['delivery_cost_to_amazon'].mean()
                            if len(daily_costs) > 7:
                                recent_trend = daily_costs.tail(7).mean() - daily_costs.iloc[:-7].mean()
                                if recent_trend > 0.5:
                                    opportunities.append(f"📈 **Cost Trend Alert:** Recent costs are ${recent_trend:.2f} higher than historical average")
                        
                        if opportunities:
                            st.markdown("**🎯 Key Optimization Opportunities:**")
                            for opportunity in opportunities:
                                st.markdown(f"""
                                <div style="background: #28a74520; padding: 0.75rem; border-radius: 6px; margin: 0.5rem 0; border-left: 4px solid #28a745;">
                                    <span style="color: rgba(255, 255, 255, 0.9);">{opportunity}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.success("✅ No major cost optimization opportunities identified in the current data.")
                        
                        # Quick metrics summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_cost = filtered_df['delivery_cost_to_amazon'].mean()
                            st.metric("Average Cost", f"${avg_cost:.2f}")
                        with col2:
                            total_cost = filtered_df['delivery_cost_to_amazon'].sum()
                            st.metric("Total Cost", f"${total_cost:,.2f}")
                        with col3:
                            high_cost_pct = (len(filtered_df[filtered_df['delivery_cost_to_amazon'] > 10]) / len(filtered_df) * 100)
                            st.metric("High-Cost Orders", f"{high_cost_pct:.1f}%")
                    else:
                        st.warning("No data available for cost optimization analysis.")

                elif selected_question == "How can we improve customer satisfaction?":
                    st.write("😊 **Customer Satisfaction Improvement Plan**")
                    st.markdown("""
                        - **Proactively communicate delays:** Inform customers about potential delays before they happen.
                        - **Offer goodwill gestures:** For significant delays, consider offering a discount or credit on a future purchase.
                        - **Prioritize high-value customers:** Ensure high CLV customers receive the best possible service.
                        - **Analyze and address root causes of delays:** Use the dashboard to identify and fix underlying issues.
                    """)
                
                elif selected_question == "What are the risk factors for delays?":
                    if not filtered_df.empty:
                        st.write("🚨 **Delay Risk Factor Analysis**")
                        
                        # Analyze risk factors from predictions
                        if 'predicted_delivery_status_class' in filtered_df.columns:
                            late_orders = filtered_df[filtered_df['predicted_delivery_status_class'] == 1]
                            if not late_orders.empty:
                                risk_factors = []
                                # Example: Check for common characteristics of late orders
                                if 'carrier' in late_orders.columns and late_orders['carrier'].nunique() > 1:
                                    late_carrier_dist = late_orders['carrier'].value_counts(normalize=True)
                                    if late_carrier_dist.iloc[0] > 0.5:
                                        risk_factors.append(f"Carrier **{late_carrier_dist.index[0]}** is associated with a high proportion of predicted delays.")
                                
                                if risk_factors:
                                    st.markdown("\n".join([f"- {rf}" for rf in risk_factors]))
                                else:
                                    st.info("No single dominant risk factor identified from the available data.")
                            else:
                                st.success("✅ No orders predicted to be late in the current dataset.")
                        else:
                            st.warning("Prediction data not available to analyze risk factors.")
                    else:
                        st.warning("No data available to analyze risk factors.")

                elif selected_question == "🗺️ How can we optimize routing and FC distances?":
                    st.write("🗺️ **Routing and Fulfillment Center Optimization**")
                    if 'destination_latitude' in filtered_df.columns and 'destination_longitude' in filtered_df.columns:
                        # Calculate distance to nearest FC for a sample of orders
                        sample_df = filtered_df.sample(min(10, len(filtered_df)))
                        distances = []
                        for _, row in sample_df.iterrows():
                            _, dist = find_nearest_fc(row['destination_latitude'], row['destination_longitude'])
                            distances.append(dist)
                        
                        avg_dist = np.mean(distances)
                        st.metric("Average Distance to Nearest FC", f"{avg_dist:.1f} km")
                        
                        if avg_dist > 200:
                            st.warning("Average distance to FC is high. Consider adding a new fulfillment center in a key demand area.")
                        else:
                            st.success("✅ Fulfillment centers seem well-positioned relative to customer locations.")
                    else:
                        st.warning("Location data not available for routing analysis.")

    # Natural Language Query Mode
    elif analysis_mode == "Natural Language Query":
        user_query = st.text_input("Ask a question about your logistics data:", "e.g., 'show me late deliveries in New York this week'")

        st.markdown("""
        <div style="background: #242A30; padding: 1.2rem; border-radius: 8px; margin-top: 1rem; border: 1px solid rgba(255, 255, 255, 0.1);">
            <h5 style="color: #FFFFFF; margin-bottom: 0.75rem;">📋 Example Questions:</h5>
            <ul style="color: rgba(255, 255, 255, 0.85); font-size: 0.95rem; margin: 0; padding-left: 1.2rem; list-style-type: '➤ '; line-height: 1.8;">
                <li>What was the delivery performance in Los Angeles last week?</li>
                <li>Which carriers are the most expensive this month?</li>
                <li>Show late deliveries in Chicago yesterday.</li>
                <li>Analyze carrier costs and performance across all cities.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if user_query and user_query != "e.g., 'show me late deliveries in New York this week'":
            st.subheader("🤖 AI Assistant Response:")
            with st.spinner("Processing your query..."):
                query_lower = user_query.lower()

                # 1. Apply filters from query
                data, time_message = get_time_filtered_data(filtered_df, query_lower)
                data, city, loc_message = get_location_filtered_data(data, query_lower)

                st.info(f"🔍 **Analysis Scope:** {time_message}. {loc_message}.")

                # 2. Determine the type of analysis requested
                analysis_found = False

                if any(k in query_lower for k in ["late", "delay"]):
                    analysis_found = True
                    st.write("📉 **Analysis of Late Deliveries**")
                    late_df = data[data['delivery_status'] == 'Late']
                    if not late_df.empty:
                        st.metric("Number of Late Deliveries", len(late_df))
                        st.dataframe(late_df[['order_id', 'carrier', 'delivery_status', 'delivery_days_actual']].head())
                    else:
                        st.success("✅ No late deliveries found for the selected criteria.")

                if "performance" in query_lower:
                    analysis_found = True
                    st.write("📊 **Delivery Performance**")
                    on_time_rate = (data['delivery_status'] == 'On-Time').mean() * 100
                    st.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%")
                    if 'delivery_days_actual' in data.columns:
                        avg_delivery_days = data['delivery_days_actual'].mean()
                        st.metric("Average Delivery Days", f"{avg_delivery_days:.1f} days")

                if "cost" in query_lower:
                    analysis_found = True
                    st.write("💰 **Cost Analysis**")
                    if 'delivery_cost_to_amazon' in data.columns:
                        # Enhanced cost analysis with more detailed breakdown
                        avg_cost = data['delivery_cost_to_amazon'].mean()
                        total_cost = data['delivery_cost_to_amazon'].sum()
                        min_cost = data['delivery_cost_to_amazon'].min()
                        max_cost = data['delivery_cost_to_amazon'].max()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Average Cost", f"${avg_cost:.2f}")
                        with col2:
                            st.metric("Total Cost", f"${total_cost:,.2f}")
                        with col3:
                            st.metric("Min Cost", f"${min_cost:.2f}")
                        with col4:
                            st.metric("Max Cost", f"${max_cost:.2f}")
                        
                        # Cost distribution analysis
                        if 'carrier' in data.columns:
                            st.markdown("**📊 Cost by Carrier**")
                            carrier_costs = data.groupby('carrier')['delivery_cost_to_amazon'].agg(['mean', 'count']).round(2)
                            carrier_costs.columns = ['Avg Cost', 'Order Count']
                            carrier_costs = carrier_costs.sort_values('Avg Cost', ascending=False)
                            st.dataframe(carrier_costs, use_container_width=True)
                        
                        # Cost efficiency insights
                        if 'delivery_status' in data.columns:
                            st.markdown("**💡 Cost Efficiency Insights**")
                            on_time_costs = data[data['delivery_status'] == 'On-Time']['delivery_cost_to_amazon'].mean()
                            late_costs = data[data['delivery_status'] == 'Late']['delivery_cost_to_amazon'].mean()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("On-Time Avg Cost", f"${on_time_costs:.2f}")
                            with col2:
                                st.metric("Late Avg Cost", f"${late_costs:.2f}")
                            
                            if late_costs > on_time_costs:
                                cost_penalty = late_costs - on_time_costs
                                st.warning(f"⚠️ Late deliveries cost ${cost_penalty:.2f} more on average than on-time deliveries")
                            else:
                                st.success("✅ On-time deliveries are cost-effective")
                    else:
                        st.warning("Cost data not available.")

                if "carrier" in query_lower:
                    analysis_found = True
                    st.write("🚚 **Carrier Analysis**")
                    carrier_metrics = calculate_carrier_metrics(data)
                    if carrier_metrics is not None and not carrier_metrics.empty:
                        st.dataframe(carrier_metrics)
                        st.markdown(f"**Best performing carrier (by efficiency):** {carrier_metrics.index[0]}")
                        st.markdown(f"**Worst performing carrier (by efficiency):** {carrier_metrics.index[-1]}")
                    else:
                        st.warning("Could not analyze carrier performance for the selected criteria.")

                if not analysis_found:
                    st.write("I'm sorry, I couldn't determine the specific analysis from your query. Here is a general overview of the data based on your filters:")
                    st.dataframe(data.head())

    # Proactive Alerts Mode
    elif analysis_mode == "Proactive Alerts":
        display_proactive_alerts(filtered_df, df_weather_alerts, df_news_alerts, df_traffic_alerts, selected_city)
        
else:
    st.info("Welcome to the Amazon Logistics Intelligence Dashboard. Please select a section from the sidebar to begin.")