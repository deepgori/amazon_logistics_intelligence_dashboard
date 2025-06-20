# utils/data_processing.py

import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Tuple, Dict, Any

@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and cache data from CSV files with proper type conversion.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with properly typed columns
    """
    try:
        df = pd.read_csv(filepath)
        
        # Basic type conversion for common columns
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        if 'delivery_datetime' in df.columns: 
            df['delivery_datetime'] = pd.to_datetime(df['delivery_datetime'], errors='coerce')
        if 'pickup_datetime' in df.columns:
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
        
        # Ensure prediction columns are correctly typed if present
        if 'predicted_delivery_status_class' in df.columns:
            df['predicted_delivery_status_class'] = df['predicted_delivery_status_class'].astype(int)
        if 'predicted_delay_probability' in df.columns:
            df['predicted_delay_probability'] = df['predicted_delay_probability'].astype(float)

        return df
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

@st.cache_data
def get_filtered_data(df: pd.DataFrame, date_range: Optional[Tuple] = None, 
                     selected_city: Optional[str] = None) -> pd.DataFrame:
    """
    Filter data based on date range and city selection.
    
    Args:
        df: Input DataFrame
        date_range: Tuple of (start_date, end_date)
        selected_city: City to filter by
        
    Returns:
        Filtered DataFrame
    """
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
def calculate_trend_metrics(filtered_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Calculate trend metrics for performance analysis.
    
    Args:
        filtered_df: Filtered DataFrame
        
    Returns:
        Dictionary containing trend metrics or None if insufficient data
    """
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
def calculate_carrier_metrics(filtered_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Calculate carrier performance metrics.
    
    Args:
        filtered_df: Filtered DataFrame
        
    Returns:
        DataFrame with carrier metrics or None if no carrier data
    """
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

def get_time_filtered_data(df: pd.DataFrame, query_lower: str) -> Tuple[pd.DataFrame, str]:
    """
    Filter data based on time-related terms in a query.
    
    Args:
        df: Input DataFrame
        query_lower: Lowercase query string
        
    Returns:
        Tuple of (filtered_dataframe, message)
    """
    try:
        if 'order_date' not in df.columns:
            return df, "Warning: No date information available for filtering."
        
        message = ""
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
        
        if len(filtered) == 0:
            return df, "No data found for the specified time period. Showing all available data instead."
        
        return filtered, message
    except Exception as e:
        return df, f"Error in time filtering: {str(e)}. Showing all available data."

def get_location_filtered_data(df: pd.DataFrame, query_lower: str) -> Tuple[pd.DataFrame, Optional[str], str]:
    """
    Filter data based on location terms in a query.
    
    Args:
        df: Input DataFrame
        query_lower: Lowercase query string
        
    Returns:
        Tuple of (filtered_dataframe, city, message)
    """
    try:
        if 'destination_city' not in df.columns:
            return df, None, "Warning: No city information available for filtering."
        
        cities = df['destination_city'].unique()
        for city in cities:
            if city.lower() in query_lower:
                filtered = df[df['destination_city'] == city]
                if len(filtered) > 0:
                    return filtered, city, f"Filtered for {city}"
        return df, None, "No specific city mentioned or found in data"
    except Exception as e:
        return df, None, f"Error in location filtering: {str(e)}"

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return quality metrics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing quality metrics
    """
    quality_metrics = {
        'total_rows': len(df),
        'missing_values': {},
        'duplicates': len(df.duplicated()),
        'data_types': {},
        'quality_score': 0.0
    }
    
    if df.empty:
        return quality_metrics
    
    # Check missing values
    missing_counts = df.isnull().sum()
    quality_metrics['missing_values'] = missing_counts[missing_counts > 0].to_dict()
    
    # Check data types
    quality_metrics['data_types'] = df.dtypes.to_dict()
    
    # Calculate quality score
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    duplicate_penalty = quality_metrics['duplicates'] * 0.1
    
    quality_score = max(0, (total_cells - missing_cells - duplicate_penalty) / total_cells * 100)
    quality_metrics['quality_score'] = round(quality_score, 2)
    
    return quality_metrics 