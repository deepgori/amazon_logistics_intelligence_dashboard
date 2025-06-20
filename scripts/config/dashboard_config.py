# config/dashboard_config.py

# Dashboard Configuration
DASHBOARD_TITLE = "Amazon Logistics Intelligence Dashboard"
DASHBOARD_DESCRIPTION = "Real-time insights and predictive analytics for optimal logistics operations"

# Color Schemes
COLORS = {
    'primary': '#FF9900',  # Amazon Orange
    'secondary': '#232F3E',  # Amazon Dark Blue
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'white': '#FFFFFF',
    'black': '#000000',
    'gray': '#6c757d',
    'gray_light': '#e9ecef',
    'gray_dark': '#495057'
}

# Route Efficiency Colors
ROUTE_COLORS = {
    'High': '#00D4AA',
    'Medium': '#FFB84D', 
    'Low': '#FF6B6B'
}

# Theme Colors
THEME_COLORS = {
    'bg_primary': '#1A1F24',
    'bg_secondary': '#242A30',
    'text_primary': '#FFFFFF',
    'text_secondary': 'rgba(255, 255, 255, 0.85)',
    'accent_primary': '#FF9900',
    'accent_secondary': '#FFB84D',
    'border_color': 'rgba(255, 255, 255, 0.1)'
}

# Page Configuration
PAGES = {
    'prime_performance': {
        'title': '📊 Prime Performance',
        'description': 'Comprehensive analysis of delivery metrics, carrier strategy, and geographical performance'
    },
    'last_mile_operations': {
        'title': '🚚 Last-Mile Operations',
        'description': 'Real-time insights on route optimization, vehicle efficiency, and delivery performance'
    },
    'cost_efficiency_analysis': {
        'title': '💰 Cost Efficiency Analysis',
        'description': 'Comprehensive cost analysis, optimization insights, and actionable recommendations for logistics cost reduction'
    },
    'amazon_purchase_trends': {
        'title': '📈 Purchase Trends',
        'description': 'Analysis of product sales, categories, and spending from research data'
    },
    'ml_prediction_demo': {
        'title': '🤖 ML Prediction Demo',
        'description': 'Enter order features to get real-time delay predictions from our ML model'
    },
    'ai_dispatcher_assistant': {
        'title': '💬 AI Assistant',
        'description': 'Your intelligent logistics advisor for actionable insights and recommendations'
    }
}

# API Configuration
API_CONFIG = {
    'prediction_url': 'http://localhost:8000/predict_delay',
    'health_url': 'http://localhost:8000/health',
    'google_routes_url': 'https://routes.googleapis.com/directions/v2:computeRoutes',
    'timeout': 10
}

# Data Configuration
DATA_CONFIG = {
    'cache_ttl': 3600,  # 1 hour cache for route data
    'max_sample_size': 1000,  # Maximum sample size for analysis
    'default_date_range_days': 30
}

# Alert Thresholds
ALERT_THRESHOLDS = {
    'on_time_rate_critical': 0.85,  # Below 85% is critical
    'on_time_rate_warning': 0.90,   # Below 90% is warning
    'cost_threshold_high': 12.0,    # Above $12 is high cost
    'cost_threshold_medium': 8.0,   # Above $8 is medium cost
    'weather_severity_critical': 4, # Severity 4+ is critical
    'weather_severity_warning': 3,  # Severity 3+ is warning
    'traffic_severity_critical': 4, # Traffic severity 4+ is critical
    'news_score_critical': 4        # News score 4+ is critical
}

# Customer Value Tiers
CUSTOMER_TIERS = {
    'High': {
        'color': '#28a745',
        'priority': 1
    },
    'Medium': {
        'color': '#ffc107', 
        'priority': 2
    },
    'Low': {
        'color': '#dc3545',
        'priority': 3
    }
}

# Carrier Configuration
CARRIERS = {
    'AMZL': {
        'name': 'Amazon Logistics',
        'color': '#FF9900',
        'priority': 1
    },
    'UPS': {
        'name': 'United Parcel Service',
        'color': '#351C15',
        'priority': 2
    },
    'FEDEX': {
        'name': 'Federal Express',
        'color': '#4D148C',
        'priority': 2
    },
    'USPS': {
        'name': 'United States Postal Service',
        'color': '#004B87',
        'priority': 3
    }
}

# Map Configuration
MAP_CONFIG = {
    'default_zoom': 4,
    'default_center': {'lat': 39.8283, 'lon': -98.5795},  # Center of USA
    'max_points': 1000  # Maximum points to display on map
}

# Chart Configuration
CHART_CONFIG = {
    'default_height': 300,
    'default_width': 'container',
    'max_bins': 15,
    'animation_duration': 300
}

# Form Configuration
FORM_CONFIG = {
    'max_quantity': 10,
    'min_cost': 1.0,
    'max_cost': 20.0,
    'min_churn_risk': 0,
    'max_churn_risk': 100,
    'min_return_rate': 0.0,
    'max_return_rate': 1.0
} 