# scripts/api_integrator.py

import requests
import os
import json
import pandas as pd
from datetime import datetime, timedelta

from logger import get_logger

logger = get_logger(__name__)

# --- Configuration (Pulled from environment variables, or uses placeholder for cost control) ---
# API Keys - Use environment variables for security
OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', '')
NEWSAPI_API_KEY = os.getenv('NEWSAPI_API_KEY', '')
MAPS_API_KEY = os.getenv('MAPS_API_KEY', '')  # Google Maps API key

# List of major US cities for API calls (from data_generator.py for consistency)
US_CITIES_FOR_API_CALLS = {
    'New York': {'state': 'NY', 'lat': 40.7128, 'lon': -74.0060},
    'Chicago': {'state': 'IL', 'lat': 41.8781, 'lon': -87.6298},
    'Los Angeles': {'state': 'CA', 'lat': 34.0522, 'lon': -118.2437},
    'Houston': {'state': 'TX', 'lat': 29.7604, 'lon': -95.3698},
    'Atlanta': {'state': 'GA', 'lat': 33.7488, 'lon': -84.3877} # Limit to a few for free tier control
}
WEATHER_ALERTS_CSV = os.path.join('data', 'weather_alerts.csv') 
NEWS_ALERTS_CSV = os.path.join('data', 'news_alerts.csv') 
TRAFFIC_DATA_CSV = os.path.join('data', 'traffic_data.csv') # <-- NEW OUTPUT FILE


def fetch_weather_alerts():
    logger.info("Fetching real-time weather alerts via OpenWeatherMap API...")
    
    # Check if the environment variable is actually set
    if not os.getenv('OPENWEATHERMAP_API_KEY'):
        logger.warning("OPENWEATHERMAP_API_KEY environment variable not set. Using simulated weather alerts.")
        simulated_alerts = [
            {'city': 'Chicago', 'state': 'IL', 'event': 'Blizzard Warning', 'severity': 'Extreme', 'description': 'Simulated: Heavy snowfall and high winds expected.', 'start': (datetime.now() - timedelta(hours=1)).isoformat(), 'end': (datetime.now() + timedelta(hours=24)).isoformat(), 'source_api': 'Simulated'},
            {'city': 'Los Angeles', 'state': 'CA', 'event': 'Flash Flood Watch', 'severity': 'Moderate', 'description': 'Simulated: Heavy rain, potential for urban flooding.', 'start': (datetime.now() - timedelta(hours=2)).isoformat(), 'end': (datetime.now() + timedelta(hours=10)).isoformat(), 'source_api': 'Simulated'},
        ]
        df_alerts = pd.DataFrame(simulated_alerts)
    else:
        all_fetched_alerts = []
        for city_name, coords in US_CITIES_FOR_API_CALLS.items():
            url = f"https://api.openweathermap.org/data/2.5/onecall?lat={coords['lat']}&lon={coords['lon']}&exclude=current,minutely,hourly,daily&appid={OPENWEATHERMAP_API_KEY}&units=imperial"
            
            try:
                response = requests.get(url, timeout=5) 
                response.raise_for_status() 
                data = response.json()
                
                alerts = data.get('alerts', []) 
                
                if alerts:
                    for alert in alerts:
                        all_fetched_alerts.append({
                            'city': city_name,
                            'state': coords['state'],
                            'event': alert.get('event', 'N/A'),
                            'severity': alert.get('severity', 'N/A'),
                            'description': alert.get('description', 'N/A'),
                            'start': datetime.fromtimestamp(alert.get('start', 0)).isoformat(),
                            'end': datetime.fromtimestamp(alert.get('end', 0)).isoformat(),
                            'source_api': 'OpenWeatherMap'
                        })
                else:
                    logger.info(f"No active alerts for {city_name} via OpenWeatherMap API.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching OpenWeatherMap alerts for {city_name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing OpenWeatherMap for {city_name}: {e}")
        
        if all_fetched_alerts:
            df_alerts = pd.DataFrame(all_fetched_alerts)
            logger.info(f"Fetched {len(df_alerts)} real weather alerts.")
        else:
            logger.warning("No real weather alerts found via API. Returning empty DataFrame for alerts.")
            df_alerts = pd.DataFrame(columns=['city', 'state', 'event', 'severity', 'description', 'start', 'end', 'source_api']) 

    os.makedirs(os.path.dirname(WEATHER_ALERTS_CSV), exist_ok=True)
    df_alerts.to_csv(WEATHER_ALERTS_CSV, index=False)
    logger.info(f"Weather alerts saved to '{WEATHER_ALERTS_CSV}'")
    
    return df_alerts

def fetch_news_alerts():
    logger.info("Fetching real-time news alerts via NewsAPI.org...")
    
    # Check if the environment variable is actually set
    if not os.getenv('NEWSAPI_API_KEY'):
        logger.warning("NEWSAPI_API_KEY environment variable not set. Using simulated news alerts.")
        simulated_news = [
            {
                'title': 'Simulated: Major Highway Accident near Dallas, TX',
                'source': 'Traffic News (Simulated)',
                'publishedAt': datetime.now().isoformat(),
                'url': 'http://simulated.news/dallas-accident',
                'source_api': 'Simulated',
                'city': 'Dallas',
                'score': 1
            },
            {
                'title': 'Simulated: Warehouse Fire Reported in Los Angeles Facility',
                'source': 'Local News (Simulated)',
                'publishedAt': (datetime.now() - timedelta(hours=2)).isoformat(),
                'url': 'http://simulated.news/la-fire',
                'source_api': 'Simulated',
                'city': 'Los Angeles',
                'score': 1
            }
        ]
        df_news = pd.DataFrame(simulated_news)
        logger.info(f"Generated {len(simulated_news)} simulated news alerts.")
    else:
        # --- Actual NewsAPI.org Call ---
        query = "delivery delay OR logistics accident OR warehouse fire OR port closure OR transport disruption OR truck crash OR road closure OR supply chain strike OR cargo delay OR freight accident OR package theft OR riot affecting delivery OR highway accident OR major road closure OR bridge collapse OR delivery truck OR Amazon delivery OR UPS strike OR FedEx delay"
        from_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d') 
        to_date = datetime.now().strftime('%Y-%m-%d')

        url = f"https://newsapi.org/v2/everything?q={query}&language=en&from={from_date}&to={to_date}&sortBy=relevancy&apiKey={NEWSAPI_API_KEY}"
        
        try:
            response = requests.get(url, timeout=5) 
            response.raise_for_status() 
            data = response.json()
            
            articles = data.get('articles', [])
            
            news_records = []
            for article in articles:
                news_records.append({
                    'title': article.get('title'),
                    'source': article.get('source', {}).get('name'),
                    'publishedAt': article.get('publishedAt'),
                    'url': article.get('url'),
                    'source_api': 'NewsAPI.org',
                    'city': article.get('source', {}).get('name'),
                    'score': 1
                })
            
            if news_records:
                df_news = pd.DataFrame(news_records)
                logger.info(f"Fetched {len(df_news)} real news alerts from NewsAPI.org.")
            else:
                logger.warning("No real news alerts found for the specified query and dates.")
                df_news = pd.DataFrame(columns=['title', 'source', 'publishedAt', 'url', 'source_api', 'city', 'score']) 
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news from NewsAPI.org: {e}")
            df_news = pd.DataFrame(columns=['title', 'source', 'publishedAt', 'url', 'source_api', 'city', 'score'])
        except Exception as e:
            logger.error(f"An unexpected error occurred during news fetching: {e}")
            df_news = pd.DataFrame(columns=['title', 'source', 'publishedAt', 'url', 'source_api', 'city', 'score'])

    os.makedirs(os.path.dirname(NEWS_ALERTS_CSV), exist_ok=True)
    df_news.to_csv(NEWS_ALERTS_CSV, index=False)
    logger.info(f"News alerts saved to '{NEWS_ALERTS_CSV}'")
    
    return df_news

def fetch_traffic_data():
    logger.info("Fetching real-time traffic data via Google Maps Routes API (Directions API)...")
    
    # Check if the environment variable is actually set
    if not os.getenv('MAPS_API_KEY'):
        logger.warning("MAPS_API_KEY environment variable not set. Using simulated traffic data.")
        simulated_traffic_data = [
            {'origin_city': 'New York', 'destination_city': 'Philadelphia', 'duration_in_traffic_seconds': 5400, 'distance_km': 160, 'incident_present': 0, 'source_api': 'Simulated'},
            {'origin_city': 'Chicago', 'destination_city': 'St. Louis', 'duration_in_traffic_seconds': 16200, 'distance_km': 480, 'incident_present': 1, 'source_api': 'Simulated'} # Increased duration for demo incident
        ]
        df_traffic = pd.DataFrame(simulated_traffic_data)
        logger.info(f"Generated {len(simulated_traffic_data)} simulated traffic data entries.")
    else:
        traffic_records = []
        route_pairs = [
            ('New York', 'Philadelphia'),
            ('Chicago', 'St. Louis'), 
            ('Los Angeles', 'San Diego')
        ]

        for origin_city, dest_city in route_pairs:
            origin_coords = US_CITIES_FOR_API_CALLS.get(origin_city)
            dest_coords = US_CITIES_FOR_API_CALLS.get(dest_city)

            if origin_coords and dest_coords:
                url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin_coords['lat']},{origin_coords['lon']}&destination={dest_coords['lat']},{dest_coords['lon']}&mode=driving&units=metric&departure_time=now&key={MAPS_API_KEY}"
                
                try:
                    response = requests.get(url, timeout=5)
                    response.raise_for_status()
                    data = response.json()

                    if data['status'] == 'OK' and data['routes']:
                        leg = data['routes'][0]['legs'][0]
                        duration_in_traffic_seconds = leg['duration_in_traffic']['value'] if 'duration_in_traffic' in leg else leg['duration']['value']
                        distance_meters = leg['distance']['value']
                        
                        incident_present = 1 if duration_in_traffic_seconds > (leg['duration']['value'] * 1.2) else 0 
                        
                        traffic_records.append({
                            'origin_city': origin_city,
                            'destination_city': dest_city,
                            'duration_in_traffic_seconds': duration_in_traffic_seconds,
                            'distance_km': round(distance_meters / 1000, 2),
                            'incident_present': incident_present,
                            'source_api': 'Google Maps'
                        })
                    else:
                        logger.warning(f"Google Maps: No route found or status not OK for {origin_city} to {dest_city}: {data.get('status', 'N/A')}")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error fetching traffic for {origin_city} to {dest_city} from Google Maps: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error processing Google Maps traffic for {origin_city} to {dest_city}: {e}")
            else:
                logger.warning(f"Missing coordinates for {origin_city} or {dest_city}. Skipping traffic fetch.")
        
        if traffic_records:
            df_traffic = pd.DataFrame(traffic_records)
            logger.info(f"Fetched {len(df_traffic)} real traffic data entries.")
        else:
            logger.warning("No real traffic data found via API. Returning empty DataFrame for traffic.")
            df_traffic = pd.DataFrame(columns=['origin_city', 'destination_city', 'duration_in_traffic_seconds', 'distance_km', 'incident_present', 'source_api'])

    os.makedirs(os.path.dirname(TRAFFIC_DATA_CSV), exist_ok=True)
    df_traffic.to_csv(TRAFFIC_DATA_CSV, index=False)
    logger.info(f"Traffic data saved to '{TRAFFIC_DATA_CSV}'")
    
    return df_traffic

def get_news_features(destination_city, news_alerts):
    # news_alerts: list of dicts, each with 'city', 'score', etc.
    relevant_alerts = [alert for alert in news_alerts if alert['city'].lower() == destination_city.lower()]
    if relevant_alerts:
        return 1, max(alert['score'] for alert in relevant_alerts)
    else:
        return 0, 0

def get_relevant_news(destination_city, news_alerts):
    return [alert for alert in news_alerts if alert['city'].lower() == destination_city.lower()]

def assign_news_impact(row):
    city = row['destination_city']
    # Find if any recent disruption news is for this city
    relevant_news = recent_disruption_news[recent_disruption_news['city'].str.lower() == city.lower()]
    if not relevant_news.empty:
        row['is_supply_chain_news_alert'] = 1
        row['news_disruption_score'] = relevant_news['score'].max() if 'score' in relevant_news.columns else 1
    return row

if __name__ == "__main__":
    fetch_weather_alerts()
    fetch_news_alerts()
    fetch_traffic_data()