import pandas as pd
import os

# Adjust these paths as needed
route_metadata_file = 'data/last_mile_raw/almrrc2021-data-training/model_build_inputs/route_metadata.json'
stops_file = 'data/last_mile_raw/almrrc2021-data-training/model_build_inputs/route_data.json'
output_dir = 'data/processed_last_mile'
os.makedirs(output_dir, exist_ok=True)

# Load route metadata (should contain 'route_id' and 'city')
route_meta = pd.read_json(route_metadata_file, lines=True)

# Load stop/location data (should contain 'route_id')
stops = pd.read_json(stops_file, lines=True)

# Merge on 'route_id'
merged = stops.merge(route_meta[['route_id', 'city']], on='route_id', how='left')

# Check available cities
print("Available cities:", merged['city'].unique())

# List of cities you want to extract
cities = ['seattle', 'los_angeles', 'austin', 'chicago', 'boston']

for city in cities:
    city_df = merged[merged['city'] == city]
    output_path = os.path.join(output_dir, f'{city}_route_data.csv')
    city_df.to_csv(output_path, index=False)
    print(f'Saved {city} data to {output_path} ({len(city_df)} rows)')

print("All done!")
