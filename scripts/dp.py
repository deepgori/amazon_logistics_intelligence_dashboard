import json
import os

# --- Configuration ---
# Adjust this path if your file is in a different location (e.g., model_apply_inputs)
TRAVEL_TIMES_FILE = 'data/last_mile_raw/almrrc2021-data-training/model_build_inputs/travel_times.json'

# --- Main Logic ---
if __name__ == "__main__":
    print(f"Attempting to read {TRAVEL_TIMES_FILE}...")
    
    if not os.path.exists(TRAVEL_TIMES_FILE):
        print(f"Error: File not found at {TRAVEL_TIMES_FILE}. Please check the path and ensure the file exists.")
        exit()

    try:
        with open(TRAVEL_TIMES_FILE, 'r') as f:
            travel_times_data = json.load(f)

        print(f"\nSuccessfully loaded {TRAVEL_TIMES_FILE}.")
        print("--- Top 4 data points ---")

        count = 0
        # travel_times.json is typically a dictionary where keys are route_ids
        # and values are another dictionary of travel times (e.g., {"route_id_A": {"(stop1,stop2)": time, ...}, ...})
        for route_id, route_times in travel_times_data.items():
            if count < 4: # Changed limit to 4
                print(f"\nRoute ID: {route_id}")
                # Print first 5 entries within this route_id's travel_times dictionary
                inner_count = 0
                for stop_pair, time in route_times.items():
                    if inner_count < 5:
                        print(f"  {stop_pair}: {time}")
                        inner_count += 1
                    else:
                        break # Limit inner prints
            else:
                break # Limit outer prints

            count += 1
        
        if count == 0:
            print("The JSON file is empty or contains no top-level entries.")
        else:
            print(f"\nTotal top-level entries found: {len(travel_times_data)}")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {TRAVEL_TIMES_FILE}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")