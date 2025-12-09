import pandas as pd
import json
import numpy as np

# Read the workouts CSV
workouts_df = pd.read_csv('data/workouts.csv')

# Replace NaN values with None (which becomes null in JSON)
workouts_df = workouts_df.replace({np.nan: None})

# Convert to JSON
workouts_json = workouts_df.to_dict('records')

# Save to JSON file
with open('../frontend/src/data/workouts.json', 'w') as f:
    json.dump(workouts_json, f, indent=2)

print(f"Converted {len(workouts_json)} workouts to JSON")
