import joblib
import json
import numpy as np
import os

# Load model and scaler
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Export cluster centers
centers = model.cluster_centers_.tolist()

# Export scaler parameters
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),  # std dev
    'feature_names': scaler.feature_names_in_.tolist()
}

# Load cluster info
with open('models/cluster_info.json', 'r') as f:
    cluster_info = json.load(f)

# Create output directory
output_dir = '../frontend/src/lib/ml'
os.makedirs(output_dir, exist_ok=True)

# Save model data to JSON
model_data = {
    'cluster_centers': centers,
    'scaler': scaler_params,
    'n_clusters': int(model.n_clusters)
}

with open(os.path.join(output_dir, 'model-data.json'), 'w') as f:
    json.dump(model_data, f, indent=2)

# Save cluster info
with open(os.path.join(output_dir, 'cluster-info.json'), 'w') as f:
    json.dump(cluster_info, f, indent=2)

print(f"Model data exported successfully!")
print(f"  - Cluster centers: {len(centers)} clusters x {len(centers[0])} features")
print(f"  - Scaler parameters: mean and scale for {len(scaler_params['feature_names'])} features")
print(f"  - Output directory: {output_dir}")
