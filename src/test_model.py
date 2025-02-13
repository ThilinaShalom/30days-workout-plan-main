import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def test_model():
    # Load model
    try:
        model = joblib.load('models/model.pkl')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    # Test data
    test_data = pd.DataFrame([{
        'question3': 70.0,  # weight
        'question4': 170.0, # height
        'question5': 30.0,  # age
        'question7': 3.0,   # days per week
        'question11': 7.0,  # sleep hours
        'dummy_feature': 0
    }])

    # Scale features
    scaler = StandardScaler()
    test_features = scaler.fit_transform(test_data)

    try:
        # Get prediction
        cluster = model.predict(test_features)[0]
        print(f"Model prediction successful. Cluster: {cluster}")

        # Verify cluster is valid
        if cluster not in [0, 1, 2]:
            print(f"Warning: Unexpected cluster value: {cluster}")
            return False

        # Map cluster to focus area
        focus = {
            0: "Cardio-focused",
            1: "Strength training-focused",
            2: "Flexibility and balance-focused"
        }.get(cluster)

        print(f"Focus area: {focus}")
        return True

    except Exception as e:
        print(f"Error during prediction: {e}")
        return False

if __name__ == "__main__":
    test_model()