import modelData from './model-data.json';
import clusterInfo from './cluster-info.json';
import { StandardScaler, KMeansPredictor } from './kmeans';

// Initialize scaler with trained parameters
const scaler = new StandardScaler(
    modelData.scaler.mean,
    modelData.scaler.scale,
    modelData.scaler.feature_names
);

// Initialize K-Means predictor with cluster centers
const kmeansPredictor = new KMeansPredictor(
    modelData.cluster_centers,
    modelData.n_clusters
);

export interface UserFeatures {
    weight: number;
    height: number;
    age: number;
    bmi: number;
    days_per_week: number;
    sleep_hours: number;
    calories: number;
    protein: number;
    carbohydrate: number;
    total_fat: number;
    fiber: number;
    intensity: number;
    exercise_type: number;
    rating: number;
}

export interface ClusterInfo {
    size: number;
    percentage: number;
    center: number[];
    focus: string;
    intensity_level: string;
    recommended_days: number;
    dominant_features: { [key: string]: number };
}

export interface PredictionResult {
    cluster: number;
    clusterInfo: ClusterInfo;
}

/**
 * Predict the cluster assignment for user features
 */
export function predictCluster(features: UserFeatures): PredictionResult {
    // Extract features in the correct order
    const featureArray: number[] = [
        features.weight,
        features.height,
        features.age,
        features.bmi,
        features.days_per_week,
        features.sleep_hours,
        features.calories,
        features.protein,
        features.carbohydrate,
        features.total_fat,
        features.fiber,
        features.intensity,
        features.exercise_type,
        features.rating,
    ];

    // Standardize features
    const standardizedFeatures = scaler.transform(featureArray);

    // Predict cluster
    const cluster = kmeansPredictor.predict(standardizedFeatures);

    // Get cluster information
    const clusterInfoData = clusterInfo[cluster.toString() as keyof typeof clusterInfo];

    return {
        cluster,
        clusterInfo: clusterInfoData,
    };
}

/**
 * Get information about a specific cluster
 */
export function getClusterInfo(clusterId: number): ClusterInfo | null {
    const info = clusterInfo[clusterId.toString() as keyof typeof clusterInfo];
    return info || null;
}

/**
 * Get all cluster information
 */
export function getAllClusters(): { [key: string]: ClusterInfo } {
    return clusterInfo;
}

/**
 * Get feature names in the order expected by the model
 */
export function getFeatureNames(): string[] {
    return scaler.getFeatureNames();
}
