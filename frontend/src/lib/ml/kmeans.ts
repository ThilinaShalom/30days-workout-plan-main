/**
 * Standard Scaler - Standardizes features by removing the mean and scaling to unit variance
 * Formula: z = (x - mean) / std_dev
 */
export class StandardScaler {
    private mean: number[];
    private scale: number[];
    private featureNames: string[];

    constructor(mean: number[], scale: number[], featureNames: string[]) {
        this.mean = mean;
        this.scale = scale;
        this.featureNames = featureNames;
    }

    /**
     * Transform input features using the scaler
     */
    transform(features: number[]): number[] {
        if (features.length !== this.mean.length) {
            throw new Error(
                `Feature count mismatch. Expected ${this.mean.length}, got ${features.length}`
            );
        }

        return features.map((value, index) => {
            return (value - this.mean[index]) / this.scale[index];
        });
    }

    getFeatureNames(): string[] {
        return this.featureNames;
    }
}

/**
 * Calculate Euclidean distance between two points
 */
function euclideanDistance(point1: number[], point2: number[]): number {
    if (point1.length !== point2.length) {
        throw new Error('Points must have the same dimensionality');
    }

    const sumOfSquares = point1.reduce((sum, val, index) => {
        const diff = val - point2[index];
        return sum + diff * diff;
    }, 0);

    return Math.sqrt(sumOfSquares);
}

/**
 * K-Means Predictor - Predicts cluster assignment for new data points
 */
export class KMeansPredictor {
    private clusterCenters: number[][];
    private nClusters: number;

    constructor(clusterCenters: number[][], nClusters: number) {
        this.clusterCenters = clusterCenters;
        this.nClusters = nClusters;
    }

    /**
     * Predict cluster assignment for a data point
     * Returns the index of the nearest cluster center
     */
    predict(point: number[]): number {
        if (point.length !== this.clusterCenters[0].length) {
            throw new Error(
                `Feature count mismatch. Expected ${this.clusterCenters[0].length}, got ${point.length}`
            );
        }

        let minDistance = Infinity;
        let nearestCluster = 0;

        for (let i = 0; i < this.nClusters; i++) {
            const distance = euclideanDistance(point, this.clusterCenters[i]);
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = i;
            }
        }

        return nearestCluster;
    }

    /**
     * Get the number of clusters
     */
    getNClusters(): number {
        return this.nClusters;
    }
}
