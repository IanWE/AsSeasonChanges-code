import pickle
import json
import numpy as np
from sklearn.feature_extraction import DictVectorizer

def load_features(feature):
    """
    Load and process features for a given feature space from the Hypercube Android Malware Dataset.

    Parameters:
        feature (str): One of {'drebin', 'malscan', 'ramda'}

    Returns:
        X (np.ndarray or sparse matrix): Feature matrix
        y (List[int]): Binary labels (1 = malware, 0 = benign)
        time (List[str]): Corresponding Google Play dates
        filtered_shas (List[str]): SHA256 hashes aligned with rows in X
    """

    feature = feature.lower()

    # Load feature file
    if feature == 'drebin':
        with open('hypercube_drebin.json', 'r') as f:
            all_features = json.load(f)

    elif feature in {'malscan', 'ramda'}:
        feature_path = f'hypercube_{feature}.pickle'
        with open(feature_path, 'rb') as f:
            all_features = pickle.load(f)

    else:
        raise ValueError("Feature must be one of: 'drebin', 'malscan', 'ramda'")

    # Load metadata
    with open('hypercube_metadata.json', 'r') as f:
        metadata = json.load(f)

    # Filter to samples that exist in both metadata and features
    filtered_shas = []
    filtered_features = []
    y = []
    time = []

    for entry in metadata:
        sha = entry['sha256']
        if sha in all_features:
            filtered_shas.append(sha)
            filtered_features.append(all_features[sha])
            y.append(1 if entry['vt_detection'] >= 4 else 0)
            time.append(entry['gp_date'])

    # Convert to feature matrix
    if feature == 'drebin':
        vec = DictVectorizer()
        X = vec.fit_transform(filtered_features)  # Returns a sparse matrix
    else:
        X = np.stack(filtered_features)  # Assumes all arrays have the same shape

    return X, y, time, filtered_shas
