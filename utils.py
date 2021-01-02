import numpy as np
from dataset import get_data


def save_features_unsupervised(labels, n_count):
    n_centers = len(labels)
    X = np.zeros([n_count * n_centers, 24])
    Y = np.zeros([n_count * n_centers], dtype=int)
    i = 0
    for label in labels:
        waves, samples, labels = get_data(label, n_count, true_p=1)
        features = get_feature(waves, samples)  # 100*24*130
        features_mean = np.mean(features, axis=2)
        X[i * n_count:(i + 1) * n_count] = features_mean
        Y[i * n_count:(i + 1) * n_count] = i
        i = i + 1

