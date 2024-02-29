import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score as ss
import itertools
from log import *
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

def tune_min_samples_parameter(X: pd.DataFrame):
    # Parmeter tuning by itertools, returns min_samples value
    '''

    Parameters
    ----------
    X
    session

    Returns
    -------
    Instead of tuning for eps and min_samples, using itertools for tuning min_samples only by utilizing silhouette
     score as metric, and using the  best_min_samples for NN-elbow method to tune epsilon
    '''
    epsil = np.arange(0.01, 10, step=1)
    min_samples = np.arange(2, 20, step=1)
    combinations = list(itertools.product(epsil, min_samples))

    scores = []
    LOGGER.info(f'Tuning hyper parameters')

    for (eps, num_samples) in combinations:
        dbscan_model = DBSCAN(eps=eps, min_samples=num_samples).fit(X)
        labels = dbscan_model.labels_
        labels_set = set(labels)
        num_clusters = len(labels_set)

        if -1 in labels_set:
            num_clusters -= 1

        if (num_clusters < 2) or (num_clusters > 50):
            scores.append(-10)
            continue

        scores.append(ss(X, labels))

    best_index = np.argmax(scores)
    best_params = combinations[best_index]
    best_score = scores[best_index]
    best_epsilon = round(float(best_params[0]), 2)
    best_min_samples = int(best_params[1])
    LOGGER.info(
        f'Best min_samples of: {best_min_samples} achieved with Silhouette score of {best_score} and eps of: {best_epsilon}')

    return best_min_samples


def tune_epsilon_parameter(X: pd.DataFrame, min_samples):
    # tuning epsilon with previous min_samples value and NN-elbow method
    '''

    Parameters
    ----------
    X
    min_samples

    Returns
    -------
    epsilon tuned by NN and  kneed LocateElbow
    '''
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, -1])
    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')

    eps_value = kneedle.elbow

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(distances)), distances, marker='o')
    plt.xlabel('Data Point')
    plt.ylabel('Distance to k-th Nearest Neighbor')
    plt.title('K-Distance Plot for DBSCAN Epsilon')
    plt.axvline(x=eps_value, color='r', linestyle='--', label=f'Elbow at data point {eps_value}')
    plt.legend()
    actual_eps_value = round(distances[eps_value], 2)
    LOGGER.info(f"The optimal epsilon value (Elbow point) for DBSCAN is: {actual_eps_value}")
    plt.show()

    return actual_eps_value