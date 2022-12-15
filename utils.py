import os
import json

from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering
import hdbscan

def read_experiments_metadata(data_folder, metadata_file):
    experiments = {}
    for experiment in os.listdir(data_folder):
        print(experiment)
        if os.path.isdir(os.path.join(data_folder, experiment)):
            # Better in a dictionary with metadata
            with open(os.path.join(data_folder, experiment, metadata_file), "r") as file:
                metadata = json.load(file)
                experiments[experiment] = metadata
                experiments[experiment]["path"] = os.path.join(data_folder, experiment, metadata_file)
    return experiments


def perform_clustering(data, method, params):
    if method == "kmeans":
        clusterer = KMeans(n_clusters=params["K"])

    elif method == "dbscan":
        clusterer = DBSCAN(eps=params["eps"], min_samples=params["min_samples"], metric=params["metric"])
        

    elif method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=params["min_cluster_size"])

    elif method == 'affinity propagation':
        clusterer = AffinityPropagation()

    elif method == 'agglomerative clustering':
        clusterer = AgglomerativeClustering(n_clusters=params["n_clusters"])

    return clusterer.fit_predict(data)

