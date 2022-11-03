
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List
import numpy as np

from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering
import hdbscan


@dataclass
class Experiment:
    """Class for keeping track of experiment configuration"""
    latent_space_folder: str = ""
    prereduction_hyp_param: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    reduction_hyp_param: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    preclustering_hyp_param: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    clustering_hyp_param: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    
    reduction_grid_search_hyp_param: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    clustering_grid_search_hyp_param: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))

    data_representation: np.ndarray = None
    reduction: np.ndarray = None 
    clusters: np.ndarray = None

    #plot:


    def calculate_clusters(self) -> float:
        if self.clustering_hyp_param["method"] == "kmeans":
            clusterer = KMeans(n_clusters=self.clustering_hyp_param["K"])

        elif self.clustering_hyp_param["method"] == "dbscan":
            clusterer = DBSCAN(eps=self.clustering_hyp_param["eps"], min_samples=self.clustering_hyp_param["min_samples"], metric='euclidean')
            

        elif self.clustering_hyp_param["method"] == "hdbscan":
            clusterer = hdbscan.HDBSCAN(min_cluster_size=self.clustering_hyp_param["min_cluster_size"])

        elif self.clustering_hyp_param["method"] == 'affinity propagation':
            clusterer = AffinityPropagation()

        elif self.clustering_hyp_param["method"] == 'agglomerative clustering':
            clusterer = AgglomerativeClustering(n_clusters=self.clustering_hyp_param["n_clusters"])

        self.clusters = clusterer.fit_predict(self.data_representation)