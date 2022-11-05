
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering
from sklearn.decomposition import PCA
import hdbscan
import pacmap
import umap

import os
import json

@dataclass
class Experiment:
    """Class for keeping track of experiment configuration"""

    # Static folders/files name
    data_folder:                str = "data"
    representations_filename:   str = "embeddings.json"
    metadata_filename:          str = "metadata.json"
    labels_filename:            str = "labels.json"
    images_folder:              str = "images"
    generated_images_folder:    str = "generated"
    
    # Dynamic folders/files name
    latent_spaces_list: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict)) 
    latent_space_folder: str = ""

    # Paths
    latent_space_path: str      = ""
    representations_path: str   = ""
    labels_path: str            = ""
    images_path: str            = ""
    generated_images_path: str  = ""

    prereduction_hyp_param:     defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    reduction_hyp_param:        defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    preclustering_hyp_param:    defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    clustering_hyp_param:       defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    
    reduction_grid_search_hyp_param:    defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    clustering_grid_search_hyp_param:   defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))

    # Data
    data_representation:    np.ndarray = None
    data_labels:            np.ndarray = None
    data_images:            list = None
    data_reduction:              np.ndarray = None
    data_clusters:               np.ndarray = None

    # Metrics
    metric_repr_silhouette: float = -1.
    metric_repr_calinski_harabasz: float = -1.
    metric_repr_davies_bouldin: float = -1.
    metric_reduct_silhouette: float = -1.
    metric_reduct_calinski_harabasz: float = -1.
    metric_reduct_davies_bouldin: float = -1.

    def create_dataframe(self):
        return None

    def load_experiment_data(self):
        """
        Funtion that loads representations, images filenames and labels associated to these
        """
        with open(self.representations_path, "r") as file:
            self.data_representation = np.array(json.load(file))
        with open(self.labels_path, "r") as file:
            temp_labels = json.load(file)

        self.data_images = temp_labels['columns']
        self.data_labels = np.array(temp_labels['data'])
        self.data_labels = self.data_labels.flatten()

    def set_current_latent_space_path(self):
        """
        Set paths on current latent space selected
        """
        self.latent_space_path      = os.path.join(self.data_folder, self.latent_space_folder)
        self.representations_path   = os.path.join(self.latent_space_path, self.representations_filename)
        self.labels_path            = os.path.join(self.latent_space_path, self.labels_filename)
        self.images_path            = os.path.join(self.latent_space_path, self.images_folder)
        self.generated_images_path  = os.path.join(self.latent_space_path, self.generated_images_folder)


    def read_experiments_metadata(self):
        """"""
        for experiment in os.listdir(self.data_folder):
            if os.path.isdir(os.path.join(self.data_folder, experiment)):
                with open(os.path.join(self.data_folder, experiment, self.metadata_filename), "r") as file:
                    metadata = json.load(file)
                    self.latent_spaces_list[experiment] = metadata
                    self.latent_spaces_list[experiment]["path"] = os.path.join(self.data_folder, experiment, self.metadata_filename)


    def calculate_clusters(self):
        """"""
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

        self.data_clusters = clusterer.fit_predict(self.data_representation)

    def calculate_clustering_metrics(self):
        """"""
        if np.unique(self.data_clusters).size > 1:
            self.metric_repr_silhouette = silhouette_score(self.data_representation, self.data_clusters)
            self.metric_repr_calinski_harabasz = calinski_harabasz_score(self.data_representation, self.data_clusters)
            self.metric_repr_davies_bouldin = davies_bouldin_score(self.data_representation, self.data_clusters)
            self.metric_reduct_silhouette = silhouette_score(self.data_reduction, self.data_clusters)
            self.metric_reduct_calinski_harabasz = calinski_harabasz_score(self.data_reduction, self.data_clusters)
            self.metric_reduct_davies_bouldin = davies_bouldin_score(self.data_reduction, self.data_clusters)
        else:
            self.metric_repr_silhouette = -1
            self.metric_repr_calinski_harabasz = -1
            self.metric_repr_davies_bouldin = -1
            self.metric_reduct_silhouette = -1
            self.metric_reduct_calinski_harabasz = -1
            self.metric_reduct_davies_bouldin = -1

    def calculate_reduction(self):
        """"""
        if self.reduction_hyp_param["method"] == "UMAP":
            reducer = umap.UMAP(n_neighbors=self.reduction_hyp_param["n_neighbors"], min_dist=self.reduction_hyp_param["min_distance"], n_components=self.reduction_hyp_param["dimensions"])
        
        elif self.reduction_hyp_param["method"] == "PACMAP":
            reducer = pacmap.PaCMAP(n_components=self.reduction_hyp_param["dimensions"], n_neighbors=self.reduction_hyp_param["n_neighbors"], MN_ratio=self.reduction_hyp_param["MN_ratio"], FP_ratio=self.reduction_hyp_param["FP_ratio"])
        
        elif self.reduction_hyp_param["method"] == 'PCA':
            reducer = PCA(n_components=self.reduction_hyp_param["dimensions"])
        
        self.data_reduction = reducer.fit_transform(self.data_representation)