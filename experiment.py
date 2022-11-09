
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import math

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering
from sklearn.decomposition import PCA
import hdbscan
import pacmap
import umap

import pandas as pd
from pandas import DataFrame

from bokeh.plotting import figure, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category20, Category10
from bokeh.io import curdoc

import os
import json

from bokeh.plotting import figure

import matplotlib.pyplot as plt

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

    # Aggregate columnar dataframe for rich plotting in bokeh
    aggregated_info: DataFrame = None
    bokeh_figure: figure = None
    grid_figure_reduction: plt = None
    grid_figure_clustering: plt = None
    
    def calculate_grid_clustering(self):
        if self.clustering_grid_search_hyp_param["method"] == "kmeans":
            K = self.clustering_grid_search_hyp_param["K"]
            K_range=[
                min(K),
                (min(K) + max(K))/2,
                max(K)
            ]
            fig, axs = plt.subplots(nrows=1, ncols=len(K_range), figsize=(10, 10), constrained_layout=True)
            fig.text(0.5, -0.03, 'K', ha='center', fontsize='medium')
            for ncol, K_iter in enumerate(K_range):
                clusterer = KMeans(n_clusters=K_iter)
                clusters = clusterer.fit_predict(self.data_representation)
                if np.unique(clusters).size < 10:
                    color = [Category10[10][i+1] for i in clusters]
                else:
                    color = [Category20[20][i+1] for i in clusters]
                axs[ncol].scatter(self.data_reduction[:,0], self.data_reduction[:,1], c=color, s=10, cmap='Spectral')
                axs[ncol].set_yticklabels([])
                axs[ncol].set_xticklabels([])
                axs[ncol].set_title('K={} '.format(K_iter), fontsize=8)
            self.grid_figure_clustering = fig

        elif self.clustering_grid_search_hyp_param["method"] == "dbscan":
            eps = self.clustering_grid_search_hyp_param["eps"]
            min_sample = self.clustering_grid_search_hyp_param["min_samples"]
            eps_range=[
                min(eps),
                (min(eps) + max(eps))/2,
                max(eps)
            ]
            min_sample_range=[
                min(min_sample),
                math.floor((min(min_sample)+max(min_sample))/2),
                max(min_sample)]
            fig, axs = plt.subplots(nrows=len(eps_range), ncols=len(min_sample_range), figsize=(10, 10), constrained_layout=True)
            fig.text(0.5, -0.03, 'eps', ha='center', fontsize='medium')
            fig.text(-0.03, 0.5, 'min_sample', va='center', rotation='vertical', fontsize='medium')
            fig.text(0.5, 1.03, 'eps', ha='center', fontsize='medium')
            fig.text(1.03, 0.5, 'min_sample', va='center', rotation='vertical', fontsize='medium')
            for nrow, eps_iter in enumerate(eps_range):
                for ncol, min_sample_iter in enumerate(min_sample_range):
                    clusterer = DBSCAN(eps=eps_iter, min_samples=min_sample_iter, metric='euclidean')
                    clusters = clusterer.fit_predict(self.data_representation)
                    if np.unique(clusters).size < 10:
                        color = [Category10[10][i+1] for i in clusters]
                    else:
                        color = [Category20[20][i+1] for i in clusters]
                    axs[nrow, ncol].scatter(self.data_reduction[:,0], self.data_reduction[:,1], c=color, s=10, cmap='Spectral')
                    axs[nrow, ncol].set_yticklabels([])
                    axs[nrow, ncol].set_xticklabels([])
                    axs[nrow, ncol].set_title('eps={} '.format(eps_iter) + 'min_sample={}'.format(min_sample_iter), fontsize=8)
            self.grid_figure_clustering = fig
            
        elif self.clustering_grid_search_hyp_param["method"] == "hdbscan":
            return None

        elif self.clustering_grid_search_hyp_param["method"] == 'agglomerative clustering':
            return None

    def calculate_grid_reduction(self):
        grid_neighbor = self.reduction_grid_search_hyp_param["grid_neighbor"]
        grid_dist = self.reduction_grid_search_hyp_param["grid_dist"]
        
        grid_neighbor_range=[
            min(grid_neighbor),
            math.floor((min(grid_neighbor) + max(grid_neighbor))/2),
            max(grid_neighbor)]
        grid_dist_range=[min(grid_dist), round((min(grid_dist)+max(grid_dist))/2, 2), max(grid_dist)]
        fig, axs = plt.subplots(nrows=len(grid_neighbor_range), ncols=len(grid_dist_range), figsize=(10, 10), constrained_layout=True)
        fig.text(0.5, -0.03, 'Minimum Distance', ha='center', fontsize='medium')
        fig.text(-0.03, 0.5, 'Number of Neighbors', va='center', rotation='vertical', fontsize='medium')
        fig.text(0.5, 1.03, 'Minimum Distance', ha='center', fontsize='medium')
        fig.text(1.03, 0.5, 'Number of Neighbors', va='center', rotation='vertical', fontsize='medium')
        for nrow, n in enumerate(grid_neighbor_range):
            for ncol, d in enumerate(grid_dist_range):
                embedding=umap.UMAP(n_components=2, n_neighbors=n, min_dist=d, random_state=42)
                reducer=embedding.fit_transform(self.data_representation)
                #axs[nrow, ncol].scatter(reducer[:,0], reducer[:,1], c=df_embedding['clusters'], s=10, cmap='Spectral')
                axs[nrow, ncol].scatter(reducer[:,0], reducer[:,1], s=10, cmap='Spectral')
                axs[nrow, ncol].set_yticklabels([])
                axs[nrow, ncol].set_xticklabels([])
                axs[nrow, ncol].set_title('n_neighbors={} '.format(n) + 'min_dist={}'.format(d), fontsize=8)
        self.grid_figure_reduction = fig


    def setup_bokeh_plot(self):
        plot_figure = figure(plot_width=800, plot_height=800, tools=('pan, wheel_zoom, reset, save'))

        if np.unique(self.data_clusters).size < 10:
            color = [Category10[10][i+1] for i in self.aggregated_info['clusters']]
        else:
            color = [Category20[20][i+1] for i in self.aggregated_info['clusters']]

        datasource =  ColumnDataSource(data=dict(index=self.aggregated_info.index,
                                                x=self.aggregated_info.x,
                                                y=self.aggregated_info.y,
                                                image=self.aggregated_info.image,
                                                clusters=self.aggregated_info.clusters,
                                                image_path=self.aggregated_info.image_path,
                                                gen_path=self.aggregated_info.gen_path,
                                                color=color))

        plot_figure.add_tools(HoverTool(tooltips="""
            <div style='text-align:center; border: 2px solid; border-radius: 2px'>
            <div style='display:flex'> 
                <div>
                    <img src='@image_path' width="192" style='display: block; margin: 2px auto auto auto;'/>
                </div>
                <div>
                    <img src='@gen_path' width="192" style='display: block; margin: 2px auto auto auto;'/>
                </div>
                </div>
                <div style='padding: 2px; font-size: 12px; color: #000000'>
                    <span>Cluster:</span>
                    <span>@clusters</span><br>
                    <span>X:</span>
                    <span>@x</span><br>
                    <span>Y:</span>
                    <span>@y</span><br>
                    <span>Image:</span>
                    <span>@image</span>
                </div>
            </div>
            """))

        datasource =  ColumnDataSource(data=dict(index=self.aggregated_info.index,
                                                    x=self.aggregated_info.x,
                                                    y=self.aggregated_info.y,
                                                    image=self.aggregated_info.image,
                                                    clusters=self.aggregated_info.clusters,
                                                    image_path=self.aggregated_info.image_path,
                                                    gen_path=self.aggregated_info.gen_path,
                                                    color=color))

        plot_figure.circle('x', 'y', source=datasource, color='color', legend_field='clusters', fill_alpha=0.5, size=12)

        plot_figure.legend.title = "Clusters"
        plot_figure.legend.label_text_color = "black"
        plot_figure.legend.background_fill_color = "white"
        plot_figure.legend.background_fill_alpha = 0.5
        self.bokeh_figure = plot_figure
    
    def aggregate_data(self):
        """
        It constructs dataframe for bokeh plot
        """
        df_gen_paths = pd.DataFrame(
            {
                'gen_path' : map(
                    lambda image: os.path.join(self.generated_images_path, image), 
                    self.data_images
                    )
            }
        )

        df_image_paths = pd.DataFrame(
            {
                'image_path' : map(
                    lambda image: os.path.join(self.images_path, image), 
                    self.data_images
                    )
            }
        )

        df_images_filename = pd.DataFrame({'image': self.data_images})
        df_embedding = pd.DataFrame(self.data_reduction)

        
        if self.reduction_hyp_param["dimensions"] == 2:
            df_embedding = df_embedding.rename(columns={0:"x", 1:"y"})
        if self.reduction_hyp_param["dimensions"] == 3:
            df_embedding = df_embedding.rename(columns={0:"x", 1:"y", 2:"z"})

        df_clusters = pd.DataFrame(self.data_clusters)
        df_clusters = df_clusters.rename(columns={0:"clusters"})
        
        df_embedding = df_embedding.join(df_images_filename)
        df_embedding = df_embedding.join(df_image_paths)
        df_embedding = df_embedding.join(df_clusters)
        df_embedding = df_embedding.join(df_gen_paths)

        self.aggregated_info = df_embedding

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