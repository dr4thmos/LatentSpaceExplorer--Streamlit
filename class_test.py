from experiment import Experiment
from utils import read_experiments_metadata

import streamlit as st
import json
import os
import numpy as np
import pandas as pd

exp = Experiment()

DATA_FOLDER = "data_demo"    

EMBEDDINGS_FILE = "embeddings.json"
METADATA_FILE = "metadata.json"
LABELS_FILE = "labels.json"
IMAGES_FOLDER = "images"
GENERATED_FOLDER = "generated"

experiments = read_experiments_metadata(DATA_FOLDER, METADATA_FILE)

st.markdown("### 1.Choose a latent space")
exp.latent_space_folder = st.selectbox(
            'Choose data',
            tuple(experiments)
        )

EXPERIMENT_FOLDER = os.path.join(DATA_FOLDER, exp.latent_space_folder)
EMBEDDINGS_PATH = os.path.join(EXPERIMENT_FOLDER, EMBEDDINGS_FILE)
LABELS_PATH = os.path.join(EXPERIMENT_FOLDER, LABELS_FILE)
IMAGE_PATH = os.path.join(EXPERIMENT_FOLDER, IMAGES_FOLDER)
GEN_PATH = os.path.join(EXPERIMENT_FOLDER, GENERATED_FOLDER)

with st.container():
    st.write('Clustering')

    exp.clustering_hyp_param["method"] = st.selectbox(
        'Clustering type',
        ('kmeans', 'dbscan', 'hdbscan', 'affinity propagation', 'agglomerative clustering')
    )

with st.container():
    if exp.clustering_hyp_param["method"] == "kmeans":
        exp.clustering_hyp_param["K"] = st.slider('Number of cluster', key="K1", step = 1, min_value = 2, max_value = 20, value = 5)

    elif exp.clustering_hyp_param["method"] == "dbscan":
        exp.clustering_hyp_param["eps"] = st.slider('Eps', key="eps1", step = 0.05, min_value = 0.0, max_value = 1.0, value = 0.1)
        exp.clustering_hyp_param["min_samples"] = st.slider('Min samples', key="min_samples1", step = 1, min_value = 2, max_value = 50, value = 5)
        
    elif exp.clustering_hyp_param["method"] == "hdbscan":
        exp.clustering_hyp_param["min_cluster_size"] = st.slider('Min cluster size', key="min_cluster_size1", step = 1, min_value = 2, max_value = 50, value = 5)

    elif exp.clustering_hyp_param["method"] == 'affinity propagation':
        st.write('No parameters for Affinity Propagation')

    elif exp.clustering_hyp_param["method"] == 'agglomerative clustering':
        exp.clustering_hyp_param["n_clusters"] = st.slider('Number of clusters', key='n_clusters1', step =1, min_value = 2, max_value = 20, value = 5)


### ------------------------

with open(EMBEDDINGS_PATH, "r") as file:
    exp.data_representation = np.array(json.load(file))
with open(LABELS_PATH, "r") as file:
    labels_data = json.load(file)

images = labels_data['columns']
labels = np.array(labels_data['data'])
labels = labels.flatten()

df_image_paths = pd.DataFrame(
    {
        'image_path' : map(
            lambda image: os.path.join(exp.latent_space_folder, IMAGES_FOLDER, image), 
            images
            )
    })
#print(df_image_paths.head())

df_images_filename = pd.DataFrame({'image': images})
df_images_filename = df_images_filename.join(df_image_paths)
#print(df_images_filename.head())


df_gen_paths = pd.DataFrame(
    {
        'gen_path' : map(
            lambda image: os.path.join(exp.latent_space_folder, GENERATED_FOLDER,image), 
            images
            )
    })

exp.calculate_clusters()

st.write(exp.clusters)
#print(df_gen_paths.head())

# Visualization
# reduce()
# if visualization_prereduction_check:
#     st.write('Visualization Pre-Reduction: ', visualization_prereduction_method)
#     if visualization_prereduction_method == "UMAP":
#         reducer = umap.UMAP(n_neighbors=vis_pre_reduction_n_neighbors, min_dist=0, n_components=vis_pre_reduction_components)
#     elif visualization_prereduction_method == "PCA":
#         reducer = PCA(n_components=vis_pre_reduction_components)
#     viz_data = reducer.fit_transform(embeddings_data)
# else:
#     viz_data = embeddings_data

# #st.write('Reduction: ', visualization_reduction_method)
# if visualization_reduction_method == "UMAP":
#     reducer = umap.UMAP(n_neighbors=vis_reduction_UMAP_n_neighbors, min_dist=vis_reduction_UMAP_min_distance, n_components=vis_reduction_components)
# elif visualization_reduction_method == "PACMAP":
#     reducer = pacmap.PaCMAP(n_components=vis_reduction_components, n_neighbors=vis_reduction_PACMAP_n_neighbors, MN_ratio=vis_reduction_PACMAP_MN_ratio, FP_ratio=vis_reduction_PACMAP_FP_ratio)
# elif visualization_reduction_method == 'PCA':
#     reducer = PCA(n_components=2)
# embedding = reducer.fit_transform(viz_data)

# df_embedding = pd.DataFrame(embedding)

# if vis_reduction_components == 2:
#     df_embedding = df_embedding.rename(columns={0:"x", 1:"y"})
# if vis_reduction_components == 3:
#     df_embedding = df_embedding.rename(columns={0:"x", 1:"y", 2:"z"})

# # Clustering
# # clustering()
# if clustering_prereduction_check:
    
#     st.write('Clustering Pre-Reduction: ', clustering_prereduction_method)
#     if clustering_prereduction_method == "UMAP":
#         reducer = umap.UMAP(n_neighbors=clustering_prereduction_n_neighbors, min_dist=0, n_components=clustering_prereduction_components)
#         clus_data = reducer.fit_transform(embeddings_data)
#     elif clustering_prereduction_method == 'PCA':
#         reducer = PCA(n_components=clustering_prereduction_components)
#         clus_data = reducer.fit_transform(embeddings_data)
#     clusters = perform_clustering(clus_data, clustering_method, clus_params)
# else:


