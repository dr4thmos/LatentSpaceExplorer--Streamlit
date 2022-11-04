from experiment import Experiment
from utils import read_experiments_metadata

import streamlit as st
import json
import os
import numpy as np
import pandas as pd

exp = Experiment(data_folder="data_demo")

exp.read_experiments_metadata()

st.markdown("### 1.Choose a latent space")
exp.latent_space_folder = st.selectbox(
        'Choose data',
        tuple(exp.latent_spaces_list)
    )

exp.set_current_latent_space_path()

with st.container():
    st.write('Clustering')

    exp.clustering_hyp_param["method"] = st.selectbox(
        'Clustering type',
        ('kmeans', 'dbscan', 'hdbscan', 'affinity propagation', 'agglomerative clustering')
    )

### ------------------------ Reduction container

with st.container():
    st.write('Reduction')

    exp.reduction_hyp_param["method"] = st.selectbox(
        'Reduction type',
        ('PCA','UMAP', 'PACMAP')
    )

    exp.reduction_hyp_param["dimensions"] = st.number_input('Output dimensions', value = 2, min_value = 2, max_value = 2)

    with st.container():
        if exp.reduction_hyp_param["method"] == "PCA":
            st.write('No more parameters for PCA')

        elif exp.reduction_hyp_param["method"] == "UMAP":
            exp.reduction_hyp_param["n_neighbors"] = st.slider('Number of neighbors', min_value = 5, max_value = 100, step = 5, value = 15)
            exp.reduction_hyp_param["min_distance"] = st.slider('Minimum distance between points', step = 0.05, min_value = 0.0, max_value = 1.0, value = 0.1)
            
        elif exp.reduction_hyp_param["method"] == "PACMAP":
            st.write('No parameters for PACMAP for now')
            exp.reduction_hyp_param["n_neighbors"] = st.slider('Number of neighbors', min_value = 5, max_value = 100, step = 5, value = 15)
            exp.reduction_hyp_param["MN_ratio"] = st.slider('Attraction between near points', step = 0.1, min_value = 0.1, max_value = 2.0, value = 0.5)
            exp.reduction_hyp_param["FP_ratio"] = st.slider('Repulsion between distance points', step = 0.5, min_value = 0.5, max_value = 5.0, value = 2.0)

### ------------------------ Clustering container

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

exp.load_experiment_data()

df_image_paths = pd.DataFrame(
    {
        'image_path' : map(
            lambda image: os.path.join(exp.images_path, image), 
            exp.data_images
            )
    })
#print(df_image_paths.head())

df_images_filename = pd.DataFrame({'image': exp.data_images})
df_images_filename = df_images_filename.join(df_image_paths)
#print(df_images_filename.head())


df_gen_paths = pd.DataFrame(
    {
        'gen_path' : map(
            lambda image: os.path.join(exp.generated_images_path, image), 
            exp.data_images
            )
    })
#print(df_gen_paths.head())
exp.calculate_clusters()

st.write(exp.data_clusters)

exp.calculate_reduction()
st.write(exp.data_reduction)

df_embedding = pd.DataFrame(exp.data_reduction)

if exp.reduction_hyp_param["dimensions"] == 2:
    df_embedding = df_embedding.rename(columns={0:"x", 1:"y"})
if exp.reduction_hyp_param["dimensions"] == 3:
    df_embedding = df_embedding.rename(columns={0:"x", 1:"y", 2:"z"})

# Visualization

