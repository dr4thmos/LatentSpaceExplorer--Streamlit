import streamlit as st
import json
import os

import numpy as np
import pandas as pd
import math

import pacmap
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category20, Category10
from bokeh.io import curdoc

from utils import read_experiments_metadata, perform_clustering

import pathlib 
import shutil
from PIL import Image

DATA_FOLDER = "data_demo"    

EMBEDDINGS_FILE = "embeddings.json"
METADATA_FILE = "metadata.json"
LABELS_FILE = "labels.json"
IMAGES_FOLDER = "images"
GENERATED_FOLDER = "generated"

SELECT_EXPERIMENT_TEXT = 'Choose data'
SELECT_EXPERIMENT_KEY = "experiment"

DEV = True

### Move Images on Streamlit static folder in order to make it available in frontend (bokeh)
if not(DEV):
    STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
    print(STREAMLIT_STATIC_PATH)
    # We create a videos directory within the streamlit static asset directory
    # and we write output files to it

    for experiment in os.listdir(DATA_FOLDER):
        STATIC_IMAGES_PATH = (os.path.join(STREAMLIT_STATIC_PATH, experiment, IMAGES_FOLDER))
        if not os.path.isdir(os.path.join(STREAMLIT_STATIC_PATH, experiment)):
            os.mkdir(os.path.join(STREAMLIT_STATIC_PATH, experiment))
        
        if not os.path.isdir(STATIC_IMAGES_PATH):
            os.mkdir(STATIC_IMAGES_PATH)

        for image in os.listdir(os.path.join(DATA_FOLDER, experiment, IMAGES_FOLDER)):
            shutil.copy(os.path.join(DATA_FOLDER, experiment, IMAGES_FOLDER, image), STATIC_IMAGES_PATH)  # For newer Python.
            pass

        STATIC_GENERATED_PATH = (os.path.join(STREAMLIT_STATIC_PATH, experiment, GENERATED_FOLDER))

        if not os.path.isdir(STATIC_GENERATED_PATH):
            os.mkdir(STATIC_GENERATED_PATH)

        for image in os.listdir(os.path.join(DATA_FOLDER, experiment, GENERATED_FOLDER)):
            shutil.copy(os.path.join(DATA_FOLDER, experiment, GENERATED_FOLDER, image), STATIC_GENERATED_PATH)  # For newer Python.
            pass

st.set_page_config(layout="wide")

experiments = read_experiments_metadata(DATA_FOLDER, METADATA_FILE)

st.markdown("### 1.Experiment selection")
current_experiment = st.selectbox(
            SELECT_EXPERIMENT_TEXT,
            tuple(experiments),
            key=SELECT_EXPERIMENT_KEY
        )

EXPERIMENT_FOLDER = os.path.join(DATA_FOLDER, current_experiment)
EMBEDDINGS_PATH = os.path.join(EXPERIMENT_FOLDER, EMBEDDINGS_FILE)
LABELS_PATH = os.path.join(EXPERIMENT_FOLDER, LABELS_FILE)
IMAGE_PATH = os.path.join(EXPERIMENT_FOLDER, IMAGES_FOLDER)
GEN_PATH = os.path.join(EXPERIMENT_FOLDER, GENERATED_FOLDER)

### Showing experiment metadata list 

with st.expander("ℹ️ - Metadata", expanded=False):
    columns = st.columns(9)

with columns[0]:
    st.write("**Name**")
    st.write(str(experiments[current_experiment]["name"]))
with columns[1]:
    st.write("**Image Size**")
    img_dim = str(experiments[current_experiment]["image"]["dim"])
    st.write("{} x {}".format(img_dim, img_dim))
with columns[2]:
    st.write("**Channels #**")
    for channel in experiments[current_experiment]["image"]["channels"]["map"]:
        st.write("{}".format(channel))
with columns[3]:
    st.write("**Image Preview**")
    for channel in experiments[current_experiment]["image"]["channels"]["preview"]:
        st.write("{}: {}".format(channel, experiments[current_experiment]["image"]["channels"]["preview"][channel]))
with columns[4]:
    st.write("**Model architecture**")
    st.write("{}".format(str(experiments[current_experiment]["architecture"]["name"])))
with columns[5]:
    st.write("**Layers #**")
    for idx, filter in enumerate(experiments[current_experiment]["architecture"]["filters"]):
        st.write("{}".format(experiments[current_experiment]["architecture"]["filters"][idx]))
with columns[6]:
    st.write("**Latent Dimension**")
    st.write("{}".format(experiments[current_experiment]["architecture"]["latent_dim"]))
with columns[7]:
    st.write("**Epochs**")
    st.write("{}".format(experiments[current_experiment]["training"]["epochs"]))
with columns[8]:
    st.write("**Batch size**")
    st.write("{}".format(experiments[current_experiment]["training"]["batch_size"]))



st.markdown("---")
st.markdown("### 2.Hyperparameter selection")


ce, c1, ce, c2, ce, c3, ce, c4, ce = st.columns([0.07, 1, 0.15, 1, 0.15, 1, 0.15, 1, 0.07])

with c1:
    
    with st.container():
        visualization_prereduction_check = st.checkbox("Visualization pre-reduction", value=False, help=None, on_change=None)

        visualization_prereduction_method = st.selectbox(
            'Reduction type algorithms',
            ('UMAP', 'PCA'),
            disabled=not(visualization_prereduction_check)
        )

        vis_pre_reduction_components = st.slider('Output dimensions', step = 1, min_value = 4, max_value = 32, value = 5, disabled=not(visualization_prereduction_check))

        with st.container():
            if visualization_prereduction_method == "UMAP":
                vis_pre_reduction_n_neighbors = st.slider('Number of neighbors', key = "vis_pre_reduction_n_neighbors1", step = 5, min_value = 5, max_value = 100, value = 15, disabled=not(visualization_prereduction_check))

            elif visualization_prereduction_method == "PCA":
                st.write('No more parameters for PCA')

with c2:
    with st.container():
        st.write('Reduction')

        visualization_reduction_method = st.selectbox(
            'Reduction type',
            ('PCA','UMAP', 'PACMAP')
        )

        vis_reduction_components = st.number_input('Output dimensions', value = 2, min_value = 2, max_value = 2)

        with st.container():
            if visualization_reduction_method == "PCA":
                st.write('No more parameters for PCA')
                

            elif visualization_reduction_method == "UMAP":
                vis_reduction_UMAP_n_neighbors = st.slider('Number of neighbors', min_value = 5, max_value = 100, step = 5, value = 15)
                vis_reduction_UMAP_min_distance = st.slider('Minimum distance between points', step = 0.05, min_value = 0.0, max_value = 1.0, value = 0.1)
                

            elif visualization_reduction_method == "PACMAP":
                st.write('No parameters for PACMAP for now')
                vis_reduction_PACMAP_n_neighbors = st.slider('Number of neighbors', min_value = 5, max_value = 100, step = 5, value = 15)
                vis_reduction_PACMAP_MN_ratio = st.slider('Attraction between near points', step = 0.1, min_value = 0.1, max_value = 2.0, value = 0.5)
                vis_reduction_PACMAP_FP_ratio = st.slider('Repulsion between distance points', step = 0.5, min_value = 0.5, max_value = 5.0, value = 2.0)
                


with c3:
    
    with st.container():
        clustering_prereduction_check = st.checkbox("(Not2Deep) Clustering prereduction", value=False, help=None, on_change=None)
        
        st.write('Clustering Pre-Reduction')
        clustering_prereduction_method = st.selectbox(
            'Reduction type',
            ('UMAP', 'PCA'),
            disabled = not(clustering_prereduction_check)
        )

        clustering_prereduction_components = st.slider(
            'Output dimensions',
            key="Output_dim_prereduction_2",
            step = 1,
            value = 5,
            min_value = 4,
            max_value = 32,
            disabled = not(clustering_prereduction_check)
        )

        with st.container():
            if clustering_prereduction_method == "UMAP":
                clustering_prereduction_n_neighbors = st.slider(
                    'Number of neighbors',
                    key="clus_prereduction_nneighbors_2",
                    step = 5,
                    min_value = 5,
                    max_value = 100,
                    value = 15,
                    disabled = not(clustering_prereduction_check)
                )

            elif clustering_prereduction_method == "PCA":
                st.write('No other parameters for PCA')
    
with c4:
    with st.container():
        st.write('Clustering')

        clustering_method = st.selectbox(
            'Clustering type',
            ('kmeans', 'dbscan', 'hdbscan', 'affinity propagation', 'agglomerative clustering')
        )

        with st.container():
            clus_params={}
            if clustering_method == "kmeans":
                clus_params["K"] = st.slider('Number of cluster', key="K", step = 1, min_value = 2, max_value = 20, value = 5)

            elif clustering_method == "dbscan":
                clus_params["eps"] = st.slider('Eps', key="eps", step = 0.05, min_value = 0.0, max_value = 1.0, value = 0.1)
                clus_params["min_samples"] = st.slider('Min samples', key="min_samples", step = 1, min_value = 2, max_value = 50, value = 5)
                
            elif clustering_method == "hdbscan":
                clus_params["min_cluster_size"] = st.slider('Min cluster size', key="min_cluster_size", step = 1, min_value = 2, max_value = 50, value = 5)

            elif clustering_method == 'affinity propagation':
                st.write('No parameters for Affinity Propagation')

            elif clustering_method == 'agglomerative clustering':
                clus_params["n_clusters"] = st.slider('Number of clusters', key='n_clusters', step =1, min_value = 2, max_value = 20, value = 5)
            
single_compute = st.button("single_compute")

if single_compute:
    #@st.cache
    def load_data(data_path):
        #print(data_path)
        with open(data_path, "r") as file:
            data = json.load(file)
            #print(data[0])
        return data

    #load_embeddings()
    #load_info()

    embeddings_data = np.array(load_data(EMBEDDINGS_PATH))
    
    labels_data = load_data(LABELS_PATH)
    images = labels_data['columns']
    labels = np.array(labels_data['data'])
    labels = labels.flatten()

    df_image_paths = pd.DataFrame(
        {
            'image_path' : map(
                lambda image: os.path.join(current_experiment, IMAGES_FOLDER,image), 
                images
                )
        })
    print(df_image_paths.head())

    df_images_filename = pd.DataFrame({'image': images})
    df_images_filename = df_images_filename.join(df_image_paths)
    print(df_images_filename.head())


    df_gen_paths = pd.DataFrame(
        {
            'gen_path' : map(
                lambda image: os.path.join(current_experiment, GENERATED_FOLDER,image), 
                images
                )
        })
    print(df_gen_paths.head())

    # Visualization
    # reduce()
    if visualization_prereduction_check:
        st.write('Visualization Pre-Reduction: ', visualization_prereduction_method)
        if visualization_prereduction_method == "UMAP":
            reducer = umap.UMAP(n_neighbors=vis_pre_reduction_n_neighbors, min_dist=0, n_components=vis_pre_reduction_components)
        elif visualization_prereduction_method == "PCA":
            reducer = PCA(n_components=vis_pre_reduction_components)
        viz_data = reducer.fit_transform(embeddings_data)
    else:
        viz_data = embeddings_data

    #st.write('Reduction: ', visualization_reduction_method)
    if visualization_reduction_method == "UMAP":
        reducer = umap.UMAP(n_neighbors=vis_reduction_UMAP_n_neighbors, min_dist=vis_reduction_UMAP_min_distance, n_components=vis_reduction_components)
    elif visualization_reduction_method == "PACMAP":
        reducer = pacmap.PaCMAP(n_components=vis_reduction_components, n_neighbors=vis_reduction_PACMAP_n_neighbors, MN_ratio=vis_reduction_PACMAP_MN_ratio, FP_ratio=vis_reduction_PACMAP_FP_ratio)
    elif visualization_reduction_method == 'PCA':
        reducer = PCA(n_components=2)
    embedding = reducer.fit_transform(viz_data)

    df_embedding = pd.DataFrame(embedding)
    
    if vis_reduction_components == 2:
        df_embedding = df_embedding.rename(columns={0:"x", 1:"y"})
    if vis_reduction_components == 3:
        df_embedding = df_embedding.rename(columns={0:"x", 1:"y", 2:"z"})

    # Clustering
    # clustering()
    if clustering_prereduction_check:
        
        st.write('Clustering Pre-Reduction: ', clustering_prereduction_method)
        if clustering_prereduction_method == "UMAP":
            reducer = umap.UMAP(n_neighbors=clustering_prereduction_n_neighbors, min_dist=0, n_components=clustering_prereduction_components)
            clus_data = reducer.fit_transform(embeddings_data)
        elif clustering_prereduction_method == 'PCA':
            reducer = PCA(n_components=clustering_prereduction_components)
            clus_data = reducer.fit_transform(embeddings_data)
        clusters = perform_clustering(clus_data, clustering_method, clus_params)
    else:
        clusters = perform_clustering(embeddings_data, clustering_method, clus_params)
    
    

    ## Metrics
    # metrics()
    if np.unique(clusters).size > 1:
        silhouette_avg = silhouette_score(embeddings_data, clusters)
        calinski_harabasz_result = calinski_harabasz_score(embeddings_data, clusters)
        davies_bouldin_result = davies_bouldin_score(embeddings_data, clusters)
    else:
        silhouette_avg = -1
        calinski_harabasz_result = -1
        davies_bouldin_result = -1

    df_clusters = pd.DataFrame(clusters)
    df_clusters = df_clusters.rename(columns={0:"clusters"})

    df_embedding = df_embedding.join(df_images_filename)
    df_embedding = df_embedding.join(df_clusters)
    df_embedding = df_embedding.join(df_gen_paths)
    #df_embedding = df_embedding.join(df_image_paths)

    # export_plot()
    output_file('plot.html')
    curdoc().theme = 'dark_minimal'

    # plot_figure()
    if np.unique(clusters).size < 10:
        color = [Category10[10][i+1] for i in df_embedding['clusters']]
    else:
        color = [Category20[20][i+1] for i in df_embedding['clusters']]

    datasource =  ColumnDataSource(data=dict(index=df_embedding.index,
                                            x=df_embedding.x,
                                            y=df_embedding.y,
                                            image=df_embedding.image,
                                            clusters=df_embedding.clusters,
                                            image_path=df_embedding.image_path,
                                            gen_path=df_embedding.gen_path,
                                            color=color))


    plot_figure = figure(plot_width=800, plot_height=800, tools=('pan, wheel_zoom, reset, save'))
    #color_mapping = CategoricalColorMapper(factors=[(x) for x in 'clusters'], palette=Category20[3])

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

    plot_figure.circle('x', 'y', source=datasource, color='color', legend_field='clusters', fill_alpha=0.5, size=12)
    plot_figure.legend.title = "Clusters"
    plot_figure.legend.label_text_color = "black"
    plot_figure.legend.background_fill_color = 'white'
    plot_figure.legend.background_fill_alpha = 0.5

    #show(plot_figure)
    c1, c2 = st.columns([0.2, 1])
    
    with c1:
        st.markdown("### Pipeline resume")
        st.markdown("""
        * Reduction
            * Umap
        * Clustering
            * Kmeans
        """)

        st.markdown("### Statistics")
        st.markdown("Silhouette score average:")
        st.write(silhouette_avg)
        st.markdown("Calinski-Harabasz score")
        st.write(calinski_harabasz_result)
        st.markdown("Davies-Bouldin score:")
        st.write(davies_bouldin_result)
        st.markdown("### Export")
        csv = df_embedding.drop(['image_path', 'gen_path'], axis=1)
        csv = csv.to_csv().encode('utf-8')
        st.download_button(label="Download clusters data as CSV", data=csv, file_name='Data_clusters.csv', mime='text/csv')
    with c2:
        st.bokeh_chart(plot_figure, use_container_width=True)

"""---"""
st.markdown('## 3.Grid search')


ce, c1, ce, c2, ce = st.columns([0.07, 1, 0.15, 1, 0.07])

with c1:
    reduction_grid_search_check = st.checkbox("Reduction grid search", value=False)
    grid_visualization_prereduction_method = st.selectbox(
    'Reduction type',
    ('UMAP', ''),
    disabled=not(reduction_grid_search_check)
    )

    grid_neighbor = st.slider('Select a range of neighbors', 5, 100, (5, 50), step=5, disabled=not(reduction_grid_search_check))
    grid_dist = st.slider('Select a range of distances', 0.0, 0.9, (0.0, 0.6), step=0.1, disabled=not(reduction_grid_search_check))

with c2:
    cluster_grid_search_check = st.checkbox("Clustering grid search", value=False)
    clustering_grid_method = st.selectbox(
        'Clustering type',
        ('kmeans', 'dbscan', 'hdbscan', 'affinity propagation', 'agglomerative clustering'),
        key="clus_type_2",
        disabled=not(cluster_grid_search_check)
    )
    if cluster_grid_search_check:
        with st.container():
            grid_clus_params={}
            if clustering_grid_method == "kmeans":
                grid_clus_params["K"] = st.slider('Number of cluster', key="K2", step = 1, min_value = 2, max_value = 20, value = 5)

            elif clustering_grid_method == "dbscan":
                grid_clus_params["eps"] = st.slider('Eps', key="eps", step = 0.05, min_value = 0.0, max_value = 1.0, value = 0.1)
                grid_clus_params["min_samples"] = st.slider('Min samples', key="min_samples2", step = 1, min_value = 2, max_value = 50, value = 5)
                
            elif clustering_grid_method == "hdbscan":
                grid_clus_params["min_cluster_size"] = st.slider('Min cluster size', key="min_cluster_size2", step = 1, min_value = 2, max_value = 50, value = 5)

            elif clustering_grid_method == 'affinity propagation':
                st.write('No parameters for Affinity Propagation')

            elif clustering_grid_method == 'agglomerative clustering':
                grid_clus_params["n_clusters"] = st.slider('Number of clusters', key='n_clusters2', step =1, min_value = 2, max_value = 20, value = 5)

cgs = st.button("Compute grid search")

if cgs:
    st.write('neighbors: '  , min(grid_neighbor), '  ', max(grid_neighbor),  'distances: ', min(grid_dist), '  ', max(grid_dist))
        
    grid_neighbor_range=[min(grid_neighbor), math.floor((min(grid_neighbor)+max(grid_neighbor))/2), max(grid_neighbor)]
    grid_dist_range=[min(grid_dist), round((min(grid_dist)+max(grid_dist))/2, 2), max(grid_dist)]
    fig, axs = plt.subplots(nrows=len(grid_neighbor_range), ncols=len(grid_dist_range), figsize=(10, 10), constrained_layout=True)
    fig.text(0.5, -0.03, 'Minimum Distance', ha='center', fontsize='medium')
    fig.text(-0.03, 0.5, 'Number of Neighbors', va='center', rotation='vertical', fontsize='medium')
    fig.text(0.5, 1.03, 'Minimum Distance', ha='center', fontsize='medium')
    fig.text(1.03, 0.5, 'Number of Neighbors', va='center', rotation='vertical', fontsize='medium')
    for nrow, n in enumerate(grid_neighbor_range):
        for ncol, d in enumerate(grid_dist_range):
            embedding=umap.UMAP(n_components=2, n_neighbors=n, min_dist=d, random_state=42)
            reducer=embedding.fit_transform(embeddings_data)
            axs[nrow, ncol].scatter(reducer[:,0], reducer[:,1], c=df_embedding['clusters'], s=10, cmap='Spectral')
            axs[nrow, ncol].set_yticklabels([])
            axs[nrow, ncol].set_xticklabels([])
            axs[nrow, ncol].set_title('n_neighbors={} '.format(n) + 'min_dist={}'.format(d), fontsize=8)
    st.pyplot(fig)

    for cluster, col in zip(np.unique(clusters), st.columns(np.unique(clusters).size)):
        with col:
            st.title('#' + str(cluster))
            for _, row in df_embedding.iterrows():
                if row.clusters == cluster:
                    
                    image = Image.open(os.path.join(DATA_FOLDER, row.gen_path))
                    st.image(image, caption=row.image)