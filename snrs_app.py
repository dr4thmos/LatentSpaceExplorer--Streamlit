from experiment import Experiment
import numpy as np
import itertools

#from skimage.metrics import structural_similarity as ssim


import streamlit as st

import time

from PIL import Image

import pathlib
import os
import shutil

from ssim import SSIM

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

exp = Experiment(data_folder="data")

exp.read_experiments_metadata()

DEV = False

# ### Move Images on Streamlit static folder in order to make it available in frontend (bokeh)
if not(DEV):
    STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static' 
    DATA_FOLDER = STREAMLIT_STATIC_PATH / exp.data_folder
    print(STREAMLIT_STATIC_PATH)

    if not os.path.isdir(os.path.join(DATA_FOLDER)):
        os.mkdir(os.path.join(DATA_FOLDER))
    # We create a videos directory within the streamlit static asset directory
    # and we write output files to it

    for experiment in os.listdir(exp.data_folder):
        STATIC_IMAGES_PATH = (os.path.join(DATA_FOLDER, experiment, exp.images_folder))
        if not os.path.isdir(os.path.join(DATA_FOLDER, experiment)):
            os.mkdir(os.path.join(DATA_FOLDER, experiment))
        
        if not os.path.isdir(STATIC_IMAGES_PATH):
            os.mkdir(STATIC_IMAGES_PATH)

        for image in os.listdir(os.path.join(exp.data_folder, experiment, exp.images_folder)):
            shutil.copy(os.path.join(exp.data_folder, experiment, exp.images_folder, image), STATIC_IMAGES_PATH)  # For newer Python.

        STATIC_GENERATED_PATH = (os.path.join(DATA_FOLDER, experiment, exp.generated_images_folder))

        if not os.path.isdir(STATIC_GENERATED_PATH):
            os.mkdir(STATIC_GENERATED_PATH)

        for image in os.listdir(os.path.join(exp.data_folder, experiment, exp.generated_images_folder)):
            shutil.copy(os.path.join(exp.data_folder, experiment, exp.generated_images_folder, image), STATIC_GENERATED_PATH)  # For newer Python.


st.sidebar.markdown("### 1.Choose a latent space")
exp.latent_space_folder = st.sidebar.selectbox(
        'Choose data',
        tuple(exp.latent_spaces_list)
    )

exp.set_current_latent_space_path()

st.sidebar.markdown("### 2.Choose which analysis to perform")

check_analysis = st.sidebar.selectbox(
    'Analysis',
    ('Interactive', 'Grid search reduction', 'Grid search clustering')
)

if check_analysis == 'Interactive':
    ### ------------------------ Reduction container
    with st.sidebar:
        st.markdown("### 3a.Choose Visualization HPs")

        exp.reduction_hyp_param["method"] = st.selectbox(
            'Reduction type',
            ('PCA','UMAP', 'PACMAP')
        )

        #exp.reduction_hyp_param["dimensions"] = st.number_input('Output dimensions', value = 2, min_value = 2, max_value = 2)
        exp.reduction_hyp_param["dimensions"] = 2

        with st.container():
            if exp.reduction_hyp_param["method"] == "PCA":
                pass

            elif exp.reduction_hyp_param["method"] == "UMAP":
                exp.reduction_hyp_param["n_neighbors"] = st.slider('Number of neighbors', min_value = 5, max_value = 100, step = 5, value = 15)
                exp.reduction_hyp_param["min_distance"] = st.slider('Minimum distance between points', step = 0.05, min_value = 0.0, max_value = 1.0, value = 0.1)
                
            elif exp.reduction_hyp_param["method"] == "PACMAP":
                st.write('No parameters for PACMAP for now')
                exp.reduction_hyp_param["n_neighbors"] = st.slider('Number of neighbors', min_value = 5, max_value = 100, step = 5, value = 15)
                exp.reduction_hyp_param["MN_ratio"] = st.slider('Attraction between near points', step = 0.1, min_value = 0.1, max_value = 2.0, value = 0.5)
                exp.reduction_hyp_param["FP_ratio"] = st.slider('Repulsion between distance points', step = 0.5, min_value = 0.5, max_value = 5.0, value = 2.0)

        with st.container():
            st.markdown("### 3b.Choose N2D reduction HPs")
            
            exp.preclustering_hyp_param["check"] = st.checkbox("N2D", value=True)
            c1, c2 = st.columns(2)
            with c1:
                exp.preclustering_hyp_param["dimensions"] = st.slider('Number of dimensions', key="predim", min_value = 2, max_value = 32, step = 2, value = 8)
            with c2:
                exp.preclustering_hyp_param["n_neighbors"] = st.slider('Number of neighbors', key="preneigh", min_value = 5, max_value = 100, step = 5, value = 15)
            exp.preclustering_hyp_param["min_distance"] = 0.0

        ### ------------------------ Clustering container

        st.markdown("### 3c.Choose Clustering HPs")

        exp.clustering_hyp_param["method"] = st.selectbox(
            'Clustering type',
            ('kmeans', 'dbscan', 'hdbscan', 'affinity propagation', 'agglomerative clustering')
        )
        if exp.clustering_hyp_param["method"] == "kmeans":
            exp.clustering_hyp_param["K"] = st.slider('Number of cluster', key="K1", step = 1, min_value = 2, max_value = 20, value = 5)

        elif exp.clustering_hyp_param["method"] == "dbscan":
            exp.clustering_hyp_param["eps"] = st.slider('Eps', key="eps1", step = 0.05, min_value = 0.05, max_value = 1.0, value = 0.1)
            exp.clustering_hyp_param["min_samples"] = st.slider('Min samples', key="min_samples1", step = 1, min_value = 2, max_value = 50, value = 5)
            
        elif exp.clustering_hyp_param["method"] == "hdbscan":
            exp.clustering_hyp_param["min_cluster_size"] = st.slider('Min cluster size', key="min_cluster_size1", step = 1, min_value = 2, max_value = 50, value = 5)

        elif exp.clustering_hyp_param["method"] == 'affinity propagation':
            st.write('No parameters for Affinity Propagation')

        elif exp.clustering_hyp_param["method"] == 'agglomerative clustering':
            exp.clustering_hyp_param["n_clusters"] = st.slider('Number of clusters', key='n_clusters1', step =1, min_value = 2, max_value = 20, value = 5)
    
    ssim_check = st.sidebar.checkbox("Perform SSIM", value=False)
    calculate_check = st.sidebar.button("Start compute interactive analysis")
    
### ------------------- Grid Search Reduction
elif check_analysis == 'Grid search reduction':
    st.markdown("### 3.Choose Grid Search hyperparameters")
    
    ### ------------------------ Hyper parameter grid reduction
    exp.reduction_grid_search_hyp_param["method"] = st.selectbox(
        'Reduction type',
        ('UMAP', '')
    )

    exp.reduction_grid_search_hyp_param["grid_neighbor"] = st.slider('Select a range of neighbors', 5, 100, (5, 50), step=5)
    exp.reduction_grid_search_hyp_param["grid_dist"] = st.slider('Select a range of distances', 0.0, 0.9, (0.0, 0.6), step=0.1)

    calculate_grid_red_check = st.button("Start compute grid reduction")

    if calculate_grid_red_check:
        ### ------------------------ Calculate Grid Reduction
        exp.load_experiment_data()
        exp.calculate_grid_reduction()
        st.pyplot(exp.grid_figure_reduction)

### ------------------- Grid Search Clustering
elif check_analysis == 'Grid search clustering':
    
    ### ------------------------ Hyper parameter grid clustering
    exp.clustering_grid_search_hyp_param["method"] = st.selectbox(
        'Clustering type',
        ('dbscan', 'kmeans', 'hdbscan', 'agglomerative clustering'),
        key="clus_type_2"
    )
    with st.container():
        if exp.clustering_grid_search_hyp_param["method"] == "kmeans":
            exp.clustering_grid_search_hyp_param["K"] = st.slider('Number of cluster', key="K2", step = 1, min_value = 2, max_value = 20, value = (3, 5))

        elif exp.clustering_grid_search_hyp_param["method"] == "dbscan":
            exp.clustering_grid_search_hyp_param["eps"] = st.slider('Eps', key="eps", step = 0.05, min_value = 0.05, max_value = 1.0, value = (0.05, 0.5))
            exp.clustering_grid_search_hyp_param["min_samples"] = st.slider('Min samples', key="min_samples2", step = 1, min_value = 2, max_value = 50, value = (2, 10))
            
        elif exp.clustering_grid_search_hyp_param["method"] == "hdbscan":
            exp.clustering_grid_search_hyp_param["min_cluster_size"] = st.slider('Min cluster size', key="min_cluster_size2", step = 1, min_value = 2, max_value = 50, value = (2, 5))

        elif exp.clustering_grid_search_hyp_param["method"] == 'agglomerative clustering':
            exp.clustering_grid_search_hyp_param["n_clusters"] = st.slider('Number of clusters', key='n_clusters2', step =1, min_value = 2, max_value = 20, value = (2,8))

    exp.clustering_grid_search_hyp_param["prereduction"]["n_neighbors"] = st.slider('(N2D) Number of neighbors', min_value = 5, max_value = 100, step = 1, value = 8)

    calculate_grid_clus_check = st.button("Start compute grid clustering")

    if calculate_grid_clus_check:
        ### ------------------------ Calculate Grid Reduction
        exp.clustering_grid_search_hyp_param["reduction"] = "PCA"

        st.markdown("### Grid search Clustering")
        exp.load_experiment_data()
        exp.calculate_grid_clustering()
        st.pyplot(exp.grid_figure_clustering)

        st.markdown("### N2D Grid search Clustering UMAP components 4")
        exp.clustering_grid_search_hyp_param["prereduction"]["n_components"] = 4
        #exp.clustering_grid_search_hyp_param["prereduction"]["n_neighbors"] = 15

        exp.calculate_grid_clustering_N2D()
        st.pyplot(exp.grid_figure_clustering_N2D)

        st.markdown("### N2D Grid search Clustering UMAP components 8")
        exp.clustering_grid_search_hyp_param["prereduction"]["n_components"] = 8
        #exp.clustering_grid_search_hyp_param["prereduction"]["n_neighbors"] = 15

        exp.calculate_grid_clustering_N2D()
        st.pyplot(exp.grid_figure_clustering_N2D)



if check_analysis == 'Interactive':
    if calculate_check:
        exp.load_experiment_data()
        exp.calculate_clusters()
        exp.calculate_reduction()
        exp.calculate_clustering_metrics()
        exp.aggregate_data()
        exp.setup_bokeh_plot()

        st.markdown("#### Silhouette: {} (Higher is better); &nbsp;&nbsp;Cal-Har: {} (Higher is better); &nbsp;&nbsp;Dav-Bou: {} (Lower is better)"
            .format(
                round(exp.metric_repr_silhouette, 2),
                round(exp.metric_repr_calinski_harabasz, 2),
                round(exp.metric_repr_davies_bouldin, 2)
            )
        )

        ### ------------------------ Visualization
        st.bokeh_chart(exp.bokeh_figure, use_container_width=True)
        
        if ssim_check:
            ssim_list = []
            # Per ogni cluster
            for cluster in np.unique(exp.data_clusters):
                cluster_rows = exp.aggregated_info[exp.aggregated_info.clusters == cluster]
                ssim_sum = 0.
                count = 0

                # Per coppia non ripetuta di immagini del cluster
                for reference_image, compare_image in itertools.combinations(cluster_rows.gen_path, 2):
                    count+=1
                    #print(reference_image)
                    #print(compare_image)
                    reference_image = Image.open(reference_image)
                    compare_image = Image.open(compare_image)
                    ssim_angle_max = 0.
                    for angle in range(0,365,5):
                        rotated_compare_image = compare_image.rotate(angle)
                        ### DEBUG method
                        # c1, c2, c3 = st.columns(3)
                        # with c1:
                        #     st.image(reference_image)
                        # with c2:
                        #     st.image(rotated_compare_image)
                        
                        #ssim_value = ssim(reference_image, rotated_compare_image, channel_axis=2)
                        ssim_value = SSIM(reference_image).cw_ssim_value(rotated_compare_image)
                        ssim_angle_max = max(ssim_value, ssim_angle_max)
                        # with c3:
                        #     st.write(round(ssim_value,2))
                    #print(ssim_angle_max)
                    ssim_sum = ssim_sum + ssim_angle_max
                    #print(ssim_sum)
                ssim_avg = ssim_sum / count
                #st.write(ssim_avg)
                ssim_list.append(ssim_avg)

        ### ------------------------ Clusters
        for cluster in np.unique(exp.data_clusters):
            if ssim_check:
                st.title('Cluster #' + str(cluster) + " ssim:  " + str(round(ssim_list[cluster], 2)))
            else:
                st.title('Cluster #' + str(cluster))
            columns = st.columns(10)
            count = 0
            for _, row in exp.aggregated_info.iterrows():
                if row.clusters == cluster:
                    image_input = Image.open(row.image_path)
                    image_gen = Image.open(row.gen_path)
                    
                    with columns[count%5]:
                        st.image(image_gen, caption=row.image)
                    with columns[(count%5)+5]:
                        st.image(image_input, caption=row.image)
                    count+=1