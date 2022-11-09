from experiment import Experiment

import streamlit as st

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

### ------------------------ Calculate reduction + clusters

exp.load_experiment_data()
exp.calculate_clusters()
exp.calculate_reduction()
exp.aggregate_data()
exp.setup_bokeh_plot()

### ------------------------ Visualization
st.bokeh_chart(exp.bokeh_figure, use_container_width=True)

### ------------------------ Hyper parameter grid reduction
reduction_grid_search_check = st.checkbox("Reduction grid search", value=True)
exp.reduction_grid_search_hyp_param["method"] = st.selectbox(
    'Reduction type',
    ('UMAP', ''),
    disabled=not(reduction_grid_search_check)
)

exp.reduction_grid_search_hyp_param["grid_neighbor"] = st.slider('Select a range of neighbors', 5, 100, (5, 50), step=5, disabled=not(reduction_grid_search_check))
exp.reduction_grid_search_hyp_param["grid_dist"] = st.slider('Select a range of distances', 0.0, 0.9, (0.0, 0.6), step=0.1, disabled=not(reduction_grid_search_check))

### ------------------------ Hyper parameter grid clustering
cluster_grid_search_check = st.checkbox("Clustering grid search", value=True)
exp.clustering_grid_search_hyp_param["method"] = st.selectbox(
    'Clustering type',
    ('dbscan', 'kmeans', 'hdbscan', 'affinity propagation', 'agglomerative clustering'),
    key="clus_type_2",
    disabled=not(cluster_grid_search_check)
)
if cluster_grid_search_check:
    with st.container():
        if exp.clustering_grid_search_hyp_param["method"] == "kmeans":
            exp.clustering_grid_search_hyp_param["K"] = st.slider('Number of cluster', key="K2", step = 1, min_value = 2, max_value = 20, value = (3, 5))

        elif exp.clustering_grid_search_hyp_param["method"] == "dbscan":
            exp.clustering_grid_search_hyp_param["eps"] = st.slider('Eps', key="eps", step = 0.05, min_value = 0.0, max_value = 1.0, value = (0.05, 0.5))
            exp.clustering_grid_search_hyp_param["min_samples"] = st.slider('Min samples', key="min_samples2", step = 1, min_value = 2, max_value = 50, value = (2, 10))
            
        elif exp.clustering_grid_search_hyp_param["method"] == "hdbscan":
            exp.clustering_grid_search_hyp_param["min_cluster_size"] = st.slider('Min cluster size', key="min_cluster_size2", step = 1, min_value = 2, max_value = 50, value = 5)

        elif exp.clustering_grid_search_hyp_param["method"] == 'agglomerative clustering':
            exp.clustering_grid_search_hyp_param["n_clusters"] = st.slider('Number of clusters', key='n_clusters2', step =1, min_value = 2, max_value = 20, value = 5)

### ------------------------ Grid Reduction
exp.calculate_grid_reduction()
st.pyplot(exp.grid_figure_reduction)

### ------------------------ Grid Clustering
exp.calculate_grid_clustering()
st.pyplot(exp.grid_figure_clustering)