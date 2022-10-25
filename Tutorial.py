#!/usr/bin/env python
# coding: utf-8



import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
from STdGCN.STdGCN import run_STdGCN

'''
This module is used to provide the path of the loading data and saving data.

Parameters:
sc_path: The path for loading single cell reference data.
ST_path: The path for loading spatial transcriptomics data.
output_path: The path for saving output files.

The relevant file name and data format for loading:
sc_data.tsv: The expression matrix of the single cell reference data with cells as rows and genes as columns. This file should be saved in "sc_path".
sc_label.tsv: The cell-type annotation of sincle cell data. The table should have two columns: The cell barcode/name and the cell-type annotation information.
            This file should be saved in "sc_path".
ST_data.tsv: The expression matrix of the spatial transcriptomics data with spots as rows and genes as columns. This file should be saved in "ST_path".
coordinates.csv: The coordinates of the spatial transcriptomics data. The table should have three columns: Spot barcode/name, X axis (column name 'x'), and Y axis (column name 'y').
            This file should be saved in "ST_path".
marker_genes.tsv [optional]: The gene list used to run STdGCN. Each row is a gene and no table header is permitted. This file should be saved in "sc_path".
ST_ground_truth.tsv [optional]: The ground truth of ST data. The data should be transformed into the cell type proportions. This file should be saved in "ST_path".
'''
paths = {
    'sc_path': './data/sc_data',
    'ST_path': './data/ST_data',
    'output_path': './output',
}



'''
This module is used to preprocess the input data and identify marker genes [optional].

Parameters:
'preprocess': [bool]. Select whether the input expression data needs to be preprocessed. This step includes normalization, logarithmization, selecting highly variable genes, 
                    regressing out mitochondrial genes, and scaling data.
'normalize': [bool]. When 'preprocess'=True, select whether you need to normalize each cell/spot by total counts = 10,000, so that every cell/spot has the same total 
                    count after normalization.
'log': [bool]. When 'preprocess'=True, select whether you need to logarithmize (X=log(X+1)) the expression matrix.
'highly_variable_genes': [bool]. When 'preprocess'=True, select whether you need to filter the highly variable genes.
'highly_variable_gene_num': [int or None]. When 'preprocess'=True and 'highly_variable_genes'=True, select the number of highly-variable genes to keep.
'regress_out': [bool]. When 'preprocess'=True, select whether you need to regress out mitochondrial genes.
'scale': [bool]. When 'preprocess'=True, select whether you need to scale each gene to unit variance and zero mean.
'PCA_components': [int]. Number of principal components to compute for principal component analysis (PCA).
'marker_gene_method': ['logreg', 'wilcoxon']. We used "scanpy.tl.rank_genes_groups" (https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.rank_genes_groups.html)
                    to identify cell type marker genes. For marker gene selection, STdGCN provides two methods, 'wilcoxon' (Wilcoxon rank-sum) and 'logreg' (uses 
                    logistic regression). 
'top_gene_per_type': [int]. The number of genes for each cell type that can be used to train STdGCN.
'filter_wilcoxon_marker_genes': [bool]. When 'marker_gene_method'='wilcoxon', select whether you need additional steps for gene filtering.
'pvals_adj_threshold': [float or None]. When 'marker_gene_method'='wilcoxon' and 'rank_gene_filter'=True, only genes with corrected p-values < 'pvals_adj_threshold' were kept.
'log_fold_change_threshold': [float or None]. When 'marker_gene_method'='wilcoxon' and 'rank_gene_filter'=True, only genes with log fold change > 'log_fold_change_threshold' were kept.
'min_within_group_fraction_threshold': [float or None]. When 'marker_gene_method'='wilcoxon' and 'rank_gene_filter'=True, only genes expressed with fraction at least
                    'min_within_group_fraction_threshold' in the cell type were kept.
'max_between_group_fraction_threshold': [float or None]. When 'marker_gene_method'='wilcoxon' and 'rank_gene_filter'=True, only genes expressed with fraction at most 
                    'max_between_group_fraction_threshold' in the union of the rest of cell types were kept.
'''
find_marker_genes_paras = {
    'preprocess': True,
    'normalize': True,
    'log': True,
    'highly_variable_genes': False,
    'highly_variable_gene_num': None,
    'regress_out': False,
    'PCA_components': 30, 
    'marker_gene_method': 'logreg',
    'top_gene_per_type': 100,
    'filter_wilcoxon_marker_genes': True,
    'pvals_adj_threshold': 0.10,
    'log_fold_change_threshold': 1,
    'min_within_group_fraction_threshold': None,
    'max_between_group_fraction_threshold': None,
}



'''
This module is used to simulate pseudo-spots.

Parameters:
'spot_num': [int]. The number of pseudo-spots.
'min_cell_num_in_spot': [int]. The minimum number of cells in a pseudo-spot.
'max_cell_num_in_spot': [int]. The maximum number of cells in a pseudo-spot.
'generation_method': ['cell' or 'celltype']. STdGCN provides two pseudo-spot simulation methods. When 'generation_method'='cell', each cell is equally selected. When 
                    'generation_method'='celltype', each cell type is equally selected. See manuscript for more details.
'max_cell_types_in_spot': [int]. When 'generation_method'='celltype', choose the maximum number of cell types in a pseudo-spot.
'''
pseudo_spot_simulation_paras = {
    'spot_num': 30000,
    'min_cell_num_in_spot': 8,
    'max_cell_num_in_spot': 12,
    'generation_method': 'celltype',
    'max_cell_types_in_spot': 4,   
}



'''
This module is used for real- and pseudo- spots normalization.

Parameters:
'normalize': [bool]. Select whether you need to normalize each cell/spot by total counts = 10,000, so that every cell/spot has the same total count after normalization.
'log': [bool]. Select whether you need to logarithmize (X=log(X+1)) the expression matrix.
'scale': [bool]. Select whether you need to scale each gene to unit variance and zero mean.
'''
data_normalization_paras = {
    'normalize': True, 
    'log': True, 
    'scale': False,
}



'''
This module is used to integrate the normalized real- and pseudo- spots together to construct the real-to-pseudo-spot link graph.

Parameters:
'batch_removal_method': ['mnn', 'scanorama', 'combat', None]. Considering batch effects, STdGCN provides four integration methods: mnn (mnnpy, DOI:10.1038/nbt.4091),
                    scanorama (Scanorama, DOI: 10.1038/s41587-019-0113-3), combat (Combat, DOI: 10.1093/biostatistics/kxj037), None (concatenation with no batch removal).
'dimensionality_reduction_method': ['PCA', 'autoencoder', 'nmf', None]. When 'batch_removal_method' is not 'scanorama', select whether the data needs dimensionality reduction, and which
                    dimensionality reduction method is applied.
'dim': [int]. When 'batch_removal_method'='scanorama', select the dimension for this method. When 'batch_removal_method' is not 'scanorama' and 'dimensionality_reduction_method' is
                    not None, select the dimension of the dimensionality reduction.
'scale': [bool]. When 'batch_removal_method' is not 'scanorama', select whether you need to scale each gene to unit variance and zero mean.
'''
integration_for_adj_paras = {
    'batch_removal_method': None, 
    'dim': 30, 
    'dimensionality_reduction_method': 'PCA',
    'scale': True,
}



'''
The module is used to construct the adjacency matrix of the expression graph, which contains three subgraphs: a real-to-pseudo-spot graph, a pseudo-spots internal graph, 
and a real-spots internal graph.

Parameters:
'find_neighbor_method' ['MNN', 'KNN']. STdGCN provides two methods for link graph construction, KNN (K-nearest neighbors) and MNN (mutual nearest neighbors, DOI: 10.1038/nbt.4091).
'dist_method': ['euclidean', 'cosine']. The metrics used for computing paired distances between spots.
'corr_dist_neighbors': [int]. The number of nearest neighbors.
'PCA_dimensionality_reduction': [bool]. For pseudo-spots internal graph and real-spots internal graph construction, select if the data needs to use PCA dimensionality reduction before
                    computing paired distances between spots.
'dim': [int]. When 'PCA_dimensionality_reduction'=True, select the dimension of the PCA.
'''
inter_exp_adj_paras = {
    'find_neighbor_method': 'MNN', 
    'dist_method': 'cosine', 
    'corr_dist_neighbors': 20, 
}
real_intra_exp_adj_paras = {
    'find_neighbor_method': 'MNN', 
    'dist_method': 'cosine',  
    'corr_dist_neighbors': 10,
    'PCA_dimensionality_reduction': False,
    'dim': 50,
}
pseudo_intra_exp_adj_paras = {
    'find_neighbor_method': 'MNN', 
    'dist_method': 'cosine', 
    'corr_dist_neighbors': 20,
    'PCA_dimensionality_reduction': False,
    'dim': 50,
}



'''
The module is used to construct the adjacency matrix of the spatial graph.

Parameters:
'space_dist_threshold': [float or None]. Only the distance between two spots smaller than 'space_dist_threshold' can be linked.
'link_method' ['soft', 'hard']. If spot i and j linked, A(i,j)=1 if 'link_method'='hard', while A(i,j)=1/distance(i,j) if 'link_method'='soft'. See manuscript for more details.
'''
spatial_adj_paras = {
    'link_method': 'soft', 
    'space_dist_threshold': 2,
}



'''
This module is used to integrate the normalized real- and pseudo- spots as the input feature for STdGCN.

Parameters:
'batch_removal_method': ['mnn', 'scanorama', 'combat', None]. Considering batch effects, STdGCN provides four integration methods: mnn (mnnpy, DOI:10.1038/nbt.4091),
                    scanorama (Scanorama, DOI: 10.1038/s41587-019-0113-3), combat (Combat, DOI: 10.1093/biostatistics/kxj037), None (concatenation with no batch removal).
'dimensionality_reduction_method': ['PCA', 'autoencoder', 'nmf', None]. When 'batch_removal_method' is not 'scanorama', select whether the data needs dimensionality reduction, and which
                    dimensionality reduction method is applied.
'dim': [int]. When 'batch_removal_method'='scanorama', select the dimension for this method. When 'batch_removal_method' is not 'scanorama' and 'dimensionality_reduction_method' is
                    not None, select the dimension of the dimensionality reduction.
'scale': [bool]. When 'batch_removal_method' is not 'scanorama', select whether you need to scale each gene to unit variance and zero mean.
'''
integration_for_feature_paras = {
    'batch_removal_method': None, 
    'dimensionality_reduction_method': None, 
    'dim': 80,
    'scale': True,
}



'''
This module is used for setting the deep learning parameters for STdGCN.

Parameters:
'epoch_n': [int]. The maximum number of epochs.
'dim': [int]. The dimension of the hidden layers.
'common_hid_layers_num': [int]. The number of GCN layers = 'common_hid_layers_num'+1.
'fcnn_hid_layers_num': [int]. The number of fully connected neural network layers = 'fcnn_hid_layers_num'+2.
'dropout': [float]. The probability of an element to be zeroed.
'learning_rate_SGD': [float]. Initial learning rate.
'weight_decay_SGD': [float]. L2 penalty.
'momentum': [float]. Momentum factor.
'dampening': [float]. Dampening for momentum.
'nesterov': [bool]. Enables Nesterov momentum.
'early_stopping_patience': [int]. Early stopping epochs.
'clip_grad_max_norm': [float]. Clips gradient norm of an iterable of parameters.
'LambdaLR_scheduler_coefficient': [float]. The coefficent of the LambdaLR scheduler fucntion:  lr(epoch) = [LambdaLR_scheduler_coefficient] ^ epoch_n × learning_rate_SGD. 
'print_loss_epoch_step': [int]. Print the loss value at every 'print_epoch_step' epoch.
'cpu_num': [int]. Set the number of threads used for intraop parallelism on CPU.
'''
GCN_paras = {
    'epoch_n': 3000,
    'dim': 80,
    'common_hid_layers_num': 1,
    'fcnn_hid_layers_num': 1,
    'dropout': 0,
    'learning_rate_SGD': 5e-1,
    'weight_decay_SGD': 3e-4,
    'momentum': 0.9,
    'dampening': 0,
    'nesterov': True,
    'early_stopping_patience': 20,
    'clip_grad_max_norm': 1,
    'LambdaLR_scheduler_coefficient': 0.997,
    'print_loss_epoch_step': 10,
    'cpu_num': 10,
}




'''
## run STdGCN

Parameters
'load_test_groundtruth': [bool]. Select whether you need to upload the ground truth file (ST_ground_truth.tsv) of the spatial transcriptomics data to track the performance of STdGCN.
'use_marker_genes': [bool]. Select whether you need the gene selection process before running STdGCN. Otherwise use common genes from single cell and spatial transcriptomics data.
'external_genes': [bool]. When "use_marker_genes"=True, you can upload your specified gene list (marker_genes.tsv) to run STdGCN.
'generate_new_pseudo_spots': [bool]. STdGCN will save the simulated pseudo-spots to "pseudo_ST.pkl". If you want to run multiple deconvolutions with the same single cell reference data,
                    you don't need to simulate new pseudo-spots and set 'generate_new_pseudo_spots'=False. When 'generate_new_pseudo_spots'=False, you need to pre-move the "pseudo_ST.pkl" 
                    to the 'output_path' so that STdGCN can directly load the pre-simulated pseudo-spots.
'fraction_pie_plot': [bool]. Select whether you need to draw the pie plot of the predicted results. Based on our experience, we do not recommend to draw the pie plot when the predicted
                    spot number is very large. For 1,000 spots, the plotting time is less than 3 minutes; for 2,000 spots, the plotting time is about 20 minutes; for 3,000 spots, it takes
                    about one hour.
'cell_type_distribution_plot': [bool]. Select whether you need to draw the scatter plot of the predicted results for each cell type.
'''
results =  run_STdGCN(paths,
                          load_test_groundtruth = True,
                          use_marker_genes = True,
                          external_genes = False,
                          find_marker_genes_paras = find_marker_genes_paras,
                          generate_new_pseudo_spots = False, 
                          pseudo_spot_simulation_paras = pseudo_spot_simulation_paras,
                          data_normalization_paras = data_normalization_paras,
                          integration_for_adj_paras = integration_for_adj_paras,
                          inter_exp_adj_paras = inter_exp_adj_paras,
                          spatial_adj_paras = spatial_adj_paras,
                          real_intra_exp_adj_paras = real_intra_exp_adj_paras,
                          pseudo_intra_exp_adj_paras = pseudo_intra_exp_adj_paras,
                          integration_for_feature_paras = integration_for_feature_paras,
                          GCN_paras = GCN_paras,
                          fraction_pie_plot = True,
                          cell_type_distribution_plot = True
                         )

results.write_h5ad(paths['output_path']+'/results.h5ad')

