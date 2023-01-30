import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import pickle

from .__init__ import *

   

def run_STdGCN(paths,
               find_marker_genes_paras,
               pseudo_spot_simulation_paras,
               data_normalization_paras,
               integration_for_adj_paras,
               inter_exp_adj_paras,
               spatial_adj_paras,
               real_intra_exp_adj_paras,
               pseudo_intra_exp_adj_paras,
               integration_for_feature_paras,
               GCN_paras,
               load_test_groundtruth = False,
               use_marker_genes = True,
               external_genes = False,
               generate_new_pseudo_spots = True,
               fraction_pie_plot = False,
               cell_type_distribution_plot = True,
               n_jobs = -1,
               GCN_device = 'CPU'
              ):

    sc_path = paths['sc_path']
    ST_path = paths['ST_path']
    output_path = paths['output_path']
    
    sc_adata = sc.read_csv(sc_path+"/sc_data.tsv", delimiter='\t')
    sc_label = pd.read_table(sc_path+"/sc_label.tsv", sep = '\t', header = 0, index_col = 0, encoding = "utf-8")
    sc_label.columns = ['cell_type']
    sc_adata.obs['cell_type'] = sc_label['cell_type'].values

    cell_type_num = len(sc_adata.obs['cell_type'].unique())
    cell_types = sc_adata.obs['cell_type'].unique()

    word_to_idx_celltype = {word: i for i, word in enumerate(cell_types)}
    idx_to_word_celltype = {i: word for i, word in enumerate(cell_types)}

    celltype_idx = [word_to_idx_celltype[w] for w in sc_adata.obs['cell_type']]
    sc_adata.obs['cell_type_idx'] = celltype_idx
    sc_adata.obs['cell_type'].value_counts()

    if use_marker_genes == True:
        if external_genes == True:
            with open(sc_path+"/marker_genes.tsv", 'r') as f:
                selected_genes = [line.rstrip('\n') for line in f]
        else:
            selected_genes, cell_type_marker_genes = find_marker_genes(sc_adata,
                                                                      preprocess = find_marker_genes_paras['preprocess'],
                                                                      highly_variable_genes = find_marker_genes_paras['highly_variable_genes'],
                                                                      PCA_components = find_marker_genes_paras['PCA_components'], 
                                                                      filter_wilcoxon_marker_genes = find_marker_genes_paras['filter_wilcoxon_marker_genes'], 
                                                                      marker_gene_method = find_marker_genes_paras['marker_gene_method'],
                                                                      pvals_adj_threshold = find_marker_genes_paras['pvals_adj_threshold'],
                                                                      log_fold_change_threshold = find_marker_genes_paras['log_fold_change_threshold'],
                                                                      min_within_group_fraction_threshold = find_marker_genes_paras['min_within_group_fraction_threshold'],
                                                                      max_between_group_fraction_threshold = find_marker_genes_paras['max_between_group_fraction_threshold'],
                                                                      top_gene_per_type = find_marker_genes_paras['top_gene_per_type'])
            with open(output_path+"/marker_genes.tsv", 'w') as f:
                for gene in selected_genes:
                    f.write(str(gene) + '\n')
            
    print("{} genes have been selected as marker genes.".format(len(selected_genes)))
    
    

    if generate_new_pseudo_spots == True:
        pseudo_adata = pseudo_spot_generation(sc_adata,
                                              idx_to_word_celltype,
                                              spot_num = pseudo_spot_simulation_paras['spot_num'],
                                              min_cell_number_in_spot = pseudo_spot_simulation_paras['min_cell_num_in_spot'],
                                              max_cell_number_in_spot = pseudo_spot_simulation_paras['max_cell_num_in_spot'],
                                              max_cell_types_in_spot = pseudo_spot_simulation_paras['max_cell_types_in_spot'],
                                              generation_method = pseudo_spot_simulation_paras['generation_method'],
                                              n_jobs = n_jobs
                                              )
        data_file = open(output_path+'/pseudo_ST.pkl','wb')
        pickle.dump(pseudo_adata, data_file)
        data_file.close()
    else:
        data_file = open(output_path+'/pseudo_ST.pkl','rb')
        pseudo_adata = pickle.load(data_file)
        data_file.close()

    ST_adata = sc.read_csv(ST_path+"/ST_data.tsv", delimiter='\t')
    ST_coor = pd.read_table(ST_path+"/coordinates.csv", sep = ',', header = 0, index_col = 0, encoding = "utf-8")
    ST_adata.obs['coor_X'] = ST_coor['x']
    ST_adata.obs['coor_Y'] = ST_coor['y']
    if load_test_groundtruth == True:
        ST_groundtruth = pd.read_table(ST_path+"/ST_ground_truth.tsv", sep = '\t', header = 0, index_col = 0, encoding = "utf-8")
        for i in cell_types:
            ST_adata.obs[i] = ST_groundtruth[i]

    ST_genes = ST_adata.var.index.values
    pseudo_genes = pseudo_adata.var.index.values
    common_genes = set(ST_genes).intersection(set(pseudo_genes))
    ST_adata_filter = ST_adata[:,list(common_genes)]
    pseudo_adata_filter = pseudo_adata[:,list(common_genes)]
    
    
    ST_adata_filter_norm = ST_preprocess(ST_adata_filter, 
                                         normalize = data_normalization_paras['normalize'], 
                                         log = data_normalization_paras['log'], 
                                         scale = data_normalization_paras['scale'],
                                        )[:,selected_genes]
    
    try:
        try:
            ST_adata_filter_norm.obs.insert(0, 'cell_num', ST_adata_filter.obs['cell_num'])
        except:
            ST_adata_filter_norm.obs['cell_num'] = ST_adata_filter.obs['cell_num']
    except:
        ST_adata_filter_norm.obs.insert(0, 'cell_num', [0]*ST_adata_filter_norm.obs.shape[0])
    for i in cell_types:
        try:
            ST_adata_filter_norm.obs[i] = ST_adata_filter.obs[i]
        except:
            ST_adata_filter_norm.obs[i] = [0]*ST_adata_filter_norm.obs.shape[0]
    try:
        ST_adata_filter_norm.obs['cell_type_num'] = (ST_adata_filter_norm.obs[cell_types]>0).sum(axis=1)
    except:
        ST_adata_filter_norm.obs['cell_type_num'] = [0]*ST_adata_filter_norm.obs.shape[0]


    pseudo_adata_norm = ST_preprocess(pseudo_adata_filter, 
                                      normalize = data_normalization_paras['normalize'], 
                                      log = data_normalization_paras['log'], 
                                      scale = data_normalization_paras['scale'],
                                     )[:,selected_genes]

    pseudo_adata_norm.obs['cell_type_num'] = (pseudo_adata_norm.obs[cell_types]>0).sum(axis=1)
    
    
    ST_integration = data_integration(ST_adata_filter_norm, 
                                      pseudo_adata_norm, 
                                      batch_removal_method = integration_for_adj_paras['batch_removal_method'], 
                                      dim = min(integration_for_adj_paras['dim'], int(ST_adata_filter_norm.shape[1]/2)), 
                                      dimensionality_reduction_method=integration_for_adj_paras['dimensionality_reduction_method'],
                                      scale=integration_for_adj_paras['scale'],
                                      cpu_num=n_jobs,
                                      AE_device=GCN_device
                                      )
    
    A_inter_exp =  inter_adj(ST_integration, 
                             find_neighbor_method=inter_exp_adj_paras['find_neighbor_method'], 
                             dist_method=inter_exp_adj_paras['dist_method'], 
                             corr_dist_neighbors=inter_exp_adj_paras['corr_dist_neighbors'], 
                            )

    A_intra_space = intra_dist_adj(ST_adata_filter_norm, 
                                   link_method=spatial_adj_paras['link_method'],
                                   space_dist_threshold=spatial_adj_paras['space_dist_threshold'],
                                  )
    
    A_real_intra_exp = intra_exp_adj(ST_adata_filter_norm, 
                                     find_neighbor_method=real_intra_exp_adj_paras['find_neighbor_method'], 
                                     dist_method=real_intra_exp_adj_paras['dist_method'],
                                     PCA_dimensionality_reduction=real_intra_exp_adj_paras['PCA_dimensionality_reduction'],
                                     corr_dist_neighbors=real_intra_exp_adj_paras['corr_dist_neighbors'],
                                    )

    A_pseudo_intra_exp = intra_exp_adj(pseudo_adata_norm, 
                                       find_neighbor_method=pseudo_intra_exp_adj_paras['find_neighbor_method'], 
                                       dist_method=pseudo_intra_exp_adj_paras['dist_method'],
                                       PCA_dimensionality_reduction=pseudo_intra_exp_adj_paras['PCA_dimensionality_reduction'],
                                       corr_dist_neighbors=pseudo_intra_exp_adj_paras['corr_dist_neighbors'],
                                      )
    
    real_num = ST_adata_filter.shape[0]
    pseudo_num = pseudo_adata.shape[0]

    adj_inter_exp = A_inter_exp.values
    adj_pseudo_intra_exp = A_intra_transfer(A_pseudo_intra_exp, 'pseudo', real_num, pseudo_num)
    adj_real_intra_exp = A_intra_transfer(A_real_intra_exp, 'real', real_num, pseudo_num)
    adj_intra_space = A_intra_transfer(A_intra_space, 'real', real_num, pseudo_num)

    adj_alpha = 1
    adj_beta = 1
    diag_power = 20
    adj_balance = (1+adj_alpha+adj_beta)*diag_power
    adj_exp = torch.tensor(adj_inter_exp+adj_alpha*adj_pseudo_intra_exp+adj_beta*adj_real_intra_exp)/adj_balance + torch.eye(adj_inter_exp.shape[0])
    adj_sp = torch.tensor(adj_intra_space)/diag_power + torch.eye(adj_intra_space.shape[0])

    norm = True
    if(norm == True):
        adj_exp = torch.tensor(adj_normalize(adj_exp, symmetry=True))
        adj_sp = torch.tensor(adj_normalize(adj_sp, symmetry=True))
        
        
    ST_integration_batch_removed = data_integration(ST_adata_filter_norm, 
                                                    pseudo_adata_norm, 
                                                    batch_removal_method=integration_for_feature_paras['batch_removal_method'], 
                                                    dim=min(int(ST_adata_filter_norm.shape[1]*1/2), integration_for_feature_paras['dim']), 
                                                    dimensionality_reduction_method=integration_for_feature_paras['dimensionality_reduction_method'], 
                                                    scale=integration_for_feature_paras['scale'],
                                                    cpu_num=n_jobs,
                                                    AE_device=GCN_device
                                                   )
    feature = torch.tensor(ST_integration_batch_removed.iloc[:, 3:].values)
    
    
    input_layer = feature.shape[1]
    hidden_layer = min(int(ST_adata_filter_norm.shape[1]*1/2), GCN_paras['dim'])
    output_layer1 = len(word_to_idx_celltype)
    epoch_n = GCN_paras['epoch_n']
    common_hid_layers_num = GCN_paras['common_hid_layers_num']
    fcnn_hid_layers_num = GCN_paras['fcnn_hid_layers_num']
    dropout = GCN_paras['dropout']
    learning_rate_SGD = GCN_paras['learning_rate_SGD']
    weight_decay_SGD = GCN_paras['weight_decay_SGD']
    momentum = GCN_paras['momentum']
    dampening = GCN_paras['dampening']
    nesterov = GCN_paras['nesterov']
    early_stopping_patience = GCN_paras['early_stopping_patience']
    clip_grad_max_norm = GCN_paras['clip_grad_max_norm']
    LambdaLR_scheduler_coefficient = 0.997
    ReduceLROnPlateau_factor = 0.1
    ReduceLROnPlateau_patience = 5
    scheduler = 'scheduler_ReduceLROnPlateau'
    print_epoch_step = GCN_paras['print_loss_epoch_step']
    cpu_num = n_jobs
    
    model = conGCN(nfeat = input_layer, 
                   nhid = hidden_layer, 
                   common_hid_layers_num = common_hid_layers_num, 
                   fcnn_hid_layers_num = fcnn_hid_layers_num, 
                   dropout = dropout, 
                   nout1 = output_layer1
                  )

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr = learning_rate_SGD, 
                                momentum = momentum, 
                                weight_decay = weight_decay_SGD, 
                                dampening = dampening, 
                                nesterov = nesterov)
    
    scheduler_LambdaLR = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                           lr_lambda = lambda epoch: LambdaLR_scheduler_coefficient ** epoch)
    scheduler_ReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                             mode='min', 
                                                                             factor=ReduceLROnPlateau_factor, 
                                                                             patience=ReduceLROnPlateau_patience, 
                                                                             threshold=0.0001, 
                                                                             threshold_mode='rel', 
                                                                             cooldown=0, 
                                                                             min_lr=0)
    if scheduler == 'scheduler_LambdaLR':
        scheduler = scheduler_LambdaLR
    elif scheduler == 'scheduler_ReduceLROnPlateau':
        scheduler = scheduler_ReduceLROnPlateau
    else:
        scheduler = None
    
    loss_fn1 = nn.KLDivLoss(reduction = 'mean')

    train_valid_len = pseudo_adata.shape[0]
    test_len = ST_adata_filter.shape[0]

    table1 = ST_adata_filter_norm.obs.copy()
    label1 = table1[pseudo_adata.obs.iloc[:,:-1].columns].append(pseudo_adata.obs.iloc[:,:-1])
    label1 = torch.tensor(label1.values)

    adjs = [adj_exp.float(), adj_sp.float()]

    output1, loss, trained_model = conGCN_train(model = model, 
                                                train_valid_len = train_valid_len,
                                                train_valid_ratio = 0.9,
                                                test_len = test_len, 
                                                feature = feature, 
                                                adjs = adjs, 
                                                label = label1, 
                                                epoch_n = epoch_n, 
                                                loss_fn = loss_fn1, 
                                                optimizer = optimizer, 
                                                scheduler = scheduler, 
                                                early_stopping_patience = early_stopping_patience,
                                                clip_grad_max_norm = clip_grad_max_norm,
                                                load_test_groundtruth = load_test_groundtruth,
                                                print_epoch_step = print_epoch_step,
                                                cpu_num = cpu_num,
                                                GCN_device = GCN_device
                                               )
    
    loss_table = pd.DataFrame(loss, columns=['train', 'valid', 'test'])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(loss_table.index, loss_table['train'], label='train')
    ax.plot(loss_table.index, loss_table['valid'], label='valid')
    if load_test_groundtruth == True:
        ax.plot(loss_table.index, loss_table['test'], label='test')
    ax.set_xlabel('Epoch', fontsize = 20)
    ax.set_ylabel('Loss', fontsize = 20)
    ax.set_title('Loss function curve', fontsize = 20)
    ax.legend(fontsize = 15)
    plt.tight_layout()
    plt.savefig(output_path+'/Loss_function.jpg', dpi=300)
    plt.close('all')
    
    predict_table = pd.DataFrame(np.exp(output1[:test_len].detach().numpy()).tolist(), index=ST_adata_filter_norm.obs.index, columns=pseudo_adata_norm.obs.columns[:-2])
    predict_table.to_csv(output_path+'/predict_result.csv', index=True, header=True)
    
    torch.save(trained_model, output_path+'/model_parameters')
    
    pred_use = np.round_(output1.exp().detach()[:test_len], decimals=4)
    cell_type_list = cell_types
    coordinates = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']]
    
    if fraction_pie_plot == True:
        plot_frac_results(pred_use, cell_type_list, coordinates, point_size=300, size_coefficient=0.0009, file_name=output_path+'/predict_results_pie_plot.jpg', if_show=False)
        
    if cell_type_distribution_plot == True:
        plot_scatter_by_type(pred_use, cell_type_list, coordinates, point_size=300, file_path=output_path, if_show=False)
    
    ST_adata_filter_norm.obsm['predict_result'] = np.exp(output1[:test_len].detach().numpy())
    
    torch.cuda.empty_cache()
    
    return ST_adata_filter_norm
