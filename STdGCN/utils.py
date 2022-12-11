import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import multiprocessing
from tqdm.notebook import tqdm
import random
from sklearn.decomposition import NMF


from .autoencoder import *



def ST_preprocess(ST_exp, 
                  normalize=True,
                  log=True,
                  highly_variable_genes=False, 
                  regress_out=False, 
                  scale=False,
                  scale_max_value=None,
                  scale_zero_center=True,
                  hvg_min_mean=0.0125,
                  hvg_max_mean=3,
                  hvg_min_disp=0.5,
                  highly_variable_gene_num=None
                 ):
    
    adata = ST_exp.copy()
    
    if normalize == True:
        sc.pp.normalize_total(adata, target_sum=1e4)
        
    if log == True:
        sc.pp.log1p(adata)
        
    adata.layers['scale.data'] = adata.X.copy()
    
    if highly_variable_genes == True:
        sc.pp.highly_variable_genes(adata, 
                                    min_mean=hvg_min_mean, 
                                    max_mean=hvg_max_mean, 
                                    min_disp=hvg_min_disp,
                                    n_top_genes=highly_variable_gene_num,
                                   )
        adata = adata[:, adata.var.highly_variable]
        
    if regress_out == True:
        mito_genes = adata.var_names.str.startswith('MT-')
        adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
        sc.pp.filter_cells(adata, min_counts=0)
        sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])
    
    if scale == True:
        sc.pp.scale(adata, max_value=scale_max_value, zero_center=scale_zero_center)
    
    return adata



def find_marker_genes(sc_exp, 
                      preprocess = True,
                      highly_variable_genes = True,
                      regress_out = False,
                      scale = False,
                      PCA_components = 50, 
                      marker_gene_method = 'wilcoxon',
                      filter_wilcoxon_marker_genes = True, 
                      top_gene_per_type = 20, 
                      pvals_adj_threshold = 0.10,
                      log_fold_change_threshold = 1,
                      min_within_group_fraction_threshold = 0.7,
                      max_between_group_fraction_threshold = 0.3,
                     ):

    if preprocess == True:
        sc_adata_marker_gene = ST_preprocess(sc_exp.copy(), 
                                             normalize=True,
                                             log=True,
                                             highly_variable_genes=highly_variable_genes, 
                                             regress_out=regress_out, 
                                             scale=scale,
                                            )
    else:
        sc_adata_marker_gene = sc_exp.copy()

    sc.tl.pca(sc_adata_marker_gene, n_comps=PCA_components, svd_solver='arpack', random_state=None)
    
    layer = 'scale.data'
    sc.tl.rank_genes_groups(sc_adata_marker_gene, 'cell_type', layer=layer, use_raw=False, pts=True, 
                            method=marker_gene_method, corr_method='benjamini-hochberg', key_added=marker_gene_method)

    if marker_gene_method == 'wilcoxon':
        if filter_wilcoxon_marker_genes == True:
            gene_dict = {}
            gene_list = []
            for name in sc_adata_marker_gene.obs['cell_type'].unique():
                data = sc.get.rank_genes_groups_df(sc_adata_marker_gene, group=name, key=marker_gene_method).sort_values('pvals_adj')
                if pvals_adj_threshold != None:
                    data = data[data['pvals_adj'] < pvals_adj_threshold]
                if log_fold_change_threshold != None:
                    data = data[data['logfoldchanges'] >= log_fold_change_threshold]
                if min_within_group_fraction_threshold != None:
                    data = data[data['pct_nz_group'] >= min_within_group_fraction_threshold]
                if max_between_group_fraction_threshold != None:
                    data = data[data['pct_nz_reference'] < max_between_group_fraction_threshold]
                gene_dict[name] = data['names'].values[:top_gene_per_type].tolist()
                gene_list = gene_list + data['names'].values[:top_gene_per_type].tolist()
                gene_list = list(set(gene_list))
        else:
            gene_table = pd.DataFrame(sc_adata_marker_gene.uns[marker_gene_method]['names'][:top_gene_per_type])
            gene_dict = {}
            for i in gene_table.columns:
                gene_dict[i] = gene_table[i].values.tolist()
            gene_list = list(set([item   for sublist in gene_table.values.tolist()   for item in sublist]))
    elif marker_gene_method == 'logreg':
        gene_table = pd.DataFrame(sc_adata_marker_gene.uns[marker_gene_method]['names'][:top_gene_per_type])
        gene_dict = {}
        for i in gene_table.columns:
            gene_dict[i] = gene_table[i].values.tolist()
        gene_list = list(set([item   for sublist in gene_table.values.tolist()   for item in sublist]))
    else:
        print("marker_gene_method should be 'logreg' or 'wilcoxon'")
    
    return gene_list, gene_dict



def generate_a_spot(sc_exp, 
                    min_cell_number_in_spot, 
                    max_cell_number_in_spot,
                    max_cell_types_in_spot,
                    generation_method,
                   ):
    
    if generation_method == 'cell':
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_list = list(sc_exp.obs.index.values)
        picked_cells = random.choices(cell_list, k=cell_num)
        return sc_exp[picked_cells]
    elif generation_method == 'celltype':
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_type_list = list(sc_exp.obs['cell_type'].unique())
        cell_type_num = random.randint(1, max_cell_types_in_spot)
        
        while(True):
            cell_type_list_selected = random.choices(sc_exp.obs['cell_type'].value_counts().keys(), k=cell_type_num)
            if len(set(cell_type_list_selected)) == cell_type_num:
                break
        sc_exp_filter = sc_exp[sc_exp.obs['cell_type'].isin(cell_type_list_selected)]
        
        picked_cell_type = random.choices(cell_type_list_selected, k=cell_num)
        picked_cells = []
        for i in picked_cell_type:
            data = sc_exp[sc_exp.obs['cell_type'] == i]
            cell_list = list(data.obs.index.values)
            picked_cells.append(random.sample(cell_list, 1)[0])
            
        return sc_exp_filter[picked_cells]
    else:
        print('generation_method should be "cell" or "celltype" ')

        

def pseudo_spot_generation(sc_exp, 
                           idx_to_word_celltype,
                           spot_num, 
                           min_cell_number_in_spot, 
                           max_cell_number_in_spot,
                           max_cell_types_in_spot,
                           generation_method,
                           n_jobs = -1
                          ):
    
    cell_type_num = len(sc_exp.obs['cell_type'].unique())
    
    cores = multiprocessing.cpu_count()
    if n_jobs == -1:
        pool = multiprocessing.Pool(processes=cores)
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
    args = [(sc_exp, min_cell_number_in_spot, max_cell_number_in_spot, max_cell_types_in_spot, generation_method) for i in range(spot_num)]
    generated_spots = pool.starmap(generate_a_spot, tqdm(args, desc='Generating pseudo-spots'))
    
    pseudo_spots = []
    pseudo_spots_table = np.zeros((spot_num, sc_exp.shape[1]), dtype=float)
    pseudo_fraction_table = np.zeros((spot_num, cell_type_num), dtype=float)
    for i in range(spot_num):
        one_spot = generated_spots[i]
        pseudo_spots.append(one_spot)
        pseudo_spots_table[i] = one_spot.X.sum(axis=0)
        for j in one_spot.obs.index:
            type_idx = one_spot.obs.loc[j, 'cell_type_idx']
            pseudo_fraction_table[i, type_idx] += 1
    pseudo_spots_table = pd.DataFrame(pseudo_spots_table, columns=sc_exp.var.index.values)
    pseudo_spots = anndata.AnnData(X=pseudo_spots_table.iloc[:,:].values)
    pseudo_spots.obs.index = pseudo_spots_table.index[:]
    pseudo_spots.var.index = pseudo_spots_table.columns[:]
    type_list = [idx_to_word_celltype[i] for i in range(cell_type_num)]
    pseudo_fraction_table = pd.DataFrame(pseudo_fraction_table, columns=type_list)
    pseudo_fraction_table['cell_num'] = pseudo_fraction_table.sum(axis=1)
    for i in pseudo_fraction_table.columns[:-1]:
        pseudo_fraction_table[i] = pseudo_fraction_table[i]/pseudo_fraction_table['cell_num']
    pseudo_spots.obs = pseudo_spots.obs.join(pseudo_fraction_table)
        
    return pseudo_spots



def data_integration(real, 
                     pseudo, 
                     batch_removal_method="combat",
                     dimensionality_reduction_method='PCA', 
                     dim=50, 
                     scale=True,
                     autoencoder_epoches=2000,
                     autoencoder_LR=1e-3,
                     autoencoder_drop=0,
                     cpu_num=-1,
                     AE_device='GPU'
                    ):
    
    if batch_removal_method == 'mnn':
        mnn = sc.external.pp.mnn_correct(pseudo, real, svd_dim=dim, k=50, batch_key='real_pseudo', save_raw=True, var_subset=None)
        adata = mnn[0]
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size)/2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size, p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction = 'mean')
            embedding = auto_train(model=nets, 
                                   epoch_n=autoencoder_epoches, 
                                   loss_fn=loss_ae, 
                                   optimizer=optimizer_ae, 
                                   data=data,
                                   cpu_num=cpu_num,
                                   device=AE_device
                                  ).detach().numpy()
            if scale == True:
                embedding = (embedding-embedding.mean(axis=0))/embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf-nmf.mean(axis=0))/nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table = table.iloc[pseudo.shape[0]:,:].append(table.iloc[:pseudo.shape[0],:])
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    elif batch_removal_method == 'scanorama':
        import scanorama
        scanorama.integrate_scanpy([real, pseudo], dimred = dim)
        table1 = pd.DataFrame(real.obsm['X_scanorama'], index=real.obs.index.values)
        table2 = pd.DataFrame(pseudo.obsm['X_scanorama'], index=pseudo.obs.index.values)
        table = table1.append(table2)
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    elif batch_removal_method == 'combat':
        aaa = real.copy()
        aaa.obs = pd.DataFrame(index = aaa.obs.index)
        bbb = pseudo.copy()
        bbb.obs = pd.DataFrame(index = bbb.obs.index)
        adata = aaa.concatenate(bbb, batch_key='real_pseudo')
        sc.pp.combat(adata, key='real_pseudo')
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size)/2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size, p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction = 'mean')
            embedding = auto_train(model=nets, 
                                   epoch_n=autoencoder_epoches, 
                                   loss_fn=loss_ae, 
                                   optimizer=optimizer_ae, 
                                   data=data,
                                   cpu_num=cpu_num,
                                   device=AE_device
                                  ).detach().numpy()
            if scale == True:
                embedding = (embedding-embedding.mean(axis=0))/embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf-nmf.mean(axis=0))/nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    else:
        aaa = real.copy()
        aaa.obs = pd.DataFrame(index = aaa.obs.index)
        bbb = pseudo.copy()
        bbb.obs = pd.DataFrame(index = bbb.obs.index)
        adata = aaa.concatenate(bbb, batch_key='real_pseudo')
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size)/2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size, p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction = 'mean')
            embedding = auto_train(model=nets, 
                                   epoch_n=autoencoder_epoches, 
                                   loss_fn=loss_ae, 
                                   optimizer=optimizer_ae, 
                                   data=data,
                                   cpu_num=cpu_num,
                                   device=AE_device
                                  ).detach().numpy()
            if scale == True:
                embedding = (embedding-embedding.mean(axis=0))/embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf-nmf.mean(axis=0))/nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=False)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    table.insert(1, 'cell_num', real.obs['cell_num'].values.tolist()+pseudo.obs['cell_num'].values.tolist())
    table.insert(2, 'cell_type_num', real.obs['cell_type_num'].values.tolist()+pseudo.obs['cell_type_num'].values.tolist())

    return table