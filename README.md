# ![Image text](https://github.com/luoyuanlab/stdgcn/blob/main/img_folder/logo-design-2.jpg)  **STdGCN: accurate cell-type deconvolution using graph convolutional networks in spatial transcriptomic data**

# <img src="https://github.com/luoyuanlab/stdgcn/blob/main/img_folder/logo-design-2.jpg" width="200px"; style="float:right; margin: 5px;" /> **STdGCN: accurate cell-type deconvolution using graph convolutional networks in spatial transcriptomic data**

Spatial Transcriptomics deconvolution using Graph Convolutional Networks (STdGCN) is a graph-based deep learning framework that leverages cell type profiles learned from single-cell data to deconvolve the cell type mixtures of spatial transcriptomics data.

![Image text](https://github.com/luoyuanlab/stdgcn/blob/main/img_folder/Figure%201.jpg)

## **Requirements**  
torch == 1.11.0  
numpy == 1.21.6  
pandas == 1.3.5  
scanpy == 1.9.1  
matplotlib == 3.5.1  
scipy == 1.7.3  
tqdm == 4.64.0  
sklearn == 1.0.2  
scanorama == 1.7.2  
random  
pickle  
time  
math  
copy  

## **The example dataset**  
The included spatial transcriptomic (ST) dataset is from Zhu et al. [1], which includes a seqFISH+ slice from the sub-ventricular zone (SVZ) of a mouse somatosensory (SS) region. The resolution of this dataset is single cell-level. We resampled the cells into multiple square pixel areas. Cells within each square pixel area were merged into a synthetic spot. We chose the 200 × 200 square pixel area for resampling. Spots with cells less than two were discarded. The raw single-cell ST data was also used as the single cell reference. All example data files are stored in “./data”

## **Run STdGCN**  
A complete guide for running cell type deconvolution using STdGCN can be found in “Toturial.ipynb” and “Toturial.py”, including the detailed introductions and annotations of using STdGCN.

## **Input files**  
•	sc_data.tsv: The expression matrix of the single cell reference data with cells as rows and genes as columns.  
•	sc_label.tsv: The cell-type annotation of sincle cell data. The table should have two columns: The cell barcode/name and the cell-type annotation information.  
•	ST_data.tsv: The expression matrix of the spatial transcriptomics data with spots as rows and genes as columns.  
•	coordinates.csv: The coordinates of the spatial transcriptomics data. The table should have three columns: Spot barcode/name, X axis (column name 'x'), and Y axis (column name 'y').  
•	marker_genes.tsv [optional]: The gene list used to run STdGCN. Each row is a gene and no table header are permitted.  
•	ST_ground_truth.tsv [optional]: The ground truth of ST data. The data should be transformed into the cell type proportions.  

## **Output files**  
•	pseudo_ST.pkl: The pseudo-spots information.  
•	marker_genes.tsv: Selected cell type marker genes for training.  
•	model_parameters: The saved deep learning parameters for STdGCN.  
•	Loss_function.jpg: The curves to display the loss changes of training, validating, and test (if ST_ground_truth.tsv is provided) datasets.  
•	predict_result.csv: The predicted cell type proportions for the ST data.  
•	results.h5ad: The predicted cell type proportions for the ST data.  
•	predict_results_pie_plot.jpg: The pie plot visualization of the predicted ST data.  
•	“cell type name”.jpg: The scatter plots show the predicted proportions of each cell type in the ST map.  

## **References**  
[1] Zhu Q, Shah S, Dries R, Cai L, Yuan GC. Identification of spatially associated subpopulations by combining scrnaseq and sequential fluorescence in situ hybridization data. *Nat Biotechnol* 2018.

