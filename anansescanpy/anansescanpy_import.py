"""
Collection of AnanseScanpy import functions
"""
import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from statistics import mean
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

def import_scanpy_scANANSE(
    anndata,
    cluster_id = 'leiden_new',
    anansnake_inf_dir = 'None',
    unified_contrast='average'
):
    """import_scanpy_scANANSE
    This function imports the influence results from anansnake into the scanpy 
    object and returns a dataframe with influence scores of bulk clusters
    
    Params:
    ---
    anndata object
    cluster_id: ID used for finding clusters of cells
    anansnake_inf_dir: directory of the anansnake output
    unified_contrast: the contrast where all populations were compared to with ANANSE
    Usage:
    ---
    >>> from anansescanpy import import_scanpy_scANANSE
    >>> df = import_scanpy_scANANSE(adata,cluster_id="predicted.id",
                       anansnake_inf_dir="/AnanseScanpy_outs/influence/")
    """
    adata = anndata
    appended_data = []
    for file in os.listdir(anansnake_inf_dir):
        if file.endswith('.tsv'):
            if unified_contrast in file:
                if not 'diffnetwork' in file:
                    filedir= anansnake_inf_dir+file
                    influence_data= pd.read_csv(filedir,sep='\t',index_col=0)
                    influence_data= influence_data[["influence_score"]]
                    cluster= file.split("_")[1]
                    influence_data= influence_data.rename(columns={'influence_score': cluster})
                    
    # store DataFrame in list
                    appended_data.append(influence_data)
        
    # Insert 0 values where Nan is seen
    appended_data= pd.concat(appended_data)
    appended_data= appended_data.replace(np.nan,0)

    # Generate a influence score dataframe for export and process the appended_data for anndata
    joined_data= appended_data.transpose()
    joined_data= joined_data.groupby(joined_data.columns, axis=1).sum()
    output= joined_data
    joined_data= joined_data.add_suffix('_influence')
    joined_data[cluster_id]= joined_data.index
    
    # Retrieve the cluster IDs and cell IDs from the anndata object
    df= pd.DataFrame(adata.obs[cluster_id])
    df["cells"]=df.index.astype("string")
    
    # Merge the processed appended_data together with the cell ID df
    df_obs= joined_data.merge(df, on=cluster_id,how='left')
    df_obs=df_obs.drop(columns=[cluster_id])
    
    # Merge the observation dataframe with anndata obs
    adata.obs["cells"]=adata.obs.index.astype("string")
    adata.obs = adata.obs.merge(df_obs, on='cells',how='left')
    adata.obs.index=adata.obs["cells"]
    adata.obs.index.name = None
    
    return output
    

def import_scanpy_maelstrom(
    anndata,
    cluster_id="predicted.id",
    maelstrom_dir="maelstrom/",
    return_df = False):
    """import_scanpy_maelstrom
    This functions imports maelstrom output to the anndata object
    Params:
    ---
    anndata object
    cluster_id: ID used for finding clusters of cells
    maelstrom_dir: directory where maelstrom output is located
    return_df: returns a df if set to True
    Usage:
    ---
    >>> from anansescanpy import scanpy_maelstrom
    >>> import_scanpy_maelstrom(adata)
    >>> maelstrom_df = import_scanpy_maelstrom(adata,return_df = True)
    """
    adata = anndata
    
    # Import the output df from maelstrom
    maelstrom_df = pd.read_csv(str(maelstrom_dir + "final.out.txt"),sep="\t",index_col=0)
    maelstrom_cols = [col for col in maelstrom_df.columns if 'z-score' in col]
    maelstrom_Zscore = maelstrom_df[maelstrom_cols]
    motif_df=maelstrom_Zscore.transpose()
    
    # Generate a maelstrom score dataframe for each motif 
    # and process the appended_data for anndata
    motif_df.index = motif_df.index.str.replace('z-score ', '')
    output=motif_df
    motif_df= motif_df.add_suffix('_maelstrom')
    motif_df[cluster_id]= motif_df.index

    # Retrieve the cluster IDs and cell IDs from the anndata object
    df= pd.DataFrame(adata.obs[cluster_id])
    df["cells"]=df.index.astype("string")

    # Merge the processed motif_df together with the cell ID df
    df_obs= motif_df.merge(df, on=cluster_id,how='left')
    df_obs=df_obs.drop(columns=[cluster_id])

    # Merge the observation dataframe with anndata obs
    adata.obs["cells"]=adata.obs.index.astype("string")
    adata.obs = adata.obs.merge(df_obs, on='cells',how='left')
    adata.obs.index=adata.obs["cells"]
    adata.obs.index.name = None

    if return_df == True:
        return output
    else:
        return
        
def per_cluster_df(
    anndata,
    assay='influence',
    cluster_id='leiden_new'):
    """per_cluster_df
    This functions creates motif-factor links & export tables for printing motif score alongside its binding factor
    Params:
    ---
    anndata object
    cluster_id: ID used for finding clusters of cells
    assay: influence or maelstrom if they were added previously to the anndata object
    Usage:
    ---
    >>> from anansescanpy import per_cluster_df
    >>> cluster_df = per_cluster_df(adata)  
    """
    cluster_names = list()
    bulk_data = list()
    adata=anndata.copy()
    clusters =adata.obs[cluster_id].astype("category").unique()

    # Check if the assay observations exist in the anndata object
    columns=adata.obs.columns.tolist()
    if not any(str("_"+assay) in s for s in columns) == True:
        raise ValueError(str('assay: '+assay+' not found in the scanpy object'))    
    columns_assay = [word for word in columns if assay in word]

    for cluster in adata.obs[cluster_id].astype("category").unique():
        adata_sel = adata[adata.obs[cluster_id].isin([cluster])].copy()
        adata_sel.obs = adata_sel.obs[columns_assay]
        adata_sel.obs

        if not mean(adata_sel.obs.nunique()) <=1:
            raise ValueError(str('not all cells of the cluster '+ cluster+' have the same value in the assay '+assay))   

        # Omit clusters that have NA since they have 0 as nunique()
        if mean(adata_sel.obs.nunique()) ==1:
            cluster_names.append(str(cluster))
            bulk_data += [adata_sel.obs.values[:1][0]]

    # Generate the bulk matrix from the assay data
    bulk_df = pd.DataFrame(bulk_data)
    bulk_df.index=cluster_names
    bulk_df.columns=columns_assay

    return bulk_df