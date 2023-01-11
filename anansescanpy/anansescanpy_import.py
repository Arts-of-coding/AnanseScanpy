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
from .anansescanpy_export import add_contrasts

def config_scANANSE(anndata,min_cells=50,outputdir="",cluster_id="scanpy_cluster",genome="./scANANSE/data/hg38",additional_contrasts=None):
    """config_scANANSE
    This functions generates a sample file and config file for running Anansnake based on the anndata object

    Params:
    ---
    anndata object
    min_cells: minimum of cells a cluster needs to be exported
    output_dir: directory where the files are outputted
    cluster_id: ID used for finding clusters of cells
    genome: genomepy name or location of the genome fastq file
    additional_contrasts: additional contrasts to add between clusters within cluster_ID
    Usage:
    ---
    >>> from anansescanpy import config_scANANSE
    >>> config_scANANSE(adata)
    """
    adata = anndata
    if not outputdir == "":
        os.makedirs(outputdir, exist_ok=True)
    cluster_names = list()
    contrast_list = list()

    # Only use ANANSE on clusters with more than minimal amount of cells
    for cluster in adata.obs[cluster_id].astype("category").unique():
        n_cells = adata.obs[cluster_id].value_counts()[cluster]

        if n_cells > min_cells:
            cluster_names.append(str(cluster))
            additional_contrasts_2 = str("anansesnake_" + cluster + "_average")
            contrast_list += [additional_contrasts_2]

    # lets generate the snakemake sample file
    cluster_names_contrast = cluster_names
    cluster_names.append("average")
    sample_file_df = pd.DataFrame(cluster_names, columns=["sample"])
    sample_file_df.index = cluster_names
    sample_file_df["assembly"] = os.path.basename(genome)
    sample_file_df["anansesnake"] = sample_file_df["sample"]
    sample_file_location = str(outputdir + "samplefile.tsv")
    sample_file_df.to_csv(sample_file_location, sep="\t", index=False)

    # lets generate the snakemake config file
    if isinstance(additional_contrasts, list):
        contrast_list=add_contrasts(contrast_list,additional_contrasts)

    # Retrieve full path from current working directory
    if outputdir == "":
        outdir = os.getcwd()
        outdir = outdir + "/"
    else:
        outdir = outputdir
    file = str(outputdir + "config.yaml")

    # Specify the absolute paths
    Peak_file = str(outdir + "Peak_Counts.tsv")
    count_file = str(outdir + "RNA_Counts.tsv")
    CPM_file = str(outdir + "TPM.tsv")
    genome = os.path.basename(genome)
    sample_file_location = str(outdir + "samplefile.tsv")
    img = "png"

    # Write to config file
    myfile = open(file, "w")
    myfile.write("rna_samples: " + str(sample_file_location).strip('"') + "\n")
    myfile.write("rna_tpms: " + str(CPM_file).strip('"') + "\n")
    myfile.write("rna_counts: " + str(count_file).strip('"') + "\n")
    myfile.write("atac_samples: " + str(sample_file_location).strip('"') + "\n")
    myfile.write("atac_counts: " + str(Peak_file).strip('"') + "\n")
    myfile.write("genome: " + str(genome).strip('"') + "\n")
    myfile.write("result_dir: " + str(outdir).strip('"') + "\n")
    myfile.write("contrasts:".strip('"') + "\n")
    # Adding the additional contrasts to config file
    for j in range(0, len(contrast_list)):
        contrast_string = contrast_list[j]
        print(contrast_string)
        myfile.write(" " + "- " + '"' + contrast_string + '"' + "\n")    
    myfile.write("database: gimme.vertebrate.v5.0".strip('"') + "\n")
    for config_parameter in ["jaccard: 0.1","edges: 500_000","padj: 0.05","get_orthologs: false"]:
        myfile.write(str(config_parameter).strip('"') + "\n")
    myfile.write("plot_type: ".strip('"') + '"' + img + '"' + "\n")
    myfile.close()


def import_scanpy_scANANSE(
    anndata,
    cluster_id = 'scanpy_cluster',
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
    cluster_id="scanpy_cluster",
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
    cluster_id='scanpy_cluster'):
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
