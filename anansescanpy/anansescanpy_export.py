"""
Collection of AnanseScanpy functions
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

def add_contrasts(condition_list,additional_contrasts):        
    print("adding additional contrasts")
    for i in range(0, len(additional_contrasts)):
        additional_contrasts_2 = str("anansesnake_" + additional_contrasts[i])
        condition_list += [additional_contrasts_2]
        
    return condition_list

def export_CPM_scANANSE(anndata, min_cells=50, outputdir="", cluster_id="scanpy_cluster"):
    """export_CPM_scANANSE
    This functions exports CPM values from an anndata object on the raw count sparce matrix: anndata.X
    This requires having the raw count matrix in anndata.X or adata.raw.X.
    Params:
    ---
    anndata object
    min_cells: minimum of cells a cluster needs to be exported
    output_dir: directory where the files are outputted
    cluster_id: ID used for finding clusters of cells
    Usage:
    ---
    >>> from anansescanpy import export_CPM_scANANSE
    >>> export_CPM_scANANSE(adata)
    """
    adata = anndata.copy()
    
    # Set the raw data as all genes and reimplement them to rawdata if needed
    if adata.raw is not None:
        adata=adata.raw.to_adata()
        adata.var_names=adata.var["_index"].tolist()
        adata.raw=adata
    else:
        adata.raw=adata
    
    if not outputdir == "":
        os.makedirs(outputdir, exist_ok=True)
    rna_count_lists = list()
    FPKM_count_lists = list()
    cluster_names = list()

    for cluster in adata.obs[cluster_id].astype("category").unique():

        # Only use ANANSE on clusters with more than minimal amount of cells
        n_cells = adata.obs[cluster_id].value_counts()[cluster]

        if n_cells > min_cells:
            cluster_names.append(str(cluster))

            # Generate the raw count file
            adata_sel = adata[adata.obs[cluster_id].isin([cluster])].copy()
            adata_sel.raw=adata_sel
            
            print(
                str("gather data from " + cluster + " with " + str(n_cells) + " cells")
            )
            
            X_clone = adata_sel.X.tocsc()
            X_clone.data = np.ones(X_clone.data.shape)
            NumNonZeroElementsByColumn = X_clone.sum(0)
            rna_count_lists += [list(np.array(NumNonZeroElementsByColumn)[0])]
            sc.pp.normalize_total(adata_sel, target_sum=1e6, inplace=True)
            X_clone2=adata_sel.X.toarray()
            NumNonZeroElementsByColumn = [X_clone2.sum(0)]
            FPKM_count_lists += [list(np.array(NumNonZeroElementsByColumn)[0])]

    # Specify the df.index
    df = adata.T.to_df()

    # Generate the count matrix df
    rna_count_lists = pd.DataFrame(rna_count_lists)
    rna_count_lists = rna_count_lists.transpose()
    rna_count_lists.columns = cluster_names
    rna_count_lists.index = df.index
    rna_count_lists["average"] = rna_count_lists.mean(axis=1)
    rna_count_lists = rna_count_lists.astype("int")

    # Generate the FPKM matrix df
    FPKM_count_lists = pd.DataFrame(FPKM_count_lists)
    FPKM_count_lists = FPKM_count_lists.transpose()
    FPKM_count_lists.columns = cluster_names
    FPKM_count_lists.index = df.index
    FPKM_count_lists["average"] = FPKM_count_lists.mean(axis=1)
    FPKM_count_lists = FPKM_count_lists.astype("int")

    count_file = str(outputdir + "RNA_Counts.tsv")
    CPM_file = str(outputdir + "TPM.tsv")
    rna_count_lists.to_csv(count_file, sep="\t", index=True, index_label=False)
    FPKM_count_lists.to_csv(CPM_file, sep="\t", index=True, index_label=False)


def export_ATAC_scANANSE(anndata, min_cells=50, outputdir="", cluster_id="scanpy_cluster"):
    """export_ATAC_scANANSE
    This functions exports peak values from an anndata object on the sparce peak count matrix: anndata.X.
    This requires setting the peak count matrix as anndata.X
    Params:
    ---
    anndata object
    min_cells: minimum of cells a cluster needs to be exported
    output_dir: directory where the files are outputted
    cluster_id: ID used for finding clusters of cells
    Usage:
    ---
    >>> from anansescanpy import export_ATAC_scANANSE
    >>> export_ATAC_scANANSE(adata)
    """
    adata = anndata.copy()
    if not outputdir == "":
        os.makedirs(outputdir, exist_ok=True)
    atac_count_lists = list()
    cluster_names = list()

    for cluster in adata.obs[cluster_id].astype("category").unique():

        # Only use ANANSE on clusters with more than minimal amount of cells
        n_cells = adata.obs[cluster_id].value_counts()[cluster]

        if n_cells > min_cells:
            cluster_names.append(str(cluster))

            # Generate the raw count file
            adata_sel = adata[adata.obs[cluster_id].isin([cluster])].copy()
            adata_sel.raw=adata_sel
            
            print(
                str("gather data from " + cluster + " with " + str(n_cells) + " cells")
            )
            
            X_clone = adata_sel.X.tocsc()
            X_clone.data = np.ones(X_clone.data.shape)
            NumNonZeroElementsByColumn = X_clone.sum(0)
            atac_count_lists += [list(np.array(NumNonZeroElementsByColumn)[0])]

    # Generate the count matrix df
    atac_count_lists = pd.DataFrame(atac_count_lists)
    atac_count_lists = atac_count_lists.transpose()
    atac_count_lists.columns = cluster_names
    df = adata.T.to_df()

    # Format the chromosome loci if supplied with "-"
    if not df.index.str.contains(":").any():
        df.index = df.index.str.replace("-", ":", n=1, case=None, flags=0, regex=None)

    atac_count_lists.index = df.index
    atac_count_lists["average"] = atac_count_lists.mean(axis=1)
    atac_count_lists = atac_count_lists.astype("int")

    count_file = str(outputdir + "Peak_Counts.tsv")
    atac_count_lists.to_csv(count_file, sep="\t", index=True, index_label=False)


def DEGS_scANANSE(anndata,min_cells=50,outputdir="",cluster_id="scanpy_cluster",additional_contrasts=None,genome_name="hg38"):
    """DEGS_scANANSE
    Calculate the differential genes needed for ANANSE influence.

    Params:
    ---
    anndata object
    min_cells: minimum of cells a cluster needs to be exported
    output_dir: directory where the files are outputted
    cluster_id: ID used for finding clusters of cells
    additional_contrasts: additional contrasts to add between clusters within cluster_ID
    genome_name: the genome used for mapping
    Usage:
    ---
    >>> from anansescanpy import DEGS_scANANSE
    >>> DEGS_scANANSE(adata)
    """
    adata = anndata
    os.makedirs(outputdir + "/deseq2/", exist_ok=True)
    adata.obs[cluster_id] = adata.obs[cluster_id].astype("category")
    cluster_names = list()
    contrast_list = list()

    # Normalize the raw count data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["normalized"] = adata.X

    for cluster in adata.obs[cluster_id].astype("category").unique():

        # Only use ANANSE on clusters with more than minimal amount of cells
        n_cells = adata.obs[cluster_id].value_counts()[cluster]

        if n_cells > min_cells:
            additional_contrasts_2 = str("anansesnake_" + cluster + "_average")
            contrast_list += [additional_contrasts_2]
            cluster_names.append(cluster)

    # Select only cells for the average that have the minimal amount of cells or more
    adata = adata[adata.obs[cluster_id].isin(cluster_names)].copy()

    # lets generate the snakemake config file
    if isinstance(additional_contrasts, list):
        contrast_list=add_contrasts(contrast_list,additional_contrasts)

    for j in range(0, len(contrast_list)):
        print("calculating DEGS for contrast " + contrast_list[j])
        contrast_string = contrast_list[j]
        comparison1 = contrast_string.split("_")[1]
        comparison2 = contrast_string.split("_")[2]

        DEG_file = str(
            outputdir
            + "deseq2/"
            + genome_name
            + "-anansesnake_"
            + comparison1
            + "_"
            + comparison2
            + ".diffexp.tsv"
        )

        if os.path.exists(DEG_file):
            print("skip")
        else:
            if comparison2 == "average":
                DEGS = sc.tl.rank_genes_groups(
                    adata,
                    cluster_id,
                    method="wilcoxon",
                    layer="normalized",
                    use_raw=False,
                )
            else:
                DEGS = sc.tl.rank_genes_groups(
                    adata,
                    cluster_id,
                    reference=comparison2,
                    method="wilcoxon",
                    layer="normalized",
                    use_raw=False,
                )
            l2fc = adata.uns["rank_genes_groups"]["logfoldchanges"][comparison1]
            padj = adata.uns["rank_genes_groups"]["pvals_adj"][comparison1]
            A = ["log2FoldChange", "padj"]
            B = adata.uns["rank_genes_groups"]['names'][comparison1]
            C = [l2fc, padj]

            DEGS_output = pd.DataFrame(C, columns=B)
            DEGS_output.index = A
            DEGS_output = DEGS_output.T
            DEGS_output.to_csv(DEG_file, sep="\t", index=True, index_label=False)


