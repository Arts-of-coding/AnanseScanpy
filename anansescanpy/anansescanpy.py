"""
A collection of functions
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc


def export_CPM_scANANSE(anndata, min_cells=50, outputdir="", cluster_id="leiden_new"):
    """export_CPM_scANANSE
    This functions exports CPM values from an anndata object on the raw count sparce matrix: anndata.X
    This requires setting the raw count matrix as anndata.X

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
    adata = anndata
    adata.layers["counts"] = adata.X
    if not outputdir == "":
        os.makedirs(outputdir, exist_ok=True)
    sc.pp.normalize_total(adata, target_sum=1e6, inplace=True, layer="counts")
    rna_count_lists = list()
    FPKM_count_lists = list()
    cluster_names = list()
    i = 0

    for cluster in adata.obs[cluster_id].astype("category").unique():

        # Only use ANANSE on clusters with more than 50 cells
        n_cells = adata.obs[cluster_id].value_counts()[i]

        if n_cells > min_cells:

            print(
                str("gather data from " + cluster + " with " + str(n_cells) + " cells")
            )
            cluster_names.append(str(cluster))

            # Generate the raw count file
            adata_pseudobulk = adata[adata.obs[cluster_id].isin([cluster])]
            adata_sel = adata[adata.obs[cluster_id].isin([cluster])].copy()
            X_clone = adata_sel.X.tocsc()
            X_clone.data = np.ones(X_clone.data.shape)
            NumNonZeroElementsByColumn = X_clone.sum(0)
            rna_count_lists += [list(np.array(NumNonZeroElementsByColumn)[0])]

            # Generate the FPKM/CPM count file
            adata_pseudobulk = adata[adata.obs[cluster_id].isin([cluster])]
            adata_sel2 = adata[adata.obs[cluster_id].isin([cluster])].copy()
            del adata_sel2.layers["counts"]
            X_clone = adata_sel2.X.tocsc()
            X_clone.data = np.ones(X_clone.data.shape)
            NumNonZeroElementsByColumn = X_clone.sum(0)
            FPKM_count_lists += [list(np.array(NumNonZeroElementsByColumn)[0])]

            i = (
                i + 1
            )  # Increase i to add clusters iteratively to the cluster_names list

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


def export_ATAC_scANANSE(anndata, min_cells=50, outputdir="", cluster_id="leiden_new"):
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
    adata = anndata
    if not outputdir == "":
        os.makedirs(outputdir, exist_ok=True)
    atac_count_lists = list()
    cluster_names = list()
    i = 0

    for cluster in adata.obs[cluster_id].astype("category").unique():

        # Only use ANANSE on clusters with more than 50 cells
        n_cells = adata.obs[cluster_id].value_counts()[i]

        if n_cells > min_cells:

            print(
                str("gather data from " + cluster + " with " + str(n_cells) + " cells")
            )
            cluster_names.append(str(cluster))

            # Generate the raw count file
            adata_sel = adata[adata.obs[cluster_id].isin([cluster])].copy()
            X_clone = adata_sel.X.tocsc()
            X_clone.data = np.ones(X_clone.data.shape)
            NumNonZeroElementsByColumn = X_clone.sum(0)
            atac_count_lists += [list(np.array(NumNonZeroElementsByColumn)[0])]

            i = (
                i + 1
            )  # Increase i to add clusters iteratively to the cluster_names list

    # Generate the count matrix df
    atac_count_lists = pd.DataFrame(atac_count_lists)
    atac_count_lists = atac_count_lists.transpose()

    atac_count_lists.columns = cluster_names
    df = adata.T.to_df()
    atac_count_lists.index = df.index
    atac_count_lists["average"] = atac_count_lists.mean(axis=1)
    atac_count_lists = atac_count_lists.astype("int")

    count_file = str(outputdir + "Peak_Counts.tsv")
    atac_count_lists.to_csv(count_file, sep="\t", index=True, index_label=False)


def config_scANANSE(
    anndata,
    min_cells=50,
    outputdir="",
    cluster_id="leiden_new",
    genome="./scANANSE/data/hg38",
    additional_contrasts=None,
):
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
    i = 0
    contrast_list = list()

    # Only use ANANSE on clusters with more than 50 cells
    for cluster in adata.obs[cluster_id].astype("category").unique():
        n_cells = adata.obs[cluster_id].value_counts()[i]

        if n_cells > min_cells:
            cluster_names.append(str(cluster))
            additional_contrasts_2 = str("anansesnake_" + cluster + "_average")
            contrast_list += [additional_contrasts_2]
            i = (
                i + 1
            )  # Increase i to add clusters iteratively to the cluster_names list

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
        print("adding additional contrasts")
        for i in range(0, len(additional_contrasts)):
            additional_contrasts_2 = str("anansesnake_" + additional_contrasts[i])
            contrast_list += [additional_contrasts_2]

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
    myfile.write("result_dir: " + str(outdir).strip('"') + "\n")
    myfile.write("genome: " + str(genome).strip('"') + "\n")
    myfile.write("database: gimme.vertebrate.v5.0".strip('"') + "\n")
    myfile.write("jaccard: 0.1".strip('"') + "\n")
    myfile.write("edges: 500_000".strip('"') + "\n")
    myfile.write("padj: 0.05".strip('"') + "\n")
    myfile.write("plot_type: ".strip('"') + '"' + img + '"' + "\n")
    myfile.write("get_orthologs: false".strip('"') + "\n")
    myfile.write("contrasts:".strip('"') + "\n")

    # Adding the additional contrasts to config file
    for j in range(0, len(contrast_list)):
        contrast_string = contrast_list[j]
        print(contrast_string)
        myfile.write("\t" + "- " + '"' + contrast_string + '"' + "\n")
    myfile.close()


def DEGS_scANANSE(
    anndata,
    min_cells=50,
    outputdir="",
    cluster_id="leiden_new",
    additional_contrasts=None,
    genome_name="hg38",
):
    """DEGS_scANANSE
    Calculate the differential genes needed for ananse influence

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
    i = 0
    contrast_list = list()

    # Normalize the raw count data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["normalized"] = adata.X

    for cluster in adata.obs[cluster_id].astype("category").unique():

        # Only use ANANSE on clusters with more than the min_cells
        n_cells = adata.obs[cluster_id].value_counts()[i]

        if n_cells > min_cells:
            additional_contrasts_2 = str("anansesnake_" + cluster + "_average")
            contrast_list += [additional_contrasts_2]
            cluster_names.append(cluster)
            i = (
                i + 1
            )  # Increase i to add clusters iteratively to the cluster_names list

    # Select only cells for the average that have min_cells or more
    adata = adata[adata.obs[cluster_id].isin(cluster_names)].copy()

    # lets generate the snakemake config file
    if isinstance(additional_contrasts, list):
        print("adding additional contrasts")
        for i in range(0, len(additional_contrasts)):
            additional_contrasts_2 = str("anansesnake_" + additional_contrasts[i])
            contrast_list += [additional_contrasts_2]

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
            B = adata.var.index
            C = [l2fc, padj]

            DEGS_output = pd.DataFrame(C, columns=B)
            DEGS_output.index = A
            DEGS_output = DEGS_output.T
            DEGS_output.to_csv(DEG_file, sep="\t", index=True, index_label=False)


def import_scanpy_scANANSE(
    anndata,
    cluster_id = 'leiden_new',
    anansnake_inf_dir = 'None',
    unified_contrast='average'
):
    """export_CPM_scANANSE
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
    >>> from anansescanpy import export_CPM_scANANSE
    >>> export_CPM_scANANSE(adata)
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
    adata.obs
    
    return output
    
