"""
Collection of AnanseScanpy functions
"""
import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
from statistics import mean
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

def export_CPM_scANANSE(anndata, min_cells=50, outputdir="", cluster_id="leiden_new"):
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
    myfile.write("genome: " + str(genome).strip('"') + "\n")
    myfile.write("result_dir: " + str(outdir).strip('"') + "\n")
    myfile.write("contrasts:".strip('"') + "\n")
    # Adding the additional contrasts to config file
    for j in range(0, len(contrast_list)):
        contrast_string = contrast_list[j]
        print(contrast_string)
        myfile.write(" " + "- " + '"' + contrast_string + '"' + "\n")    
    myfile.write("database: gimme.vertebrate.v5.0".strip('"') + "\n")
    myfile.write("jaccard: 0.1".strip('"') + "\n")
    myfile.write("edges: 500_000".strip('"') + "\n")
    myfile.write("padj: 0.05".strip('"') + "\n")
    myfile.write("plot_type: ".strip('"') + '"' + img + '"' + "\n")
    myfile.write("get_orthologs: false".strip('"') + "\n")
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
            B = adata.uns["rank_genes_groups"]['names'][comparison1]
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
    

def export_ATAC_maelstrom(anndata, min_cells=50, outputdir="", 
                          cluster_id="leiden_new",select_top_rows=True,
                         n_top_rows=100000):
    """export_ATAC_maelstrom
    This functions exports normalized peak values from an anndata object 
    on the sparce peak count matrix: anndata.X required for maelstrom. 
    This requires setting the peak count matrix as anndata.X
    Params:
    ---
    anndata object
    min_cells: minimum of cells a cluster needs to be exported
    output_dir: directory where the files are outputted
    cluster_id: ID used for finding clusters of cells
    select_top_rows: only output the top variable rows, or all rows if false
    n_top_rows: amount of variable rows to export
    Usage:
    ---
    >>> from anansescanpy import export_ATAC_maelstrom
    >>> export_ATAC_maelstrom(adata)
    """
    adata = anndata.copy()
    if not outputdir == "":
        os.makedirs(outputdir, exist_ok=True)
    atac_count_lists = list()
    cluster_names = list()

    for cluster in adata.obs[cluster_id].astype("category").unique():

        # Only use appending to df on clusters with more than minimal amount of cells
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
    
    # Normalize the raw counts
    activity_matrix= atac_count_lists
    activity_matrix=np.log2(activity_matrix+1)
    
    # Import the scaler function from sklearn and scale
    sc = StandardScaler()
    sc.fit(activity_matrix)
    sc.scale_ = np.std(activity_matrix, axis=0, ddof=1).to_list()
    activity_matrix=sc.transform(activity_matrix)
    activity_matrix=pd.DataFrame(activity_matrix)
    activity_matrix.columns = atac_count_lists.columns
    activity_matrix.index = atac_count_lists.index
    
    # Select the n_top_rows with the highest row variance
    if np.shape(activity_matrix)[0]>n_top_rows:
        print(
            str("large dataframe detected, selecting top variable rows n = " + str(n_top_rows))
            )
        print("if entire dataframe is required, add select_top_rows = False as a parameter")
        print("or change ammount of rows via the n_top_rows parameter")
        activity_matrix["rowvar"]=activity_matrix.var(axis='columns')
        activity_matrix=activity_matrix.sort_values(by='rowvar', ascending=False)
        activity_matrix=activity_matrix.head(n_top_rows)
        activity_matrix=activity_matrix.drop(['rowvar'], axis=1)
        
    # Save the activity matrix and peak_file to the output dir
    activity_file = str(outputdir + "Peaks_scaled.tsv")
    activity_matrix.to_csv(activity_file, sep="\t", index=True, index_label=False)
    

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


def Maelstrom_Motif2TF(
    anndata,
    mot_mat=None,
    m2f_df=None,
    cluster_id="leiden_new",
    maelstrom_dir = 'maelstrom/',
    combine_motifs = 'means',
    expr_tresh = 10,
    cor_tresh = 0.30,
    curated_motifs = False,
    cor_method = "pearson",
    save_output= False,
    outputdir=""):
    """Maelstrom_Motif2TF
    This functions creates motif-factor links & export tables for printing motif score alongside its binding factor.
    The anndata object requires raw expression data in either the anndata.X slot or the anndata.raw.X slot for the correlation.
    Params:
    ---
    anndata object
    cluster_id: ID used for finding clusters of cells
    mot_mat: motif matrix, for instance output from per_cluster_df with the maelstrom option
    m2f_df: motif to factors dataframe for instance from gimmemotifs containing Motif, Factor and Curated columns
    maelstrom_dir: directory where maelstrom output is located
    combine_motifs: combining motifs to individual transcription factors can be done by calculating the means of all linked motifs,
    by selecting the most correlating motif with gene expression and the most variable motif across input data
    curated_motifs: option to only select curated motifs
    cor_method: pearson or spearman
    save_output: save the output of the correlations
    outputdir: the output directory where data of scaling is stored
    Usage:
    ---
    >>> from anansescanpy import Maelstrom_Motif2TF
    >>> adata = Maelstrom_Motif2TF(adata)  
    """
    adata=anndata.copy()
    pd.options.mode.chained_assignment = None # set SettingWithCopyWarning to None
    
    # Check if m2f_df object provided contains the right columns
    fields = ['Motif', 'Factor']
    try:
        maelstrom_df = pd.read_csv(str(maelstrom_dir + "nonredundant.motifs.motif2factors.txt"),sep="\t", comment='#',
                                   skipinitialspace=True, usecols=fields)
    except:
        print("Provide m2f_df with at least 2 columns with names Motif and Factor.")


    # Check when Curated is True if there is a Curated column

    if curated_motifs == True:
        fields = ['Motif', 'Factor', 'Curated']
        try:
            m2f_df = pd.read_csv(str(maelstrom_dir + "nonredundant.motifs.motif2factors.txt"),
                                 sep="\t", comment='#',
                                 skipinitialspace=True, usecols=fields)
            print('using only curated motifs from database')
            m2f_df=m2f_df[m2f_df['Curated'].isin(['Y'])]

        except:
            print("Provide m2f_df with at least 3 columns with names Motif, Factor and Curated.")
    else:
        m2f_df = maelstrom_df

    ## Load needed objects
    if mot_mat is None:
        print(str('loading maelstrom values from maelstrom assay using the cluster identifier '+cluster_id))
        mot_mat = per_cluster_df(anndata,
                                 assay = 'maelstrom',
                                 cluster_id = cluster_id)
    
    # Set the raw data as all genes and reimplement them to rawdata if needed
    if adata.raw is not None:
        adata=adata.raw.to_adata()
        adata.raw=adata
    
    res = pd.DataFrame(columns=adata.var_names, index=adata.obs[cluster_id].astype("category").unique())                                                                                          

    ## Set up scanpy object based on expression treshold
    for clust in adata.obs[cluster_id].astype("category").unique():
        if adata.raw is not None:
            res.loc[clust] = adata[adata.obs[cluster_id].isin([clust]),:].raw.X.mean(0)
        else:
            res.loc[clust] = adata[adata.obs[cluster_id].isin([clust]),:].X.mean(0)
    res.loc["sum"]=np.sum(res,axis=0).tolist()
    res=res.transpose()
    res=res.loc[res['sum'] > expr_tresh]
    genes_expressed = res.index.tolist()
    adata_sel = adata[:, genes_expressed].copy()

    ## Select motifs with binding TFs present in object
    m2f_df = m2f_df[m2f_df["Factor"].isin(genes_expressed)]

    print(str("Seurat NormalizeData with default settings will be run on all the genes"))
    sc.pp.normalize_total(adata_sel,inplace=True)

    ## Generate the df with mean normalized expression
    exp_mat = pd.DataFrame(columns=adata.var_names, index=adata.obs[cluster_id].astype("category").unique())                                                                                          

    for clust in adata.obs[cluster_id].astype("category").unique(): 
        exp_mat.loc[clust] = adata[adata.obs[cluster_id].isin([clust]),:].X.mean(0)

    ## make sure that all genes in matrix have mean expression > 0
    exp_mat.loc["sum"]=np.sum(exp_mat,axis=0).tolist()
    exp_mat=exp_mat.transpose()
    exp_mat=exp_mat.loc[exp_mat['sum'] > 0]

    ## Select the same exp_mat columns as in mot_mat columns (if the grouping var is the same)
    exp_mat=exp_mat[mot_mat.index.tolist()]
    exp_mat=exp_mat.transpose()

    ## limit table to motifs and TFs present in dataset
    mot_mat.columns=mot_mat.columns.str.removesuffix("_maelstrom")
    mot_mat = mot_mat[m2f_df["Motif"].unique().tolist()]
    TF_mat = exp_mat[m2f_df["Factor"].unique().tolist()]

    # Correlating the expression data and motif data
    m2f_df_match = m2f_df
    cor_list=list()
    for i in range(len(m2f_df_match.index)):
        mot=m2f_df_match["Motif"].tolist()[i]
        tf=m2f_df_match["Factor"].tolist()[i]
        
        if cor_method == "pearson":
            cor_list.append(pearsonr(mot_mat[mot],exp_mat[tf])[0])
            
        if cor_method == "spearman":
            cor_list.append(spearmanr(mot_mat[mot],exp_mat[tf])[0])
            
    m2f_df_match["cor"] = cor_list
    
    # Calculate the variance of the motifs
    var_list=mot_mat.var(axis=0)
    var_list=var_list.to_frame(name="var")
    var_list["Motif"]=var_list.index
    m2f_df_match = m2f_df_match.merge(var_list, on='Motif',how='left')

    ## Only keep motif-TF combinations with an absolute R higher than treshold
    print(str("Only keep motif-TF combinations with an R > "+str(cor_tresh)))
    m2f_df_match=m2f_df_match.loc[abs(m2f_df_match['cor']) > cor_tresh]

    # Select highest absolute correlation of TF and motif
    m2f_df_match["abscor"]=abs(m2f_df_match["cor"])
    m2f_df_unique=m2f_df_match.groupby(["abscor"], as_index=False).max()
    print(str('total length m2f_df_unique '+ str(len(m2f_df_unique.index))))

    # Select only positive correlations or only negative correlations (repressors)
    for typeTF in ['TFcor','TFanticor']:
        m2f = m2f_df_unique
        if typeTF == 'TFanticor':
            print("Selecting anticorrelating TFs")
            m2f= m2f_df_unique.loc[m2f_df_unique['cor'] < 0]
            print(str('total m2f: '+ str(len(m2f.index))))
        else:
            print("Selecting correlating TFs")
            m2f= m2f_df_unique.loc[m2f_df_unique['cor'] > 0]
            print(str('total m2f: '+ str(len(m2f.index))))

        ## Order motifs according to m2f
        mot_plot = mot_mat[m2f["Motif"]]
        mot_plot = mot_plot.transpose()
        
        ## Replace motif name by TF name
        mot_plot.index= m2f["Factor"]
        
        # Extract metadata of motif and factors
        metadata=m2f
        m2f=m2f.set_index(mot_plot.index)
        
        # Make motif score per TF (selecting most variable motif per TF or make mean of all motifs associated)
        # Take mean of motifs linked to the same TF
        if combine_motifs == 'means':
            print("Take mean motif score of all binding motifs per TF")
            mot_plot=mot_plot.groupby(["Factor"], as_index=True).mean()
            
        # Take the highest correlating   
        if combine_motifs == 'max_cor':
            print("Motif best (absolute)correlated to expression is selected per TF")
            mot_plot["cor"] = m2f["cor"]
            idx = mot_plot.groupby(['Factor'])['cor'].transform(max) == mot_plot['cor']
            mot_plot=mot_plot[idx]
            mot_plot=mot_plot.drop(columns=["cor"])
        
        # Take the highest variable motif
        if combine_motifs == 'max_var':
            print("Motif best (absolute)correlated to expression is selected per TF")
            mot_plot["var"] = m2f["var"]
            idx = mot_plot.groupby(['Factor'])['var'].transform(max) == mot_plot['var']
            mot_plot=mot_plot[idx]
            mot_plot=mot_plot.drop(columns=["var"])
            
        # order expression matrix and motif matrix the same way
        mot_plot = mot_plot.transpose()
        exp_plot = TF_mat[mot_plot.columns.tolist()]

        # Import the scaler function from sklearn and scale
        exp_plot_scale=exp_plot
        mot_plot_scale=mot_plot

        scs = StandardScaler()
        scs.fit(exp_plot_scale)
        scs.scale_ = np.std(exp_plot_scale, axis=0, ddof=1).to_list()
        exp_plot_scale=scs.transform(exp_plot_scale)
        exp_plot_scale=pd.DataFrame(exp_plot_scale)
        exp_plot_scale.columns = exp_plot.columns
        exp_plot_scale.index = exp_plot.index

        scs.fit(mot_plot_scale)
        scs.scale_ = np.std(mot_plot_scale, axis=0, ddof=1).to_list()
        mot_plot_scale=scs.transform(mot_plot_scale)
        mot_plot_scale=pd.DataFrame(mot_plot_scale)
        mot_plot_scale.columns = mot_plot.columns
        mot_plot_scale.index = mot_plot.index
        
        if save_output==True:
            expression_file = str(outputdir+typeTF+"_expression_means_scaled.tsv")
            exp_plot_scale.to_csv(expression_file, sep="\t", index=True, index_label=False)

            motif_file = str(outputdir+typeTF+"_motif_intensities_scaled.tsv")
            mot_plot_scale.to_csv(motif_file, sep="\t", index=True, index_label=False)

        # Generate a scaled TF motif score dataframe and process the appended_data for anndata
        mot_plot_scale= mot_plot_scale.add_suffix(str('_'+typeTF+'_score'))
        mot_plot_scale[cluster_id]= mot_plot_scale.index

        # Retrieve the cluster IDs and cell IDs from the anndata object
        df= pd.DataFrame(adata.obs[cluster_id])
        df["cells"]=df.index.astype("string")

        # Merge the processed motif_df together with the cell ID df
        df_obs= mot_plot_scale.merge(df, on=cluster_id,how='left')
        df_obs=df_obs.drop(columns=[cluster_id])

        # Merge the observation dataframe with anndata obs
        adata.obs["cells"]=adata.obs.index.astype("string")
        adata.obs = adata.obs.merge(df_obs, on='cells',how='left')
        adata.obs.index=adata.obs["cells"]
        adata.obs.index.name = None
        
        # Generate a scaled TF expression score dataframe and process the appended_data for anndata
        exp_plot_scale= exp_plot_scale.add_suffix(str('_'+typeTF+'_expression_score'))
        exp_plot_scale[cluster_id]= exp_plot_scale.index

        # Retrieve the cluster IDs and cell IDs from the anndata object
        df= pd.DataFrame(adata.obs[cluster_id])
        df["cells"]=df.index.astype("string")

        # Merge the processed motif_df together with the cell ID df
        df_obs= exp_plot_scale.merge(df, on=cluster_id,how='left')
        df_obs=df_obs.drop(columns=[cluster_id])

        # Merge the observation dataframe with anndata obs
        adata.obs["cells"]=adata.obs.index.astype("string")
        adata.obs = adata.obs.merge(df_obs, on='cells',how='left')
        adata.obs.index=adata.obs["cells"]
        adata.obs.index.name = None
        
        # Add the metadata of motifs to factors in a dataframe in uns
        adata.uns[str(typeTF+"_"+combine_motifs)]=metadata
        
    return adata


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

