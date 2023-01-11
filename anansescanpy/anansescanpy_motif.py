"""
Collection of AnanseScanpy motif functions
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
from .anansescanpy_import import per_cluster_df

def Maelstrom_Motif2TF(
    anndata,
    mot_mat=None,
    m2f_df=None,
    cluster_id="scanpy_cluster",
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
    if curated_motifs == True:
        fields = ['Motif', 'Factor', 'Curated']
    else:
        fields = ['Motif', 'Factor']
    try:
        maelstrom_df = pd.read_csv(str(maelstrom_dir + "nonredundant.motifs.motif2factors.txt"),sep="\t", comment='#',
                                   skipinitialspace=True, usecols=fields)
    except:
        print("Provide m2f_df with at least two columns containing names Motif and Factor.")

    # Check when Curated is True if there is a Curated column

    if curated_motifs == True:
        fields = ['Motif', 'Factor', 'Curated']
        try:
            m2f_df = maelstrom_df
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
        adata.var_names=list(adata.var["features"].values)
        rawnames=list(adata.var["features"].values)
        res = pd.DataFrame(columns=rawnames, index=adata.obs[cluster_id].astype("category").unique())
    else:
        res = pd.DataFrame(columns=adata.var_names.tolist(), index=adata.obs[cluster_id].astype("category").unique())
    
    ## Set up scanpy object based on expression treshold
    for clust in adata.obs[cluster_id].astype("category").unique():
        if adata.raw is not None:
            res.loc[clust] = adata[adata.obs[cluster_id].isin([clust]),:].raw.X.sum(0)
        else:
            res.loc[clust] = adata[adata.obs[cluster_id].isin([clust]),:].X.sum(0)
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
    exp_mat = pd.DataFrame(columns=adata_sel.var_names.tolist(), index=adata_sel.obs[cluster_id].astype("category").unique())                                                                                          
    for clust in adata_sel.obs[cluster_id].astype("category").unique(): 
        exp_mat.loc[clust] = adata_sel[adata_sel.obs[cluster_id].isin([clust]),:].X.mean(0)

    ## Ensure that all genes in matrix have mean expression > 0
    exp_mat.loc["sum"]=np.sum(exp_mat,axis=0).tolist()
    exp_mat_transposed=exp_mat.transpose()
    exp_mat_transposed=exp_mat_transposed.loc[exp_mat_transposed['sum'] > 0]

    ## Select the same exp_mat columns as in mot_mat columns (if the grouping var is the same)
    exp_mat_transposed=exp_mat_transposed[mot_mat.index.tolist()]
    exp_mat=exp_mat_transposed.transpose()

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
            cor_test = pearsonr(mot_mat[mot],exp_mat[tf])
            
        elif cor_method == "spearman":
            cor_test = spearmanr(mot_mat[mot],exp_mat[tf])
            
        cor_list.append(cor_test[0])
            
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
        #m2f = m2f_df_unique
        if typeTF == 'TFanticor':
            print("Selecting anticorrelating TFs")
            m2f= m2f_df_unique.loc[m2f_df_unique['cor'] < 0]

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
            
        if combine_motifs == 'max_cor' or combine_motifs == 'max_var':
            mot_plot = mot_plot.assign(var=m2f["var"],abscor=m2f["abscor"],TF=m2f["Factor"],Motif=m2f["Motif"])
            
            # Take the highest correlating motif
            if combine_motifs == 'max_cor':
                print("Motif best (absolute)correlated to expression is selected per TF")
                sorted_value="abscor"
                
            # Take the highest variable motif    
            if combine_motifs == 'max_var':
                print("The highest variable motif associated is selected per TF")
                sorted_value="var"

            mot_plot = mot_plot.sort_values(by=sorted_value, ascending=False)
            mot_plot = mot_plot.drop_duplicates('Motif', keep='first')
            mot_plot = mot_plot.drop_duplicates('TF', keep='first')
            mot_plot = mot_plot.drop(columns=["var","abscor","TF","Motif"])
            
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
            for i in ["expression","motif"]:
                file = str(outputdir+typeTF+"_"+str(i)+"_"+combine_motifs+"_scaled.tsv")
                if i == "expression":
                    exp_plot_scale.to_csv(file, sep="\t", index=True, index_label=False)
                if i == "motif":
                    mot_plot_scale.to_csv(file, sep="\t", index=True, index_label=False)

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


def Factor_Motif_Plot(
    anndata,
    factor_list,
    assay_maelstrom = 'TFanticor',
    combine_motifs='max_cor',
    logo_dir = 'maelstrom/logos/'):
    """Factor_Motif_Plot2
    This functions plots expression, the associated motif on the UMAP and the motif itself
    ---
    anndata object
    factor_list: list of transcription factors in the assay of interest
    assay_maelstrom: either TFanticor (default) or TFcor
    combine_motifs: max_cor (default) or max_var (does not work for means) see Motif2Factors function
    logo_dir: the directory where logos of the maelstrom output are located
    Usage:
    ---
    >>> from anansescanpy import Factor_Motif_Plot
    >>> Factor_Motif_Plot(adata,factor_list=factors)  
    """

    adata=anndata
    TF_list=factor_list
    
    for i in TF_list:
        if combine_motifs == "max_var":
            subset_parameter = "var"
            
        if combine_motifs == "max_cor":
            subset_parameter = "cor"
            
        subset=adata.uns[str(assay_maelstrom+"_"+str(combine_motifs))][adata.uns[str(assay_maelstrom+"_"+str(combine_motifs))]["Factor"]==i]
        subset_max=subset[subset[subset_parameter]==subset[subset_parameter].max()]
        title=subset_max["Motif"].tolist()[0]
        title = title.replace('.', '_')

        # Plot figure together
        image = plt.imread(str(logo_dir+title+'.png'))
        fig, axes = plt.subplots(1,3, figsize=(16,4))

        axes[2].axis('off')
        axes[2].imshow(image)

        sc.pl.umap(adata,show=False, color=[str(i)], cmap="Blues", ax=axes[0])
        sc.pl.umap(adata,show=False, color=[str(i+"_"+assay_maelstrom+"_score")], cmap="RdGy",ax=axes[1], title=title)

    return
