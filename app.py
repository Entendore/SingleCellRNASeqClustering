import streamlit as st
import scanpy as sc
import anndata
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import io
import zipfile

# Optional: suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

st.set_option('deprecation.showPyplotGlobalUse', False)

# --- Marker genes dictionary for annotation ---
CELL_TYPE_MARKERS = {
    "T cells": ["CD3D", "CD3E", "CD2"],
    "B cells": ["MS4A1", "CD79A"],
    "Monocytes": ["LYZ", "CD14", "FCGR3A"],
    "NK cells": ["GNLY", "NKG7"],
    "Dendritic cells": ["FCER1A", "CST3"],
    "Platelets": ["PPBP"]
}

@st.cache(allow_output_mutation=True)
def load_sample_data():
    return sc.datasets.pbmc3k()

def preprocess(adata, n_pcs=20, leiden_res=0.5, batch_key=None, batch_correction=None):
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    adata = adata[adata.obs.n_genes_by_counts > 200, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)

    adata = adata[:, adata.var.highly_variable]

    # Batch correction
    if batch_key and batch_correction and batch_key in adata.obs.columns:
        if batch_correction == "Combat":
            sc.pp.combat(adata, key=batch_key)
        elif batch_correction == "Harmony":
            import harmonypy as hm
            sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)
            ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, batch_key)
            adata.obsm['X_pca_harmony'] = ho.Z_corr.T
            adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
        else:
            st.warning(f"Unknown batch correction: {batch_correction}")

    if batch_correction != "Harmony":
        sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)

    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=n_pcs)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=leiden_res)

    # Save raw data for visualization if not present
    if adata.raw is None:
        adata.raw = adata

    return adata

def plot_umap(adata, color_by, raw_expr=False):
    fig, ax = plt.subplots(figsize=(8,6))
    if raw_expr and color_by in adata.var_names:
        sc.pl.umap(adata, color=color_by, use_raw=True, ax=ax, show=False)
    else:
        sc.pl.umap(adata, color=color_by, ax=ax, show=False)
    st.pyplot(fig)
    return fig

def plot_violin(adata, genes, groupby='leiden'):
    fig, ax = plt.subplots(figsize=(10, 6))
    sc.pl.violin(adata, keys=genes, groupby=groupby, ax=ax, show=False, rotation=90)
    st.pyplot(fig)
    return fig

def plot_dotplot(adata, genes, groupby='leiden'):
    fig, ax = plt.subplots(figsize=(10, 6))
    sc.pl.dotplot(adata, var_names=genes, groupby=groupby, ax=ax, show=False)
    st.pyplot(fig)
    return fig

def plot_heatmap(adata, genes, groupby='leiden'):
    fig, ax = plt.subplots(figsize=(10, max(len(genes)*0.4+2, 4)))
    sc.pl.heatmap(adata, var_names=genes, groupby=groupby, ax=ax, show=False)
    st.pyplot(fig)
    return fig

def annotate_clusters(adata, top_n=3):
    if 'rank_genes_groups' not in adata.uns:
        sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')

    markers_df = sc.get.rank_genes_groups_df(adata, group=None)
    cluster_ids = adata.obs['leiden'].cat.categories

    cluster_annotations = {}
    for clust in cluster_ids:
        top_genes = markers_df[markers_df['group'] == clust].head(top_n)['names'].values
        matched_types = []
        for cell_type, markers in CELL_TYPE_MARKERS.items():
            if any(g in markers for g in top_genes):
                matched_types.append(cell_type)
        cluster_annotations[clust] = matched_types if matched_types else ["Unknown"]
    return cluster_annotations

def load_10x_from_folder(folder):
    adata = sc.read_10x_mtx(folder, var_names='gene_symbols', cache=True)
    return adata

def load_loom(file_path):
    adata = sc.read_loom(file_path)
    return adata

def load_csv(file_path):
    df = pd.read_csv(file_path, index_col=0)
    adata = anndata.AnnData(df.T)
    return adata

# Sidebar inputs
st.sidebar.header("Input Data")
input_type = st.sidebar.radio("Select input type", ["Sample dataset (PBMC)", "10X folder (.mtx)", ".loom file", ".csv expression matrix"])

st.sidebar.header("Clustering Parameters")
leiden_res = st.sidebar.slider("Leiden clustering resolution", 0.1, 2.0, 0.5, 0.1)
n_pcs = st.sidebar.slider("Number of PCs", 5, 50, 20, 1)
show_raw = st.sidebar.checkbox("Show raw expression on UMAP", False)

st.sidebar.header("Batch Correction")
batch_key = st.sidebar.text_input("Batch column name (optional)")
batch_correction = st.sidebar.selectbox("Batch correction method", options=["None", "Combat", "Harmony"])

adata = None

if input_type == "Sample dataset (PBMC)":
    st.sidebar.info("Using sample PBMC dataset")
    adata = load_sample_data()

elif input_type == "10X folder (.mtx)":
    uploaded_zip = st.sidebar.file_uploader("Upload zipped 10X folder", type=["zip"])
    if uploaded_zip:
        with tempfile.TemporaryDirectory() as tmpdirname:
            with zipfile.ZipFile(uploaded_zip) as zip_ref:
                zip_ref.extractall(tmpdirname)
            adata = load_10x_from_folder(tmpdirname)
            st.sidebar.success("10X data loaded!")

elif input_type == ".loom file":
    uploaded_loom = st.sidebar.file_uploader("Upload .loom file", type=["loom"])
    if uploaded_loom:
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(uploaded_loom.read())
            tmpfile.flush()
            adata = load_loom(tmpfile.name)
        st.sidebar.success(".loom file loaded!")

elif input_type == ".csv expression matrix":
    uploaded_csv = st.sidebar.file_uploader("Upload .csv file", type=["csv"])
    if uploaded_csv:
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(uploaded_csv.read())
            tmpfile.flush()
            adata = load_csv(tmpfile.name)
        st.sidebar.success(".csv file loaded!")

if adata is not None:
    st.sidebar.write("Preprocessing data ...")
    adata = preprocess(adata, n_pcs=n_pcs, leiden_res=leiden_res,
                       batch_key=batch_key if batch_key else None,
                       batch_correction=batch_correction if batch_correction != "None" else None)

    st.header("UMAP Visualization")
    fig_clusters = plot_umap(adata, 'leiden', raw_expr=False)

    st.header("Marker Gene Expression Visualization")
    marker_gene = st.text_input("Enter marker gene (e.g. CD3D, MS4A1, LYZ)", "CD3D")
    if marker_gene in adata.var_names:
        fig_marker = plot_umap(adata, marker_gene, raw_expr=show_raw)
    else:
        st.warning(f"Gene '{marker_gene}' not found in dataset.")

    st.header("Additional Plots")
    plot_type = st.selectbox("Select plot type", ["Violin plot", "Dot plot", "Heatmap"])

    # Compute marker genes if needed
    if 'rank_genes_groups' not in adata.uns:
        sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')

    markers_df = sc.get.rank_genes_groups_df(adata, group=None)
    cluster_ids = adata.obs['leiden'].cat.categories
    top_genes_all = []
    for clust in cluster_ids:
        top_genes_all.extend(markers_df[markers_df['group'] == clust].head(5)['names'].values.tolist())
    top_genes_all = list(set(top_genes_all))

    st.write(f"Top marker genes (up to 5 per cluster): {', '.join(top_genes_all[:20])}")

    if plot_type == "Violin plot":
        fig_additional = plot_violin(adata, top_genes_all)
    elif plot_type == "Dot plot":
        fig_additional = plot_dotplot(adata, top_genes_all)
    else:
        fig_additional = plot_heatmap(adata, top_genes_all)

    st.header("Cluster Annotation Suggestions")
    annotations = annotate_clusters(adata)
    for clust, types in annotations.items():
        st.write(f"Cluster {clust}: likely cell type(s): {', '.join(types)}")

    # Download processed AnnData
    with tempfile.NamedTemporaryFile(suffix=".h5ad") as tmpfile:
        adata.write(tmpfile.name)
        tmpfile.seek(0)
        adata_bytes = tmpfile.read()

    st.download_button("Download processed AnnData (.h5ad)",
                       data=adata_bytes,
                       file_name="processed_data.h5ad",
                       mime="application/octet-stream")

    # Download marker genes CSV
    csv_markers = markers_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download marker genes table (CSV)",
                       data=csv_markers,
                       file_name="marker_genes_per_cluster.csv",
                       mime="text/csv")

    # Download UMAP cluster plot PNG
    buf_clusters = io.BytesIO()
    fig_clusters.savefig(buf_clusters, format="png")
    buf_clusters.seek(0)
    st.download_button("Download UMAP clusters plot (PNG)",
                       data=buf_clusters,
                       file_name="umap_clusters.png",
                       mime="image/png")

    # Download marker gene UMAP plot PNG (if valid gene)
    if marker_gene in adata.var_names:
        buf_marker = io.BytesIO()
        fig_marker.savefig(buf_marker, format="png")
        buf_marker.seek(0)
        st.download_button(f"Download UMAP plot for {marker_gene} (PNG)",
                           data=buf_marker,
                           file_name=f"umap_{marker_gene}.png",
                           mime="image/png")
else:
    st.info("Upload or select a dataset to start.")

st.write("""
---
#### Notes:
- Upload zipped 10X folder (matrix.mtx, barcodes.tsv, features.tsv), .loom, or .csv files.  
- Batch correction requires batch column in metadata and (optional) `harmonypy` for Harmony method.  
- Marker genes dictionary for cell type annotation can be extended to your dataset.  
- Plots and data can be downloaded using buttons below each section.
""")
