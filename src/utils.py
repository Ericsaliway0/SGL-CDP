## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python __pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --net_type CPDB --score_threshold 0.99 --hidden_feats 1024 --learning_rate 0.001 --num_epochs 105
## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python __pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --net_type HIPPIE --score_threshold 0.99 --in_feats 2048 --hidden_feats 256 --learning_rate 0.001 --num_epochs 105

## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python __pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --score_threshold 0.99 --learning_rate 0.001 --num_epochs 204
## p_value in average predicted score
## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python __pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --net_type STRING --score_threshold 0.99 --learning_rate 0.001 --num_epochs 505
## python _gene_label_prediction_tsne_pertag.py --model_type ChebNet --net_type pathnet --score_threshold 0.4 --learning_rate 0.001 --num_epochs 65 
## (kg39) ericsali@erics-MBP-4 gnn_pathways % python _gene_label_prediction_tsne_sage.py --model_type EMOGI --net_type ppnet --score_threshold 0.5 --learning_rate 0.001 --num_epochs 100 
## (kg39) ericsali@erics-MBP-4 gnn_pathways % python _gene_label_prediction_tsne_pertag.py --model_type ATTAG --net_type ppnet --score_threshold 0.9 --learning_rate 0.001 --num_epochs 201

import json
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from torch_geometric.nn import GCNConv
from .models import ACGNN, HGDC, EMOGI, MTGCN, GCN, GAT, GraphSAGE, GIN, ChebNet, ChebNetII, DMGNN, FocalLoss
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from sklearn.preprocessing import normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
from sklearn.cluster import KMeans, SpectralBiclustering

import os
import math
import csv
from collections import Counter
import time
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import psutil
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import entropy, ttest_ind, fisher_exact
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from sklearn.cluster import KMeans, SpectralBiclustering
from sklearn.preprocessing import normalize
from scipy.ndimage import gaussian_filter
#from sklearn_extra.cluster import KMeansConstrained
import umap
from matplotlib import rcParams
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, confusion_matrix
)
from networkx.algorithms.community import greedy_modularity_communities
from matplotlib import ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.cm import get_cmap
import dgl
import networkx as nx
from dgl.nn import GNNExplainer
from torch_geometric.nn import GCNConv
from .models import HGDC, EMOGI, MTGCN, GCN, GAT, GraphSAGE, GIN, ChebNet, ChebNetII, FocalLoss
# from .utils import (
#     choose_model, plot_roc_curve, plot_pr_curve, load_graph_data,
#     load_oncokb_genes, plot_and_analyze, save_and_plot_results
# )
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, ListedColormap, to_rgba, to_rgb
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
from venn import venn
import holoviews as hv
hv.extension('bokeh')
from holoviews import opts
from bokeh.io import output_notebook, export_png
from bokeh.plotting import show
from bokeh.io.export import get_screenshot_as_png
import plotly.graph_objects as go
from gprofiler import GProfiler
import glob
from pathlib import Path   
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
import numpy as np  # required for -log10
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics.pairwise import rbf_kernel
import random
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LinearSegmentedColormap, to_rgba

from lifelines.statistics import logrank_test, multivariate_logrank_test, pairwise_logrank_test
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.lines as mlines
from scipy.stats import zscore
from sklearn.decomposition import NMF

from matplotlib.ticker import PercentFormatter

CLUSTER_COLORS = {
    0: '#0077B6',   1: '#0000FF',   2: '#00B4D8',   3: '#48EAC4',
    4: '#F1C0E8',   5: '#B9FBC0',   6: '#32CD32',   7: '#bee1e6',
    8: '#8A2BE2',   9: '#E377C2',  10: '#8EECF5',  11: '#A3C4F3',
    12: '#FFB347', 13: '#FFD700',  14: '#FF69B4',  15: '#CD5C5C',
    16: '#7FFFD4', 17: '#FF7F50',  18: '#C71585',  19: '#20B2AA',
    20: '#6A5ACD', 21: '#40E0D0',  22: '#FF8C00',  23: '#DC143C',
    24: '#9ACD32'
}

def save_model_details(model, args, model_csv_path, in_feats, hidden_feats, out_feats):
    """
    Extracts model details and saves them to a CSV file.

    Parameters:
    - model: The neural network model.
    - args: Arguments containing model configuration.
    - model_csv_path: File path to save the model details.
    - in_feats: Number of input features.
    - hidden_feats: Number of hidden layer features.
    - out_feats: Number of output features.
    """
    # Count layers and parameters
    num_layers = sum(1 for _ in model.children())  # Count layers
    total_params = sum(p.numel() for p in model.parameters())  # Count parameters

    # Detect attention layers
    attention_layer_nodes = None
    for layer in model.children():
        if hasattr(layer, 'heads'):  # Assuming attention layers have 'heads' attribute
            attention_layer_nodes = layer.heads

    # Detect residual connections
    has_residual = any(isinstance(layer, nn.Identity) for layer in model.modules())

    # Prepare data for CSV
    model_data = {
        "Method": [args.model_type],
        "Number of Layers": [num_layers],
        "Input Layer Nodes": [in_feats],
        "Hidden Layer Nodes": [hidden_feats],
        "Attention Layer Nodes": [attention_layer_nodes if attention_layer_nodes else "N/A"],
        "Output Layer Nodes": [out_feats],
        "Total Parameters": [total_params],
        "Residual Connection": ["Yes" if has_residual else "No"]
    }

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(model_data)
    df.to_csv(model_csv_path, index=False)
    print(f"Model architecture saved to {model_csv_path}")

def choose_model(model_type, in_feats, hidden_feats, out_feats):
    if model_type == 'GraphSAGE':
        return GraphSAGE(in_feats, hidden_feats, out_feats)
    elif model_type == 'GAT':
        return GAT(in_feats, hidden_feats, out_feats, num_heads=1)
    elif model_type == 'GCN':
        return GCN(in_feats, hidden_feats, out_feats)
    elif model_type == 'GIN':
        return GIN(in_feats, hidden_feats, out_feats)
    elif model_type == 'HGDC':
        return GAT(in_feats, hidden_feats, out_feats, num_heads=1)
    elif model_type == 'EMOGI':
        return GAT(in_feats, hidden_feats, out_feats, num_heads=1)
    elif model_type == 'MTGCN':
        return GCN(in_feats, hidden_feats, out_feats)
    elif model_type == 'ChebNet':
        return ChebNet(in_feats, hidden_feats, out_feats)
    elif model_type == 'ChebNetII':
        return ChebNet(in_feats, hidden_feats, out_feats)    
    elif model_type == 'DMGNN':
        return DMGNN(
            in_feat_dim=in_feats,
            hidden_dim=hidden_feats,
            out_dim=out_feats,
            heads=4,
            dropout=0.5
        )
    elif model_type == 'ACGNN':
        return ACGNN(in_feats, hidden_feats, out_feats)
    else:
        raise ValueError("Invalid model type. Choose from ['GraphSAGE', 'GAT', 'EMOGI', 'HGDC', 'MTGCN', 'GCN', 'GIN', 'ChebNet', 'ChebNetII', 'DMGNN', 'ACGNN'].")

def save_and_plot_results_no_error_bar_pass(predicted_above, predicted_below, degrees_above, degrees_below, avg_above, avg_below, args):

    # Save predictions and degrees
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    def save_csv(data, filename, header):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)
            csvwriter.writerows(data)
        print(f"File saved: {filepath}")

    save_csv(predicted_above, f'{args.model_type}_above_threshold.csv', ['Gene', 'Score'])
    save_csv(predicted_below, f'{args.model_type}_below_threshold.csv', ['Gene', 'Score'])
    save_csv(degrees_above.items(), f'{args.model_type}_degrees_above.csv', ['Gene', 'Degree'])
    save_csv(degrees_below.items(), f'{args.model_type}_degrees_below.csv', ['Gene', 'Degree'])

    # Degree comparison barplot
    data = pd.DataFrame({
        'Threshold': ['Above', 'Below'],
        'Average Degree': [avg_above, avg_below]
    })
    plt.figure(figsize=(8, 6))
    sns.barplot(data=data, x='Threshold', y='Average Degree', palette="viridis")
    plt.title('Average Degree Comparison')
    plt.savefig(os.path.join(output_dir, f'{args.model_type}_degree_comparison.png'))
    plt.close()

def plot_roc_curve(labels, scores, filename):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="blue")
    plt.plot([0, 1], [0, 1], color="salmon", linestyle="--")
    plt.title("Receiver Operating Characteristic Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    ##plt.grid(alpha=0.4)
    plt.savefig(filename)
    plt.close()
    print(f"ROC Curve saved to {filename}")

def plot_all_roc_curves_grid(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all ROC curves (one per cancer type) in a single figure.
    """
    plt.figure(figsize=(10, 8))
    
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={auc_score:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUROC Curves for All 16 Cancer Types")
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    plt.grid(False, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    print(f"Combined AUROC plot saved to {output_path}")

def plot_all_roc_curves_ori(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all AUROC curves (one per cancer type) with scientific-style formatting.
    """
    plt.figure(figsize=(10, 8))

    # Plot each ROC curve
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} AUC = {auc_score:.3f}")

    # Random classifier diagonal
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1.2)

    # Axis labels and title
    plt.xlabel("Specificity", fontsize=14)
    plt.ylabel("Sensitivity", fontsize=14)
    plt.title("AUROC Curves for All Cancer Types", fontsize=16, weight='bold')

    # Custom AUC summary box (no frame)
    median_auc = round(np.median(all_aucs), 3)
    auc_range = (round(min(all_aucs), 3), round(max(all_aucs), 3))
    textstr = f"Median AUC = {median_auc}\nRange = {auc_range[0]}–{auc_range[1]}"
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    # Legend (no frame)
    plt.legend(loc='lower right', fontsize=9, ncol=2, frameon=False)

    # Clean plot
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"AUROC plot saved to {output_path}")

def plot_all_roc_curves(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all AUROC curves (one per cancer type) with scientific-style formatting.
    """
    plt.figure(figsize=(10, 8))

    # Plot each ROC curve
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} AUC = {auc_score:.4f}")

    # Random classifier diagonal
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1.2)

    # Axis labels and title
    plt.xlabel("Specificity", fontsize=26)
    plt.ylabel("Sensitivity", fontsize=26)
    plt.title("AUROC Curves for All Cancer Types", fontsize=14, weight='bold')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Custom AUC summary box (no frame)
    median_auc = round(np.median(all_aucs), 3)
    auc_range = (round(min(all_aucs), 3), round(max(all_aucs), 3))
    textstr = f"Median AUC = {median_auc}\nRange = {auc_range[0]}–{auc_range[1]}"
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=16, verticalalignment='top')

    # Legend (no frame)
    plt.legend(loc='lower right', fontsize=14, ncol=2, frameon=False)

    # Clean plot
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"AUROC plot saved to {output_path}")

def plot_all_roc_curves_with_frame(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all AUROC curves (one per cancer type) with scientific-style formatting.
    """
    plt.figure(figsize=(10, 8))

    # Plot each ROC curve
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} AUC = {auc_score:.3f}")

    # Random classifier diagonal
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1.2)

    # Axis labels and title
    plt.xlabel("Specificity", fontsize=14)
    plt.ylabel("Sensitivity", fontsize=14)
    plt.title("AUROC Curves for All Cancer Types", fontsize=16, weight='bold')

    # Custom AUC summary box
    median_auc = round(np.median(all_aucs), 3)
    auc_range = (round(min(all_aucs), 3), round(max(all_aucs), 3))
    textstr = f"Median AUC = {median_auc}\nRange = {auc_range[0]}–{auc_range[1]}"
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    # Legend
    plt.legend(loc='lower right', fontsize=9, ncol=2, frameon=True)

    # Clean plot
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"AUROC plot saved to {output_path}")

def plot_all_roc_curves_mda_comparision_two_columns(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all AUROC curves (one per cancer type) with scientific-style formatting.
    """
    plt.figure(figsize=(10, 8))

    # Plot each ROC curve
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} AUC = {auc_score:.4f}")

    # Random classifier diagonal
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1.2)

    # Axis labels and title
    # plt.xlabel("Specificity", fontsize=26)
    # plt.ylabel("Sensitivity", fontsize=26)
    plt.xlabel("False Positive Rate", fontsize=26)
    plt.ylabel("True Positive Rate", fontsize=26)
    plt.title("AUROC Curves for All Cancer Types", fontsize=14, weight='bold')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Custom AUC summary box (no frame)
    # median_auc = round(np.median(all_aucs), 3)
    # auc_range = (round(min(all_aucs), 3), round(max(all_aucs), 3))
    # textstr = f"Median AUC = {median_auc}\nRange = {auc_range[0]}–{auc_range[1]}"
    # plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
    #          fontsize=16, verticalalignment='top')

    # Legend (no frame)
    plt.legend(loc='lower right', fontsize=14, ncol=2, frameon=False)

    # Clean plot
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"AUROC plot saved to {output_path}")

def plot_all_roc_curves_mda_comparision(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all AUROC curves (one per cancer type) with scientific-style formatting.
    """
    plt.figure(figsize=(10, 8))

    # Plot each ROC curve
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC = {auc_score:.4f})")

    # Random classifier diagonal
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1.2)

    # Axis labels and title
    plt.xlabel("False Positive Rate", fontsize=20)
    plt.ylabel("True Positive Rate", fontsize=20)
    plt.title("AUROC Curves for TarBase", fontsize=20)#, weight='bold')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Legend in one column
    plt.legend(loc='lower right', fontsize=16, ncol=1, frameon=False)

    # Clean plot
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"AUROC plot saved to {output_path}")

def plot_all_pr_curves_mda_comparision_(
    all_precisions, all_recalls, all_auprcs, model_labels, output_path
):
    """
    Plots all AUPRC curves (one per model) with scientific-style formatting, matching AUROC plot style.
    """
    plt.figure(figsize=(10, 8))

    # Plot each PR curve
    for precision, recall, auprc, label in zip(all_precisions, all_recalls, all_auprcs, model_labels):
        # Sort by recall to ensure correct plotting
        recall_sorted, precision_sorted = zip(*sorted(zip(recall, precision)))
        recall_sorted = np.array(recall_sorted)
        precision_sorted = np.array(precision_sorted)

        plt.plot(recall_sorted, precision_sorted, lw=1.5, label=f"{label} AUPRC = {auprc:.4f}")

    # Axis labels and title
    plt.xlabel("Recall", fontsize=26)
    plt.ylabel("Precision", fontsize=26)
    plt.title("AUPRC Curves for mirBase", fontsize=18)#, weight='bold')

    # Tick font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Legend
    plt.legend(loc='lower left', fontsize=14, ncol=1, frameon=False)

    # Clean plot
    plt.grid(False)
    plt.tight_layout()

    # Save and show
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"AUPRC plot saved to {output_path}")

def plot_all_pr_curves_mda_comparision(
    all_precisions, all_recalls, all_auprcs, model_labels, output_path
):
    """
    Plots Precision-Recall curves (AUPRC) for multiple models with legend at bottom left and no grid.
    """
    plt.figure(figsize=(10, 8))

    for precision, recall, auprc, label in zip(all_precisions, all_recalls, all_auprcs, model_labels):
        # Sort by recall to ensure a correct PR curve
        recall_sorted, precision_sorted = zip(*sorted(zip(recall, precision)))
        recall_sorted = np.array(recall_sorted)
        precision_sorted = np.array(precision_sorted)

        plt.plot(
            recall_sorted,
            precision_sorted,
            lw=1.2,
            label=f"{label} (AUPRC = {auprc:.4f})"
        )

    plt.title("AUPRC Curve for miRTarBase", fontsize=20)
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.legend(loc="lower left", fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Precision-Recall Comparison Curve saved to {output_path}")

import matplotlib.pyplot as plt
import numpy as np

def plot_all_pr_curves_mda_comparision__(
    all_precisions, all_recalls, all_auprcs, model_labels, output_path
):
    """
    Plots Precision-Recall curves (AUPRC) for multiple models with proper sorting.

    Parameters:
        - all_precisions: list of arrays, precision values per model
        - all_recalls: list of arrays, recall values per model
        - all_auprcs: list of floats, AUPRC scores
        - model_labels: list of str, names of the models
        - output_path: str, where to save the figure
    """
    plt.figure(figsize=(8, 6))

    for precision, recall, auprc, label in zip(all_precisions, all_recalls, all_auprcs, model_labels):
        # Sort by recall to ensure proper curve
        recall_sorted, precision_sorted = zip(*sorted(zip(recall, precision)))

        # Convert to numpy arrays for safe handling
        recall_sorted = np.array(recall_sorted)
        precision_sorted = np.array(precision_sorted)

        plt.plot(
            recall_sorted,
            precision_sorted,
            lw=2,
            label=f"{label} (AUPRC = {auprc:.4f})"
        )

    plt.title("Precision-Recall Curve Comparison", fontsize=14)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Precision-Recall Comparison Curve saved to {output_path}")

def plot_all_pr_curves_mda_comparision(
    all_precisions, all_recalls, all_auprcs, model_labels, output_path
):
    """
    Plots Precision-Recall curves (AUPRC) for multiple models with proper sorting.

    Parameters:
        - all_precisions: list of arrays, precision values per model
        - all_recalls: list of arrays, recall values per model
        - all_auprcs: list of floats, AUPRC scores
        - model_labels: list of str, names of the models
        - output_path: str, where to save the figure
    """
    plt.figure(figsize=(10, 8))

    for precision, recall, auprc, label in zip(all_precisions, all_recalls, all_auprcs, model_labels):
        # Sort by recall to ensure proper curve
        recall_sorted, precision_sorted = zip(*sorted(zip(recall, precision)))

        # Convert to numpy arrays for safe handling
        recall_sorted = np.array(recall_sorted)
        precision_sorted = np.array(precision_sorted)

        plt.plot(
            recall_sorted,
            precision_sorted,
            lw=1.2,
            label=f"{label} (AUPRC = {auprc:.4f})"
        )

    plt.title("AUPRC Curve for miRTarBase", fontsize=20)
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.legend(loc="lower left", fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Precision-Recall Comparison Curve saved to {output_path}")

def plot_all_pr_curves(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all ROC curves (one per cancer type) in a single figure with scientific-style formatting.
    """
    plt.figure(figsize=(10, 8))

    # Plot each ROC curve
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} AUC = {auc_score:.3f}")

    # Plot diagonal (random)
    plt.plot([0, 1], [0, 1], color='grey', linestyle='-', linewidth=1.2)

    # Axis labels and title
    plt.xlabel("1 - Specificity", fontsize=14)
    plt.ylabel("Sensitivity", fontsize=14)
    plt.title("Performance on Test Datasets", fontsize=16, weight='bold')

    # Invert X-axis to match the shared plot (Specificity on X, decreasing)
    plt.gca().invert_xaxis()

    # Custom AUC summary box
    median_auc = round(np.median(all_aucs), 3)
    auc_range = (round(min(all_aucs), 3), round(max(all_aucs), 3))
    textstr = f"Median AUC = {median_auc}\nRange = {auc_range[0]}–{auc_range[1]}"
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    # Legend
    plt.legend(loc='lower left', fontsize=9, ncol=2, frameon=True)

    # Remove grid, set tight layout
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Combined AUROC plot saved to {output_path}")

def plot_all_roc_curves_pa(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all ROC curves (one per cancer type) in a single figure.
    """
    plt.figure(figsize=(10, 8))
    
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={auc_score:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUROC Curves for All 16 Cancer Types")
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    # Grid line removed
    # plt.grid(False, linestyle='--', alpha=0.4)
    plt.grid(False)

    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    print(f"Combined AUROC plot saved to {output_path}")

def plot_pr_curve(labels, scores, filename):
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.4f})", color="green")
    ##plt.plot([0, 1], [1, 0], color="salmon", linestyle="--")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    ##plt.grid(alpha=0.4)
    plt.savefig(filename)
    plt.close()
    print(f"Precision-Recall Curve saved to {filename}")

def load_graph_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    nodes = {}
    edges = []
    labels = []
    embeddings = []

    for entry in data:
        source = entry["source"]["properties"]
        target = entry["target"]["properties"]
        relation = entry["relation"]["type"]

        # Add source node
        if source["name"] not in nodes:
            nodes[source["name"]] = len(nodes)
            embeddings.append(source["embedding"])
            labels.append(source.get("label", -1) if source.get("label") is not None else -1)

        # Add target node
        if target["name"] not in nodes:
            nodes[target["name"]] = len(nodes)
            embeddings.append(target["embedding"])
            labels.append(target.get("label", -1) if target.get("label") is not None else -1)

        # Add edge
        edges.append((nodes[source["name"]], nodes[target["name"]]))

    # Convert embeddings and labels to tensors
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return nodes, edges, embeddings_tensor, labels_tensor

def load_oncokb_genes(filepath):
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f)

def plot_and_analyze_ori(args):
    # File path for the saved predictions
    csv_file_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_scores_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    results = []

    # Read the CSV file
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            node_name, score, label = row
            score = float(score)
            label = int(label)
            results.append((node_name, score, label))
            
            # Check if label is 0 and print the row
            if label == 0:
                print(f"Node Name: {node_name}, Score: {score}, Label: {label}")


    # Extract scores and labels
    scores = np.array([row[1] for row in results])
    labels = np.array([row[2] for row in results])
    

    # Define group labels in the desired order
    group_labels = [1, 2, 0, 3]
    average_scores = []

    # Calculate average scores for each group
    for label in group_labels:
        group_scores = scores[labels == label]
        avg_score = np.mean(group_scores) if len(group_scores) > 0 else 0.0
        average_scores.append(avg_score)

    # Perform statistical tests to calculate p-values
    p_values = {}
    comparisons = [(1, 2), (1, 0), (1, 3)]  # Pairs to compare
    for group1, group2 in comparisons:
        scores1 = scores[labels == group1]
        scores2 = scores[labels == group2]
        if len(scores1) > 1 and len(scores2) > 1:
            _, p_value = ttest_ind(scores1, scores2, equal_var=False)
            p_values[(group1, group2)] = p_value
        else:
            p_values[(group1, group2)] = np.nan

    # Save average scores and p-values to another CSV
    avg_csv_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_pvalues_epo{args.num_epochs}_2048.csv'
    )
    os.makedirs(os.path.dirname(avg_csv_path), exist_ok=True)
    with open(avg_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Group Label', 'Average Score', 'Comparison', 'P-Value'])
        for label, avg_score in zip(group_labels, average_scores):
            writer.writerow([f'Group {label}', avg_score, '', ''])
        for (group1, group2), p_value in p_values.items():
            writer.writerow(['', '', f'Group {group1} vs Group {group2}', p_value])
    print(f"Average scores and p-values saved to {avg_csv_path}")

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(group_labels)), average_scores,
                   color=['green', 'red', 'blue', 'orange'], edgecolor='black', alpha=0.8)
    for bar, avg_score in zip(bars, average_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{avg_score:.4f}',
                 ha='center', va='bottom', fontsize=12)
    plt.xticks(range(len(group_labels)), ['Ground-truth (1)', 'Predicted (2)', 'Non-driver (0)', 'Other (3)'])
    plt.xlabel('Gene Groups', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.title('Average Scores for Each Gene Group', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    bar_plot_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_barplot_epo{args.num_epochs}_2048.png'
    )
    plt.savefig(bar_plot_path)
    print(f"Bar plot saved to {bar_plot_path}")
    plt.close()

    '''
    
    # Convert gene_labels to NumPy array
    ##gene_labels = gene_labels.cpu().numpy()

    # Calculate average scores for each group in order of label 1, 2, 0, 3
    group_labels = [1, 2, 0, 3]  # Define the groups in the desired order
    average_scores = [np.mean(scores[gene_labels == label]) if (gene_labels == label).sum() > 0 else 0.0
                    for label in group_labels]

    p_values = {}
    for g1, g2 in [(1, 2), (1, 0), (1, 3)]:
        scores1 = scores[gene_labels == g1]
        scores2 = scores[gene_labels == g2]
        p_values[(g1, g2)] = ttest_ind(scores1, scores2, equal_var=False).pvalue if len(scores1) > 1 and len(scores2) > 1 else np.nan

    # Save averages and p-values
    avg_csv_path = os.path.join('results/gene_prediction/',
                                f'{args.model_type}_avg_scores_pvalues.csv')
    with open(avg_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Group Label', 'Average Score', 'Comparison', 'P-Value'])
        for label, avg_score in zip(group_labels, average_scores):
            writer.writerow([f'Group {label}', avg_score, '', ''])
        for (g1, g2), p_val in p_values.items():
            writer.writerow(['', '', f'Group {g1} vs Group {g2}', p_val])
    print(f"Average scores and p-values saved to {avg_csv_path}")

    # Plot average scores
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(group_labels)), average_scores,
                   color=['green', 'red', 'blue', 'orange'], edgecolor='black', alpha=0.8)
    for bar, avg in zip(bars, average_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{avg:.4f}',
                 ha='center', va='bottom', fontsize=12)
    plt.xticks(range(len(group_labels)), ['Ground-truth (1)', 'Predicted (2)', 'Non-driver (0)', 'Other (3)'])
    plt.xlabel('Gene Groups', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.title('Average Scores by Gene Group', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    bar_plot_path = os.path.join('results/gene_prediction/',
                                 f'{args.model_type}_group_avg_scores.png')
    plt.savefig(bar_plot_path)
    print(f"Bar plot saved to {bar_plot_path}")
    plt.close()'''

def plot_and_analyze_(args):
    # File path for the saved predictions
    csv_file_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_scores_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    results = []

    # Read the CSV file
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            node_name, score, label = row
            score = float(score)
            label = int(label)
            results.append((node_name, score, label))
    
    # Extract scores and labels as NumPy arrays
    scores = np.array([row[1] for row in results])
    labels = np.array([row[2] for row in results])
    
    # Define group labels in the desired order
    group_labels = [1, 2, 0, 3]
    average_scores = []

    # Calculate average scores for each group
    for label in group_labels:
        group_scores = scores[labels == label]
        avg_score = np.mean(group_scores) if len(group_scores) > 0 else 0.0
        average_scores.append(avg_score)

    # Perform statistical tests to calculate p-values
    p_values = {}
    comparisons = [(1, 2), (1, 0), (1, 3), (2, 3)]  # Added (2 vs. 3)
    for group1, group2 in comparisons:
        scores1 = scores[labels == group1]
        scores2 = scores[labels == group2]
        if len(scores1) > 1 and len(scores2) > 1:
            _, p_value = ttest_ind(scores1, scores2, equal_var=False)
            p_values[(group1, group2)] = p_value
        else:
            p_values[(group1, group2)] = np.nan

    # Save average scores and p-values to another CSV
    avg_csv_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_pvalues_threshold{args.score_threshold}_epo{args.num_epochs}_2048.csv'
    )
    os.makedirs(os.path.dirname(avg_csv_path), exist_ok=True)
    with open(avg_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Group Label', 'Average Score', 'Comparison', 'P-Value'])
        for label, avg_score in zip(group_labels, average_scores):
            writer.writerow([f'Group {label}', avg_score, '', ''])
        for (group1, group2), p_value in p_values.items():
            writer.writerow(['', '', f'Group {group1} vs Group {group2}', p_value])
    
    print(f"Average scores and p-values saved to {avg_csv_path}")

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(group_labels)), average_scores,
                   color=['green', 'red', 'blue', 'orange'], edgecolor='black', alpha=0.8)
    for bar, avg_score in zip(bars, average_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{avg_score:.4f}',
                 ha='center', va='bottom', fontsize=12)
    plt.xticks(range(len(group_labels)), ['Ground-truth (1)', 'Predicted (2)', 'Non-driver (0)', 'Other (3)'])
    plt.xlabel('Gene Groups', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.title('Average Scores for Each Gene Group', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the bar plot
    bar_plot_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_barplot_threshold{args.score_threshold}_epo{args.num_epochs}_2048.png'
    )
    plt.savefig(bar_plot_path)
    print(f"Bar plot saved to {bar_plot_path}")
    plt.close()

def plot_and_analyze(args):
    # File path for the saved predictions
    csv_file_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_scores_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    results = []

    # Read the CSV file
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            node_name, score, label = row
            score = float(score)
            label = int(label)
            results.append((node_name, score, label))
    
    # Extract scores and labels as NumPy arrays
    scores = np.array([row[1] for row in results])
    labels = np.array([row[2] for row in results])
    
    # Define group labels in the desired order
    group_labels = [1, 2, 0, 3]
    average_scores = []

    # Calculate average scores for each group
    for label in group_labels:
        group_scores = scores[labels == label]
        avg_score = np.mean(group_scores) if len(group_scores) > 0 else 0.0
        average_scores.append(avg_score)

    # Perform statistical tests to calculate p-values
    p_values = {}
    comparisons = [(1, 2), (1, 0), (1, 3), (2, 3)]  # Added (2 vs. 3)
    for group1, group2 in comparisons:
        scores1 = scores[labels == group1]
        scores2 = scores[labels == group2]
        if len(scores1) > 1 and len(scores2) > 1:
            _, p_value = ttest_ind(scores1, scores2, equal_var=False)
            p_values[(group1, group2)] = p_value
        else:
            p_values[(group1, group2)] = np.nan

    # Save average scores and p-values to another CSV
    avg_csv_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_pvalues_threshold{args.score_threshold}_epo{args.num_epochs}_2048.csv'
    )
    os.makedirs(os.path.dirname(avg_csv_path), exist_ok=True)
    with open(avg_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Group Label', 'Average Score', 'Comparison', 'P-Value'])
        for label, avg_score in zip(group_labels, average_scores):
            writer.writerow([f'Group {label}', avg_score, '', ''])
        for (group1, group2), p_value in p_values.items():
            writer.writerow(['', '', f'Group {group1} vs Group {group2}', p_value])
    
    print(f"Average scores and p-values saved to {avg_csv_path}")

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    bar_width = 0.3  # Adjust the width (default is usually around 0.8)
    bars = plt.bar(range(len(group_labels)), average_scores, width=bar_width, 
                color=['green', 'red', 'blue', 'orange'], edgecolor='black', alpha=0.8)

    # Add text labels on top of bars
    for bar, avg_score in zip(bars, average_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{avg_score:.4f}',
                ha='center', va='bottom', fontsize=12)

    plt.xticks(range(len(group_labels)), ['Ground-truth (1)', 'Predicted (2)', 'Non-driver (0)', 'Other (3)'])
    plt.xlabel('Gene Groups', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.title('Average Scores for Each Gene Group', fontsize=16)
    ##plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the bar plot
    bar_plot_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_barplot_threshold{args.score_threshold}_epo{args.num_epochs}_2048.png'
    )
    plt.savefig(bar_plot_path)
    print(f"Bar plot saved to {bar_plot_path}")
    plt.close()

def save_and_plot_results(predicted_above, predicted_below, degrees_above, degrees_below, avg_above, avg_below, avg_error_above, avg_error_below, args):

    # Save predictions and degrees
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    def save_csv(data, filename, header):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)
            csvwriter.writerows(data)
        print(f"File saved: {filepath}")

    save_csv(predicted_above, f'{args.model_type}_{args.net_type}_above_threshold.csv', ['Gene', 'Score'])
    save_csv(predicted_below, f'{args.model_type}_{args.net_type}_below_threshold.csv', ['Gene', 'Score'])
    save_csv(degrees_above.items(), f'{args.model_type}_{args.net_type}_degrees_above.csv', ['Gene', 'Degree'])
    save_csv(degrees_below.items(), f'{args.model_type}_{args.net_type}_degrees_below.csv', ['Gene', 'Degree'])

    # Degree comparison barplot with error bars
    data = pd.DataFrame({
        'Threshold': ['Above', 'Below'],
        'Average Degree': [avg_above, avg_below],
        'Error': [avg_error_above, avg_error_below]  # Add error values
    })
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(data['Threshold'], data['Average Degree'], yerr=data['Error'], capsize=5, color=['green', 'red'], edgecolor='black', alpha=0.8)

    # Add error bars explicitly (optional, can be done directly in the bar plot)
    for bar, error in zip(bars, data['Error']):
        plt.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=error, fmt='none', color='black', capsize=5, linestyle='--')

    plt.title('Average Degree Comparison with Error Bars')
    plt.savefig(os.path.join(output_dir, f'{args.model_type}_{args.net_type}_degree_comparison_with_error_bars.png'))
    plt.close()
    print(f"Degree comparison plot saved to {os.path.join(output_dir, f'{args.model_type}_{args.net_type}_degree_comparison_with_error_bars.png')}")


def load_ground_truth_cancer_genes(file_path):
    """
    Load ground truth cancer genes from a file.
    
    Args:
        file_path (str): Path to the ground truth cancer gene file.
    
    Returns:
        set: A set containing ground truth cancer gene names.
    """
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)


def save_model_details(model, args, model_csv_path, in_feats, hidden_feats, out_feats):
    """
    Extracts model details and saves them to a CSV file.

    Parameters:
    - model: The neural network model.
    - args: Arguments containing model configuration.
    - model_csv_path: File path to save the model details.
    - in_feats: Number of input features.
    - hidden_feats: Number of hidden layer features.
    - out_feats: Number of output features.
    """
    # Count layers and parameters
    num_layers = sum(1 for _ in model.children())  # Count layers
    total_params = sum(p.numel() for p in model.parameters())  # Count parameters

    # Detect attention layers
    attention_layer_nodes = None
    for layer in model.children():
        if hasattr(layer, 'heads'):  # Assuming attention layers have 'heads' attribute
            attention_layer_nodes = layer.heads

    # Detect residual connections
    has_residual = any(isinstance(layer, nn.Identity) for layer in model.modules())

    # Prepare data for CSV
    model_data = {
        "Method": [args.model_type],
        "Number of Layers": [num_layers],
        "Input Layer Nodes": [in_feats],
        "Hidden Layer Nodes": [hidden_feats],
        "Attention Layer Nodes": [attention_layer_nodes if attention_layer_nodes else "N/A"],
        "Output Layer Nodes": [out_feats],
        "Total Parameters": [total_params],
        "Residual Connection": ["Yes" if has_residual else "No"]
    }

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(model_data)
    df.to_csv(model_csv_path, index=False)
    print(f"Model architecture saved to {model_csv_path}")



def save_predicted_scores(scores, labels, nodes, args):
    """
    Saves predicted scores and labels to a CSV file.

    Parameters:
    - scores: List of predicted scores.
    - labels: List of ground-truth labels.
    - nodes: Dictionary of node names.
    - args: Arguments containing model configuration.
    """
    # Initialize variables to calculate average scores and standard deviations
    label_scores = {0: [], 1: [], 2: [], 3: []}  # Groups for each label

    # Define CSV file path
    csv_file_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_scores_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Save results to CSV
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Node Name', 'Score', 'Label'])  # Header

        for i, score in enumerate(scores):
            label = int(labels[i].item())  # Ensure label is an integer
            
            if label in [1, 0]:  # Ground-truth labels
                writer.writerow([list(nodes.keys())[i], score, label])
                label_scores[label].append(score)
            elif label == -1 and score >= args.score_threshold:  # Predicted driver genes
                writer.writerow([list(nodes.keys())[i], score, 2])
                label_scores[2].append(score)
            else:  # Non-labeled nodes or other
                writer.writerow([list(nodes.keys())[i], score, 3])
                label_scores[3].append(score)

    print(f"Predicted scores and labels saved to {csv_file_path}")

    return label_scores  # Returning for further analysis if needed



def save_average_scores(label_scores, args):
    """
    Calculates and saves the average score, standard deviation, and number of nodes per label.

    Parameters:
    - label_scores: Dictionary with labels as keys and lists of scores as values.
    - args: Arguments containing model configuration.
    """
    # Define CSV file path
    average_scores_file = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_average_scores_by_label_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(average_scores_file), exist_ok=True)

    # Save average scores to CSV
    with open(average_scores_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Label', 'Average Score', 'Standard Deviation', 'Number of Nodes'])  # Header

        for label, scores_list in label_scores.items():
            if scores_list:  # Check if the list is not empty
                avg_score = np.mean(scores_list)
                std_dev = np.std(scores_list)
                num_nodes = len(scores_list)
            else:
                avg_score = 0.0  # Default if no nodes in the label group
                std_dev = 0.0
                num_nodes = 0

            writer.writerow([label, avg_score, std_dev, num_nodes])

    print(f"Average scores by label saved to {average_scores_file}")


def plot_average_scores(label_scores, args):
    """
    Plots average scores with error bars and saves the figure.

    Parameters:
    - label_scores: Dictionary with labels as keys and lists of scores as values.
    - args: Arguments containing model configuration.
    """
    labels_list = []
    avg_scores = []
    std_devs = []

    for label, scores_list in label_scores.items():
        if scores_list:
            labels_list.append(label)
            avg_scores.append(np.mean(scores_list))
            std_devs.append(np.std(scores_list))

    if not labels_list:
        print("No valid scores to plot.")
        return

    # Define plot save path
    plot_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_average_scores_with_error_bars_threshold{args.score_threshold}_epo{args.num_epochs}.png'
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.bar(labels_list, avg_scores, yerr=std_devs, capsize=5, color='skyblue', alpha=0.7)
    plt.xlabel('Label')
    plt.ylabel('Average Score')
    plt.title('Average Scores by Label with Error Bars')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Save and close plot
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Error bar plot saved to {plot_path}")

def plot_score_distributions(label_scores, args):
    """
    Plots score distributions for each label and saves the figures.

    Parameters:
    - label_scores: Dictionary with labels as keys and lists of scores as values.
    - args: Arguments containing model configuration.
    """
    for label, scores_list in label_scores.items():
        if scores_list:
            plt.figure(figsize=(8, 6))
            plt.hist(scores_list, bins=20, alpha=0.7, color='#98f5e1', edgecolor='black')

            # Set labels and tick sizes
            plt.xlabel('Score', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            # Customize grid and tick appearance
            plt.tick_params(axis='both', which='major', length=6, width=2, direction='inout', grid_color='gray', grid_alpha=0.5)
            plt.grid(axis='y', linestyle='--', alpha=0.6)

            # Define plot save path
            plot_path = os.path.join(
                'results/gene_prediction/',
                f'{args.model_type}_{args.net_type}_score_distribution_label{label}_threshold{args.score_threshold}_epo{args.num_epochs}.png'
            )

            # Ensure directory exists
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)

            # Save and close plot
            plt.savefig(plot_path)
            plt.close()

            print(f"Score distribution for label {label} saved to {plot_path}")

def save_performance_metrics(epoch_times, cpu_usages, gpu_usages, args):
    """
    Saves performance metrics per epoch, including time per epoch, CPU, and GPU usage.

    Parameters:
    - epoch_times: List of epoch durations (in seconds).
    - cpu_usages: List of CPU memory usage per epoch (in MB).
    - gpu_usages: List of GPU memory usage per epoch (in MB).
    - args: Arguments containing model and training configuration.
    - output_dir: Directory to save the metrics CSV file.
    """

    # Compute total and average performance metrics
    total_time = sum(epoch_times)
    avg_time_per_epoch = total_time / args.num_epochs
    avg_cpu_usage = sum(cpu_usages) / args.num_epochs
    avg_gpu_usage = sum(gpu_usages) / args.num_epochs

    # Create DataFrame with per-epoch metrics
    df_metrics = pd.DataFrame({
        "Epoch": range(1, args.num_epochs + 1),
        "Time per Epoch (s)": epoch_times,
        "CPU Usage (MB)": cpu_usages,
        "GPU Usage (MB)": gpu_usages
    })

    # Define CSV path
    metrics_csv_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_performance_metrics_epo{args.num_epochs}_2048.csv'
    )

    # Save to CSV
    df_metrics.to_csv(metrics_csv_path, index=False)

    print(f"Epoch performance metrics saved to {metrics_csv_path}")

    # Print summary statistics
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Average Time per Epoch: {avg_time_per_epoch:.2f} seconds")
    print(f"Average CPU Usage: {avg_cpu_usage:.2f} MB")
    print(f"Average GPU Usage: {avg_gpu_usage:.2f} MB")


def save_overall_metrics(total_time, average_time_per_epoch, average_cpu_usage, average_gpu_usage, args, output_dir):
    """
    Save the overall performance metrics to a CSV file.

    Args:
    - total_time: Total training time in seconds.
    - average_time_per_epoch: Average time per epoch in seconds.
    - average_cpu_usage: Average CPU usage in MB.
    - average_gpu_usage: Average GPU usage in MB.
    - args: Argument object containing model and network type.
    - output_dir: The directory where the results will be saved.
    """
    # Save overall metrics
    df_overall_metrics = pd.DataFrame([{
        "Model Type": args.model_type,
        "Total Time": f"{total_time:.4f}s",
        "Average Time per Epoch": f"{average_time_per_epoch:.4f}s",
        "Average CPU Usage (MB)": f"{average_cpu_usage:.2f}",
        "Average GPU Usage (MB)": f"{average_gpu_usage:.2f}"
    }])
    
    # Define path to save the CSV
    overall_metrics_csv_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_overall_performance_epo{args.num_epochs}_2048.csv')
    
    # Save to CSV
    df_overall_metrics.to_csv(overall_metrics_csv_path, index=False)
    print(f"Overall performance metrics saved to {overall_metrics_csv_path}")

def process_predictions(ranking, args, drivers_file_path, oncokb_file_path, ongene_file_path, ncg_file_path, intogen_file_path, node_names, non_labeled_nodes):
    """
    Process and save the predicted driver genes, confirmed sources, and known drivers.
    
    Args:
    - ranking: List of tuples (gene, score) representing ranked predictions.
    - args: Argument object containing model and network type, and score threshold.
    - drivers_file_path, oncokb_file_path, ongene_file_path, ncg_file_path, intogen_file_path: Paths to the confirmation gene files.
    - node_names, non_labeled_nodes: Information about node names and indices for matching.
    """
    # Load data from the confirmation files
    oncokb_genes = load_gene_set(oncokb_file_path)
    ongene_genes = load_gene_set(ongene_file_path)
    ncg_genes = load_gene_set(ncg_file_path)
    intogen_genes = load_gene_set(intogen_file_path)

    # Threshold for the score
    score_threshold = args.score_threshold

    confirmed_predictions = []
    predicted_genes = []

    for node, score in ranking:
        if score >= score_threshold:
            sources = []  # Accumulate sources confirming the gene
            if node in oncokb_genes:
                sources.append("OncoKB")
            if node in ongene_genes:
                sources.append("OnGene")
            if node in ncg_genes:
                sources.append("NCG")
            if node in intogen_genes:
                sources.append("IntOGen")
            if sources:  # If the gene is confirmed by at least one source
                confirmed_predictions.append((node, score, ", ".join(sources)))
            predicted_genes.append((node, score, ", ".join(sources) if sources else ""))

    # Save predictions to a CSV file
    save_predictions_to_csv(predicted_genes, 'results/gene_prediction/', args.model_type, args.net_type, args.num_epochs)
    save_confirmed_predictions_to_csv(confirmed_predictions, 'results/gene_prediction/', args.model_type, args.net_type, args.num_epochs)

    # Load known cancer driver genes
    with open(drivers_file_path, 'r') as f:
        known_drivers = set(line.strip() for line in f)

    # Collect predicted cancer driver genes that match the known drivers
    predicted_driver_genes = [node_names[i] for i in non_labeled_nodes if node_names[i] in known_drivers]

    # Save the predicted known cancer driver genes to a CSV file
    save_predicted_known_drivers(predicted_driver_genes, 'results/gene_prediction/', args.model_type, args.net_type, args.num_epochs)

def load_gene_set(file_path):
    """
    Load a gene list from a file and return as a set.
    
    Args:
    - file_path: Path to the file containing genes, one per line.
    
    Returns:
    - Set of gene names.
    """
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)


def save_predictions_to_csv(predicted_genes, output_dir, model_type, net_type, num_epochs):
    """
    Save the predicted genes with their sources to a CSV file.
    
    Args:
    - predicted_genes: List of tuples (gene, score, sources) to save.
    - output_dir: Directory to save the CSV file.
    - model_type, net_type, num_epochs: For naming the output file.
    """
    os.makedirs(output_dir, exist_ok=True)
    predicted_genes_csv_path = os.path.join(output_dir, f'{model_type}_{net_type}_predicted_driver_genes_epo{num_epochs}_2048.csv')
    df_predictions = pd.DataFrame(predicted_genes, columns=["Gene", "Score", "Confirmed Sources"])
    df_predictions.to_csv(predicted_genes_csv_path, index=False)
    print(f"Predicted driver genes with confirmed sources saved to {predicted_genes_csv_path}")

def save_confirmed_predictions_to_csv(confirmed_predictions, output_dir, model_type, net_type, num_epochs):
    """
    Save confirmed predicted genes to a CSV file.
    
    Args:
    - confirmed_predictions: List of tuples (gene, score, sources).
    - output_dir: Directory to save the CSV file.
    - model_type, net_type, num_epochs: For naming the output file.
    """
    confirmed_predictions_csv_path = os.path.join(output_dir, f'{model_type}_{net_type}_confirmed_predicted_genes_epo{num_epochs}_2048.csv')
    df_confirmed = pd.DataFrame(confirmed_predictions, columns=["Gene", "Score", "Source"])
    df_confirmed.to_csv(confirmed_predictions_csv_path, index=False)
    print(f"Confirmed predicted genes saved to {confirmed_predictions_csv_path}")

def save_predicted_known_drivers(predicted_driver_genes, output_dir, model_type, net_type, num_epochs):
    """
    Save predicted known cancer driver genes to a CSV file.
    
    Args:
    - predicted_driver_genes: List of predicted cancer driver genes.
    - output_dir: Directory to save the CSV file.
    - model_type, net_type, num_epochs: For naming the output file.
    """
    predicted_drivers_csv_path = os.path.join(output_dir, f'{model_type}_{net_type}_predicted_known_drivers_epo{num_epochs}_2048.csv')
    df = pd.DataFrame(predicted_driver_genes, columns=["Gene"])
    df.to_csv(predicted_drivers_csv_path, index=False)
    print(f"Predicted known driver genes saved to {predicted_drivers_csv_path}")

def extract_summary_features_np_bio(bio_embeddings_np):
    """
    Extracts summary features from just the 1024 biological features (bio only).

    Args:
        bio_embeddings_np (np.ndarray): shape [num_nodes, 1024]

    Returns:
        np.ndarray: shape [num_nodes, 64]
    """
    num_nodes, num_features = bio_embeddings_np.shape
    summary_features = []

    assert num_features == 1024, f"Expected 1024 bio features, got {num_features}"

    for o_idx in range(4):  # 4 omics types
        for c_idx in range(16):  # 16 cancer types
            base = o_idx * 16 * 16 + c_idx * 16
            group = bio_embeddings_np[:, base:base + 16]  # [num_nodes, 16]
            max_vals = group.max(axis=1, keepdims=True)
            summary_features.append(max_vals)

    return np.concatenate(summary_features, axis=1)

def extract_summary_features_np_topo(topo_features_np):
    """
    Extracts summary features from the topological embedding section (features 1024–2047)
    by computing the max over each 16-dimensional segment.

    Args:
        features_np (np.ndarray): shape [num_nodes, 2048]

    Returns:
        np.ndarray: shape [num_nodes, 64]
    """
    num_nodes, num_features = topo_features_np.shape
    assert num_features == 1024, f"Expected 2048 features, got {num_features}"

    # Select topological features only
    ##topo_features = features_np[:, 1024:]  # shape: [num_nodes, 1024]
    ##topo_features = topo_features_np[:, 1024:2048]
    topo_features = topo_features_np  # already 1024 features


    summary_features = []

    # Pool over 64 chunks of 16 features
    for i in range(64):
        start = i * 16
        end = start + 16
        group = topo_features[:, start:end]  # shape: [num_nodes, 16]
        max_vals = group.max(axis=1, keepdims=True)  # shape: [num_nodes, 1]
        summary_features.append(max_vals)

    return np.concatenate(summary_features, axis=1)  # shape: [num_nodes, 64]

def compute_relevance_scores(model, graph, features, node_indices=None, method="saliency", use_abs=True, baseline=None, steps=50):
    """
    Computes relevance scores for selected nodes using either saliency (gradients) or integrated gradients (IG).

    Args:
        model: Trained GNN model
        graph: DGL graph
        features: Input node features (torch.Tensor or np.ndarray)
        node_indices: List/Tensor of node indices to compute relevance for. If None, auto-select using probs > 0.0
        method: "saliency" or "integrated_gradients"
        use_abs: Whether to use absolute values of gradients
        baseline: Baseline input for IG (default: zero vector)
        steps: Number of steps for IG approximation

    Returns:
        relevance_scores: Tensor of shape [num_nodes, num_features] (0s for nodes not analyzed)
    """
    model.eval()
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)

    features = features.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        logits = model(graph, features)
        probs = torch.sigmoid(logits.squeeze())

        if node_indices is None:
            node_indices = torch.nonzero(probs > 0.0, as_tuple=False).squeeze()
            if node_indices.ndim == 0:
                node_indices = node_indices.unsqueeze(0)

        relevance_scores = torch.zeros_like(features)

        for i, idx in enumerate(tqdm(node_indices, desc=f"Computing relevance ({method})", leave=True)):
            model.zero_grad()
            if features.grad is not None:
                features.grad.zero_()

            if method == "saliency":
                probs[idx].backward(retain_graph=(i != len(node_indices) - 1))
                grads = features.grad[idx]
                relevance_scores[idx] = grads.abs().detach() if use_abs else grads.detach()

            elif method == "integrated_gradients":
                # Define baseline
                if baseline is None:
                    baseline_input = torch.zeros_like(features)
                else:
                    baseline_input = baseline.clone().detach()

                # Generate scaled inputs
                total_grad = torch.zeros_like(features)
                for alpha in range(1, steps + 1):
                    scaled_input = baseline_input + (alpha / steps) * (features - baseline_input)
                    scaled_input.requires_grad_()

                    out = model(graph, scaled_input)
                    prob = torch.sigmoid(out.squeeze())[idx]

                    model.zero_grad()
                    if scaled_input.grad is not None:
                        scaled_input.grad.zero_()

                    prob.backward(retain_graph=True)
                    grad = scaled_input.grad
                    total_grad += grad

                avg_grad = total_grad / steps
                ig = (features - baseline_input) * avg_grad
                relevance_scores[idx] = ig[idx].abs() if use_abs else ig[idx]

            else:
                raise ValueError(f"Unknown method: {method}. Use 'saliency' or 'integrated_gradients'.")

    return relevance_scores




def filter_topk_midrange_relevance_genes(
    node_names,
    labels,
    scores,
    relevance_scores,
    output_dir,
    top_k=1000,
    relevance_q_low=0.1,
    relevance_q_high=0.9
):
    """
    Retains exactly top_k predicted genes (unlabeled) with relevance in mid quantile range.
    Plots score vs. relevance and returns names and relevance vectors.
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Compute relevance sum and normalize
    relevance_sums = relevance_scores.sum(dim=1).cpu().numpy()
    relevance_sums_norm = (relevance_sums - relevance_sums.min()) / (relevance_sums.max() - relevance_sums.min() + 1e-8)

    # Define relevance thresholds
    low_q, high_q = np.quantile(relevance_sums_norm, [relevance_q_low, relevance_q_high])

    # Only use unlabeled nodes
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    scored_relevance_data = [
        (i, node_names[i], scores[i], relevance_sums_norm[i])
        for i in non_labeled_nodes
    ]

    # Sort all by prediction score
    scored_relevance_data.sort(key=lambda x: x[2], reverse=True)

    # Retain top-k with mid-range relevance
    retained = []
    for i, name, score, rel in scored_relevance_data:
        if low_q <= rel <= high_q:
            retained.append((i, name, score, rel))
        if len(retained) == top_k:
            break

    if len(retained) < top_k:
        print(f"[Warning] Only {len(retained)} genes found in mid-relevance range.")

    # Plot
    discarded = [d for d in scored_relevance_data if d not in retained]

    plt.figure(figsize=(10, 6))
    plt.scatter([r for _, _, _, r in discarded], [s for _, _, s, _ in discarded], alpha=0.3, label='Discarded', color='lightgray')
    plt.scatter([r for _, _, _, r in retained], [s for _, _, s, _ in retained], alpha=0.8, label='Retained', color='#0077B6')
    plt.xlabel("Normalized Relevance Sum")
    plt.ylabel("Predicted Score (Sigmoid)")
    plt.title(f"Top-{top_k} Predicted Genes (Mid-range Relevance)")
    plt.legend()
    plot_path = os.path.join(output_dir, f"top{top_k}_mid_relevance_scatter.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"[Saved] Scatter plot → {plot_path}")

    # Return results
    retained_gene_names = [name for _, name, _, _ in retained]
    retained_indices = [i for i, _, _, _ in retained]
    retained_relevance = relevance_scores[retained_indices]

    return retained_gene_names, retained_relevance, retained_indices


def plot_predicted_genes_distribution(pred_counts, output_path):
    """
    Visualize the number of predicted cancer genes in each cluster.

    Args:
        pred_counts (dict): Dictionary mapping cluster_id -> number of predicted genes
        output_path (str): Path to save the plot
    """
    clusters = list(pred_counts.keys())
    counts = [pred_counts[c] for c in clusters]
    colors = [CLUSTER_COLORS.get(c, "#808080") for c in clusters]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(clusters, counts, color=colors, edgecolor='black')

    ax.set_title("Predicted Cancer Genes per Cluster", fontsize=16)
    ax.set_xlabel("Cluster ID", fontsize=14)
    ax.set_ylabel("Number of Predicted Genes", fontsize=14)
    ax.set_xticks(clusters)
    ax.set_xticklabels(clusters, rotation=0, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Add value labels on top
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def apply_biclustering_with_heatmap_plot_bio(
    graph,
    saliency_matrix: np.ndarray,
    node_names_topk: list,
    topk_node_indices,
    predicted_cancer_genes: list,
    cluster_colors: dict,
    omics_colors: dict,
    omics_splits,
    output_dir: str,
    args,
    n_clusters_row: int = 4,
    n_clusters_col: int = 4,
    top_k: int = 10,
    n_trials: int = 10,
    cmap: str = None
):
    """
    Full biclustering, graph annotation, saving, and heatmap plot.
    Mirrors `apply_full_spectral_biclustering_bio` but with heatmap figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    assert saliency_matrix.shape[0] == len(node_names_topk), "Row count mismatch"

    # === Normalize
    ##saliency_matrix_norm = (saliency_matrix - saliency_matrix.min()) / (saliency_matrix.max() - saliency_matrix.min()) * 10
    ## Darker plot $##################################################################################
    saliency_matrix_norm = normalize(saliency_matrix, axis=1)
    
    # === Plot unsorted (raw) heatmap with input-style layout
    unsorted_output_path = os.path.join(output_dir, "bio_biclustering_unsorted_input_style_heatmap.png")

    ##saliency_matrix_norm = normalize(saliency_matrix, axis=1)
    vmin, vmax = 0, np.percentile(saliency_matrix_norm, 99)

    # Reuse or define omics + feature labels again for consistency
    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'KidneyPap',
        'Liver', 'LungAd', 'LungSc', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    omics_order = ['CNA', 'GE', 'METH', 'MF']
    feature_names = [f"{omics}: {cancer}" for omics in omics_order for cancer in cancer_names]
    feature_names = feature_names[:saliency_matrix.shape[1]]
    reordered_feature_labels = [f.split(": ")[1] for f in feature_names]
    reordered_omics_labels = [f.split(": ")[0] for f in feature_names]

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])

    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.4, 19], hspace=0.0)

    # Dummy top bar (white, just for layout alignment)
    ax_col_dummy = fig.add_subplot(gs[0])
    ax_col_dummy.set_xlim([0, saliency_matrix.shape[1]])
    ax_col_dummy.set_xticks([])
    ax_col_dummy.set_yticks([])
    ax_col_dummy.set_frame_on(False)
    ax_col_dummy.imshow(
        np.ones((1, saliency_matrix.shape[1], 3)),
        extent=[0, saliency_matrix.shape[1], 0, 1],
        aspect='auto'
    )

    # Raw heatmap
    ax_raw = fig.add_subplot(gs[1])
    sns.heatmap(
        saliency_matrix_norm,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax_raw
    )

    # X-axis labels (cancer types) and color by omics
    ax_raw.set_xticks(np.arange(len(reordered_feature_labels)) + 0.5)
    ax_raw.set_xticklabels(reordered_feature_labels, rotation=90, fontsize=24)
    ax_raw.tick_params(axis='x', which='both', bottom=True, top=False, length=5)

    for label, omics in zip(ax_raw.get_xticklabels(), reordered_omics_labels):
        label.set_color(omics_colors.get(omics.upper(), 'black'))

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.0)
    plt.savefig(unsorted_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Unsorted input-style heatmap → {unsorted_output_path}")

    # === Feature labels
    # cancer_names = [
    #     'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'KidneyPap',
    #     'Liver', 'LungAd', 'LungSc', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    # ]
    # omics_order = ['CNA', 'GE', 'METH', 'MF']
    # feature_names = [f"{omics}: {cancer}" for omics in omics_order for cancer in cancer_names]
    # feature_names = feature_names[:saliency_matrix.shape[1]]
    
    saliency_matrix = normalize(saliency_matrix, axis=1)

    # === Run biclustering trials
    best_model, best_score = None, np.inf
    valid = False
    for i in tqdm(range(n_trials * 3), desc="Biclustering Trials", ncols=80):
        model = SpectralBiclustering(n_clusters=(n_clusters_row, n_clusters_col), method='log', random_state=i)
        model.fit(saliency_matrix)
        reordered = saliency_matrix[np.argsort(model.row_labels_)][:, np.argsort(model.column_labels_)]
        mse = mean_squared_error(saliency_matrix, reordered)

        col_labels = model.column_labels_
        cluster_to_cancers = defaultdict(set)
        for idx, cid in enumerate(col_labels):
            cancer = feature_names[idx].split(": ")[1]
            cluster_to_cancers[cid].add(cancer)

        if all(len(cancers) >= 2 for cancers in cluster_to_cancers.values()) and mse < best_score:
            best_model, best_score = model, mse
            valid = True

    if not valid:
        raise RuntimeError("❌ Valid biclustering not found.")

    model = best_model
    row_labels = model.row_labels_
    col_labels = model.column_labels_

    # === Sort
    row_order = []
    for cid in np.unique(row_labels):
        cluster_idx = np.where(row_labels == cid)[0]
        row_sums = saliency_matrix[cluster_idx].sum(axis=1)
        sorted_idx = cluster_idx[np.argsort(-row_sums)]
        row_order.extend(sorted_idx)

    feature_avgs = saliency_matrix.mean(axis=0)
    
    # === Build group → indices mapping
    omics_groups = {}
    for i, fname in enumerate(feature_names):
        omics = fname.split(":")[0].strip().upper()
        if omics not in omics_groups:
            omics_groups[omics] = []
        omics_groups[omics].append(i)

    # === Compute mean per omics group
    omics_means = {}
    for omics, indices in omics_groups.items():
        omics_means[omics] = feature_avgs[indices].mean()

    # === Sort omics groups descending by mean
    sorted_omics = [o for o, _ in sorted(omics_means.items(), key=lambda x: -x[1])]

    print(f"🔍 Omics groups sorted by mean: {sorted_omics}")

    # === Now build final col_order
    col_order = []
    reordered_feature_labels, reordered_omics_labels, feature_colors = [], [], []
    for omics in sorted_omics:
        indices = omics_groups[omics]
        sorted_within = [i for _, i in sorted(zip(feature_avgs[indices], indices), reverse=True)]
        col_order.extend(sorted_within)
        reordered_feature_labels.extend([feature_names[i].split(": ")[1] for i in sorted_within])
        reordered_omics_labels.extend([feature_names[i].split(": ")[0] for i in sorted_within])
        feature_colors.extend([omics_colors.get(omics.upper(), 'gray')] * len(sorted_within))


    clustered_matrix = saliency_matrix[row_order][:, col_order]
    reordered_gene_names = [node_names_topk[i] for i in row_order]
    reordered_cluster_labels = row_labels[row_order]

    # === Save output tables
    pd.DataFrame({"Gene": reordered_gene_names, "Cluster": reordered_cluster_labels})\
        .to_csv(os.path.join(output_dir, "bio_biclustering_row_labels.csv"), index=False)
    pd.DataFrame(clustered_matrix, index=reordered_gene_names, columns=reordered_feature_labels)\
        .to_csv(os.path.join(output_dir, "bio_biclustering_heatmap_matrix.csv"))
    pd.DataFrame({
        gene: [reordered_feature_labels[i] for i in np.argsort(-clustered_matrix[i])[:top_k]]
        for i, gene in enumerate(reordered_gene_names)
    }).T.rename(columns=lambda i: f"Top{i+1}")\
     .to_csv(os.path.join(output_dir, f"bio_biclustering_top{top_k}_features.csv"))

    # === Figure
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
    ##vmin, vmax = 0, np.percentile(clustered_matrix, 99)

    '''fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 0.1, 20], hspace=0.0)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=[20, 1], wspace=0.0)
    gs_lower = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], width_ratios=[20, 1], wspace=0.0)
    ax_top_bar = fig.add_subplot(gs_top[0])
    ax_heatmap = fig.add_subplot(gs_lower[0])
    ax_curve = fig.add_subplot(gs_lower[1], sharey=ax_heatmap)'''
    # fig = plt.figure(figsize=(22, 16))
    # ##gs = gridspec.GridSpec(2, 2, height_ratios=[1, 20], width_ratios=[20, 1], hspace=0.0, wspace=0.0)
    # gs = gridspec.GridSpec(2, 2, height_ratios=[2, 20], width_ratios=[20, 1], hspace=0.0, wspace=0.0)
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.8, 20], width_ratios=[20, 0.5], hspace=0.0, wspace=0.0)
    
    ax_top_bar = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[1, 0])
    ax_curve = fig.add_subplot(gs[1, 1], sharey=ax_heatmap)

    # ax_top_bar.bar(
    #     x=np.arange(len(feature_avgs[col_order])) + 0.5,
    #     height=feature_avgs[col_order],
    #     width=1.0,
    #     color=feature_colors,
    #     edgecolor='black',
    #     linewidth=0.3
    # )
    
    # Normalize feature averages to [0, 1]
    heights = feature_avgs[col_order]
    norm_heights = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)

    # Plot with alpha-scaled bars
    for i, (norm_h, color) in enumerate(zip(norm_heights, feature_colors)):
        ax_top_bar.bar(
            x=i + 0.5,
            height=norm_h,#*0.5,
            width=1.0,
            color=color,
            edgecolor='none',
            linewidth=0,
            alpha=0.3 + 0.7 * norm_h
        )

    ax_top_bar.set_ylim(0, 1.1)
    # Safer xlim
    ax_top_bar.set_xlim(-0.5, len(feature_avgs[col_order]) + 0.5)
    # Turn off clutter
    ax_top_bar.axis("off")

    # ax_top_bar.set_xlim(0, len(feature_avgs[col_order]))
    # #ax_top_bar.set_ylim(0, 1.05)
    # ax_top_bar.set_ylim(0, feature_avgs[col_order].max() * 1.2)
    ax_top_bar.set_xticks([])
    ax_top_bar.set_yticks([])
    for spine in ax_top_bar.spines.values():
        spine.set_visible(False)

    sns.heatmap(
        clustered_matrix,
        cmap=cmap,
        # vmin=vmin,
        # vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax_heatmap
    )
    for i, cluster in enumerate(reordered_cluster_labels):
        ax_heatmap.add_patch(
            plt.Rectangle(
                (-1.5, i), 1.5, 1,
                linewidth=0,
                facecolor=to_rgba(cluster_colors.get(cluster, '#FFFFFF')),
                clip_on=False
            )
        )
    unique_clusters, cluster_sizes = np.unique(reordered_cluster_labels, return_counts=True)
    start = 0
    for cluster, size in zip(unique_clusters, cluster_sizes):
        center_y = start + size / 2
        ax_heatmap.text(-2.0, center_y, f"{size}", va='center', ha='right', fontsize=22)
        start += size

    ax_heatmap.set_xticks(np.arange(len(reordered_feature_labels)) + 0.5)
    ax_heatmap.set_xticklabels(reordered_feature_labels, rotation=90, fontsize=20)
    ax_heatmap.tick_params(axis='x', which='both', bottom=True, top=False, length=5, pad=1)
    for label, omics in zip(ax_heatmap.get_xticklabels(), reordered_omics_labels):
        label.set_color(omics_colors.get(omics.upper(), 'black'))

    saliency_sums = clustered_matrix.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())
    ax_curve.xaxis.set_ticks_position('top')
    ax_curve.xaxis.set_label_position('top')
    ax_curve.set_xlim([0, 1])
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=18)
    ax_curve.set_ylim(0, len(saliency_sums))
    ax_curve.invert_yaxis()
    for spine in ax_curve.spines.values():
        spine.set_visible(False)
    ax_curve.tick_params(axis='y', left=False, labelleft=False)

    current_idx = 0
    for cluster_id, cluster_size in zip(unique_clusters, cluster_sizes):
        y_vals = np.arange(current_idx, current_idx + cluster_size)
        cluster_saliency = saliency_sums[current_idx: current_idx + cluster_size]
        cluster_color = cluster_colors.get(cluster_id, '#CCCCCC')
        ax_curve.fill_betweenx(y_vals, 0, cluster_saliency, color=cluster_color, alpha=0.9, linewidth=0)
        current_idx += cluster_size

    fig.subplots_adjust(top=0.99, bottom=0.01, hspace=0.0)
    out_fig_path = os.path.join(output_dir, "bio_biclustering_heatmap_curve.png")
    plt.savefig(out_fig_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"[Saved] Heatmap → {out_fig_path}")

    # === Graph + stats
    row_labels_tensor = torch.full((graph.num_nodes(),), -1, dtype=torch.long)
    row_labels_tensor[topk_node_indices] = torch.tensor(row_labels, dtype=torch.long)
    graph.ndata['cluster_bio_summary'] = row_labels_tensor

    total_genes_per_cluster = {cid: int((row_labels == cid).sum()) for cid in np.unique(row_labels)}
    pred_counts = {}
    for cid in np.unique(row_labels):
        idx = np.where(row_labels == cid)[0]
        names_in_cluster = [node_names_topk[i] for i in idx]
        pred_counts[cid] = sum(1 for g in names_in_cluster if g in predicted_cancer_genes)

    pd.DataFrame(list(total_genes_per_cluster.items()), columns=["Cluster", "TotalGenes"])\
        .to_csv(os.path.join(output_dir, "bio_biclustering_total_genes.csv"), index=False)
    pd.DataFrame(list(pred_counts.items()), columns=["Cluster", "PredictedGenes"])\
        .to_csv(os.path.join(output_dir, "bio_biclustering_predicted_counts.csv"), index=False)

    #################
    # === Save detailed cluster assignment ===
    assignment_rows = []
    for i, (idx, cluster) in enumerate(zip(topk_node_indices, row_labels)):
        gene_name = node_names_topk[i] if i < len(node_names_topk) else f"node_{idx}"
        rel_score = saliency_matrix[i].sum().item()
        assignment_rows.append([gene_name, gene_name, idx, round(rel_score, 5), cluster])

    assignment_rows.sort(key=lambda x: (x[4], -x[3]))

    assignment_df = pd.DataFrame(
        assignment_rows,
        columns=["Source_Gene", "Neighbor_Name", "Neighbor_Index", "Relevance_Score", "Cluster"]
    )
    assignment_csv_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_bio_biclustering_gene_assignments.csv"
    )
    assignment_df.to_csv(assignment_csv_path, index=False)
    print(f"📄 Saved gene–neighbor cluster assignments: {assignment_csv_path}")

    # === Extra plots ===
    plot_bio_biclustering_clustermap(
        args=args,
        relevance_scores=saliency_matrix,
        omics_splits=omics_splits,
        output_path=os.path.join(output_dir, "spectral_biclustering_bio_fixed_clustermap.png"),
        row_labels=row_labels,
        col_labels=col_labels,
        # cluster_colors=cluster_colors
    )

    plot_predicted_genes_distribution(
        pred_counts=pred_counts,
        output_path=os.path.join(output_dir, "predicted_genes_per_cluster.png")
    )

    return graph, row_labels, col_labels, total_genes_per_cluster, pred_counts


def apply_biclustering_with_heatmap_plot_topo(
    graph,
    saliency_matrix: np.ndarray,
    node_names_topk: list,
    topk_node_indices,
    predicted_cancer_genes: list,
    cluster_colors: dict,
    output_dir: str,
    args,
    n_clusters_row: int = 4,
    n_clusters_col: int = 4,
    top_k: int = 10,
    n_trials: int = 10,
    cmap: str = None
):
    # best_model, best_score = None, np.inf
    # valid = False

    # for i in tqdm(range(n_trials * 10), desc="Biclustering Trials", ncols=80):
    #     model = SpectralBiclustering(
    #         n_clusters=(n_clusters_row, n_clusters_col),
    #         method='log',
    #         random_state=i
    #     )
    #     model.fit(saliency_matrix)

    #     row_labels = model.row_labels_
    #     col_labels = model.column_labels_

    #     # === Check: Does every COLUMN cluster have at least 2 predicted genes/features? ===
    #     pass_check = True
    #     for cid in np.unique(col_labels):
    #         cluster_idx = np.where(col_labels == cid)[0]
    #         # For columns, you usually check the feature *names* — here I assume you have that:
    #         # If your columns have names, supply them.
    #         names_in_cluster = [f"feat_{j}" for j in cluster_idx]  # Replace with real names if you have them!
    #         pred_count = sum(1 for f in names_in_cluster if f in predicted_cancer_genes)
    #         if pred_count < 2:
    #             pass_check = False
    #             break

    #     if not pass_check:
    #         continue  # Reject and try next

    #     # === Evaluate MSE for tie-breaking ===
    #     reordered = saliency_matrix[np.argsort(row_labels)][:, np.argsort(col_labels)]
    #     mse = mean_squared_error(saliency_matrix, reordered)

    #     if mse < best_score:
    #         best_model, best_score = model, mse
    #         valid = True

    # if not valid:
    #     raise RuntimeError("❌ Valid biclustering not found that satisfies min predicted genes/features per column cluster.")

    os.makedirs(output_dir, exist_ok=True)
    assert saliency_matrix.shape[0] == len(node_names_topk), "Row count mismatch"

    saliency_matrix = normalize(saliency_matrix, axis=1)

    best_model, best_score = None, np.inf
    valid = False
    for i in tqdm(range(n_trials * 3), desc="Biclustering Trials", ncols=80):
        model = SpectralBiclustering(n_clusters=(n_clusters_row, n_clusters_col), method='log', random_state=i)
        model.fit(saliency_matrix)
        reordered = saliency_matrix[np.argsort(model.row_labels_)][:, np.argsort(model.column_labels_)]
        mse = mean_squared_error(saliency_matrix, reordered)
        if mse < best_score:
            best_model, best_score = model, mse
            valid = True

    if not valid:
        raise RuntimeError("❌ Valid biclustering not found.")

    model = best_model
    row_labels = model.row_labels_
    col_labels = model.column_labels_

    row_order = []
    for cid in np.unique(row_labels):
        cluster_idx = np.where(row_labels == cid)[0]
        row_sums = saliency_matrix[cluster_idx].sum(axis=1)
        sorted_idx = cluster_idx[np.argsort(-row_sums)]
        row_order.extend(sorted_idx)

    col_order = []
    for cid in np.unique(col_labels):
        cluster_idx = np.where(col_labels == cid)[0]
        col_sums = saliency_matrix[:, cluster_idx].sum(axis=0)
        sorted_idx = cluster_idx[np.argsort(-col_sums)]
        col_order.extend(sorted_idx)

    clustered_matrix = saliency_matrix[row_order][:, col_order]
    reordered_gene_names = [node_names_topk[i] for i in row_order]
    reordered_cluster_labels = row_labels[row_order]
    reordered_col_labels = col_labels[col_order]

    pd.DataFrame({"Gene": reordered_gene_names, "Cluster": reordered_cluster_labels})\
        .to_csv(os.path.join(output_dir, "topo_biclustering_row_labels.csv"), index=False)
    pd.DataFrame(clustered_matrix, index=reordered_gene_names)\
        .to_csv(os.path.join(output_dir, "topo_biclustering_heatmap_matrix.csv"))

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])

    # Get col_order by grouping features by their column clusters
    col_order = []
    col_cluster_order = []
    for cid in np.unique(col_labels):
        idxs = np.where(col_labels == cid)[0]
        col_order.extend(idxs)
        col_cluster_order.extend([cid] * len(idxs))

    col_avgs = saliency_matrix.mean(axis=0)

    # Reorder matrix and cluster labels
    clustered_matrix = saliency_matrix[row_order][:, col_order]
    reordered_gene_names = [node_names_topk[i] for i in row_order]
    reordered_cluster_labels = row_labels[row_order]

    fig = plt.figure(figsize=(20, 16))
    # gs = gridspec.GridSpec(3, 2, height_ratios=[0.6, 0.5, 20], width_ratios=[20, 0.5], hspace=0.0, wspace=0.0)
    
    gs = gridspec.GridSpec(3, 2,
        height_ratios=[0.8, 0.3, 20],
        width_ratios=[20, 0.5],
        hspace=0.0, wspace=0.0)

    # === Top bar goes in row 0 ===
    ax_top_bar = fig.add_subplot(gs[0, 0])

    # === Color stripe goes in row 1 ===
    ax_col_cluster = fig.add_subplot(gs[1, 0])



    # === Top stripe showing column cluster colors
    # ax_col_cluster = fig.add_subplot(gs[0, 0])
    col_cluster_colors = [cluster_colors.get(cid, '#FFFFFF') for cid in col_cluster_order]
    col_cluster_rgb = np.array([[to_rgb(c) for c in col_cluster_colors]])
    ax_col_cluster.imshow(col_cluster_rgb, aspect='auto', extent=[0, len(col_order), 0, 1])
    ax_col_cluster.set_xlim([0, len(col_order)])
    ax_col_cluster.set_xticks([])
    ax_col_cluster.set_yticks([])
    ax_col_cluster.set_frame_on(False)

    # === Top bar with column feature values
    # ax_top_bar = fig.add_subplot(gs[1, 0])
    ax_top_bar.bar(
        x=np.arange(clustered_matrix.shape[1]) + 0.5,
        height=col_avgs[col_order],
        width=1.0,
        color='#4682B4',
        edgecolor='black',
        linewidth=0.3
    )
    ax_top_bar.set_xlim(0, clustered_matrix.shape[1])
    ax_top_bar.set_ylim(0, clustered_matrix[:, col_order].max() * 1.2)
    ax_top_bar.set_xticks([])
    ax_top_bar.set_yticks([])
    ax_top_bar.set_frame_on(False)

    # === Main heatmap
    ax_heatmap = fig.add_subplot(gs[2, 0])
    sns.heatmap(
        clustered_matrix,
        cmap=cmap,
        xticklabels=True,
        yticklabels=False,
        cbar=False,
        ax=ax_heatmap
    )

    # Add row cluster color patches
    for i, cluster in enumerate(reordered_cluster_labels):
        ax_heatmap.add_patch(
            plt.Rectangle((-1.5, i), 1.5, 1, linewidth=0, facecolor=to_rgba(cluster_colors.get(cluster, '#FFFFFF')), clip_on=False)
        )

    # Add cluster size labels on left
    unique_clusters, cluster_sizes = np.unique(reordered_cluster_labels, return_counts=True)
    start = 0
    for cluster, size in zip(unique_clusters, cluster_sizes):
        center_y = start + size / 2
        ax_heatmap.text(-2.0, center_y, f"{size}", va='center', ha='right', fontsize=22)
        start += size

    # === X-tick: Original feature indices, colored by cluster
    ax_heatmap.set_xticks(np.arange(len(col_order)) + 0.5)
    ax_heatmap.set_xticklabels([str(i) for i in col_order], rotation=90, fontsize=20)
    for label, cid in zip(ax_heatmap.get_xticklabels(), col_cluster_order):
        label.set_color(cluster_colors.get(cid, 'black'))

    ax_heatmap.tick_params(axis='x', which='both', bottom=True, top=False, length=5, pad=2)

    # === Right saliency curves
    ax_curve = fig.add_subplot(gs[2, 1], sharey=ax_heatmap)
    saliency_sums = clustered_matrix.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())

    ax_curve.xaxis.set_ticks_position('top')
    ax_curve.xaxis.set_label_position('top')
    ax_curve.set_xlim([0, 1])
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=14)
    ax_curve.set_ylim(0, len(saliency_sums))
    ax_curve.invert_yaxis()
    ax_curve.tick_params(axis='y', left=False, labelleft=False)
    for spine in ax_curve.spines.values():
        spine.set_visible(False)

    current_idx = 0
    for cluster_id, cluster_size in zip(unique_clusters, cluster_sizes):
        y_vals = np.arange(current_idx, current_idx + cluster_size)
        cluster_saliency = saliency_sums[current_idx: current_idx + cluster_size]
        cluster_color = cluster_colors.get(cluster_id, '#CCCCCC')
        ax_curve.fill_betweenx(y_vals, 0, cluster_saliency, color=cluster_color, alpha=0.9, linewidth=0)
        current_idx += cluster_size

    # === Final adjustments
    fig.subplots_adjust(top=0.98, bottom=0.01)
    out_fig_path = os.path.join(output_dir, "topo_biclustering_heatmap_curve.png")
    plt.savefig(out_fig_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"[Saved] Topo Heatmap → {out_fig_path}")

    # === Stats
    row_labels_tensor = torch.full((graph.num_nodes(),), -1, dtype=torch.long)
    row_labels_tensor[topk_node_indices] = torch.tensor(row_labels, dtype=torch.long)
    graph.ndata['cluster_topo_summary'] = row_labels_tensor

    total_genes_per_cluster = {cid: int((row_labels == cid).sum()) for cid in np.unique(row_labels)}
    pred_counts = {}
    for cid in np.unique(row_labels):
        idx = np.where(row_labels == cid)[0]
        names_in_cluster = [node_names_topk[i] for i in idx]
        pred_counts[cid] = sum(1 for g in names_in_cluster if g in predicted_cancer_genes)

    pd.DataFrame(list(total_genes_per_cluster.items()), columns=["Cluster", "TotalGenes"])\
        .to_csv(os.path.join(output_dir, "topo_biclustering_total_genes.csv"), index=False)
    pd.DataFrame(list(pred_counts.items()), columns=["Cluster", "PredictedGenes"])\
        .to_csv(os.path.join(output_dir, "topo_biclustering_predicted_counts.csv"), index=False)

    # ✅✅✅ You MUST have:
    return graph, row_labels, col_labels, total_genes_per_cluster, pred_counts


def plot_interactions_with_kcgs(data, output_path):
    """
    Creates a box plot with individual data points showing 
    the number of interactions with known cancer genes (KCGs).
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    sns.set_style("white")

    unique_clusters = sorted(data['Cluster'].unique())
    cluster_color_map = {cluster_id: CLUSTER_COLORS[cluster_id] for cluster_id in unique_clusters}

    ax = sns.boxplot(
        x='Cluster', y='Interactions', data=data, 
        hue='Cluster', palette=cluster_color_map, showfliers=False
    )

    sns.stripplot(
        x='Cluster', y='Interactions', data=data, 
        color='black', alpha=0.2, jitter=True, size=1.5
    )

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # 👉 Adjust x-axis limits for more space on the left
    num_clusters = len(unique_clusters)
    ax.set_xlim(-0.55, num_clusters - 0.65)  # Or adjust -0.4, -0.5, etc. for more space

    plt.ylabel("Number of interactions with KCGs", fontsize=20)
    plt.xlabel("")
    plt.xticks(rotation=0, ha="right", fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 50)

    sns.despine()  # 🔻 Remove top/right spines

    plt.savefig(output_path, dpi=300, bbox_inches="tight")  
    plt.close()

    print(f"✅ Plot saved to {output_path}")


def plot_interactions_with_pcgs(data, output_path):
    """
    Creates a box plot with individual data points showing 
    the number of interactions with predicted cancer genes (PCGs).
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    sns.set_style("white")

    unique_clusters = sorted(data['Cluster'].unique())
    cluster_color_map = {cluster_id: CLUSTER_COLORS[cluster_id] for cluster_id in unique_clusters}

    ax = sns.boxplot(
        x='Cluster', y='Interactions', data=data, 
        hue='Cluster', palette=cluster_color_map, showfliers=False
    )

    sns.stripplot(
        x='Cluster', y='Interactions', data=data, 
        color='black', alpha=0.2, jitter=True, size=1.5
    )

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # 👉 Adjust x-axis limits for more space on the left
    num_clusters = len(unique_clusters)
    ax.set_xlim(-0.55, num_clusters - 0.65)  # Or adjust -0.4, -0.5, etc. for more space

    plt.ylabel("Number of interactions with PCGs", fontsize=20)
    plt.xlabel("")
    plt.xticks(rotation=0, ha="right", fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 50)

    sns.despine()  # 🔻 Remove top/right spines

    plt.savefig(output_path, dpi=300, bbox_inches="tight")  
    plt.close()

    print(f"✅ Plot saved to {output_path}")


def plot_enriched_term_counts(enrichment_results, output_path, model_type, net_type, num_epochs, bio_color='#1f77b4', topo_color='#ff7f0e'):
    """
    Plot bar chart of the number of enriched terms per cluster for bio and topo clusters,
    with x-axis labels colored according to their type.

    Parameters:
        enrichment_results (dict): Dictionary with enrichment results for 'bio' and 'topo' clusters.
        output_path (str): Path to save the output plot.
        model_type (str): Model type for naming the file.
        net_type (str): Network type for naming the file.
        num_epochs (int): Number of epochs for naming the file.
        bio_color (str): Color for bio bars and labels.
        topo_color (str): Color for topo bars and labels.
        
        plt.plot(bio_scores_rel, label='Bio', color='#1f77b4')
        plt.plot(topo_scores_rel, label='Topo', color='#ff7f0e')
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = []
    xtick_colors = []
    xtick_labels = []

    for cluster_type in ['bio', 'topo']:
        cluster_ids = list(enrichment_results[cluster_type].keys())
        term_counts = [len(res) for res in enrichment_results[cluster_type].values()]
        labels = [f"{cluster_type.capitalize()}_{i}" for i in cluster_ids]

        color = bio_color if cluster_type == 'bio' else topo_color
        bar_container = ax.bar(labels, term_counts, label=cluster_type, color=color)
        bars.extend(bar_container)

        xtick_labels.extend(labels)
        xtick_colors.extend([color] * len(labels))

    # Adjust x-axis limits
    ax.set_xlim(-0.65, len(bars) - 0.5)

    # Labeling
    ax.set_ylabel("Number of enriched terms", fontsize=28)
    ##ax.set_title("Functional Coherence: Enriched Term Counts per Cluster", fontsize=28)

    # Set custom x-tick labels and colors
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=20)
    for tick_label, color in zip(ax.get_xticklabels(), xtick_colors):
        tick_label.set_color(color)

    # Set y-tick font size
    ax.tick_params(axis='y', labelsize=20)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ##ax.legend(frameon=False, fontsize=24)

    plt.tight_layout()
    filename = f"{model_type}_{net_type}_term_counts_barplot_epo{num_epochs}.png"
    plt.savefig(os.path.join(output_path, filename), dpi=300)
    plt.close()

    print(f"Bar plot saved to {os.path.join(output_path, filename)}")

def plot_shared_enriched_pathways_venn(enrichment_results, output_path, model_type, net_type, num_epochs, bio_color='#1f77b4', topo_color='#ff7f0e'):
    """
    Plots a Venn diagram showing the overlap of enriched pathways between bio and topo clusters.

    Parameters:
        enrichment_results (dict): Dictionary with enrichment DataFrames under 'bio' and 'topo'.
        output_path (str): Path to save the output plot.
        model_type (str): Model type for naming the file.
        net_type (str): Network type for naming the file.
        num_epochs (int): Number of epochs for naming the file.
        bio_color (str): Color for the bio set in the Venn diagram.
        topo_color (str): Color for the topo set in the Venn diagram.
    """
    bio_terms = set(
        sum([df['name'].tolist() for df in enrichment_results['bio'].values() if not df.empty], [])
    )
    topo_terms = set(
        sum([df['name'].tolist() for df in enrichment_results['topo'].values() if not df.empty], [])
    )

    plt.figure(figsize=(8, 8))
    venn = venn2(
        [bio_terms, topo_terms],
        set_labels=('Bio', 'Topo'),
        set_colors=(bio_color, topo_color),
        alpha=0.7
    )

    # Set font size for all Venn labels and subset counts
    for text in venn.set_labels:
        if text:
            text.set_fontsize(28)
    for text in venn.subset_labels:
        if text:
            text.set_fontsize(28)

    plt.title("Overlap of enriched pathways", fontsize=16)

    filename = f"{model_type}_{net_type}_shared_pathways_venn_epo{num_epochs}.png"
    venn_path = os.path.join(output_path, filename)
    plt.savefig(venn_path, dpi=300)
    plt.close()

    print(f"Venn diagram saved to {venn_path}")

def plot_omics_barplot_bio(df, output_path=None):
    """
    Plot omics relevance for biological features with format like 'MF:BRCA'.
    """
    omics_order = ['cna', 'ge', 'meth', 'mf']
    omics_colors = {
        'cna': '#9370DB',    # purple
        'ge': '#228B22',      # dark green
        'meth': '#00008B',   # dark blue
        'mf': '#b22222',     # dark red
    }

    # Extract 'Omics' and 'Cancer' from features like 'MF:BRCA'
    df[['Omics', 'Cancer']] = df['Feature'].str.split(':', expand=True)
    df['Omics'] = df['Omics'].str.lower()

    omics_relevance = df.groupby('Omics')['Relevance'].sum().reindex(omics_order)

    _plot_bar(omics_relevance, omics_colors, omics_order, output_path)

def plot_omics_barplot_topo(df, output_path=None):
    """
    Plot omics relevance for topological features with format like 'BRCA_mf'.
    """
    omics_order = ['cna', 'ge', 'meth', 'mf']
    omics_colors = {
        'cna': '#9370DB',    # purple
        'ge': '#228B22',      # dark green
        'meth': '#00008B',   # dark blue
        'mf': '#b22222',     # dark red
    }

    # Extract 'Cancer' and 'Omics' from features like 'BRCA_mf'
    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)
    df['Omics'] = df['Omics'].str.lower()

    omics_relevance = df.groupby('Omics')['Relevance'].sum().reindex(omics_order)

    _plot_bar(omics_relevance, omics_colors, omics_order, output_path)



def plot_pcg_cancer_genes(
    clusters,
    predicted_cancer_genes_count,
    total_genes_per_cluster,
    node_names,
    row_labels,
    output_path):
    """
    Plots the percentage of predicted cancer genes per cluster.
    """

    # Convert to sorted array
    clusters = np.array(sorted(total_genes_per_cluster.keys()))
    total_genes_array = np.array([total_genes_per_cluster[c] for c in clusters])
    predicted_counts = np.array([predicted_cancer_genes_count.get(c, 0) for c in clusters])

    # Compute percentages
    percent_predicted = np.divide(predicted_counts, total_genes_array, where=total_genes_array > 0)

    # Prepare bar colors
    colors = [CLUSTER_COLORS.get(c, '#333333') for c in clusters]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(clusters, percent_predicted, color=colors, edgecolor='black')

    # Annotate with raw count
    for bar, cluster_id in zip(bars, clusters):
        height = bar.get_height()
        count = predicted_cancer_genes_count.get(cluster_id, 0)
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(count),
                ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Left margin space
    num_clusters = len(clusters)
    ax.set_xlim(-0.55, num_clusters - 0.65)

    # Labels and formatting
    ax.set_ylabel("Percent of PCGs", fontsize=20)
    plt.xlabel("")  # No xlabel here
    plt.xticks(clusters, fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_ylim(0, max(percent_predicted) + 0.1)

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Plot saved to {output_path}")

def plot_kcg_cancer_genes(clusters, kcg_count, total_genes_per_cluster, node_names, row_labels, output_path):

    cluster_ids = sorted(clusters)
    total = [total_genes_per_cluster[c] for c in cluster_ids]
    kcgs = [kcg_count.get(c, 0) for c in cluster_ids]
    proportions = [k / t if t > 0 else 0 for k, t in zip(kcgs, total)]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(cluster_ids, proportions, 
                   color=[CLUSTER_COLORS.get(c, '#333333') for c in cluster_ids],
                   edgecolor='black')

    # Annotate each bar with the raw KCG count
    for bar, cluster_id in zip(bars, cluster_ids):
        height = bar.get_height()
        count = kcg_count.get(cluster_id, 0)
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(count), 
                 ha='center', va='bottom', fontsize=16, fontweight='bold')

    ax = plt.gca()
    num_clusters = len(cluster_ids)
    ax.set_xlim(-0.55, num_clusters - 0.65)

    # Formatting
    ##plt.xlabel("Cluster ID", fontsize=16)
    plt.ylabel("Percent of KCGs", fontsize=20)
    plt.xlabel("")
    plt.xticks(cluster_ids, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, max(proportions) + 0.1)

    sns.despine()  # 🔻 Remove top/right spines

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_enriched_term_counts(enrichment_results, output_path, model_type, net_type, num_epochs, bio_color='#1f77b4', topo_color='#ff7f0e'):
    """
    Plot bar chart of the number of enriched terms per cluster for bio and topo clusters,
    with x-axis labels colored according to their type.

    Parameters:
        enrichment_results (dict): Dictionary with enrichment results for 'bio' and 'topo' clusters.
        output_path (str): Path to save the output plot.
        model_type (str): Model type for naming the file.
        net_type (str): Network type for naming the file.
        num_epochs (int): Number of epochs for naming the file.
        bio_color (str): Color for bio bars and labels.
        topo_color (str): Color for topo bars and labels.
        
        plt.plot(bio_scores_rel, label='Bio', color='#1f77b4')
        plt.plot(topo_scores_rel, label='Topo', color='#ff7f0e')
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = []
    xtick_colors = []
    xtick_labels = []

    for cluster_type in ['bio', 'topo']:
        cluster_ids = list(enrichment_results[cluster_type].keys())
        term_counts = [len(res) for res in enrichment_results[cluster_type].values()]
        labels = [f"{cluster_type.capitalize()}_{i}" for i in cluster_ids]

        color = bio_color if cluster_type == 'bio' else topo_color
        bar_container = ax.bar(labels, term_counts, label=cluster_type, color=color)
        bars.extend(bar_container)

        xtick_labels.extend(labels)
        xtick_colors.extend([color] * len(labels))

    # Adjust x-axis limits
    ax.set_xlim(-0.65, len(bars) - 0.5)

    # Labeling
    ax.set_ylabel("Number of enriched terms", fontsize=28)
    ##ax.set_title("Functional Coherence: Enriched Term Counts per Cluster", fontsize=28)

    # Set custom x-tick labels and colors
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=20)
    for tick_label, color in zip(ax.get_xticklabels(), xtick_colors):
        tick_label.set_color(color)

    # Set y-tick font size
    ax.tick_params(axis='y', labelsize=20)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ##ax.legend(frameon=False, fontsize=24)

    plt.tight_layout()
    filename = f"{model_type}_{net_type}_term_counts_barplot_epo{num_epochs}.png"
    plt.savefig(os.path.join(output_path, filename), dpi=300)
    plt.show()

    print(f"Bar plot saved to {os.path.join(output_path, filename)}")

def plot_contingency_matrix(row_labels_bio_topk, row_labels_topo_topk, ari_score, nmi_score, output_dir, args):
    """
    Plots a contingency matrix comparing bio and topo cluster labels.

    Parameters:
    - row_labels_bio_topk: Cluster labels from bio features.
    - row_labels_topo_topk: Cluster labels from topo features.
    - ari_score: Adjusted Rand Index between the clusterings.
    - nmi_score: Normalized Mutual Information score.
    - output_dir: Directory to save the plot.
    - args: Arguments containing model type, net type, and epoch count.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    # === Contingency matrix ===
    contingency = confusion_matrix(row_labels_bio_topk, row_labels_topo_topk)

    # Plot heatmap of confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='BuPu', cbar=False, annot_kws={"fontsize": 20})

    plt.title(f"Contingency Matrix: Bio vs Topo\n(ARI={ari_score:.2f}, NMI={nmi_score:.2f})", fontsize=30)
    plt.xlabel("Topo clusters", fontsize=28)
    plt.ylabel("Bio clusters", fontsize=28)

    plt.xticks([])
    plt.yticks([])

    # Save plot
    contingency_plot_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_contingency_matrix_epo{args.num_epochs}.png"
    )
    plt.tight_layout()
    plt.savefig(contingency_plot_path, dpi=300)
    plt.show()

    print(f"✅ Contingency matrix saved to {contingency_plot_path}")

def plot_omics_barplot_bio(df, output_path=None):
    """
    Plot omics relevance for biological features with format like 'MF:BRCA'.
    """
    omics_order = ['cna', 'ge', 'meth', 'mf']
    omics_colors = {
        'cna': '#9370DB',    # purple
        'ge': '#228B22',      # dark green
        'meth': '#00008B',   # dark blue
        'mf': '#b22222',     # dark red
    }

    # Extract 'Omics' and 'Cancer' from features like 'MF:BRCA'
    df[['Omics', 'Cancer']] = df['Feature'].str.split(':', expand=True)
    df['Omics'] = df['Omics'].str.lower()

    omics_relevance = df.groupby('Omics')['Relevance'].sum().reindex(omics_order)

    _plot_bar(omics_relevance, omics_colors, omics_order, output_path)

def plot_omics_barplot_topo(df, output_path=None):
    """
    Plot omics relevance for topological features with format like 'BRCA_mf'.
    """
    omics_order = ['cna', 'ge', 'meth', 'mf']
    omics_colors = {
        'cna': '#9370DB',    # purple
        'ge': '#228B22',      # dark green
        'meth': '#00008B',   # dark blue
        'mf': '#b22222',     # dark red
    }

    # Extract 'Cancer' and 'Omics' from features like 'BRCA_mf'
    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)
    df['Omics'] = df['Omics'].str.lower()

    omics_relevance = df.groupby('Omics')['Relevance'].sum().reindex(omics_order)

    _plot_bar(omics_relevance, omics_colors, omics_order, output_path)

def plot_novel_predicted_cancer_genes(
    clusters,
    novel_predicted_cancer_genes,
    total_genes_per_cluster,
    node_names,
    row_labels,
    output_path):
    """
    Plots the percentage of novel predicted cancer genes per cluster.
    """

    # Convert to array and sort clusters
    clusters = np.array(sorted(total_genes_per_cluster.keys()))
    total_genes_array = np.array([total_genes_per_cluster[c] for c in clusters])

    # Count novel predicted genes per cluster
    cluster_to_novel_count = {c: 0 for c in clusters}
    for i, name in enumerate(node_names):
        if name in novel_predicted_cancer_genes:
            cluster = row_labels[i]
            cluster_to_novel_count[cluster] += 1

    # Get counts in cluster order
    predicted_counts = np.array([cluster_to_novel_count.get(c, 0) for c in clusters])
    percent_predicted = np.divide(predicted_counts, total_genes_array, where=total_genes_array > 0)

    # Prepare bar colors
    colors = [CLUSTER_COLORS.get(c, '#333333') for c in clusters]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(clusters, percent_predicted, color=colors, edgecolor='black')

    # Annotate bars
    for bar, cluster_id in zip(bars, clusters):
        height = bar.get_height()
        count = cluster_to_novel_count.get(cluster_id, 0)
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(count),
                ha='center', va='bottom', fontsize=16, fontweight='bold')

    ax.set_xlim(-0.55, len(clusters) - 0.65)
    ax.set_ylabel("Percent of NPCGs", fontsize=20)
    plt.xticks(clusters, fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_ylim(0, max(percent_predicted) + 0.1)

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Novel PCG plot saved to {output_path}")

def plot_collapsed_clusterfirst_multilevel_sankey_bio(
    args,
    graph,
    node_names,
    name_to_index,
    node_names_topk,
    row_labels,
    total_clusters,
    relevance_scores,
    output_dir,
    CLUSTER_COLORS
):


    topk_name_to_index = {name: i for i, name in enumerate(node_names)}
    node_id_to_name = {i: name for i, name in enumerate(node_names)} 
    cluster_assignments = graph.ndata["cluster_bio"].numpy()

    top_scored_genes = sorted(
        [g for g in node_names_topk if g in topk_name_to_index and topk_name_to_index[g] < relevance_scores.shape[0]],
        key=lambda g: relevance_scores[topk_name_to_index[g]].sum(),
        reverse=True
    )

    top_scored_genes = [g for g in top_scored_genes if not g.startswith("MED")]

    selected_known_genes = ["EGFR", "BRCA2"]
    selected_novel_genes = ["EIF2B3", "TOM1L2", "SYNCRIP", "GEMIN5", "WDR24", "ZNF598", "TAF8", "SEC61A1", "FUBP1", "TRIP13"]

    combined_genes = []
    seen = set()
    # for g in top_scored_genes:
    for g in selected_known_genes + top_scored_genes:
        if g in name_to_index and g not in seen:
            combined_genes.append(g)
            seen.add(g)
        if len(combined_genes) == 30:
            break

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, combined_genes)

    label_to_idx = {}
    all_labels = []
    all_colors = []
    font_sizes = []

    cluster_to_genes = {}
    gene_to_neighbors = {}
    highlight_node_indices = []
    highlight_node_saliency = []

    for gene in combined_genes:
        if gene not in topk_name_to_index:
            continue

        # node_idx = topk_name_to_index[gene]
        # if node_idx >= len(row_labels):
        #     continue

        rel_idx = topk_name_to_index[gene]
        if rel_idx >= len(row_labels):
            continue
        
        # graph_node_idx = name_to_index[gene]
        # gene_cluster = cluster_assignments[graph_node_idx]
        gene_cluster = row_labels[rel_idx]
        
        cluster_label = f"Cluster {gene_cluster}"
        cluster_to_genes.setdefault(cluster_label, []).append(gene)

        neighbors = neighbors_dict.get(gene, [])
        neighbor_scores = {}

        for n in neighbors:
            if n == gene:
                continue
            if n in combined_genes:
                continue
            if n not in topk_name_to_index:
                continue

            # rel_idx = topk_name_to_index[n]
            # if rel_idx >= relevance_scores.shape[0]:
            #     continue
            neighbor_rel_idx = topk_name_to_index[n]
            if neighbor_rel_idx >= len(row_labels):
                continue
            
            rel_score = relevance_scores[neighbor_rel_idx].sum().item()
            neighbor_scores[neighbor_rel_idx] = rel_score


        if neighbor_scores:
            neighbor_scores = dict(sorted(neighbor_scores.items(), key=lambda x: -x[1])[:2])
        gene_to_neighbors[gene] = neighbor_scores

    # 📁 Save gene-neighbor relationships (CSV export)
    all_neighbor_rows = []
    for gene, neighbors in gene_to_neighbors.items():
        # graph_node_idx = name_to_index[gene]
        # gene_cluster = int(cluster_assignments[graph_node_idx])
        gene_rel_idx = topk_name_to_index[gene]
        gene_cluster = int(row_labels[gene_rel_idx])
        for neighbor_idx, score in neighbors.items():
            neighbor_name = node_names_topk[neighbor_idx]
            neighbor_cluster = int(row_labels[neighbor_idx])
            all_neighbor_rows.append([
                gene, 
                neighbor_name, 
                neighbor_idx, 
                float(score), 
                neighbor_cluster
            ])
        # for neighbor_idx, score in neighbors.items():
        #     if neighbor_idx >= len(cluster_assignments):
        #         continue
        #     neighbor_name = node_id_to_name[neighbor_idx]
        #     neighbor_cluster = int(cluster_assignments[neighbor_idx])
        #     # 🧼 Skip neighbors with cluster -1
        #     if neighbor_cluster == -1:
        #         continue
        #     all_neighbor_rows.append([
        #         gene,
        #         neighbor_name,
        #         str(neighbor_idx),
        #         float(score),
        #         neighbor_cluster
        #     ])

    # 💾 Write CSV
    os.makedirs(output_dir, exist_ok=True)
    neighbors_csv_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_gene_neighbors_epo{args.num_epochs}.csv"
    )
    with open(neighbors_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Gene", "Neighbor", "Neighbor_Index", "Relevance_Score", "Neighbor_Cluster"])
        writer.writerows(all_neighbor_rows)
    print(f"📄 Saved gene-neighbor relevance data to: {neighbors_csv_path}")

    source = []
    target = []
    value = []
    link_colors = []

    existing_edges = set()  # Track (source, target) to avoid A→B and B→A cycles

    for cluster_label, genes in cluster_to_genes.items():
        if cluster_label not in label_to_idx:
            label_to_idx[cluster_label] = len(all_labels)
            all_labels.append(cluster_label)
            cluster_id = int(cluster_label.split()[-1])
            all_colors.append(CLUSTER_COLORS.get(cluster_id, "#000000"))
            font_sizes.append(24)

        cluster_idx = label_to_idx[cluster_label]
        # genes_sorted = sorted(genes, key=lambda g: relevance_scores[name_to_index[g]].sum(), reverse=True)[:10]
        genes_sorted = sorted(genes, key=lambda g: relevance_scores[topk_name_to_index[g]].sum(), reverse=True)[:10]

        for gene in genes_sorted:
            if gene not in label_to_idx:
                label_to_idx[gene] = len(all_labels)
                all_labels.append(gene)
                rel_idx = topk_name_to_index[gene]
                # saliency = relevance_scores[rel_idx].sum().item()
                # graph_node_idx = name_to_index[gene]
                # gene_cluster = cluster_assignments[graph_node_idx]
                gene_cluster = row_labels[rel_idx]
                color = CLUSTER_COLORS.get(gene_cluster, "#000000")
                all_colors.append(color)
                font_sizes.append(18)

            gene_idx = label_to_idx[gene]

            # Cluster → Gene link
            if (cluster_idx, gene_idx) not in existing_edges:
                source.append(cluster_idx)
                target.append(gene_idx)
                value.append(1)
                link_colors.append(hex_to_rgba(CLUSTER_COLORS.get(int(cluster_label.split()[-1]), "#000000"), 0.4))
                existing_edges.add((cluster_idx, gene_idx))

            neighbors = gene_to_neighbors.get(gene, {})
            # for neighbor_idx, neighbor_score in neighbors.items():
            #     neighbor_name = node_id_to_name[neighbor_idx]

            #     # ✅ NEW: Only allow neighbor if in top-k
            #     rel_idx = topk_name_to_index.get(neighbor_name)
            #     if rel_idx is None or rel_idx >= len(row_labels):
            #         continue  # skip: neighbor not top-k or no cluster

            #     neighbor_cluster = row_labels[rel_idx]

            for neighbor_idx, neighbor_score in gene_to_neighbors.get(gene, {}).items():
                neighbor_name = node_names_topk[neighbor_idx]

                if neighbor_name not in topk_name_to_index:
                    continue

                # Skip self-link just in case
                if neighbor_name == gene:
                    continue

                # Add neighbor label if new
                if neighbor_name not in label_to_idx:
                    label_to_idx[neighbor_name] = len(all_labels)
                    all_labels.append(neighbor_name)
                    saliency = relevance_scores[rel_idx].sum().item()
                    neighbor_cluster = row_labels[neighbor_idx]
                    color = CLUSTER_COLORS.get(neighbor_cluster, "#000000")
                    all_colors.append(color)
                    font_sizes.append(16 if saliency > 0.5 else 10)
                    # if saliency > 0.5:
                    #     highlight_node_indices.append(label_to_idx[neighbor_name])
                    #     highlight_node_saliency.append(saliency)

                neighbor_node_idx = label_to_idx[neighbor_name]
                neighbor_cluster = row_labels[neighbor_idx]

                # Add neighbor cluster label if new
                neighbor_cluster_label = f"nClust {neighbor_cluster}"
                if neighbor_cluster_label not in label_to_idx:
                    label_to_idx[neighbor_cluster_label] = len(all_labels)
                    all_labels.append(neighbor_cluster_label)
                    all_colors.append(CLUSTER_COLORS.get(neighbor_cluster, "#000000"))
                    font_sizes.append(18)

                neighbor_cluster_idx = label_to_idx[neighbor_cluster_label]

                # Gene → Neighbor
                if (gene_idx, neighbor_node_idx) not in existing_edges:
                    source.append(gene_idx)
                    target.append(neighbor_node_idx)
                    value.append(neighbor_score)
                    link_colors.append("rgba(160,160,160,0.5)")
                    existing_edges.add((gene_idx, neighbor_node_idx))

                # Neighbor → Cluster
                if (neighbor_node_idx, neighbor_cluster_idx) not in existing_edges:
                    source.append(neighbor_node_idx)
                    target.append(neighbor_cluster_idx)
                    value.append(neighbor_score)
                    link_colors.append(hex_to_rgba(CLUSTER_COLORS.get(neighbor_cluster, "#000000"), 0.6))
                    existing_edges.add((neighbor_node_idx, neighbor_cluster_idx))

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=30,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=all_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])

    fig.update_layout(
        font_size=16,
        margin=dict(l=20, r=20, t=20, b=20),
        width=1200,
        height=1200,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    output_dir = "results/gene_prediction/bio_collapsed_clusterfirst_multilevel_sankey/"
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_bio_collapsed_clusterfirst_multilevel_sankey_epo{args.num_epochs}.html"
    )
    fig.write_html(save_path)
    print(f"✅ Collapsed Cluster-First Multi-level Sankey saved: {save_path}")

    try:
        import plotly.io as pio
        png_save_path = os.path.join(
            output_dir,
            f"{args.model_type}_{args.net_type}_bio_collapsed_clusterfirst_multilevel_sankey_epo{args.num_epochs}.png"
        )
        fig.write_image(png_save_path, scale=2, width=600, height=1200)
        print(f"🖼️ PNG also saved to: {png_save_path}")
    except Exception as e:
        print(f"⚠️ Failed to save PNG: {e}")
        print("Tip: Install 'kaleido' via pip to enable static image export: pip install kaleido")

    sankey_stats = analyze_sankey_structure(
        source,
        target,
        value,
        label_to_idx,
        node_names,
        cluster_to_genes,
        gene_to_neighbors,
        row_labels,
        name_to_index,
        relevance_scores 
    )

    entropy = sankey_stats["cluster_entropy"]
    entropy_df = pd.DataFrame(list(entropy.items()), columns=["Cluster", "Entropy"]).sort_values("Entropy", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Entropy", y="Cluster", data=entropy_df, palette="coolwarm")
    plt.title("Cluster Entropy (Gene Participation Diversity)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{args.model_type}_{args.net_type}_entropy_bar_epo{args.num_epochs}.png"))
    plt.close()

    jaccard = sankey_stats["cluster_jaccard"]
    jaccard_df = pd.DataFrame([
        {"Cluster1": c1, "Cluster2": c2, "Jaccard": val}
        for (c1, c2), val in jaccard.items()
    ])
    pivot_df = jaccard_df.pivot(index="Cluster1", columns="Cluster2", values="Jaccard").fillna(0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, cmap="viridis", annot=True, fmt=".2f", square=True, cbar_kws={'label': 'Jaccard Index'})
    plt.title("Jaccard Similarity Between Confirmed Gene Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{args.model_type}_{args.net_type}_jaccard_heatmap_epo{args.num_epochs}.png"))
    plt.close()

    centrality_df = pd.DataFrame(list(sankey_stats["gene_degree_centrality"].items()), columns=["Gene", "Centrality"])
    centrality_df = centrality_df.sort_values("Centrality", ascending=False)
    plt.figure(figsize=(10, 15))
    sns.barplot(x="Centrality", y="Gene", data=centrality_df, palette="magma")
    plt.title("Neighbor Centrality (Sum of Relevance)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{args.model_type}_{args.net_type}_centrality_bar_epo{args.num_epochs}.png"))
    plt.close()

    return sankey_stats

def plot_collapsed_clusterfirst_multilevel_sankey_topo(
    args,
    graph,
    node_names,
    name_to_index,
    novel_predicted_genes, #confirmed_genes,
    scores,
    row_labels,
    total_clusters,
    relevance_scores,
    CLUSTER_COLORS
):
    topk_name_to_index = {name: i for i, name in enumerate(node_names)}
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    # Top 10 by model score
    top_scored_genes = sorted(
        novel_predicted_genes,  # or confirmed_genes
        key=lambda g: scores[name_to_index[g]],
        reverse=True
    )

    # Exclude MED subunits
    top_scored_genes = [
        g for g in top_scored_genes
        if not g.startswith("MED")
    ]


    # Manually selected important genes
    selected_genes = ["BRCA1", "TP53", "PIK3CA", "KRAS", "ALK"]
    # selected_known_genes = [
    #     "BRCA2",   # Breast, ovarian, pancreatic cancer
    #     "CDK4",    # Melanoma, sarcoma
    #     "BCL6",    # B-cell lymphomas
    #     "E2F3",    # Bladder, prostate cancer
    #     "CUL1",    # Cell cycle, implicated in various cancers
    #     "FOXM1",   # Proliferation driver, overexpressed in many tumors
    #     "ETS1",    # Leukemia, lymphomas
    #     "SKP2",    # Regulates cell cycle, implicated in prostate, breast, etc.
    #     "ATR",     # DNA repair gene, involved in many cancers
    #     "HDAC2"    # Epigenetic regulator, therapeutic target in hematologic cancers
    # ]
    selected_known_genes = [
        "ACTB", "ATR", "BCL6", "BRCA2", "CDK4", "CUL1",
        "E2F3", "EGFR", "ETS1", "FOXM1", "HDAC2", "RBM39",
        "SKP2", "SRC"
    ]

    selected_known_genes = ["EGFR", "SRC", "ACTB", "RBM39", "BRCA2", "HDAC2", "ETS1", "ATR"]
    
    selected_known_genes = ["BRCA2", "HDAC2"]

    selected_novel_genes = ["EIF2B3", "TOM1L2", "SYNCRIP", "GEMIN5", "WDR24", "ZNF598", "TAF8", "SEC61A1", "FUBP1", "TRIP13"]


    # Combine and deduplicate while preserving order
    combined_genes = []
    seen = set()
    # for g in selected_novel_genes + selected_known_genes:
    for g in top_scored_genes:
        if g in name_to_index and g not in seen:
            combined_genes.append(g)
            seen.add(g)
        if len(combined_genes) == 20:
            break

    #combined_genes = [g for g in combined_genes if g != "IRF2"]

    #confirmed_genes = combined_genes

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, combined_genes)

    label_to_idx = {}
    all_labels = []
    all_colors = []
    font_sizes = []

    cluster_to_genes = {}
    gene_to_neighbors = {}

    highlight_node_indices = []
    highlight_node_saliency = []

    for gene in combined_genes:
        if gene not in topk_name_to_index:
            continue

        node_idx = topk_name_to_index[gene]
        if node_idx >= len(row_labels):
            continue
        gene_cluster = row_labels[node_idx]

        cluster_label = f"Confirmed Cluster {gene_cluster}"

        cluster_to_genes.setdefault(cluster_label, []).append(gene)

        neighbors = neighbors_dict.get(gene, [])
        neighbor_scores = {}

        for n in neighbors:
            if n in topk_name_to_index:
                rel_idx = topk_name_to_index[n]
                if rel_idx < relevance_scores.shape[0]:
                    rel_score = relevance_scores[rel_idx].sum().item()
                    neighbor_scores[rel_idx] = rel_score

        if neighbor_scores:
            neighbor_scores = dict(sorted(neighbor_scores.items(), key=lambda x: -x[1])[:5])

        gene_to_neighbors[gene] = neighbor_scores

    source = []
    target = []
    value = []
    link_colors = []

    for cluster_label, genes in cluster_to_genes.items():
        if cluster_label not in label_to_idx:
            label_to_idx[cluster_label] = len(all_labels)
            all_labels.append(cluster_label)
            cluster_id = int(cluster_label.split()[-1])
            all_colors.append(CLUSTER_COLORS.get(cluster_id, "#000000"))
            font_sizes.append(24)

        cluster_idx = label_to_idx[cluster_label]
        genes_sorted = sorted(genes, key=lambda g: scores[name_to_index[g]], reverse=True)[:10]

        for gene in genes_sorted:
            if gene not in label_to_idx:
                label_to_idx[gene] = len(all_labels)
                all_labels.append(gene)

                rel_idx = topk_name_to_index[gene]
                saliency = relevance_scores[rel_idx].sum().item()
                gene_cluster = row_labels[rel_idx]
                color = CLUSTER_COLORS.get(gene_cluster, "#000000")
                all_colors.append(color)
                font_sizes.append(18 if saliency > 0.5 else 10)

                if saliency > 0.5:
                    highlight_node_indices.append(label_to_idx[gene])
                    highlight_node_saliency.append(saliency)

            gene_idx = label_to_idx[gene]

            source.append(cluster_idx)
            target.append(gene_idx)
            value.append(1)
            link_colors.append(hex_to_rgba(CLUSTER_COLORS.get(int(cluster_label.split()[-1]), "#000000"), 0.4))

            neighbors = gene_to_neighbors.get(gene, {})
            for neighbor_idx, neighbor_score in neighbors.items():
    
                # if neighbor_idx == node_idx:
                #     continue
                neighbor_name = node_id_to_name[neighbor_idx]
                
                if neighbor_name == gene:
                    continue
                
                neighbor_cluster = row_labels[neighbor_idx]
                neighbor_cluster_label = f"Cluster {neighbor_cluster}"

                if neighbor_name not in label_to_idx:
                    label_to_idx[neighbor_name] = len(all_labels)
                    all_labels.append(neighbor_name)

                    saliency = relevance_scores[neighbor_idx].sum().item()
                    color = CLUSTER_COLORS.get(neighbor_cluster, "#000000")
                    all_colors.append(color)
                    font_sizes.append(16 if saliency > 0.5 else 10)

                    if saliency > 0.5:
                        highlight_node_indices.append(label_to_idx[neighbor_name])
                        highlight_node_saliency.append(saliency)

                neighbor_node_idx = label_to_idx[neighbor_name]

                if neighbor_cluster_label not in label_to_idx:
                    label_to_idx[neighbor_cluster_label] = len(all_labels)
                    all_labels.append(neighbor_cluster_label)
                    all_colors.append(CLUSTER_COLORS.get(neighbor_cluster, "#000000"))
                    font_sizes.append(18)

                neighbor_cluster_idx = label_to_idx[neighbor_cluster_label]

                source.append(gene_idx)
                target.append(neighbor_node_idx)
                value.append(neighbor_score)
                link_colors.append("rgba(160,160,160,0.5)")

                source.append(neighbor_node_idx)
                target.append(neighbor_cluster_idx)
                value.append(neighbor_score)
                link_colors.append(hex_to_rgba(CLUSTER_COLORS.get(neighbor_cluster, "#000000"), 0.6))

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=30,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=all_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])

    if highlight_node_indices:
        x_positions = [0.1 + (idx % 6) * 0.15 for idx in highlight_node_indices]
        y_positions = [0.9 - (idx // 6) * 0.1 for idx in highlight_node_indices]

        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='none',
            marker=dict(
                size=[30 + 40 * (s-0.5) for s in highlight_node_saliency],
                color="rgba(255,0,0,0.3)",
                line=dict(width=2, color="rgba(255,0,0,0.7)"),
                sizemode='diameter'
            ),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title=None,
        font_size=16,
        margin=dict(l=20, r=20, t=20, b=20),
        width=1200,
        height=1200,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
    )

    output_dir = "results/gene_prediction/topo_collapsed_clusterfirst_multilevel_sankey/"
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_topo_collapsed_clusterfirst_multilevel_sankey_epo{args.num_epochs}.html"
    )
    fig.write_html(save_path)
    print(f"✅ Collapsed Cluster-First Multi-level Sankey saved: {save_path}")

    try:
        import plotly.io as pio
        png_save_path = os.path.join(
            output_dir,
            f"{args.model_type}_{args.net_type}_topo_collapsed_clusterfirst_multilevel_sankey_epo{args.num_epochs}.png"
        )
        fig.write_image(png_save_path, scale=2, width=1200, height=1200)
        print(f"🖼️ PNG also saved to: {png_save_path}")
    except Exception as e:
        print(f"⚠️ Failed to save PNG: {e}")
        print("Tip: Install 'kaleido' via pip to enable static image export: pip install kaleido")

    # === Run Structure Analysis
    sankey_stats = analyze_sankey_structure(
        source,
        target,
        value,
        label_to_idx,
        node_names,
        cluster_to_genes,
        gene_to_neighbors,
        row_labels,
        name_to_index,
        scores
    )

    # === Summary Plot: Entropy per Cluster
    entropy = sankey_stats["cluster_entropy"]

    entropy_df = pd.DataFrame(list(entropy.items()), columns=["Cluster", "Entropy"])
    entropy_df = entropy_df.sort_values("Entropy", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Entropy", y="Cluster", data=entropy_df, palette="coolwarm")
    plt.title("Cluster Entropy (Gene Participation Diversity)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{args.model_type}_{args.net_type}_entropy_bar_epo{args.num_epochs}.png"))
    plt.close()

    # === Summary Plot: Jaccard Heatmap
    jaccard = sankey_stats["cluster_jaccard"]
    ##jaccard_df = pd.DataFrame(jaccard).fillna(0)
    jaccard_df = pd.DataFrame([
        {"Cluster1": c1, "Cluster2": c2, "Jaccard": val}
        for (c1, c2), val in jaccard.items()
    ])

    pivot_df = jaccard_df.pivot(index="Cluster1", columns="Cluster2", values="Jaccard").fillna(0)


    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, cmap="viridis", annot=True, fmt=".2f", square=True, cbar_kws={'label': 'Jaccard Index'})
    plt.title("Jaccard Similarity Between Confirmed Gene Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{args.model_type}_{args.net_type}_jaccard_heatmap_epo{args.num_epochs}.png"))
    plt.close()

    # === (Optional) Summary Plot: Centrality per Gene
    #centrality_df = pd.DataFrame(list(sankey_stats["centrality"].items()), columns=["Gene", "Centrality"])
    centrality_df = pd.DataFrame(list(sankey_stats["gene_degree_centrality"].items()), columns=["Gene", "Centrality"])

    centrality_df = centrality_df.sort_values("Centrality", ascending=False)

    plt.figure(figsize=(10, 15))
    sns.barplot(x="Centrality", y="Gene", data=centrality_df, palette="magma")
    plt.title("Neighbor Centrality (Sum of Relevance)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{args.model_type}_{args.net_type}_centrality_bar_epo{args.num_epochs}.png"))
    plt.close()

    return sankey_stats


def get_neighbors_gene_names(graph, node_names, name_to_index, genes):
    neighbors_dict = {}
    for gene in genes:
        if gene in name_to_index:
            idx = name_to_index[gene]
            neighbors = graph.successors(idx).tolist()
            neighbor_names = [node_names[n] for n in neighbors]
            neighbors_dict[gene] = neighbor_names
    return neighbors_dict

def plot_dynamic_sankey_bio_clusterlevel(
    args,
    graph,
    node_names_topk,
    name_to_index,
    confirmed_genes,
    scores,
    row_labels,
    total_clusters,
    relevance_scores
):
    # Build safe top-k index mapping
    topk_name_to_index = {name: i for i, name in enumerate(node_names_topk)}
    node_id_to_name = {i: name for i, name in enumerate(node_names_topk)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names_topk, name_to_index, confirmed_genes)

    # Mapping cluster-to-cluster scores
    cluster_to_cluster_score = {}

    for gene in confirmed_genes:
        if gene not in topk_name_to_index:
            print(f"⚠️ Gene {gene} not in top-k node list. Skipping.")
            continue

        node_idx = topk_name_to_index[gene]
        gene_cluster = row_labels[node_idx].item()

        neighbors = neighbors_dict.get(gene, [])

        neighbor_scores_dict = {}
        for n in neighbors:
            if n in topk_name_to_index:
                rel_idx = topk_name_to_index[n]
                if rel_idx < relevance_scores.shape[0]:
                    rel_score = relevance_scores[rel_idx].sum().item()
                    neighbor_scores_dict[rel_idx] = rel_score

        if not neighbor_scores_dict:
            print(f"⚠️ No valid neighbors for {gene}.")
            continue

        top_neighbors = dict(sorted(neighbor_scores_dict.items(), key=lambda x: -x[1])[:10])

        for rel_idx, rel_score in top_neighbors.items():
            neighbor_cluster = row_labels[rel_idx].item()

            key = (gene_cluster, neighbor_cluster)
            cluster_to_cluster_score[key] = cluster_to_cluster_score.get(key, 0) + rel_score

    if not cluster_to_cluster_score:
        print("⚠️ No cluster-to-cluster links to plot.")
        return

    # Now prepare Sankey inputs
    clusters_involved = set()
    for (src_c, tgt_c) in cluster_to_cluster_score.keys():
        clusters_involved.add(src_c)
        clusters_involved.add(tgt_c)
    clusters_involved = sorted(list(clusters_involved))

    cluster_id_to_label = {c: f"C{c}" for c in clusters_involved}
    label_to_index = {f"C{c}": i for i, c in enumerate(clusters_involved)}

    source = []
    target = []
    value = []
    label = [f"C{c}" for c in clusters_involved]
    color = [CLUSTER_COLORS.get(c, "#CCCCCC") for c in clusters_involved]  # 🛠 fixed color

    for (src_c, tgt_c), score in cluster_to_cluster_score.items():
        source.append(label_to_index[f"C{src_c}"])
        target.append(label_to_index[f"C{tgt_c}"])
        value.append(score)

    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label,
            color=color
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])

    sankey_fig.update_layout(
        title_text=f"Confirmed Cluster → Neighbor Cluster (Bio) - {args.model_type}_{args.net_type}",
        font_size=10,
        width=1200,
        height=800
    )

    output_dir = "results/gene_prediction/bio_dynamic_sankey_clusterlevel/"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_bio_confirmed_dynamic_sankey_clusterlevel_epo{args.num_epochs}.html"
    )
    sankey_fig.write_html(plot_path)
    print(f"✅ Cluster-level Sankey diagram saved to {plot_path}")

def plot_dynamic_sankey_topo_clusterlevel(
    args,
    graph,
    node_names_topk,
    name_to_index,
    confirmed_genes,
    scores,
    relevance_scores,
    row_labels,
    total_clusters
):
    # Mapping node IDs and names
    topk_name_to_index = {name: i for i, name in enumerate(node_names_topk)}
    node_id_to_name = {i: name for i, name in enumerate(node_names_topk)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names_topk, name_to_index, confirmed_genes)

    # Mapping cluster-to-cluster scores
    cluster_to_cluster_score = {}

    for gene in confirmed_genes:
        if gene not in topk_name_to_index:
            print(f"⚠️ Gene {gene} not in top-k node list. Skipping.")
            continue

        node_idx = topk_name_to_index[gene]
        gene_score = scores[node_idx]

        # ✅ Get topo cluster from graph
        gene_cluster = graph.ndata["cluster_topo"][node_idx].item()

        neighbors = neighbors_dict.get(gene, [])
        neighbor_scores_dict = {}

        for n in neighbors:
            if n in topk_name_to_index:
                rel_idx = topk_name_to_index[n]
                if rel_idx < relevance_scores.shape[0]:  # bounds check
                    rel_score = relevance_scores[rel_idx].sum().item()
                    neighbor_scores_dict[rel_idx] = rel_score

        if not neighbor_scores_dict:
            print(f"⚠️ No valid neighbors for {gene}.")
            continue

        # Top neighbors (limit to 10)
        top_neighbors = dict(sorted(neighbor_scores_dict.items(), key=lambda x: -x[1])[:10])

        for rel_idx, rel_score in top_neighbors.items():
            neighbor_cluster = row_labels[rel_idx].item()

            key = (gene_cluster, neighbor_cluster)
            cluster_to_cluster_score[key] = cluster_to_cluster_score.get(key, 0) + rel_score

    if not cluster_to_cluster_score:
        print("⚠️ No cluster-to-cluster links to plot.")
        return

    # Prepare Sankey inputs
    clusters_involved = set()
    for (src_c, tgt_c) in cluster_to_cluster_score.keys():
        clusters_involved.add(src_c)
        clusters_involved.add(tgt_c)
    clusters_involved = sorted(list(clusters_involved))

    cluster_id_to_label = {c: f"C{c}" for c in clusters_involved}
    label_to_index = {f"C{c}": i for i, c in enumerate(clusters_involved)}

    source = []
    target = []
    value = []
    label = [f"C{c}" for c in clusters_involved]
    color = [CLUSTER_COLORS.get(c, "#CCCCCC") for c in clusters_involved]  # 🛠 fixed color

    for (src_c, tgt_c), score in cluster_to_cluster_score.items():
        source.append(label_to_index[f"C{src_c}"])
        target.append(label_to_index[f"C{tgt_c}"])
        value.append(score)

    # Build Sankey figure
    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label,
            color=color
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
        ))])

    sankey_fig.update_layout(
        title_text=f"Confirmed Cluster → Neighbor Cluster (Topo) - {args.model_type}_{args.net_type}",
        font_size=10,
        width=1200,
        height=800
    )

    # Save
    output_dir = "results/gene_prediction/topo_dynamic_sankey_clusterlevel/"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_topo_confirmed_dynamic_sankey_clusterlevel_epo{args.num_epochs}.html"
    )
    sankey_fig.write_html(plot_path)
    print(f"✅ Cluster-level Sankey diagram saved to {plot_path}")

def plot_multilevel_sankey_bio_clusterlevel(
    args,
    graph,
    node_names_topk,
    name_to_index,
    confirmed_genes,
    scores,
    row_labels,
    total_clusters,
    relevance_scores
):
    topk_name_to_index = {name: i for i, name in enumerate(node_names_topk)}
    node_id_to_name = {i: name for i, name in enumerate(node_names_topk)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names_topk, name_to_index, confirmed_genes)

    for gene in confirmed_genes:
        if gene not in topk_name_to_index:
            print(f"⚠️ Gene {gene} not in top-k node list. Skipping.")
            continue

        node_idx = topk_name_to_index[gene]
        neighbors = neighbors_dict.get(gene, [])
        
        if not neighbors:
            print(f"⚠️ No neighbors found for {gene}.")
            continue

        neighbor_scores = {}
        for n in neighbors:
            if n in topk_name_to_index:
                rel_idx = topk_name_to_index[n]
                if rel_idx < relevance_scores.shape[0]:
                    rel_score = relevance_scores[rel_idx].sum().item()
                    neighbor_scores[rel_idx] = rel_score

        if not neighbor_scores:
            print(f"⚠️ No valid neighbor relevance for {gene}.")
            continue

        # Limit to top 10 neighbors
        neighbor_scores = dict(sorted(neighbor_scores.items(), key=lambda x: -x[1])[:10])

        # Now build node list and mapping
        all_labels = []
        all_colors = []
        label_to_idx = {}

        # 1. Confirmed gene node
        confirmed_label = f"{gene}"
        all_labels.append(confirmed_label)
        all_colors.append("gray")
        label_to_idx[confirmed_label] = 0

        # 2. Neighbor nodes
        for neighbor_idx in neighbor_scores.keys():
            neighbor_name = node_id_to_name[neighbor_idx]
            neighbor_label = f"{neighbor_name}"
            if neighbor_label not in label_to_idx:
                label_to_idx[neighbor_label] = len(all_labels)
                all_labels.append(neighbor_label)
                all_colors.append("lightgray")

        # 3. Cluster nodes
        neighbor_clusters = set()
        for neighbor_idx in neighbor_scores.keys():
            neighbor_cluster = row_labels[neighbor_idx]
            neighbor_clusters.add(neighbor_cluster)

        for cluster_id in sorted(neighbor_clusters):
            cluster_label = f"Cluster {cluster_id}"
            if cluster_label not in label_to_idx:
                label_to_idx[cluster_label] = len(all_labels)
                all_labels.append(cluster_label)
                all_colors.append(CLUSTER_COLORS.get(cluster_id, "#000000"))

        # Build Sankey source-target-value-linkcolor
        source = []
        target = []
        value = []
        link_colors = []

        # Gene -> Neighbor
        for neighbor_idx, score in neighbor_scores.items():
            neighbor_name = node_id_to_name[neighbor_idx]
            source.append(label_to_idx[confirmed_label])
            target.append(label_to_idx[neighbor_name])
            value.append(score)
            link_colors.append("rgba(128,128,128,0.4)")  # gray links

        # Neighbor -> Cluster
        for neighbor_idx, score in neighbor_scores.items():
            neighbor_name = node_id_to_name[neighbor_idx]
            neighbor_cluster = row_labels[neighbor_idx]
            cluster_label = f"Cluster {neighbor_cluster}"

            source.append(label_to_idx[neighbor_name])
            target.append(label_to_idx[cluster_label])
            value.append(score)
            # Link colored by cluster
            cluster_color = CLUSTER_COLORS.get(neighbor_cluster, "#000000")
            link_colors.append(cluster_color.replace("#", "rgba(") + ",0.6)")  # lighter

        # Build figure
        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_labels,
                color=all_colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors
            ))])

        fig.update_layout(
            title_text=f"Multi-Level Sankey: {gene} → Neighbors → Clusters (Bio)",
            font_size=10,
            width=1000,
            height=700
        )

        # Save
        output_dir = "results/gene_prediction/bio_multilevel_sankey/"
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(
            output_dir,
            f"{args.model_type}_{args.net_type}_{gene}_bio_multilevel_sankey_epo{args.num_epochs}.html"
        )
        fig.write_html(save_path)
        print(f"✅ Multi-level Sankey saved: {save_path}")

def compute_combined_relevance_scores(
    model,
    graph,
    features,
    node_indices=None,
    use_abs=True,
    normalize=False,
    prob_threshold=0.5
):
    """
    Computes gradient-based relevance (saliency) scores for selected nodes.

    Args:
        model: Trained GNN model
        graph: DGLGraph
        features: Node features (Tensor or numpy array)
        node_indices: List/Tensor of node indices to compute relevance for. If None, select using prob_threshold
        use_abs: Whether to use absolute value of gradients
        normalize: Whether to normalize relevance scores per node (optional)
        prob_threshold: Probability threshold for auto-selecting nodes

    Returns:
        relevance_scores: Tensor of shape [num_nodes, num_features]
    """
    model.eval()

    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)

    features = features.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        logits = model(graph, features)
        probs = torch.sigmoid(logits.squeeze())

        if node_indices is None:
            node_indices = torch.nonzero(probs > prob_threshold, as_tuple=False).squeeze()
            if node_indices.ndim == 0:
                node_indices = node_indices.unsqueeze(0)

        relevance_scores = torch.zeros_like(features)

        for i, idx in enumerate(node_indices):
            model.zero_grad()
            if features.grad is not None:
                features.grad.zero_()

            probs[idx].backward(retain_graph=(i != len(node_indices) - 1))

            grads = features.grad[idx]
            relevance = grads.abs() if use_abs else grads

            if normalize:
                norm = relevance.norm(p=1)  # L1 norm
                if norm > 0:
                    relevance = relevance / norm

            relevance_scores[idx] = relevance.detach()

    return relevance_scores

def saliency_to_color(saliency, min_saliency=0.0, max_saliency=1.0):
    saliency = np.clip((saliency - min_saliency) / (max_saliency - min_saliency), 0, 1)

    # Interpolate from blue (low) to red (high)
    r = int(0 + saliency * (255 - 0))    # Red from 0 to 255
    g = int(0)                           # Green stays 0
    b = int(255 - saliency * (255 - 0))  # Blue from 255 to 0

    return f"rgb({r},{g},{b})"

def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

def saliency_to_grayscale(saliency, min_saliency=0.0, max_saliency=1.0):
    saliency = np.clip((saliency - min_saliency) / (max_saliency - min_saliency), 0, 1)
    gray_value = int(238 - saliency * (238 - 17))  # 238: #eeeeee (light gray), 17: #111111 (dark gray)
    return f"rgb({gray_value},{gray_value},{gray_value})"

def plot_top_confirmed_gene_neighbors_chord_(
    graph,
    node_names,
    name_to_index,
    scores,
    confirmed_genes,
    output_html="top10_confirmed_gene_neighbors_chord.html",
    output_png="top10_confirmed_gene_neighbors_chord.png",
    top_k_genes=10,
    top_k_neighbors=10,
    min_edge_score=0.05
):
    """
    Plots a chord diagram of top confirmed genes and their neighbors.
    Saves both interactive HTML and static PNG versions.

    Parameters:
        graph: DGLGraph
        node_names: list mapping node indices to gene names
        name_to_index: dict mapping gene names to indices
        scores: numpy array of predicted scores
        confirmed_genes: list of confirmed gene names
        output_html: HTML path to save the interactive chord diagram
        output_png: PNG path to save the static image
        top_k_genes: number of confirmed genes to include
        top_k_neighbors: number of top neighbors per gene
        min_edge_score: minimum score threshold for edges
    """

    # Step 1: Filter top confirmed genes
    confirmed_gene_scores = [
        (gene, scores[name_to_index[gene]])
        for gene in confirmed_genes if gene in name_to_index
    ]
    top_confirmed_genes = sorted(confirmed_gene_scores, key=lambda x: x[1], reverse=True)[:top_k_genes]

    # Step 2: Build edges to top neighbors
    chord_links = []
    seen_edges = set()

    for gene, _ in top_confirmed_genes:
        idx = name_to_index[gene]
        neighbors = graph.successors(idx).tolist()
        neighbor_scores = [
            (node_names[n], scores[n]) for n in neighbors if node_names[n] != gene
        ]
        top_neighbors = sorted(neighbor_scores, key=lambda x: x[1], reverse=True)[:top_k_neighbors]

        for neighbor, score in top_neighbors:
            if score >= min_edge_score:
                edge_key = tuple(sorted((gene, neighbor)))
                if edge_key not in seen_edges:
                    chord_links.append((gene, neighbor, score))
                    seen_edges.add(edge_key)

    if not chord_links:
        print("[Chord Diagram] No valid edges found. Try lowering `min_edge_score`.")
        return

    # Step 3: Create Chord diagram
    chord = hv.Chord(chord_links).select(value=(min_edge_score, None))
    chord.opts(
        opts.Chord(
            cmap='Category20',
            edge_color='source',
            node_color='index',
            labels='name',
            edge_alpha=0.7,
            edge_line_width=hv.dim('value') * 5,
            width=900,
            height=900,
            title="Top Confirmed Genes and Their Neighbors"
        )
    )

    # Step 4: Save HTML
    hv.save(chord, output_html)
    print(f"[✔] HTML saved to: {output_html}")

    # Step 5: Save PNG using Bokeh backend
    try:
        from bokeh.io.export import export_png
        from bokeh.io import curdoc
        from holoviews.plotting.bokeh import render

        plot = render(chord)
        export_png(plot, filename=output_png)
        print(f"[✔] PNG saved to: {output_png}")
    except Exception as e:
        print(f"[⚠] PNG export failed: {e}")
        print("To enable PNG export, make sure you have installed: selenium, pillow, and a compatible web driver like chromedriver.")

def plot_top_confirmed_gene_neighbors_chord_not_gene_name(
    graph,
    node_names,
    name_to_index,
    scores,
    confirmed_genes,
    output_path="top10_confirmed_gene_neighbors_chord.html",
    top_k_genes=10,
    top_k_neighbors=10,
    min_edge_score=0.05  # filter out weak edges for clarity
):
    """
    Plots a chord diagram of top K confirmed genes and their top K neighbors by predicted cancer score.

    Parameters:
        graph: DGL graph
        node_names: list of all node names
        name_to_index: dict mapping names to indices
        scores: numpy array of cancer scores
        confirmed_genes: list of confirmed gene names
        output_path: where to save the HTML file
        top_k_genes: how many confirmed genes to show
        top_k_neighbors: how many neighbors per gene
        min_edge_score: minimum score threshold for showing edges
    """

    # Step 1: Rank confirmed genes by model score
    confirmed_gene_scores = [
        (gene, scores[name_to_index[gene]])
        for gene in confirmed_genes if gene in name_to_index
    ]
    top_confirmed_genes = sorted(confirmed_gene_scores, key=lambda x: x[1], reverse=True)[:top_k_genes]

    # Step 2: For each gene, get top neighbors by score
    chord_links = []
    seen_edges = set()

    for gene, _ in top_confirmed_genes:
        idx = name_to_index[gene]
        neighbors = graph.successors(idx).tolist()
        neighbor_scores = [
            (node_names[n], scores[n]) for n in neighbors if node_names[n] != gene
        ]
        top_neighbors = sorted(neighbor_scores, key=lambda x: x[1], reverse=True)[:top_k_neighbors]

        for neighbor, score in top_neighbors:
            if score >= min_edge_score:
                edge_key = tuple(sorted((gene, neighbor)))
                if edge_key not in seen_edges:
                    chord_links.append((gene, neighbor, score))
                    seen_edges.add(edge_key)

    if not chord_links:
        print("[Chord Diagram] No valid edges found. Try lowering `min_edge_score`.")
        return

    # Step 3: Create Chord Diagram
    chord = hv.Chord(chord_links).select(value=(min_edge_score, None))
    chord.opts(
        opts.Chord(
            cmap='Category20',
            edge_color='source',
            node_color='index',
            labels='name',
            edge_alpha=0.7,
            edge_line_width=hv.dim('value') * 5,
            width=900,
            height=900,
            title="Top Confirmed Genes and Neighbors"
        )
    )

    # Save
    hv.save(chord, output_path)
    print(f"[✔] Chord diagram saved to: {output_path}")

def plot_chord_diagram_topo(
    args,
    source_labels,
    target_labels,
    matrix,
    CLUSTER_COLORS
):
    """
    A stylized approximation of a chord diagram using Plotly's Sankey layout.
    """

    all_labels = list(set(source_labels) | set(target_labels))
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}
    
    source = []
    target = []
    value = []
    link_colors = []

    for i, src in enumerate(source_labels):
        for j, tgt in enumerate(target_labels):
            if matrix[i][j] > 0:
                source_idx = label_to_index[src]
                target_idx = label_to_index[tgt]
                source.append(source_idx)
                target.append(target_idx)
                value.append(matrix[i][j])

                # Get color from source if available
                color = CLUSTER_COLORS.get(i, "#999999")
                link_colors.append(hex_to_rgba(color, 0.5))

    node_colors = [CLUSTER_COLORS.get(i, "#888888") for i in range(len(all_labels))]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])

    fig.update_layout(
        title_text="Chord Diagram (Approx) - Topological Gene Interactions",
        font_size=12,
        margin=dict(l=200, r=200, t=100, b=100),
        width=1200,
        height=1000,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Save output
    output_dir = "results/gene_prediction/topo_chord_diagram/"
    os.makedirs(output_dir, exist_ok=True)

    html_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_topo_chord_epo{args.num_epochs}.html"
    )
    fig.write_html(html_path)
    print(f"✅ Chord diagram saved as HTML: {html_path}")

    # Save PNG
    try:
        import plotly.io as pio
        png_path = os.path.join(
            output_dir,
            f"{args.model_type}_{args.net_type}_topo_chord_epo{args.num_epochs}.png"
        )
        fig.write_image(png_path, format="png", scale=2, width=1200, height=1000)
        print(f"🖼️ PNG also saved: {png_path}")
    except Exception as e:
        print(f"⚠️ Failed to save PNG: {e}")
        print("Tip: Install 'kaleido' to enable image export:\n  pip install kaleido")

def plot_top_confirmed_gene_neighbors_chord(
    graph,
    node_names,
    name_to_index,
    scores,
    confirmed_genes,
    row_labels=None,
    cluster_colors=None,
    output_html="results/chord/top10_confirmed_gene_neighbors_chord.html",
    output_png="results/chord/top10_confirmed_gene_neighbors_chord.png",
    top_k_genes=10,
    top_k_neighbors=10,
    min_edge_score=0.05
):
    """
    Plots a chord diagram of top confirmed genes and their neighbors.
    Saves both interactive HTML and static PNG versions.

    Parameters:
        graph: DGLGraph
        node_names: list mapping node indices to gene names
        name_to_index: dict mapping gene names to indices
        scores: numpy array of predicted scores
        confirmed_genes: list of confirmed gene names
        row_labels: list or array of cluster labels for each node (optional)
        cluster_colors: dict mapping cluster label to hex color (optional)
        output_html: path to save interactive HTML
        output_png: path to save static PNG
        top_k_genes: number of confirmed genes to include
        top_k_neighbors: number of top neighbors per gene
        min_edge_score: minimum score threshold for edges
    """
    # Ensure output dirs exist
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    # Step 1: Filter top confirmed genes
    confirmed_gene_scores = [
        (gene, scores[name_to_index[gene]])
        for gene in confirmed_genes if gene in name_to_index
    ]
    top_confirmed_genes = sorted(confirmed_gene_scores, key=lambda x: x[1], reverse=True)[:top_k_genes]

    # Step 2: Build edges to top neighbors
    chord_links = []
    seen_edges = set()

    for gene, _ in top_confirmed_genes:
        idx = name_to_index[gene]
        neighbors = graph.successors(idx).tolist()
        neighbor_scores = [
            (node_names[n], scores[n]) for n in neighbors if node_names[n] != gene
        ]
        top_neighbors = sorted(neighbor_scores, key=lambda x: x[1], reverse=True)[:top_k_neighbors]

        for neighbor, score in top_neighbors:
            if score >= min_edge_score:
                edge_key = tuple(sorted((gene, neighbor)))
                if edge_key not in seen_edges:
                    chord_links.append((gene, neighbor, score))
                    seen_edges.add(edge_key)

    if not chord_links:
        print("[Chord Diagram] No valid edges found. Try lowering `min_edge_score`.")
        return

    # Step 3: Create node color mapping if cluster info is provided
    node_set = set([g for edge in chord_links for g in edge[:2]])
    df_nodes = pd.DataFrame({'name': list(node_set)})

    if row_labels is not None and cluster_colors is not None:
        #df_nodes['cluster'] = df_nodes['name'].map(lambda g: row_labels[name_to_index[g]])
        df_nodes['cluster'] = df_nodes['name'].map(
            lambda g: row_labels[name_to_index[g]] if g in name_to_index else -1
        )

        df_nodes['color'] = df_nodes['cluster'].map(lambda c: cluster_colors.get(c, "#cccccc"))
    else:
        df_nodes['color'] = "#cccccc"

    name_to_color = dict(zip(df_nodes['name'], df_nodes['color']))

    # Step 4: Create Chord diagram
    chord = hv.Chord(chord_links).select(value=(min_edge_score, None))
    chord.opts(
        opts.Chord(
            cmap='Category20',
            edge_color='source',
            node_color=hv.dim('name').categorize(name_to_color, default="#cccccc"),
            labels='name',
            edge_alpha=0.7,
            edge_line_width=hv.dim('value') * 5,
            width=900,
            height=900,
            title="Top Confirmed Genes and Their Neighbors"
        )
    )

    # Step 5: Save HTML
    hv.save(chord, output_html)
    print(f"[✔] HTML saved to: {output_html}")

    # Step 6: Save PNG using Bokeh backend
    try:
        from bokeh.io.export import export_png
        from holoviews.plotting.bokeh import render
        plot = render(chord)
        export_png(plot, filename=output_png)
        print(f"[✔] PNG saved to: {output_png}")
    except Exception as e:
        print(f"[⚠] PNG export failed: {e}")
        print("To enable PNG export, install dependencies: `pip install selenium pillow` and configure a headless browser like `chromedriver`.")

def plot_enriched_pathways_heatmap(
    enrichment_results,
    output_dir,
    model_type,
    net_type,
    num_epochs,
    max_term_len=40,
    max_topo_term_len=60,
    max_rows=50,
    top_n_terms_per_cluster=None,
    return_data=False
):

    heatmap_data = pd.DataFrame()

    for cluster_type in ['bio', 'topo']:
        for cid, df in enrichment_results[cluster_type].items():
            if top_n_terms_per_cluster:
                df = df[df['p_value'] < 0.05].sort_values(by='p_value').head(top_n_terms_per_cluster)
            colname = f"{cluster_type.capitalize()}_{cid}"
            vals = {}
            for _, row in df.iterrows():
                p = row['p_value']
                name = row['name']
                if p < 0.05 and len(name) <= max_term_len:
                    term = f"{name} ({row['source']})"
                    vals[term] = -np.log10(p)
            heatmap_data[colname] = pd.Series(vals)

    heatmap_data = heatmap_data.fillna(0)
    heatmap_data = heatmap_data[heatmap_data.max(axis=1) > 1]

    enrichment_csv_path = os.path.join(
        output_dir,
        f"{model_type}_{net_type}_enrichment_matrix_epo{num_epochs}.csv"
    )
    heatmap_data.to_csv(enrichment_csv_path, index_label='Enriched Pathway')

    # Topo terms export
    topo_terms = []
    for cid, df in enrichment_results['topo'].items():
        if top_n_terms_per_cluster:
            df = df[df['p_value'] < 0.05].sort_values(by='p_value').head(top_n_terms_per_cluster)
        for _, row in df.iterrows():
            if row['p_value'] < 0.05 and len(row['name']) <= max_topo_term_len:
                topo_terms.append({
                    "Cluster": f"Topo_{cid}",
                    "Term": row['name'],
                    "Source": row['source'],
                    "p_value": row['p_value'],
                    "-log10(p)": -np.log10(row['p_value']),
                })

    topo_terms_df = pd.DataFrame(topo_terms)
    topo_terms_path = os.path.join(
        output_dir,
        f"{model_type}_{net_type}_topo_cluster_top_terms_epo{num_epochs}.csv"
    )
    topo_terms_df.to_csv(topo_terms_path, index=False)

    if heatmap_data.shape[0] > max_rows:
        step = max(1, heatmap_data.shape[0] // max_rows)
        selected_indices = heatmap_data.index[::step][:max_rows]
        heatmap_data = heatmap_data.loc[selected_indices]

    norm_data = heatmap_data / heatmap_data.max().replace(0, 1)

    colormaps = {
        'bio': get_cmap('Blues'),
        'topo': get_cmap('YlOrRd'),
    }

    colors = np.zeros((heatmap_data.shape[0], heatmap_data.shape[1], 4))
    col_types = []

    for i, col in enumerate(norm_data.columns):
        group = 'bio' if col.lower().startswith("bio") else 'topo'
        col_types.append(group)
        cmap = colormaps[group]
        colors[:, i, :] = cmap(norm_data[col].values)

    fig, ax = plt.subplots(figsize=(0.5 * len(norm_data.columns), 0.2 * len(norm_data)))

    ax.imshow(colors, aspect='auto')

    ax.set_xticks(np.arange(len(norm_data.columns)))
    ax.set_xticklabels(norm_data.columns, rotation=90, fontsize=14)
    ax.set_yticks(np.arange(len(norm_data.index)))
    ax.set_yticklabels(norm_data.index, fontsize=14)

    ax.set_ylabel("Enriched Pathway", fontsize=16, labelpad=20)

    for xtick, col in zip(ax.get_xticklabels(), col_types):
        xtick.set_color('darkblue' if col == 'bio' else 'darkred')

    # ax.set_title("Top Enriched Pathways per Cluster (p < 0.05)", fontsize=15, pad=16)
    ax.set_xlabel("Cluster", fontsize=14)

    legend_patches = [
        Patch(color='cornflowerblue', label='Bio'),
        Patch(color='salmon', label='Topo')
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))

    sns.despine(ax=ax, trim=True)
    ax.tick_params(axis='both', which='both', length=0)
    plt.tight_layout(rect=[0, 0, 0.95, 0.93])

    enriched_terms_heatmap_path = os.path.join(
        output_dir,
        f"{model_type}_{net_type}_enriched_terms_heatmap_epo{num_epochs}.png"
    )
    plt.savefig(enriched_terms_heatmap_path, dpi=300)
    plt.close()

    if return_data:
        return heatmap_data, topo_terms_df

def plot_enriched_term_counts(enrichment_results, output_path, model_type, net_type, num_epochs):
    """
    Plot bar chart of the number of enriched terms per cluster for bio and topo clusters,
    with x-axis labels colored according to their type.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = []
    xtick_colors = []
    xtick_labels = []

    colormaps = {
        'bio': get_cmap('Blues')(0.6),     # medium blue
        'topo': get_cmap('YlOrRd')(0.6),   # medium orange-red
    }
        
    for cluster_type in ['bio', 'topo']:
        cluster_ids = list(enrichment_results[cluster_type].keys())
        term_counts = [len(res) for res in enrichment_results[cluster_type].values()]
        labels = [f"{cluster_type.capitalize()}_{i}" for i in cluster_ids]

        color = colormaps[cluster_type]
        bar_container = ax.bar(labels, term_counts, label=cluster_type, color=color)
        bars.extend(bar_container)

        xtick_labels.extend(labels)
        xtick_colors.extend([color] * len(labels))

    ax.set_xlim(-0.65, len(bars) - 0.5)
    ax.set_ylabel("Number of enriched terms", fontsize=28)

    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=20)
    for tick_label, color in zip(ax.get_xticklabels(), xtick_colors):
        tick_label.set_color(color)

    ax.tick_params(axis='y', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    filename = f"{model_type}_{net_type}_term_counts_barplot_epo{num_epochs}.png"
    plt.savefig(os.path.join(output_path, filename), dpi=300)
    plt.show()

    print(f"Bar plot saved to {os.path.join(output_path, filename)}")

def plot_shared_enriched_pathways_venn(enrichment_results, output_path, model_type, net_type, num_epochs):
    """
    Plots a Venn diagram showing the overlap of enriched pathways between bio and topo clusters.
    """
    bio_terms = set(
        sum([df['name'].tolist() for df in enrichment_results['bio'].values() if not df.empty], [])
    )
    topo_terms = set(
        sum([df['name'].tolist() for df in enrichment_results['topo'].values() if not df.empty], [])
    )

    colormaps = {
        'bio': get_cmap('Blues')(0.6),     # medium blue
        'topo': get_cmap('YlOrRd')(0.6),   # medium orange-red
    }
    
    plt.figure(figsize=(8, 8))
    venn = venn2(
        [bio_terms, topo_terms],
        set_labels=('Bio', 'Topo'),
        set_colors=(colormaps['bio'], colormaps['topo']),
        alpha=0.7
    )

    for text in venn.set_labels:
        if text:
            text.set_fontsize(28)
    for text in venn.subset_labels:
        if text:
            text.set_fontsize(28)

    plt.title("Overlap of enriched pathways", fontsize=16)

    filename = f"{model_type}_{net_type}_shared_pathways_venn_epo{num_epochs}.png"
    venn_path = os.path.join(output_path, filename)
    plt.savefig(venn_path, dpi=300)
    plt.show()

    print(f"Venn diagram saved to {venn_path}")


def extract_summary_features_np_topo(topo_features_np):
    """
    Extracts summary features from the topological embedding section (features 1024–2047)
    by computing the max over each 16-dimensional segment.

    Args:
        features_np (np.ndarray): shape [num_nodes, 2048]

    Returns:
        np.ndarray: shape [num_nodes, 64]
    """
    num_nodes, num_features = topo_features_np.shape
    assert num_features == 1024, f"Expected 1024 features, got {num_features}"

    # Select topological features only
    ##topo_features = features_np[:, 1024:]  # shape: [num_nodes, 1024]
    ##topo_features = topo_features_np[:, 1024:2048]
    topo_features = topo_features_np  # already 1024 features


    summary_features = []

    # Pool over 64 chunks of 16 features
    for i in range(64):
        start = i * 16
        end = start + 16
        group = topo_features[:, start:end]  # shape: [num_nodes, 16]
        max_vals = group.max(axis=1, keepdims=True)  # shape: [num_nodes, 1]
        summary_features.append(max_vals)

    return np.concatenate(summary_features, axis=1)  # shape: [num_nodes, 64]

def extract_specific_omics_cancer(bio_embeddings_np, omics_target='mf', cancer_target='BRCA'):
    """
    Extracts 16 features for a specific omics-cancer pair from the 1024 bio features.

    Args:
        bio_embeddings_np (np.ndarray): shape [num_nodes, 1024]
        omics_target (str): one of ['cna', 'ge', 'meth', 'mf']
        cancer_target (str): one of 16 cancer types like 'BRCA'

    Returns:
        np.ndarray: shape [num_nodes, 16]
    """
    omics_types = ['cna', 'ge', 'meth', 'mf']
    # cancer_names = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUAD',
    #                 'LUSC', 'PRAD', 'READ', 'STAD', 'THCA', 'UCEC']
    cancer_names = [
        'BLADDER', 'BREAST', 'CERVIX', 'COLON', 'ESOPHAGUS', 'HEADNECK', 'KIDNEYCC', 'KIDNEYPC',
        'LIVER', 'LUNGAD', 'LUNGSC', 'PROSTATE', 'RECTUM', 'STOMACH', 'THYROID', 'UTERUS'
    ]
    o_idx = omics_types.index(omics_target)
    c_idx = cancer_names.index(cancer_target)

    start = o_idx * 16 * 16 + c_idx * 16
    end = start + 16

    return bio_embeddings_np[:, start:end]

def plot_top_gene_ridge_from_specific_omics_(
    bio_embeddings_np,
    node_names,
    row_labels,
    output_path,
    omics_target='mf',
    cancer_target='BRCA',
    top_n=12
):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    # 1. Extract 16-dim features for the selected omics-cancer pair
    features_16 = extract_specific_omics_cancer(bio_embeddings_np, omics_target, cancer_target)

    # 2. Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # 3. Build DataFrame
    df = pd.DataFrame(features_16)
    df["Gene"] = node_names
    df["Cluster"] = row_labels
    df["MeanFeatureValue"] = df.iloc[:, :-2].mean(axis=1)

    clusters = sorted(df["Cluster"].unique())

    for cluster in clusters:
        cluster_df = df[df["Cluster"] == cluster].copy()
        top_genes = cluster_df.nlargest(top_n, "MeanFeatureValue")

        # Prepare melted long-form data for seaborn
        plot_data = []
        for _, row in top_genes.iterrows():
            gene = row["Gene"]
            for i in range(16):
                plot_data.append({
                    "Gene": gene,
                    "FeatureIndex": i,
                    "Value": row[i]
                })
        plot_df = pd.DataFrame(plot_data)

        # Sort gene order
        gene_order = plot_df.groupby("Gene")["Value"].mean().sort_values().index
        plot_df["Gene"] = pd.Categorical(plot_df["Gene"], categories=gene_order, ordered=True)

        # Compute proper x-axis limits with padding
        xmin = plot_df["Value"].min()
        xmax = plot_df["Value"].max()
        x_range = xmax - xmin
        xmin -= x_range * 0.5
        xmax += x_range * 0.5


        # Plot with ridge style
        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        g = sns.FacetGrid(
            plot_df,
            row="Gene",
            hue="Gene",
            aspect=12,
            height=0.4,
            palette="Spectral",
            sharex=True
        )

        g.map(
            sns.kdeplot,
            "Value",
            bw_adjust=0.5,
            fill=True,
            alpha=0.8,
            cut=100,
            clip=(xmin, xmax)
        )
        g.map(
            sns.kdeplot,
            "Value",
            bw_adjust=0.5,
            color="black",
            lw=1,
            cut=100,
            clip=(xmin, xmax)
        )


        g.set_titles("")
        g.set(xlim=(xmin, xmax), xlabel="Feature Value", ylabel="", yticks=[])

        # Label genes on the left side
        for ax, gene in zip(g.axes.flat, gene_order):
            ax.set_ylabel(gene, rotation=0, ha='right', va='top', fontsize=18, labelpad=10)

        g.despine(bottom=True, left=True)
        g.fig.subplots_adjust(hspace=-0.3, left=0.3, right=0.95, top=0.93)
        g.fig.suptitle(f"Cluster {cluster}", x=0.6, fontsize=20)

        # After g.set(...) and before plt.savefig(...)
        for ax in g.axes.flat:
            ax.tick_params(axis='x', labelsize=16)  # or 14, 16, etc.

        # Save
        out_path = os.path.join(
            output_path,
            f"{omics_target}_{cancer_target}_cluster{cluster}_top{top_n}_genes_ridge.png"
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {out_path}")

def extract_all_omics_for_cancer(bio_embeddings_np, cancer_target='BRCA'):
    """
    Extracts 64 features (4 omics × 16 features) for a specific cancer type from the 1024 bio features.

    Args:
        bio_embeddings_np (np.ndarray): shape [num_nodes, 1024]
        cancer_target (str): one of 16 cancer types like 'BRCA'

    Returns:
        np.ndarray: shape [num_nodes, 64]
    """
    omics_types = ['cna', 'ge', 'meth', 'mf']
    cancer_names = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP',
                    'LIHC', 'LUAD', 'LUSC', 'PRAD', 'READ', 'STAD', 'THCA', 'UCEC']
    # cancer_names = [
    #     'BLADDER', 'BREAST', 'CERVIX', 'COLON', 'ESOPHAGUS', 'HEADNECK', 'KIDNEYCC', 'KIDNEYPC',
    #     'LIVER', 'LUNGAD', 'LUNGSC', 'PROSTATE', 'RECTUM', 'STOMACH', 'THYROID', 'UTERUS'
    # ]    
    c_idx = cancer_names.index(cancer_target)

    feature_blocks = []
    for o_idx in range(len(omics_types)):
        start = o_idx * 16 * 16 + c_idx * 16
        end = start + 16
        feature_blocks.append(bio_embeddings_np[:, start:end])

    return np.concatenate(feature_blocks, axis=1)  # shape: [num_nodes, 64]



def run_gprofiler_enrichment(cluster_dict, cancer_type, tag):
    gp = GProfiler(return_dataframe=True)
    output_files = []
    for cluster, genes in cluster_dict.items():
        try:
            result = gp.profile(
                organism="hsapiens",
                query=genes,
                sources=["REAC", "KEGG", "GO:BP", "HP"],
                user_threshold=0.05,
                significance_threshold_method="fdr"
            )
            if not result.empty:
                for source in ["REAC", "KEGG", "GO:BP", "HP"]:
                    filtered = result[result["source"] == source]
                    path = f"results/gene_prediction/enrichment/{cancer_type}_{tag}_Cluster_{cluster}_{source}_enrichment.csv"
                    dir_path = os.path.dirname(path)
                    os.makedirs(dir_path, exist_ok=True)

                    filtered.to_csv(path, index=False)
                    output_files.append(path)
        except Exception as e:
            print(f"Enrichment failed for {cancer_type} {cluster}: {e}")
    return output_files

def plot_dot_enrichment_per_cluster(cancer_type, tag, source="REAC", top_n=10):
    files = glob.glob(f"results/gene_prediction/{cancer_type}_{tag}_Cluster_*_{source}_enrichment.csv")
    if not files:
        print(f"No enrichment files found for {cancer_type.upper()} [{tag}] and source {source}")
        return

    for f in files:
        try:
            cluster = Path(f).stem.split("_")[2]  # Cluster number
        except IndexError:
            print(f"Filename parsing failed: {f}")
            continue

        df = pd.read_csv(f)
        if df.empty or "intersection_size" not in df or "query_size" not in df:
            print(f"Invalid dataframe for {f}")
            continue

        df["gene_ratio"] = df["intersection_size"] / df["query_size"]
        df["-log10(FDR)"] = -np.log10(df["p_value"].clip(lower=1e-300))
        top_df = df.sort_values("p_value").head(top_n)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=top_df,
            x="gene_ratio",
            y="name",
            hue="-log10(FDR)",
            size="-log10(FDR)",
            sizes=(40, 200),
            palette="viridis",
            legend="brief"
        )
        plt.title(f"{source} Enrichment for {cancer_type.upper()} [{tag}] - Cluster {cluster}")
        plt.xlabel("Gene Ratio")
        plt.ylabel("Pathway")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"results/gene_prediction/{cancer_type}_{tag}_Cluster_{cluster}_{source}_dotplot.png", dpi=300)
        plt.close()

def plot_dot_enrichment_from_results(enrichment_results, top_n=10, source_filter=["REAC", "KEGG", "GO:BP", "HP"]):
    """
    Plot dot plots for enrichment results stored in memory.

    Args:
        enrichment_results (dict): Nested dictionary like {'bio': {0: df, 1: df, ...}, 'topo': {0: df, ...}}.
        top_n (int): Number of top enriched terms to display.
        source_filter (list): List of enrichment sources to include.
    """
    for cluster_type, cluster_data in enrichment_results.items():
        for cluster_id, df in cluster_data.items():
            if df.empty or "p_value" not in df or "intersection_size" not in df or "query_size" not in df:
                print(f"Skipping {cluster_type} cluster {cluster_id}: invalid or empty data")
                continue

            # Optional: filter by source
            if source_filter:
                df = df[df["source"].isin(source_filter)]

            # Compute additional metrics
            df["gene_ratio"] = df["intersection_size"] / df["query_size"]
            df["-log10(FDR)"] = -np.log10(df["p_value"].clip(lower=1e-300))
            df_plot = df.sort_values("p_value").head(top_n)

            if df_plot.empty:
                print(f"No significant enrichment to plot for {cluster_type} cluster {cluster_id}")
                continue

            # Plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=df_plot,
                x="gene_ratio",
                y="name",
                hue="-log10(FDR)",
                size="-log10(FDR)",
                sizes=(40, 200),
                palette="viridis",
                legend="brief"
            )
            plt.title(f"{cluster_type.capitalize()} Cluster {cluster_id} Enrichment")
            plt.xlabel("Gene Ratio")
            plt.ylabel("Pathway")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            # Save plot
            out_path = f"results/enrichment_dotplots/{cluster_type}_cluster_{cluster_id}_dotplot.png"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved: {out_path}")

def plot_dot_enrichment_per_cluster_all(
    cancer_names,
    tag="bio",
    source="REAC",
    top_n=10,
    n_clusters=10,
    base_path="results/gene_prediction/enrichment"
):
    base_path = Path(base_path)

    for cancer_type in cancer_names:
        print(f"\n🔍 Processing {cancer_type.upper()} [{tag}] enrichment dotplots...")
        
        for cluster_id in range(n_clusters):
            file_path = base_path / f"{cancer_type}_{tag}_Cluster_{cluster_id}_{source}_enrichment.csv"

            if not file_path.exists():
                print(f"  ✗ Missing: {file_path.name}")
                continue

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"  ✗ Error reading {file_path.name}: {e}")
                continue

            if df.empty or "intersection_size" not in df or "query_size" not in df:
                print(f"  ✗ Invalid content: {file_path.name}")
                continue

            # Calculate gene ratio and transformed FDR
            df["gene_ratio"] = df["intersection_size"] / df["query_size"]
            df["-log10(FDR)"] = -np.log10(df["p_value"].clip(lower=1e-300))
            top_df = df.sort_values("p_value").head(top_n)

            # Plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=top_df,
                x="gene_ratio",
                y="name",
                hue="-log10(FDR)",
                size="-log10(FDR)",
                sizes=(40, 200),
                palette="viridis",
                legend="brief"
            )
            plt.title(f"{source} Enrichment: {cancer_type.upper()} [{tag}] - Cluster {cluster_id}")
            plt.xlabel("Gene Ratio")
            plt.ylabel("Pathway")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            # Save plot
            out_path = base_path / f"{cancer_type}_{tag}_Cluster_{cluster_id}_{source}_dotplot.png"
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"  ✓ Saved: {out_path.name}")

def apply_full_spectral_biclustering_cancer(cancer_feature, n_clusters):
    from sklearn.cluster import SpectralBiclustering
    import numpy as np

    print(f"Running Spectral Biclustering on 64-dim summary bio features with {n_clusters} clusters...")

    assert cancer_feature.shape[1] == 64, f"Expected 64 summary features, got {cancer_feature.shape[1]}"

    # Perform spectral biclustering
    bicluster = SpectralBiclustering(n_clusters=n_clusters, method='log', random_state=42)
    bicluster.fit(cancer_feature)

    row_labels = bicluster.row_labels_
    print("✅ Spectral Biclustering complete.")

    return row_labels

def plot_all_cancer_ridges_all_omics(
    bio_embeddings_np,
    node_names,
    #row_labels,
    n_clusters_row,
    output_base_path,
    top_n=12
):
    cancer_list = [
        'BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP',
        'LIHC', 'LUAD', 'LUSC', 'PRAD', 'READ', 'STAD', 'THCA', 'UCEC'
    ]

    for cancer in cancer_list:
        print(f"\n📊 Plotting ridge for ALL_OMICS — {cancer}...")
        output_path = os.path.join(output_base_path, f"ALL_OMICS_{cancer}")
        plot_top_gene_ridge_from_all_omics(
            bio_embeddings_np=bio_embeddings_np,
            node_names=node_names,
            #row_labels=row_labels,
            n_clusters_row=n_clusters_row,
            output_path=output_path,
            cancer_target=cancer,
            top_n=top_n
        )

def plot_top_gene_ridge_from_all_omics(
    bio_embeddings_np,
    node_names,
    #row_labels,
    n_clusters_row,
    output_path,
    cancer_target='BRCA',
    top_n=12
):
    # 1. Extract 64-dim features
    #features_64 = extract_all_omics_for_cancer(bio_embeddings_np, cancer_target)
    cancer_feature = extract_all_omics_for_cancer(bio_embeddings_np, cancer_target)
    row_labels = apply_full_spectral_biclustering_cancer(cancer_feature, n_clusters=n_clusters_row)
    
    # 2. Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # 3. Build DataFrame
    df = pd.DataFrame(cancer_feature)
    df["Gene"] = node_names
    df["Cluster"] = row_labels
    df["MeanFeatureValue"] = df.iloc[:, :-2].mean(axis=1)

    clusters = sorted(df["Cluster"].unique())
    cluster_dict = defaultdict(list)  # Save top genes for each cluster

    for cluster in clusters:
        cluster_df = df[df["Cluster"] == cluster].copy()
        top_genes = cluster_df.nlargest(top_n, "MeanFeatureValue")
        cluster_dict[cluster] = top_genes["Gene"].tolist()

        # Prepare long-form data for seaborn
        plot_data = []
        for _, row in top_genes.iterrows():
            gene = row["Gene"]
            for i in range(64):
                plot_data.append({
                    "Gene": gene,
                    "FeatureIndex": i,
                    "Value": row[i]
                })
        plot_df = pd.DataFrame(plot_data)

        gene_order = plot_df.groupby("Gene")["Value"].mean().sort_values().index
        plot_df["Gene"] = pd.Categorical(plot_df["Gene"], categories=gene_order, ordered=True)

        xmin = plot_df["Value"].min()
        xmax = plot_df["Value"].max()
        x_range = xmax - xmin
        xmin -= x_range * 0.5
        xmax += x_range * 0.5

        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        g = sns.FacetGrid(
            plot_df,
            row="Gene",
            hue="Gene",
            aspect=12,
            height=0.4,
            palette="Spectral",
            sharex=True
        )

        g.map(sns.kdeplot, "Value", bw_adjust=0.5, fill=True, alpha=0.8, cut=100, clip=(xmin, xmax))
        g.map(sns.kdeplot, "Value", bw_adjust=0.5, color="black", lw=1, cut=100, clip=(xmin, xmax))

        g.set_titles("")
        g.set(xlim=(xmin, xmax), xlabel="Feature Value", ylabel="", yticks=[])

        for ax, gene in zip(g.axes.flat, gene_order):
            ax.set_ylabel(gene, rotation=0, ha='right', va='top', fontsize=18, labelpad=10)

        g.despine(bottom=True, left=True)
        g.fig.subplots_adjust(hspace=-0.3, left=0.3, right=0.95, top=0.93)
        g.fig.suptitle(f"Cluster {cluster}", x=0.6, fontsize=20)

        for ax in g.axes.flat:
            ax.tick_params(axis='x', labelsize=16)

        out_path = os.path.join(output_path, f"ALL_OMICS_{cancer_target}_cluster{cluster}_top{top_n}_genes_ridge.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {out_path}")

    # Save cluster_dict as JSON
    '''cluster_json_path = os.path.join(output_path, f"ALL_OMICS_{cancer_target}_cluster_genes.json")

    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_safe(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Convert NumPy scalars to native Python types
        else:
            return obj

    cluster_dict_safe = make_json_safe(cluster_dict)

    with open(cluster_json_path, "w") as f:
        json.dump(cluster_dict_safe, f, indent=2)

    print(f"💾 Saved cluster gene dictionary: {cluster_json_path}")'''

def extract_all_omics_for_cancer(bio_embeddings_np, cancer_target='BRCA'):
    """
    Extracts 64 features (4 omics × 16 features) for a specific cancer type from the 1024 bio features.

    Args:
        bio_embeddings_np (np.ndarray): shape [num_nodes, 1024]
        cancer_target (str): one of 16 cancer types like 'BRCA'

    Returns:
        np.ndarray: shape [num_nodes, 64]
    """
    omics_types = ['cna', 'ge', 'meth', 'mf']
    # cancer_names = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP',
    #                 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'READ', 'STAD', 'THCA', 'UCEC']
    cancer_names = [
        'BLADDER', 'BREAST', 'CERVIX', 'COLON', 'ESOPHAGUS', 'HEADNECK', 'KIDNEYCC', 'KIDNEYPC',
        'LIVER', 'LUNGAD', 'LUNGSC', 'PROSTATE', 'RECTUM', 'STOMACH', 'THYROID', 'UTERUS'
    ]    
    c_idx = cancer_names.index(cancer_target)

    feature_blocks = []
    for o_idx in range(len(omics_types)):
        start = o_idx * 16 * 16 + c_idx * 16
        end = start + 16
        feature_blocks.append(bio_embeddings_np[:, start:end])

    return np.concatenate(feature_blocks, axis=1)  # shape: [num_nodes, 64]

def make_cluster_dict(row_labels, node_names):
    cluster_dict = defaultdict(list)
    for idx, label in enumerate(row_labels):
        cluster_dict[label].append(node_names[idx])
    return cluster_dict

def collect_top_enrichments(cancer_type, tag="bio", source="REAC", top_n=10):
    base_path = Path("results/gene_prediction/enrichment")
    base_path.mkdir(parents=True, exist_ok=True)
    terms = []

    for cluster_id in range(10):  # Clusters 0–9
        file = base_path / f"{cancer_type}_{tag}_Cluster_{cluster_id}_{source}_enrichment.csv"
        if not file.exists():
            continue

        df = pd.read_csv(file).sort_values("p_value").head(top_n)
        for _, row in df.iterrows():
            term = f"{row['name']} (C{cluster_id})"
            score = -np.log10(row["p_value"] + 1e-10)
            terms.append((term, score, cluster_id))

    return sorted(terms, key=lambda x: x[1], reverse=True)

def draw_horizontal_bar_plot(terms, cancer_type, source):
    if not terms:
        print(f"No enrichment terms for {cancer_type.upper()} ({source})")
        return

    # Sort and select top 20 terms by score
    terms_sorted = sorted(terms, key=lambda x: x[1], reverse=True)[:20]

    labels = [t[0] for t in terms_sorted]
    scores = [t[1] for t in terms_sorted]
    clusters = [t[2] for t in terms_sorted]
    colors = [CLUSTER_COLORS[c] for c in clusters]

    fig, ax = plt.subplots(figsize=(12, 0.5 * len(terms_sorted)))
    y_pos = np.arange(len(terms_sorted))

    ax.barh(y_pos, scores, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=18)  # Larger font size
    ax.invert_yaxis()  # Highest scores on top
    ax.set_xlabel("-log10(p-value)", fontsize=18)
    ax.set_title(f"Top Enriched Pathways — {cancer_type.upper()} ({source})", fontsize=18)
    ax.tick_params(axis='x', labelsize=16) 
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1) 

    out_path = Path(f"results/gene_prediction/{cancer_type}_{source}_bar_plot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved bar plot: {out_path}")

def collect_enrichment_with_ratios(cancer_type, tag="bio", source="REAC", top_n=10):


    base_path = Path("results/gene_prediction/enrichment")
    base_path.mkdir(parents=True, exist_ok=True)
    terms = []

    for cluster_id in range(10):  # Clusters 0–9
        file = base_path / f"{cancer_type}_{tag}_Cluster_{cluster_id}_{source}_enrichment.csv"
        if not file.exists():
            continue

        df = pd.read_csv(file).sort_values("p_value").head(top_n)
        for _, row in df.iterrows():
            raw_name = row['name']
            # Always define `term`, truncate if needed
            term = raw_name if len(raw_name) <= 60 else raw_name[:57] + "..."
            pval = row['p_value']
            intersection = row['intersection_size']
            input_size = row.get('effective_domain_size', 1)  # Prevent zero division
            gene_ratio = intersection / input_size if input_size > 0 else 0
            terms.append((term, gene_ratio, -np.log10(pval + 1e-10), cluster_id))


    return sorted(terms, key=lambda x: x[2], reverse=True)

def draw_dot_plot_with_ratio(terms, cancer_type, source, top_n=20):
    if not terms:
        print(f"No enrichment terms for {cancer_type.upper()} ({source})")
        return

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(terms, columns=["Term", "GeneRatio", "LogP", "Cluster"])
    df["Color"] = df["Cluster"].map(CLUSTER_COLORS)

    # 🔢 Select top N by LogP
    df = df.sort_values("LogP", ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 0.4 * len(df)))
    scatter = sns.scatterplot(
        data=df,
        x="GeneRatio", y="Term",
        size="LogP", hue="Cluster",
        palette=CLUSTER_COLORS,
        sizes=(50, 300),
        edgecolor="black",
        linewidth=0.5
    )

    # Remove legend
    if scatter.legend_:
        scatter.legend_.remove()
        
    #plt.xlabel("Gene Ratio", fontsize=16)
    plt.ylabel("Enriched Pathway", fontsize=18)
    plt.title(f"Dot Plot (Ratio) — {cancer_type.upper()} ({source})", fontsize=18)
    plt.xscale("log")
    plt.xticks(fontsize=18)
    plt.xlabel("Gene Ratio (log scale)", fontsize=18)

    plt.yticks(fontsize=18)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_path = Path(f"results/gene_prediction/enrichment/{cancer_type}_{source}_dot_plot_ratio.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved dot plot with gene ratio: {out_path}")

def plot_bio_clusterwise_feature_contributions(
    args,
    relevance_scores,           # 2D array (samples x features)
    row_labels,             # 1D array of cluster assignments
    feature_names,              # List of feature names (e.g., MF: BRCA, ...)
    per_cluster_feature_contributions_output_dir, 
    omics_colors):             # Dict of omics type colors (e.g., 'mf': '#D62728')
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    os.makedirs(per_cluster_feature_contributions_output_dir, exist_ok=True)

    def get_omics_color(feature_name):
        prefix = feature_name.split(":")[0].lower()
        return omics_colors.get(prefix, "#AAAAAA")

    def get_omics_prefix(feature_name):
        return feature_name.split(":")[0].lower()

    # Group features by omics type and preserve their indices
    omics_groups = defaultdict(list)
    for idx, fname in enumerate(feature_names):
        omics_groups[get_omics_prefix(fname)].append((idx, fname))

    # Follow omics_colors ordering if possible
    ordered_features = []
    for omics in omics_colors:
        ordered_features.extend(omics_groups.get(omics, []))
    for omics in omics_groups:
        if omics not in omics_colors:
            ordered_features.extend(omics_groups[omics])

    ordered_indices = [idx for idx, _ in ordered_features]
    ordered_feature_names = [name for _, name in ordered_features]

    unique_clusters = np.unique(row_labels)

    for cluster_id in sorted(unique_clusters):
        indices = np.where(row_labels == cluster_id)[0]
        cluster_scores = relevance_scores[indices]
        avg_contribution = np.mean(cluster_scores, axis=0)
        total_score = np.sum(avg_contribution)

        fig, ax = plt.subplots(figsize=(10, 2.5))

        x = np.linspace(0, 1, len(ordered_feature_names))
        bar_width = 1 / len(ordered_feature_names) * 0.95

        bars = ax.bar(
            x,
            avg_contribution[ordered_indices],
            width=bar_width,
            color=[get_omics_color(name) for name in ordered_feature_names],
            align='center'
        )

        ax.set_title(
            fr"Cluster {cluster_id} $\mathregular{{({len(indices)}\ genes,\ avg = {total_score:.2f})}}$",
            fontsize=14
        )

        clean_labels = [name.split(":")[1].strip() if ":" in name else name for name in ordered_feature_names]
        ax.set_xticks(x)
        ax.set_xticklabels(clean_labels, rotation=90)

        for label, feature_name in zip(ax.get_xticklabels(), ordered_feature_names):
            label.set_color(get_omics_color(feature_name))

        ax.tick_params(axis='x', labelsize=9)
        ax.set_xlim(-bar_width, 1 + bar_width)

        plt.tight_layout()
        save_path = os.path.join(
            per_cluster_feature_contributions_output_dir,
            f"{args.model_type}_{args.net_type}_BIO_cluster_{cluster_id}_feature_contributions_epo{args.num_epochs}.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved BIO feature contribution barplot for Cluster {cluster_id} to {save_path}")

def save_graph_with_clusters(graph, save_path):
    torch.save({
        'edges': graph.edges(),
        'features': graph.ndata['feat'],
        'labels': graph.ndata.get('label', None),
        'cluster_bio_summary': graph.ndata['cluster_bio_summary']
    }, save_path)

def compute_total_genes_per_cluster(row_labels, n_clusters):
    return {i: np.sum(row_labels == i) for i in range(n_clusters)}

def compute_relevance_scores_norm(
    model,
    graph,
    features,
    node_indices,
    normalize=True,
    feature_groups=None):  # e.g., {"bio": (0, 1024), "topo": (1024, 2048)}):
    """
    Compute saliency-based relevance scores with optional normalization and feature group selection.

    Args:
        model (torch.nn.Module): Your trained GNN model.
        graph (DGLGraph): The graph structure.
        features (torch.Tensor): Node feature matrix (N x F).
        node_indices (list[int]): Node indices to compute relevance for.
        normalize (bool): Whether to normalize saliency per node.
        feature_groups (dict): Dict of feature group name → (start, end) slice indices.

    Returns:
        dict: node_idx → dict of {group_name: relevance_tensor} or "all": full saliency
    """
    model.eval()
    features = features.clone().detach().requires_grad_(True)

    output = model(graph, features)
    relevance_dict = {}

    for node_idx in node_indices:
        model.zero_grad()
        node_score = output[node_idx].squeeze()
        node_score.backward(retain_graph=True)

        saliency = features.grad[node_idx].detach().abs()  # (F,)
        
        if normalize:
            saliency = saliency / (saliency.sum() + 1e-9)

        if feature_groups:
            group_scores = {}
            for group_name, (start, end) in feature_groups.items():
                group_scores[group_name] = saliency[start:end]
            relevance_dict[node_idx] = group_scores
        else:
            relevance_dict[node_idx] = {"all": saliency}

        features.grad.zero_()

    return relevance_dict

def count_predicted_genes_per_cluster(row_labels, node_names, predicted_cancer_genes, n_clusters):
    pred_counts = {i: 0 for i in range(n_clusters)}
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    predicted_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]
    for idx in predicted_indices:
        if 0 <= idx < len(row_labels):
            pred_counts[row_labels[idx]] += 1
    return pred_counts, predicted_indices

def plot_bio_heatmap_unsort_no_legend_patches(summary_bio_features, row_labels, col_labels, predicted_indices, output_path):
    from matplotlib.colors import LinearSegmentedColormap
    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'KidneyCC', 'KidneyPC',
        'Liver', 'LungAD', 'LungSC', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    bio_feat_names_64 = [f"{omics}_{cancer}" for omics in ['cna', 'ge', 'meth', 'mf'] for cancer in cancer_names]
    topk = 1000
    predicted_indices = predicted_indices[:topk]
    summary_topk = summary_bio_features[predicted_indices]
    row_labels_topk = row_labels[predicted_indices]
    sorted_indices = np.argsort(row_labels_topk)
    sorted_matrix = summary_topk[sorted_indices, :]
    sorted_labels = row_labels_topk[sorted_indices]

    # === Heatmap layout with narrower cluster bar and no gap
    fig = plt.figure(figsize=(17, 17))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[0.25, 13.7], wspace=0)

    ax_cluster = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[0, 1])

    # === Cluster color bar
    cluster_colors = [to_rgb(CLUSTER_COLORS[label]) for label in sorted_labels]
    cluster_colors_array = np.array(cluster_colors).reshape(-1, 1, 3)
    ax_cluster.imshow(cluster_colors_array, aspect='auto')
    ax_cluster.axis("off")

    cluster_counts = Counter(sorted_labels)
    start_idx = 0
    for cluster_id in sorted(cluster_counts):
        count = cluster_counts[cluster_id]
        mid_idx = start_idx + count // 2
        ax_cluster.text(-0.75, mid_idx, f'{count}', va='center', ha='right',
                        fontsize=14, fontweight='bold', color='black')  # larger cluster number
        start_idx += count

        # === Heatmap
        bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
        sns.heatmap(sorted_matrix, cmap=bluish_gray_gradient, center=0, vmin=-2, vmax=2,
                    ax=ax_heatmap, cbar=False)
        ax_heatmap.set_title("Heatmap of Summary Bio Features Sorted by Spectral Biclusters", fontsize=18)
        ax_heatmap.set_yticks([])

        xticks = np.arange(len(bio_feat_names_64)) + 0.5
        ax_heatmap.set_xticks(xticks)

        # Set only cancer names, big and bold
        cancer_labels_only = [name.split('_')[1] for name in bio_feat_names_64]
        ax_heatmap.set_xticklabels(cancer_labels_only, rotation=90, fontsize=18)#, weight='bold')

        # Color each tick label based on omics type
        omics_colors = {
            'cna': '#9370DB', 'ge': '#228B22', 'meth': '#00008B', 'mf': '#b22222',
        }
        for tick_label, name in zip(ax_heatmap.get_xticklabels(), bio_feat_names_64):
            omics = name.split('_')[0]
            tick_label.set_color(omics_colors.get(omics, 'black'))

        ax_heatmap.set_xlabel("Summary Bio Features (Grouped by Omics)", fontsize=18)
        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"✅ Summary bio heatmap saved to {output_path}")

def plot_tsne(features, row_labels, predicted_indices, n_clusters, output_path):
    tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1), random_state=42)
    reduced_embeddings = tsne.fit_transform(features)
    plt.figure(figsize=(12, 10))
    for cluster_id in range(n_clusters):
        idx = np.where(row_labels == cluster_id)[0]
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1],
                    color=CLUSTER_COLORS.get(cluster_id, "#777777"),
                    edgecolor='k', s=100, alpha=0.8)
    for idx in predicted_indices:
        x, y = reduced_embeddings[idx]
        plt.scatter(x, y, facecolors='none', edgecolors='red', s=50, linewidths=2)
    plt.xlabel("t-SNE Dimension 1", fontsize=18)
    plt.ylabel("t-SNE Dimension 2", fontsize=18)
    plt.title("t-SNE of Spectral Biclustering (Summary Bio Features)")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"✅ t-SNE visualization saved to {output_path}")

def plot_bio_heatmap_raw(summary_bio_features, predicted_indices, save_path):
    """
    Plot raw (unclustered) heatmap of summary bio features for predicted genes,
    with omics bar and consistent styling.

    Parameters:
    - summary_bio_features: np.ndarray, shape (num_nodes, 64)
    - predicted_indices: list of indices for predicted cancer genes
    - save_path: path to save the heatmap PNG
    """
    print("📊 Plotting raw bio heatmap (unclustered)...")
    assert summary_bio_features.shape[1] == 64, "Expected 64 summary bio features."

    # === Select only predicted cancer genes
    data = summary_bio_features[predicted_indices]

    # === Omics group boundaries for 64-dim summary bio features
    omics_group_sizes = {
        "expression": 16,
        "methylation": 16,
        "mutation": 16,
        "copy_number": 16
    }
    omics_colors = {
        "expression": "#76D7C4",
        "methylation": "#F7DC6F",
        "mutation": "#F5B7B1",
        "copy_number": "#C39BD3"
    }

    # === Feature names
    feature_names = [f"{omics}_{i}" for omics, size in omics_group_sizes.items() for i in range(size)]
    omics_color_bar = []
    for group, size in omics_group_sizes.items():
        omics_color_bar.extend([omics_colors[group]] * size)

    # === Colormap
    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient",
        ["#F0F3F4", "#85929e"]
    )

    # === Plotting
    fig = plt.figure(figsize=(14, 10))
    grid_spec = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[0.15, 0.85])
    ax_contrib = fig.add_subplot(grid_spec[0])
    ax_heatmap = fig.add_subplot(grid_spec[1])

    # === Column contribution bar
    contrib = data.sum(axis=0)
    ax_contrib.bar(np.arange(data.shape[1]), contrib, color="#85929e", width=1.0)
    ax_contrib.axis("off")

    # === Heatmap
    sns.heatmap(
        data,
        cmap=bluish_gray_gradient,
        ax=ax_heatmap,
        cbar_kws={"label": "Feature Relevance"},
        xticklabels=feature_names,
        yticklabels=[f"Gene{i}" for i in range(data.shape[0])],
        linewidths=0.2,
        linecolor='gray'
    )
    ax_heatmap.set_xlabel("Biological Features")
    ax_heatmap.set_ylabel("Predicted Cancer Genes")
    ax_heatmap.tick_params(axis='x', rotation=90)

    # === Omics group bar
    for x, color in enumerate(omics_color_bar):
        ax_heatmap.add_patch(plt.Rectangle((x, -1), 1, 0.5, color=color, transform=ax_heatmap.transData, clip_on=False))

    # === Omics legend
    handles = [Patch(facecolor=color, label=label) for label, color in omics_colors.items()]
    ax_heatmap.legend(
        handles=handles,
        title="Omics Group",
        loc='upper right',
        bbox_to_anchor=(1.15, 1.0),
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Raw bio heatmap saved to: {save_path}")

def plot_bio_heatmap_raw_unsorted(summary_bio_features, predicted_indices, output_path):
    """
    Plot unclustered raw heatmap of 64-dim summary bio features for top predicted genes.
    Features are grouped into 4 omics types x 16 cancers and are color-coded on x-axis.

    Parameters:
    - summary_bio_features: np.ndarray of shape (num_nodes, 64)
    - predicted_indices: list or array of top predicted gene indices
    - output_path: str, path to save the heatmap image
    """
    print("📊 Plotting raw bio heatmap without clustering...")

    # === Top-K predicted genes to show
    topk = 1000
    predicted_indices = predicted_indices[:topk]
    summary_topk = summary_bio_features[predicted_indices]

    # === Bio feature labels (4 omics × 16 cancers)
    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'KidneyCC', 'KidneyPC',
        'Liver', 'LungAD', 'LungSC', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    bio_feat_names_64 = [f"{omics}_{cancer}" for omics in ['cna', 'ge', 'meth', 'mf'] for cancer in cancer_names]

    # === Figure layout
    fig = plt.figure(figsize=(17, 18))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[17, 2], width_ratios=[0.25, 13.7], hspace=0.3, wspace=0.00)

    ax_cluster = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')

    # === Fake cluster panel (empty) for visual consistency
    ax_cluster.axis("off")

    # === Colormap and heatmap
    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
    sns.heatmap(summary_topk, cmap=bluish_gray_gradient, center=0, vmin=-2, vmax=2,
                ax=ax_heatmap, cbar=False)

    ax_heatmap.set_title("Raw Heatmap of Summary Bio Features (Top Predicted Genes)", fontsize=18)
    ax_heatmap.set_yticks([])

    # === X-axis feature names (with omics coloring)
    xticks = np.arange(len(bio_feat_names_64)) + 0.5
    ax_heatmap.set_xticks(xticks)
    cancer_labels_only = [name.split('_')[1] for name in bio_feat_names_64]
    ax_heatmap.set_xticklabels(cancer_labels_only, rotation=90, fontsize=18)

    # === Omics coloring for x-tick labels
    omics_colors = {'cna': '#9370DB', 'ge': '#228B22', 'meth': '#00008B', 'mf': '#b22222'}
    for tick_label, name in zip(ax_heatmap.get_xticklabels(), bio_feat_names_64):
        omics = name.split('_')[0]
        tick_label.set_color(omics_colors.get(omics, 'black'))

    # === Legend bar for omics types
    omics_patches = [Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()]
    ax_legend.legend(handles=omics_patches, loc='center', ncol=len(omics_patches),
                     fontsize=18, frameon=False)

    # === Final adjustments and save
    fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.03, hspace=0.2, wspace=0.01)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Raw bio heatmap saved to {output_path}")

def plot_topo_heatmap_raw_unsorted(summary_topo_features, predicted_indices, output_path):
    """
    Plot unclustered raw heatmap of 64-dim topological summary features for top predicted genes.

    Parameters:
    - summary_topo_features: np.ndarray of shape (num_nodes, 64)
    - predicted_indices: list or array of top predicted gene indices
    - output_path: str, path to save the heatmap image
    """
    print("📊 Plotting raw topo heatmap without clustering...")

    # === Top-K predicted genes to show
    topk = 1000
    predicted_indices = predicted_indices[:topk]
    summary_topk = summary_topo_features[predicted_indices]

    # === Topo feature labels: "00" to "63"
    topo_feat_names_64 = [f"{i:02d}" for i in range(64)]

    # === Layout
    fig = plt.figure(figsize=(17, 18))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[17, 2], width_ratios=[0.25, 13.7], hspace=0.3, wspace=0.00)
    
    ax_cluster = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')

    # === Empty cluster column for layout consistency
    ax_cluster.axis("off")

    # === Heatmap
    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
    sns.heatmap(
        summary_topk,
        cmap=bluish_gray_gradient,
        center=0,
        vmin=-2, vmax=2,
        ax=ax_heatmap,
        cbar=False
    )
    ax_heatmap.set_title("Raw Topological Feature Heatmap (Top Predicted Genes)", fontsize=18, pad=12)
    ax_heatmap.set_yticks([])

    xticks = np.arange(len(topo_feat_names_64)) + 0.5
    ax_heatmap.set_xticks(xticks)
    ax_heatmap.set_xticklabels(topo_feat_names_64, rotation=90, fontsize=12)

    # === Save
    fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.05)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Raw topo heatmap saved to {output_path}")

def plot_topo_heatmap_unsort(summary_topo_features, row_labels, col_labels, predicted_indices, output_path):

    topo_feat_names_64 = [f"{i:02d}" for i in range(64)]
    topk = 1000
    predicted_indices = predicted_indices[:topk]
    summary_topk = summary_topo_features[predicted_indices]
    row_labels_topk = row_labels[predicted_indices]
    sorted_indices = np.argsort(row_labels_topk)
    sorted_matrix = summary_topk[sorted_indices, :]
    sorted_labels = row_labels_topk[sorted_indices]

    fig = plt.figure(figsize=(17, 18))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[17, 2], width_ratios=[0.25, 13.7], hspace=0.3, wspace=0.00)
    ax_cluster = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')

    cluster_colors = [to_rgb(CLUSTER_COLORS[label]) for label in sorted_labels]
    cluster_colors_array = np.array(cluster_colors).reshape(-1, 1, 3)
    ax_cluster.imshow(cluster_colors_array, aspect='auto')
    ax_cluster.axis("off")

    # Add cluster size labels
    cluster_counts = Counter(sorted_labels)
    start_idx = 0
    for cluster_id in sorted(cluster_counts):
        count = cluster_counts[cluster_id]
        mid_idx = start_idx + count // 2
        ax_cluster.text(-0.75, mid_idx, f'{count}', va='center', ha='right',
                        fontsize=14, fontweight='bold', color='black')
        start_idx += count

    # Heatmap
    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
    sns.heatmap(
        sorted_matrix,
        cmap=bluish_gray_gradient,
        center=0,
        vmin=-2, vmax=2,
        ax=ax_heatmap,
        cbar=False
    )
    ax_heatmap.set_title("Topological Feature Heatmap (Sorted by Cluster)", fontsize=18, pad=12)
    ax_heatmap.set_yticks([])

    xticks = np.arange(len(topo_feat_names_64)) + 0.5
    ax_heatmap.set_xticks(xticks)
    ax_heatmap.set_xticklabels(topo_feat_names_64, rotation=90, fontsize=12)

    # Save
    fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.05)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Summary topo heatmap saved to {output_path}")

def plot_bio_heatmap_unsort(
    summary_bio_features, 
    row_labels, 
    col_labels, 
    predicted_indices, 
    output_path
    ):

    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'KidneyCC', 'KidneyPC',
        'Liver', 'LungAD', 'LungSC', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    bio_feat_names_64 = [f"{omics}_{cancer}" for omics in ['cna', 'ge', 'meth', 'mf'] for cancer in cancer_names]
    topk = 1000
    predicted_indices = predicted_indices[:topk]
    summary_topk = summary_bio_features[predicted_indices]
    row_labels_topk = row_labels[predicted_indices]
    sorted_indices = np.argsort(row_labels_topk)
    sorted_matrix = summary_topk[sorted_indices, :]
    sorted_labels = row_labels_topk[sorted_indices]

    fig = plt.figure(figsize=(17, 18))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[17, 2], width_ratios=[0.25, 13.7], hspace=0.3, wspace=0.00)
    ax_cluster = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')

    cluster_colors = [to_rgb(CLUSTER_COLORS[label]) for label in sorted_labels]
    cluster_colors_array = np.array(cluster_colors).reshape(-1, 1, 3)
    ax_cluster.imshow(cluster_colors_array, aspect='auto')
    ax_cluster.axis("off")

    cluster_counts = Counter(sorted_labels)
    start_idx = 0
    for cluster_id in sorted(cluster_counts):
        count = cluster_counts[cluster_id]
        mid_idx = start_idx + count // 2
        ax_cluster.text(-0.75, mid_idx, f'{count}', va='center', ha='right',
                        fontsize=14, fontweight='bold', color='black')
        start_idx += count

    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
    sns.heatmap(sorted_matrix, cmap=bluish_gray_gradient, center=0, vmin=-2, vmax=2,
                ax=ax_heatmap, cbar=False)
    ax_heatmap.set_title("Heatmap of Summary Bio Features Sorted by Spectral Biclusters", fontsize=18)
    ax_heatmap.set_yticks([])

    xticks = np.arange(len(bio_feat_names_64)) + 0.5
    ax_heatmap.set_xticks(xticks)
    cancer_labels_only = [name.split('_')[1] for name in bio_feat_names_64]
    ax_heatmap.set_xticklabels(cancer_labels_only, rotation=90, fontsize=18)

    omics_colors = {'cna': '#9370DB', 'ge': '#228B22', 'meth': '#00008B', 'mf': '#b22222'}
    for tick_label, name in zip(ax_heatmap.get_xticklabels(), bio_feat_names_64):
        omics = name.split('_')[0]
        tick_label.set_color(omics_colors.get(omics, 'black'))

    omics_patches = [Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()]
    ax_legend.legend(handles=omics_patches, loc='center', ncol=len(omics_patches),
                     fontsize=18, frameon=False)

    fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.03, hspace=0.2, wspace=0.01)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Summary bio heatmap saved to {output_path}")

def plot_bio_biclustering_heatmap_unsort_random_clusterbar(
    args,
    relevance_scores,
    row_labels,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None,
    col_labels=None
):  
    
    # 🔹 Extract and normalize relevance scores
    #relevance_scores = extract_summary_features_np_bio(relevance_scores)
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min()) * 20

    if omics_colors is None:
        omics_colors = {
            'cna': '#9370DB',    # purple
            'ge': '#228B22',     # dark green
            'meth': '#00008B',   # dark blue
            'mf': '#b22222',     # dark red
        }

    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'KidneyCC', 'KidneyPC',
        'Liver', 'LungAD', 'LungSC', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    omics_order = ['cna', 'ge', 'meth', 'mf']
    feature_names = [f"{omics.upper()}: {cancer}" for omics in omics_order for cancer in cancer_names]

    # If col_labels provided, reorder columns accordingly
    # if col_labels is not None:
    #     sorted_order = np.argsort(col_labels)
    #     relevance_scores = relevance_scores[:, sorted_order]
    #     feature_names = [feature_names[i] for i in sorted_order]

    # Build feature color bar
    feature_colors = []
    for i in range(len(feature_names)):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")  # fallback color

    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50, wspace=0.0, hspace=0.0)

    ax_bar    = fig.add_subplot(gs[0, 2:45])      
    ax        = fig.add_subplot(gs[1:13, 2:45])    
    ax_curve  = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar   = fig.add_subplot(gs[5:9, 49])

    ax_bar.axis("off")
    ax_bar.set_xlim(0, len(feature_names))
    ax_bar.set_ylim(0, 1.6)

    # Normalize per-feature means
    feature_means = relevance_scores.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min() + 1e-6)

    for i, (mean_val, color) in enumerate(zip(feature_means, feature_colors)):
        ax_bar.bar(
            x=i + 0.5,
            height=mean_val,
            width=1.0,
            bottom=0,
            color=color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.3 + 0.7 * mean_val
        )

    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient",
        ["#F0F3F4", "#85929e"]
    )

    vmin = 0
    vmax = np.percentile(relevance_scores, 99)

    sns.heatmap(
        relevance_scores,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Relevance Score",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )

    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=18)
    ax_cbar.yaxis.label.set_size(18)

    # Cluster stripe
    for i, cluster in enumerate(row_labels):
        ax.add_patch(plt.Rectangle((-1.5, i), 1.5, 1, linewidth=0, facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')), clip_on=False))

    # Cluster size text
    unique_clusters, cluster_sizes = np.unique(row_labels, return_counts=True)
    start_idx = 0
    for cluster, size in zip(unique_clusters, cluster_sizes):
        center_y = start_idx + size / 2
        ax.text(-2.0, center_y, f"{size}", va='center', ha='right', fontsize=18, fontweight='bold')
        start_idx += size

    # Add xtick labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels([c.split(": ")[1] for c in feature_names], rotation=90, fontsize=14)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # LRP curve
    saliency_sums = relevance_scores.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())
    y = np.arange(len(saliency_sums))
    ax_curve.fill_betweenx(
        y, 0, saliency_sums,
        color='#a9cce3',
        alpha=0.8,
        linewidth=3
    )

    # Omics bar below
    omics_means = {}
    for omics, (start, end) in omics_splits.items():
        group_scores = relevance_scores[:, start:end+1]
        omics_means[omics] = group_scores.mean()

    group_centers = {
        omics: (omics_splits[omics][0] + omics_splits[omics][1]) / 2 + 0.5
        for omics in omics_order
    }

    mean_vals = np.array([omics_means[om] for om in omics_order])
    min_mean, max_mean = mean_vals.min(), mean_vals.max()
    normalized_means = (mean_vals - min_mean) / (max_mean - min_mean + 1e-6)

    for i, omics in enumerate(omics_order):
        ax.bar(
            x=group_centers[omics],
            height=0.15,
            width=(omics_splits[omics][1] - omics_splits[omics][0] + 1),
            bottom=len(relevance_scores) + 1.5,
            color=omics_colors[omics],
            edgecolor='black',
            linewidth=1,
            alpha=0.3 + 0.7 * normalized_means[i]
        )

    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.hlines(
        y=1.01, xmin=0, xmax=1,
        color='black', linewidth=1.5, transform=ax_curve.get_xaxis_transform()
    )
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')
    ax_curve.set_ylim(0, len(saliency_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.set_yticks([])
    ax_curve.set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_topo_heatmap_unsorted(
    args,
    relevance_scores,
    output_path,
    gene_names=None,
    col_labels=None
):
    """
    Plots a topological heatmap (unsorted) using the same visual formatting
    as the biclustering version, but with rows and columns in original order.

    Args:
        args: CLI or config object with settings.
        relevance_scores (np.ndarray): shape [num_nodes, 2048], full embedding.
        output_path (str): Path to save the figure.
        gene_names (list of str, optional): Gene name labels for heatmap index.
        col_labels (list of str, optional): Optional column label annotations.
    Returns:
        pd.DataFrame: heatmap matrix with genes as rows and topo features as columns.
    """

    # 🔹 Extract 64D summary of topological features
    relevance_scores = extract_summary_features_np_topo(relevance_scores)
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min())

    # 🔹 Feature names
    feature_names = [f"{i+1:02d}" for i in range(relevance_scores.shape[1])]
    if col_labels is not None:
        col_labels = np.array(col_labels)

    # 🔹 Apply log1p for better contrast
    scores_log = np.log1p(relevance_scores)

    # 🔹 Set up figure
    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50, wspace=0.0, hspace=0.0)


    ax_bar = fig.add_subplot(gs[0, 2:45])
    ax = fig.add_subplot(gs[1:13, 2:45])
    ax_curve = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[5:9, 49])
    #ax_legend = fig.add_subplot(gs[14, 2:45])

    # 🔹 Colorbar range
    vmin, vmax = 0, np.percentile(scores_log, 99)

    # 🔹 Feature contribution bar (column saliency)
    feature_means = scores_log.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min()) * 0.04

    ax_bar.bar(
        np.arange(len(feature_means)) + 0.5,
        feature_means,
        width=1.0,
        color="#B0BEC5",
        linewidth=0,
        alpha=0.6
    )
    ax_bar.set_xlim(0, len(feature_means))
    ax_bar.set_ylim(0, 0.04)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    for spine in ['left', 'bottom', 'top', 'right']:
        ax_bar.spines[spine].set_visible(False)

    # 🔹 Colormap
    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient", ["#F0F3F4", "#85929e"]
    )

    # 🔹 Heatmap
    sns.heatmap(
        scores_log,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Log-Scaled Relevance",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )
    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=16)
    ax_cbar.yaxis.label.set_size(18)

    # 🔹 Tick labels (X)
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=90, fontsize=16)
    ax.tick_params(axis='x', bottom=True, labelbottom=True)

    # 🔹 Omics/Saliency Legend
    '''ax_legend.axis("off")
    lrp_patch = Patch(facecolor='#a9cce3', alpha=0.8, label='Saliency Sum')
    ax_legend.legend(
        handles=[lrp_patch],
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=16,
        handleheight=1.5,
        handlelength=3
    )'''

    # 🔹 Saliency sum curve (row-wise sum)
    saliency_sums = scores_log.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())
    y = np.arange(len(saliency_sums))

    ax_curve.fill_betweenx(
        y, 0, saliency_sums,
        color='#a9cce3',
        alpha=0.8,
        linewidth=3
    )
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.set_ylim(0, len(saliency_sums))
    for spine in ['right', 'left', 'bottom', 'top']:
        ax_curve.spines[spine].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    # 🔹 Final layout
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved unsorted topological heatmap to {output_path}")

    return pd.DataFrame(scores_log, index=gene_names, columns=feature_names)


def analyze_sankey_structure(
    source,
    target,
    value,
    label_to_idx,
    node_names,
    cluster_to_genes,
    gene_to_neighbors,
    row_labels,
    name_to_index,
    relevance_scores
):
    # === Build Directed Graph
    G = nx.DiGraph()
    for s, t, v in zip(source, target, value):
        G.add_edge(s, t, weight=v)

    node_id_to_name = {v: k for k, v in label_to_idx.items()}

    # === 1. Degree Centrality of Genes
    gene_nodes = [label_to_idx[g] for g in node_names if g in label_to_idx]
    degree_centrality = nx.degree_centrality(G)
    gene_centrality_scores = {
        node_id_to_name[n]: degree_centrality[n]
        for n in gene_nodes if n in degree_centrality
    }

    # === 2. Gene-to-Cluster Participation Count
    gene_cluster_participation = defaultdict(set)
    for cluster_label, genes in cluster_to_genes.items():
        for gene in genes:
            gene_cluster_participation[gene].add(cluster_label)
    gene_cluster_counts = {gene: len(clusters) for gene, clusters in gene_cluster_participation.items()}

    # === 3. Entropy of Flow Distributions (per Confirmed Cluster)
    # cluster_entropy = {}
    # for cluster_label, genes in cluster_to_genes.items():
    #     scores_arr = np.array([relevance_scores[name_to_index[g]] for g in genes])
    #     probs = scores_arr / scores_arr.sum() if scores_arr.sum() > 0 else np.ones_like(scores_arr) / len(scores_arr)
    #     cluster_entropy[cluster_label] = entropy(probs)

    # === 3. Entropy of Flow Distributions (per Confirmed Cluster)
    cluster_entropy = {}
    for cluster_label, genes in cluster_to_genes.items():
        scores_arr = np.array([np.linalg.norm(relevance_scores[name_to_index[g]]) for g in genes])  # ensure 1D
        if scores_arr.sum() > 0:
            probs = scores_arr / scores_arr.sum()
        else:
            probs = np.ones_like(scores_arr) / len(scores_arr)
        cluster_entropy[cluster_label] = float(entropy(probs))  # ensure scalar

    # === 4. Jaccard Similarity Between Clusters Based on Shared Downstream Clusters
    cluster_to_downstream = defaultdict(set)
    for gene, neighbors in gene_to_neighbors.items():
        if gene not in name_to_index:
            continue
        gene_idx = name_to_index[gene]
        cluster = f"Confirmed Cluster {row_labels[gene_idx]}"
        for neighbor_idx in neighbors:
            neighbor_cluster = f"Cluster {row_labels[neighbor_idx]}"
            cluster_to_downstream[cluster].add(neighbor_cluster)

    cluster_jaccard = {}
    for c1, c2 in combinations(cluster_to_downstream.keys(), 2):
        s1 = cluster_to_downstream[c1]
        s2 = cluster_to_downstream[c2]
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        if union > 0:
            cluster_jaccard[(c1, c2)] = intersection / union

    return {
        "gene_degree_centrality": gene_centrality_scores,
        "gene_cluster_counts": gene_cluster_counts,
        "cluster_entropy": cluster_entropy,
        "cluster_jaccard": cluster_jaccard,  # <- changed from cluster_jaccard_similarity
        "cluster_participation": gene_cluster_counts
    }

def save_metrics_to_csv(metrics_dict, output_dir='sankey_metrics'):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Degree Centrality
    pd.DataFrame.from_dict(metrics_dict['gene_degree_centrality'], orient='index', columns=['degree_centrality'])\
        .sort_values('degree_centrality', ascending=False)\
        .to_csv(f"{output_dir}/gene_degree_centrality.csv")

    # 2. Cluster Participation Count
    pd.DataFrame.from_dict(metrics_dict['gene_cluster_counts'], orient='index', columns=['cluster_count'])\
        .sort_values('cluster_count', ascending=False)\
        .to_csv(f"{output_dir}/gene_cluster_counts.csv")

    # 3. Entropy per Cluster
    pd.DataFrame.from_dict(metrics_dict['cluster_entropy'], orient='index', columns=['entropy'])\
        .sort_values('entropy', ascending=False)\
        .to_csv(f"{output_dir}/cluster_entropy.csv")

    # 4. Jaccard Similarity between Clusters
    # Jaccard Similarity between Clusters
    jaccard_df = pd.DataFrame([
        {'Cluster 1': k[0], 'Cluster 2': k[1], 'Jaccard Similarity': v}
        for k, v in metrics_dict['cluster_jaccard'].items()
    ])

    jaccard_df.sort_values(by='Jaccard Similarity', ascending=False)\
        .to_csv(f"{output_dir}/cluster_jaccard_similarity.csv", index=False)

def save_predictions_to_csv(predicted_genes, output_dir, model_type, net_type, num_epochs):
    """
    Save the predicted genes with their sources to a CSV file.
    
    Args:
    - predicted_genes: List of tuples (gene, score, sources) to save.
    - output_dir: Directory to save the CSV file.
    - model_type, net_type, num_epochs: For naming the output file.
    """
    os.makedirs(output_dir, exist_ok=True)
    predicted_genes_csv_path = os.path.join(output_dir, f'{model_type}_{net_type}_predicted_driver_genes_epo{num_epochs}_2048.csv')
    df_predictions = pd.DataFrame(predicted_genes, columns=["Gene", "Score", "Confirmed Sources"])
    df_predictions.to_csv(predicted_genes_csv_path, index=False)
    print(f"Predicted driver genes with confirmed sources saved to {predicted_genes_csv_path}")

def save_predicted_known_drivers(predicted_driver_genes, output_dir, model_type, net_type, num_epochs):
    """
    Save predicted known cancer driver genes to a CSV file.
    
    Args:
    - predicted_driver_genes: List of predicted cancer driver genes.
    - output_dir: Directory to save the CSV file.
    - model_type, net_type, num_epochs: For naming the output file.
    """
    predicted_drivers_csv_path = os.path.join(output_dir, f'{model_type}_{net_type}_predicted_known_drivers_epo{num_epochs}_2048.csv')
    df = pd.DataFrame(predicted_driver_genes, columns=["Gene"])
    df.to_csv(predicted_drivers_csv_path, index=False)
    print(f"Predicted known driver genes saved to {predicted_drivers_csv_path}")
        
def visualize_feature_relevance_heatmaps(relevance_df, clusters, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Merge relevance with cluster labels
    merged = relevance_df.copy()
    merged['gene'] = merged.index
    merged = pd.merge(merged, clusters, on='gene')

    for cluster_type in clusters.columns[1:]:
        for view in ['cancer', 'omics']:
            cluster_groups = merged.groupby(cluster_type)
            all_cluster_heatmaps = []
            fig, axes = plt.subplots(len(cluster_groups), 1, figsize=(12, 4 * len(cluster_groups)))
            if len(cluster_groups) == 1:
                axes = [axes]

            for ax, (label, group) in zip(axes, cluster_groups):
                data = group.drop(columns=['gene'] + list(clusters.columns[1:]))

                if view == 'cancer':
                    # Collapse omics features per cancer (max-over-omics per cancer)
                    cancer_names = list(set([col.split('_')[0] for col in data.columns]))
                    cancer_view = pd.DataFrame(index=data.index, columns=cancer_names)
                    for ct in cancer_names:
                        cols = [col for col in data.columns if col.startswith(ct + '_')]
                        cancer_view[ct] = data[cols].max(axis=1)
                    plot_data = cancer_view
                    title = f"{cluster_type.capitalize()} cluster {label} — Cancer view"
                    fname = f"{cluster_type.capitalize()}_cluster_{label}_cancer_heatmap.png"

                else:
                    # Collapse across cancer types for each omics type (max-over-cancers per omics)
                    omics_types = list(set([col.split('_')[-1] for col in data.columns]))
                    omics_view = pd.DataFrame(index=data.index, columns=omics_types)
                    for om in omics_types:
                        cols = [col for col in data.columns if col.endswith('_' + om)]
                        omics_view[om] = data[cols].max(axis=1)
                    plot_data = omics_view
                    title = f"{cluster_type.capitalize()} cluster {label} — Omics view"
                    fname = f"{cluster_type.capitalize()}_cluster_{label}_omics_heatmap.png"

                # Normalize per row for better heatmap contrast
                plot_data = pd.DataFrame(StandardScaler().fit_transform(plot_data.T).T,
                                         index=plot_data.index, columns=plot_data.columns)
                sns.heatmap(plot_data, cmap='viridis', ax=ax, cbar=True)
                ax.set_title(title)
                ax.set_xlabel('Features')
                ax.set_ylabel('Genes')

                ##fig.tight_layout()
                fig.subplots_adjust(hspace=0.0, top=0.98, bottom=0.01)

                fig.savefig(os.path.join(output_dir, fname))
                plt.close(fig)

def cluster_and_visualize_predicted_genes(graph, predicted_cancer_genes, node_names, 
                                          output_path_genes_clusters, num_clusters=12):
    """
    Clusters gene embeddings into groups using KMeans, visualizes them with t-SNE, 
    and marks predicted cancer genes with red circles (half the size of non-cancer dots).

    Returns:
        row_labels (np.ndarray): Cluster assignments for each gene.
        total_genes_per_cluster (dict): Total number of genes per cluster.
        pred_counts (dict): Number of predicted cancer genes per cluster.
    """
    # Extract embeddings
    embeddings = graph.ndata['feat'].cpu().numpy()

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=12)
    row_labels = kmeans.fit_predict(embeddings)

    # Store cluster labels in graph
    graph.ndata['cluster'] = torch.tensor(row_labels, dtype=torch.long, device=graph.device)

    # Calculate the total number of genes per cluster
    total_genes_per_cluster = {i: np.sum(row_labels == i) for i in range(num_clusters)}

    # Calculate the number of predicted cancer genes per cluster
    pred_counts = {i: 0 for i in range(num_clusters)}
    '''for gene_idx in predicted_cancer_genes:
        print(f"gene_idx type: {type(gene_idx)}, value: {gene_idx}")
        print(f"row_labels type: {type(row_labels)}, length: {len(row_labels)}")

        cluster_id = row_labels[gene_idx]
        pred_counts[cluster_id] += 1'''

    # Convert gene names to their corresponding indices
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    predicted_cancer_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    # Ensure valid indices before using them
    for gene_idx in predicted_cancer_gene_indices:
        if 0 <= gene_idx < len(row_labels):  
            cluster_id = row_labels[gene_idx]
            pred_counts[cluster_id] += 1
        else:
            print(f"Skipping invalid index: {gene_idx}")


    # Reduce dimensions with t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))

    # Plot clusters (Non-cancer genes)
    non_cancer_dot_size = 100  # Default dot size
    red_circle_size = non_cancer_dot_size / 2  # Half the size

    for cluster_id in range(num_clusters):
        idx = np.where(row_labels == cluster_id)[0]  # Get indices of this cluster
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
                    color=CLUSTER_COLORS[cluster_id], 
                    edgecolor='k', s=non_cancer_dot_size, alpha=0.8)

    # Mark predicted cancer genes with red circles (⚪, half the size)
    for gene_idx in predicted_cancer_gene_indices:
        x, y = reduced_embeddings[gene_idx]
        plt.scatter(x, y, facecolors='none', edgecolors='red', s=red_circle_size, linewidths=2)


    # Labels and title
    ##plt.title("Gene clustering with predicted cancer genes")
    plt.xlabel("t-SNE Dimension 1", fontsize=18)
    plt.ylabel("t-SNE Dimension 2", fontsize=18)

    # Save plot
    plt.savefig(output_path_genes_clusters, bbox_inches="tight")  # Ensure proper cropping
    plt.close()

    print(f"Cluster visualization saved to {output_path_genes_clusters}")

    return row_labels, total_genes_per_cluster, pred_counts

def compute_lrp_scores(model, graph, features, node_indices=None):
    model.eval()
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)

    features.requires_grad_(True)


    with torch.enable_grad():
        logits = model(graph, features)
        probs = torch.sigmoid(logits.squeeze())

        # Select the nodes to analyze (e.g., predicted cancer genes)
        if node_indices is None:
            node_indices = torch.nonzero((probs > 0.0)).squeeze()

        relevance_scores = torch.zeros_like(features)

        for idx in node_indices:
            model.zero_grad()
            ##probs[idx].backward(retain_graph=True)
            probs[idx].backward(retain_graph=(idx != node_indices[-1]))

            relevance_scores[idx] = features.grad[idx].detach()

    return relevance_scores

def load_gene_set(file_path):
    """
    Load a gene list from a file and return as a set.
    
    Args:
    - file_path: Path to the file containing genes, one per line.
    
    Returns:
    - Set of gene names.
    """
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)

def save_row_labels(row_labels, save_path):
    ##row_labels_path = os.path.join(os.path.dirname(save_path), "row_labels.npy")
    np.save(save_path, row_labels)
    print(f"Cluster labels saved to {save_path}")

# Save total genes per cluster
def save_total_genes_per_cluster(total_genes_per_cluster, save_path):
    ##total_genes_path = os.path.join(os.path.dirname(save_path), "total_genes_per_cluster.npy")
    np.save(save_path, total_genes_per_cluster)
    print(f"Total genes per cluster saved to {save_path}")

# Save predicted cancer genes per cluster
def save_predicted_counts(pred_counts, save_path):
    ##pred_counts_path = os.path.join(os.path.dirname(save_path), "predicted_counts.npy")
    np.save(save_path, pred_counts)
    print(f"Predicted cancer genes per cluster saved to {save_path}")

def reload_row_labels(save_path):
    ##row_labels_path = os.path.join(os.path.dirname(save_path), "row_labels.npy")
    row_labels = np.load(save_path)
    print(f"Cluster labels loaded from {save_path}")
    return row_labels

# Function to reload total genes per cluster
def reload_total_genes_per_cluster(save_path):
    ##total_genes_path = os.path.join(os.path.dirname(save_path), "total_genes_per_cluster.npy")
    total_genes_per_cluster = np.load(save_path, allow_pickle=True).item()
    print(f"Total genes per cluster loaded from {save_path}")
    return total_genes_per_cluster

# Function to reload predicted cancer genes per cluster
def reload_predicted_counts(save_path):
    ##pred_counts_path = os.path.join(os.path.dirname(save_path), "predicted_counts.npy")
    pred_counts = np.load(save_path, allow_pickle=True).item()
    print(f"Predicted cancer genes per cluster loaded from {save_path}")
    return pred_counts

def load_bioclustered_graph(save_path):
    """
    Loads a previously saved DGL graph that includes features, labels, and cluster assignments.

    Args:
        save_path (str): Path to the saved graph file (.pth)

    Returns:
        dgl.DGLGraph: The reconstructed graph with restored node data.
    """
    data = torch.load(save_path)

    graph = dgl.graph(data['edges'])
    graph.ndata['feat'] = data['features']

    if data.get('label') is not None:
        graph.ndata['label'] = data['label']

    if data.get('cluster') is not None:
        graph.ndata['cluster'] = data['cluster']

    if 'degree' in data:
        graph.ndata['degree'] = data['degree']
    if 'train_mask' in data:
        graph.ndata['train_mask'] = data['train_mask']
    if 'test_mask' in data:
        graph.ndata['test_mask'] = data['test_mask']

    print("✅ Clustered graph loaded successfully.")
    return graph

def save_bioclustered_gene_info_csv(
    row_labels,
    total_genes_per_cluster,
    pred_counts,
    node_names,
    predicted_gene_indices,
    output_csv_path
):
    """
    Saves clustered gene information to a CSV file.

    Args:
        row_labels (np.ndarray): Cluster assignments.
        total_genes_per_cluster (dict): Total genes in each cluster.
        pred_counts (dict): Number of predicted cancer genes per cluster.
        node_names (list): List of all gene names by index.
        predicted_gene_indices (list): Indices of predicted cancer genes.
        output_csv_path (str): Path to save the CSV file.
    """
    gene_data = []
    for idx, name in enumerate(node_names):
        cluster = row_labels[idx]
        is_predicted = 1 if idx in predicted_gene_indices else 0
        gene_data.append({
            "Gene": name,
            "Cluster": cluster,
            "IsPredictedCancerGene": is_predicted,
            "TotalGenesInCluster": total_genes_per_cluster[cluster],
            "PredictedInCluster": pred_counts[cluster]
        })

    df = pd.DataFrame(gene_data)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Clustered gene information saved to {output_csv_path}")

def save_reduced_feature_relevance_scores(
    relevance_scores,      # Original 2048D matrix, shape [N, 2048]
    gene_names,            # List of gene symbols, length N
    output_csv_path        # File to save the reduced 64D matrix
):
    """
    Reduce 2048D LRP scores to 64D using max-over-16 logic,
    and save to CSV with columns like 'BRCA_mf', ..., 'KIRP_meth'.
    """
    import numpy as np
    import pandas as pd
    import os

    # -- Step 1: Reduce features to 64D
    reduced_scores = extract_summary_features_np_skip(relevance_scores)  # shape [N, 64]

    # -- Step 2: Define feature names in cancer x omics order
    cancer_names = [
        'BLADDER', 'BREAST', 'CERVIX', 'COLON', 'ESOPHAGUS', 'HEADNECK', 'KIDNEYCC', 'KIDNEYPC',
        'LIVER', 'LUNGAD', 'LUNGSC', 'PROSTATE', 'RECTUM', 'STOMACH', 'THYROID', 'UTERUS'
    ]
    omics_types = ['cna', 'ge', 'meth', 'mf']
    column_labels = [f"{cancer}_{omics}" for omics in omics_types for cancer in cancer_names]

    # -- Step 3: Build DataFrame and save
    df = pd.DataFrame(reduced_scores, index=gene_names, columns=column_labels)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path)
    print(f"✅ Saved reduced feature relevance matrix to {output_csv_path}")

def plot_top_predicted_genes_tsne(graph, node_names, scores, output_path, top_k=1000):
    cluster_ids = graph.ndata['cluster_bio'].cpu().numpy()
    embeddings = graph.ndata['feat'].cpu().numpy()
    scores = scores##.cpu().numpy()

    # Get top K predicted genes
    top_indices = np.argsort(scores)[-top_k:]
    top_embeddings = embeddings[top_indices]
    top_clusters = cluster_ids[top_indices]
    top_scores = scores[top_indices]
    top_names = [node_names[i] for i in top_indices]

    # t-SNE projection
    tsne = TSNE(n_components=2, random_state=42)
    tsne_coords = tsne.fit_transform(top_embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    rcParams['pdf.fonttype'] = 42  # prevent font issues in vector graphics

    for c in np.unique(top_clusters):
        mask = top_clusters == c
        coords = tsne_coords[mask]
        plt.scatter(
            coords[:, 0], coords[:, 1],
            color=CLUSTER_COLORS.get(c, "#555555"),
            edgecolors='k',
            s=60,
            alpha=0.8
        )

        # Top 1 in this cluster (within top_k)
        cluster_scores = top_scores[mask]
        if cluster_scores.size > 0:
            top_idx_in_cluster = np.argmax(cluster_scores)
            name = np.array(top_names)[mask][top_idx_in_cluster]
            x, y = coords[top_idx_in_cluster]
            # Highlight the top 1 with a yellow circle (half the size of the original dot)
            plt.scatter(x, y, s=60, edgecolors='yellow', alpha=0.6, linewidth=1, marker='o', color='red')
            # Change label text color to red
            plt.text(x, y, name, fontsize=9, fontweight='bold', ha='center', va='center', color='black')

    plt.title("t-SNE of Top 1000 Predicted Genes by Cluster", fontsize=14)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    # Remove legend
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Top predicted gene t-SNE plot saved to:\n{output_path}")

def plot_tsne_predicted_genes(graph, node_names, scores, output_path, args):
    cluster_ids = graph.ndata['cluster_bio'].cpu().numpy()
    embeddings = graph.ndata['feat'].cpu().numpy()
    scores = scores##.cpu().numpy()

    predicted_mask = scores >= args.score_threshold
    predicted_indices = np.where(predicted_mask)[0]
    predicted_scores = scores[predicted_indices]
    predicted_clusters = cluster_ids[predicted_indices]
    predicted_embeddings = embeddings[predicted_indices]

    tsne = TSNE(n_components=2, random_state=42)
    tsne_coords = tsne.fit_transform(predicted_embeddings)

    # Gather top 2 genes per cluster
    top_genes = []
    for c in np.unique(predicted_clusters):
        cluster_mask = predicted_clusters == c
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            continue
        top_indices = cluster_indices[np.argsort(predicted_scores[cluster_indices])[-2:]]  # top 2
        for idx in top_indices:
            top_genes.append((predicted_indices[idx], tsne_coords[idx], c))

    # Plot
    plt.figure(figsize=(10, 7))
    for idx, (node_idx, coord, cluster_id) in enumerate(top_genes):
        color = CLUSTER_COLORS.get(cluster_id, "#333333")
        plt.scatter(coord[0], coord[1], color=color, s=120, edgecolor='k')
        plt.text(coord[0]+1.5, coord[1], node_names[node_idx], fontsize=9, color=color)

    # Legend
    unique_predicted_clusters = np.unique(predicted_clusters)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=f"Cluster {c}", markersize=10)
        for c, color in CLUSTER_COLORS.items()
        if c in unique_predicted_clusters
    ]
    plt.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("t-SNE of Top 2 Predicted Genes per Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_importance(
    relevance_vector,
    feature_names=None,
    node_name=None,
    top_k=9,
    gene_names=None,
    output_path="plots/feature_importance.png"):
    # Convert to NumPy array
    if isinstance(relevance_vector, torch.Tensor):
        relevance_vector = relevance_vector.detach().cpu().numpy()
    else:
        relevance_vector = np.array(relevance_vector)

    # --- 🔄 Min-Max Normalization ---
    min_val = relevance_vector.min()
    max_val = relevance_vector.max()
    norm_scores = (relevance_vector - min_val) / (max_val - min_val + 1e-8)

    # --- 🔝 Top-K selection ---
    top_indices = np.argsort(norm_scores)[-top_k:]
    top_scores = norm_scores[top_indices]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(relevance_vector))]

    # --- 🧬 Labeling ---
    if gene_names is not None:
        top_labels = [gene_names[i].capitalize() if i < len(gene_names) else f"Unknown {i}" for i in top_indices]
    else:
        top_labels = [feature_names[i] for i in top_indices]

    # Construct DataFrame
    df = pd.DataFrame({
        "feature": top_labels,
        "relevance": top_scores
    })

    # --- 📊 Plot ---
    plt.figure(figsize=(2.5, 2.5))
    sns.set_style("white")
    sns.barplot(
        data=df,
        x="feature",
        y="relevance",
        palette="Blues_d",
        dodge=False,
        legend=False,
        width=0.6
    )

    plt.ylabel("Relevance score", fontsize=12)
    plt.xlabel("", fontsize=12)
    plt.title(f"{node_name}", fontsize=13)

    plt.xticks(rotation=90, ha='center', fontsize=11)
    plt.yticks(fontsize=11)

    sns.despine()
    plt.tight_layout()

    # --- 💾 Ensure directory exists and save ---
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved barplot to {output_path}")
    else:
        plt.close()

def _plot_bar(omics_relevance, omics_colors, omics_order, output_path):
    plt.figure(figsize=(1.4, 2))
    sns.barplot(
        x=omics_relevance.index,
        y=omics_relevance.values,
        palette=[omics_colors[o] for o in omics_order],
        width=0.6
    )

    # Capitalize x-axis labels
    plt.xticks(
        ticks=range(len(omics_relevance.index)),
        labels=[label.upper() for label in omics_relevance.index],
        rotation=90,
        fontsize=9
    )
    plt.yticks(fontsize=10)
    plt.xlabel('', fontsize=10)
    plt.ylabel('', fontsize=10)
    plt.tick_params(axis='both', which='both', length=0)
    sns.despine()

    # Use scientific notation on y-axis
    '''ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))'''
    ax = plt.gca()

    # Set scientific formatter (plain 'e' format)
    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)

    # Reduce the font size of the offset (e.g., '1e-3' label)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(6)
    ax.yaxis.offsetText.set_horizontalalignment('left')
        
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved barplot to {output_path}")
    else:
        plt.close()

def extract_summary_features_np_skip(features_np):
    num_nodes = features_np.shape[0]
    total_dim = features_np.shape[1]
    summary_features = []

    for o_idx in range(4):  # omics
        for c_idx in range(16):  # cancer
            base = o_idx * 16 * 16 + c_idx * 16
            if base + 16 > total_dim:
                continue  # skip invalid slice
            group = features_np[:, base:base + 16]  # [num_nodes, 16]
            max_vals = group.max(axis=1, keepdims=True)
            summary_features.append(max_vals)

    return np.concatenate(summary_features, axis=1)

def compute_integrated_gradients(
    model, graph, features, node_indices=None, baseline=None, steps=50):
    model.eval()

    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)
    features = features.clone().detach()

    if baseline is None:
        baseline = torch.zeros_like(features)

    assert baseline.shape == features.shape, "Baseline must match feature shape"

    # Scale inputs and compute gradients
    scaled_inputs = [
        baseline + (float(i) / steps) * (features - baseline)
        for i in range(1, steps + 1)
    ]
    scaled_inputs = torch.stack(scaled_inputs)  # Shape: (steps, num_nodes, num_features)

    # Integrated gradients trainedization
    integrated_grads = torch.zeros_like(features)

    for step_input in scaled_inputs:
        step_input.requires_grad_(True)
        logits = model(graph, step_input)
        probs = torch.sigmoid(logits.squeeze())

        if node_indices is None:
            node_indices = torch.nonzero(probs > 0.0, as_tuple=False).squeeze()
            if node_indices.ndim == 0:
                node_indices = node_indices.unsqueeze(0)

        grads = torch.autograd.grad(
            outputs=probs[node_indices],
            inputs=step_input,
            grad_outputs=torch.ones_like(probs[node_indices]),
            retain_graph=True,
            create_graph=False,
        )[0]

        integrated_grads += grads.detach()

    # Average the gradients and scale by the input difference
    avg_grads = integrated_grads / steps
    ig_attributions = (features - baseline) * avg_grads

    return ig_attributions

def get_two_hop_neighbors(graph, node_id):
    one_hop = set(graph.neighbors(node_id)) if node_id in graph else set()
    two_hop = set()
    
    for neighbor in one_hop:
        two_hop.update(graph.neighbors(neighbor))
    
    # Remove the original node and one-hop neighbors from two-hop set
    two_hop.difference_update(one_hop)
    two_hop.discard(node_id)
    
    return sorted(two_hop)

def save_cluster_legend(output_path_legend, cluster_colors, num_clusters=12):
    """
    Creates and saves a separate legend image for cluster colors in a single row.

    Args:
        output_path_legend (str): Path to save the legend image.
        cluster_colors (list): List of colors for each cluster.
        num_clusters (int): Number of clusters.
    """
    fig, ax = plt.subplots(figsize=(12, 1.5))  # Wider and shorter for single-row legend

    # Create legend handles labeled from Cluster 0 to Cluster 11
    legend_patches = [mpatches.Patch(color=cluster_colors[i], label=f"Cluster {i}") 
                      for i in range(num_clusters)]

    # Display legend with one row
    ax.legend(handles=legend_patches, loc='center', ncol=num_clusters,
              frameon=False, fontsize=14)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Save legend image
    plt.savefig(output_path_legend, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Legend saved to {output_path_legend}")

def compute_saliency_all_nodes(model, g, features, target_classes=None):
    """
    Compute saliency (relevance scores) for all nodes using gradients.

    Args:
        model: Trained GNN model
        g: DGL graph
        features: Input node features (torch.Tensor or np.ndarray)
        target_classes: Optional tensor/list of target classes per node. If None, use predicted class.

    Returns:
        relevance_scores: Tensor of shape [num_nodes, num_features] with saliency values.
    """
    model.eval()
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)
    
    features = features.clone().detach().requires_grad_(True)
    logits = model(g, features)

    num_nodes = features.shape[0]
    relevance_scores = torch.zeros_like(features)

    if target_classes is None:
        target_classes = torch.argmax(logits, dim=1)

    for node_id in range(num_nodes):
        model.zero_grad()
        if features.grad is not None:
            features.grad.zero_()

        score = logits[node_id, target_classes[node_id]]
        score.backward(retain_graph=True)

        relevance_scores[node_id] = features.grad[node_id].abs().detach()

    return relevance_scores


def plot_bio_biclustering_heatmap_kcg(
    args,
    relevance_scores,
    row_labels,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None,
    col_labels=None,
    kcg_list=None
):
    # Normalize relevance scores
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min()) * 10

    if omics_colors is None:
        omics_colors = {
            'cna': '#9370DB',
            'ge': '#228B22',
            'meth': '#00008B',
            'mf': '#b22222',
        }

    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'KidneyCC', 'KidneyPC',
        'Liver', 'LungAD', 'LungSC', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    omics_order = ['cna', 'ge', 'meth', 'mf']
    feature_names = [f"{omics.upper()}: {cancer}" for omics in omics_order for cancer in cancer_names]

    # Column sorting (omics and per-feature)
    feature_avgs = relevance_scores.mean(axis=0)
    omics_group_means = {}
    for omics in omics_order:
        start, end = omics_splits[omics]
        group_indices = list(range(start, end + 1))
        group_mean = feature_avgs[group_indices].mean()
        omics_group_means[omics] = group_mean
    sorted_omics_order = sorted(omics_order, key=lambda x: omics_group_means[x], reverse=True)

    sorted_col_indices = []
    sorted_feature_names = []
    sorted_feature_colors = []
    new_omics_splits = {}
    col_cursor = 0

    for omics in sorted_omics_order:
        start, end = omics_splits[omics]
        group_indices = list(range(start, end + 1))
        group_avgs = feature_avgs[group_indices]
        group_sorted = [i for _, i in sorted(zip(group_avgs, group_indices), reverse=True)]

        new_omics_splits[omics] = (col_cursor, col_cursor + len(group_sorted) - 1)
        col_cursor += len(group_sorted)

        sorted_col_indices.extend(group_sorted)
        sorted_feature_names.extend([feature_names[i] for i in group_sorted])
        sorted_feature_colors.extend([omics_colors[omics]] * len(group_sorted))

    relevance_scores = relevance_scores[:, sorted_col_indices]
    feature_names = sorted_feature_names
    feature_colors = sorted_feature_colors

    # Row sorting: by cluster, then by saliency within cluster
    cluster_ids = np.unique(row_labels)
    ordered_row_indices = []
    for cluster_id in np.sort(cluster_ids):
        cluster_mask = (row_labels == cluster_id)
        cluster_scores = relevance_scores[cluster_mask]
        saliency_sums = cluster_scores.sum(axis=1)
        intra_cluster_order = np.argsort(-saliency_sums)
        cluster_indices = np.where(cluster_mask)[0][intra_cluster_order]
        ordered_row_indices.extend(cluster_indices)

    sorted_scores = relevance_scores[ordered_row_indices]
    sorted_clusters = row_labels[ordered_row_indices]

    # Plotting
    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
    vmin, vmax = 0, np.percentile(sorted_scores, 99)



    # Construct row labels: use gene names if available, else fallback to generic
    row_gene_names = [gene_names[i] if gene_names is not None else f"Gene_{i}" for i in ordered_row_indices]

    # ➤ Filter only known cancer genes
    if kcg_list is not None:
        kcg_mask = [name in kcg_list for name in row_gene_names]
        sorted_scores = sorted_scores[kcg_mask]
        sorted_clusters = sorted_clusters[kcg_mask]
        row_gene_names = [name for name, keep in zip(row_gene_names, kcg_mask) if keep]

    # Build DataFrame for ordered scores
    df_ordered_scores = pd.DataFrame(sorted_scores, index=row_gene_names, columns=feature_names)

    # Add cluster information
    df_ordered_scores.insert(0, "Cluster", sorted_clusters)
    
    base_dir = os.path.dirname(output_path)
    cluster_output_dir = os.path.join(base_dir, "cluster_csvs")
    os.makedirs(cluster_output_dir, exist_ok=True)

    cluster_output_dir = output_path.replace(".png", "_cluster_csvs")
    os.makedirs(cluster_output_dir, exist_ok=True)

    for cluster_id in np.unique(sorted_clusters):
        cluster_df = df_ordered_scores[df_ordered_scores["Cluster"] == cluster_id]
        cluster_csv_path = os.path.join(cluster_output_dir, f"cluster_{cluster_id}_genes.csv")
        cluster_df.to_csv(cluster_csv_path)
        print(f"Saved cluster {cluster_id} gene scores to {cluster_csv_path}")

    # Save to CSV
    csv_output_path = output_path.replace(".png", "_ordered_scores.csv")
    df_ordered_scores.to_csv(csv_output_path)
    print(f"Ordered relevance score matrix with cluster info saved to {csv_output_path}")



    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50, wspace=0.0, hspace=0.0)
    ax_bar = fig.add_subplot(gs[0, 2:45])
    ax = fig.add_subplot(gs[1:13, 2:45])
    ax_curve = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[5:9, 49])

    # Top bar
    feature_means = sorted_scores.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min() + 1e-6)
    ax_bar.axis("off")
    ax_bar.set_xlim(0, len(feature_names))
    ax_bar.set_ylim(0, 1.1)

    for i, (val, color) in enumerate(zip(feature_means, feature_colors)):
        ax_bar.bar(
            x=i + 0.5, 
            height=val, 
            width=1.0,
            color=color, 
            edgecolor='black', 
            linewidth=0.5, 
            alpha=0.3 + 0.7 * val
        )

    # Heatmap
    sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={"label": "Relevance Score", "shrink": 0.1, "aspect": 12, "pad": 0.02, "orientation": "vertical", "location": "right"},
        ax=ax
    )
    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=18)
    ax_cbar.yaxis.label.set_size(18)

    # Cluster stripes
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle((-1.5, i), 1.5, 1, linewidth=0, facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')), clip_on=False))

    # Cluster size labels
    unique_clusters, cluster_sizes = np.unique(sorted_clusters, return_counts=True)
    start_idx = 0
    for cluster, size in zip(unique_clusters, cluster_sizes):
        center_y = start_idx + size / 2
        ax.text(-2.0, center_y, f"{size}", va='center', ha='right', fontsize=18, fontweight='bold')
        start_idx += size

    # X-axis labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels([f.split(": ")[1] for f in feature_names], rotation=90, fontsize=14)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, length=5)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    # Saliency curve
    saliency_sums = sorted_scores.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())
    ax_curve.fill_betweenx(
        np.arange(len(saliency_sums)), 
        0, 
        saliency_sums, 
        color='#a9cce3', 
        alpha=0.8, 
        linewidth=3)

    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.set_ylim(0, len(saliency_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.hlines(y=1.01, xmin=0, xmax=1, color='black', linewidth=1.5, transform=ax_curve.get_xaxis_transform())
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')

    # Omics group bars
    for omics in sorted_omics_order:
        start, end = new_omics_splits[omics]
        group_center = (start + end) / 2 + 0.5
        mean_val = sorted_scores[:, start:end+1].mean()
        norm_mean = (mean_val - np.min(feature_means)) / (np.max(feature_means) - np.min(feature_means) + 1e-6)
        ax.bar(
            x=group_center,
            height=0.15,
            width=end - start + 1,
            bottom=len(sorted_scores) + 1.5,
            color=omics_colors[omics],
            edgecolor='black',
            linewidth=1,
            alpha=min(1.0, 0.3 + 0.7 * norm_mean)
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 🔹 Optional: Cluster-wise contributions
    plot_bio_clusterwise_feature_contributions(
        args=args,
        relevance_scores=relevance_scores,
        row_labels=row_labels,
        feature_names=feature_names,
        per_cluster_feature_contributions_output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions_bio"),
        omics_colors=omics_colors
    )

def plot_bio_biclustering_heatmap_npcg(
    args,
    relevance_scores,
    row_labels,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None,
    col_labels=None,
    kcg_list=None,                  # List of known cancer gene symbols
    plot_only_novel=True 
):
    # Normalize relevance scores
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min()) * 10

    if omics_colors is None:
        omics_colors = {
            'cna': '#9370DB',
            'ge': '#228B22',
            'meth': '#00008B',
            'mf': '#b22222',
        }

    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'KidneyCC', 'KidneyPC',
        'Liver', 'LungAD', 'LungSC', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    omics_order = ['cna', 'ge', 'meth', 'mf']
    feature_names = [f"{omics.upper()}: {cancer}" for omics in omics_order for cancer in cancer_names]

    # Column sorting (omics and per-feature)
    feature_avgs = relevance_scores.mean(axis=0)
    omics_group_means = {}
    for omics in omics_order:
        start, end = omics_splits[omics]
        group_indices = list(range(start, end + 1))
        group_mean = feature_avgs[group_indices].mean()
        omics_group_means[omics] = group_mean
    sorted_omics_order = sorted(omics_order, key=lambda x: omics_group_means[x], reverse=True)

    sorted_col_indices = []
    sorted_feature_names = []
    sorted_feature_colors = []
    new_omics_splits = {}
    col_cursor = 0

    for omics in sorted_omics_order:
        start, end = omics_splits[omics]
        group_indices = list(range(start, end + 1))
        group_avgs = feature_avgs[group_indices]
        group_sorted = [i for _, i in sorted(zip(group_avgs, group_indices), reverse=True)]

        new_omics_splits[omics] = (col_cursor, col_cursor + len(group_sorted) - 1)
        col_cursor += len(group_sorted)

        sorted_col_indices.extend(group_sorted)
        sorted_feature_names.extend([feature_names[i] for i in group_sorted])
        sorted_feature_colors.extend([omics_colors[omics]] * len(group_sorted))

    relevance_scores = relevance_scores[:, sorted_col_indices]
    feature_names = sorted_feature_names
    feature_colors = sorted_feature_colors

    # Row sorting: by cluster, then by saliency within cluster
    cluster_ids = np.unique(row_labels)
    ordered_row_indices = []
    for cluster_id in np.sort(cluster_ids):
        cluster_mask = (row_labels == cluster_id)
        cluster_scores = relevance_scores[cluster_mask]
        saliency_sums = cluster_scores.sum(axis=1)
        intra_cluster_order = np.argsort(-saliency_sums)
        cluster_indices = np.where(cluster_mask)[0][intra_cluster_order]
        ordered_row_indices.extend(cluster_indices)

    sorted_scores = relevance_scores[ordered_row_indices]
    sorted_clusters = row_labels[ordered_row_indices]

    # Plotting
    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
    vmin, vmax = 0, np.percentile(sorted_scores, 99)


    # Construct row labels: use gene names if available, else fallback
    row_gene_names = [gene_names[i] if gene_names is not None else f"Gene_{i}" for i in ordered_row_indices]

    # ➤ Filter to plot only novel predicted cancer genes
    if kcg_list is not None and plot_only_novel:
        novel_mask = [name not in kcg_list for name in row_gene_names]
        sorted_scores = sorted_scores[novel_mask]
        sorted_clusters = sorted_clusters[novel_mask]
        row_gene_names = [name for name, keep in zip(row_gene_names, novel_mask) if keep]


    # Build DataFrame for ordered scores
    df_ordered_scores = pd.DataFrame(sorted_scores, index=row_gene_names, columns=feature_names)

    # Add cluster information
    df_ordered_scores.insert(0, "Cluster", sorted_clusters)
    
    base_dir = os.path.dirname(output_path)
    cluster_output_dir = os.path.join(base_dir, "cluster_csvs")
    os.makedirs(cluster_output_dir, exist_ok=True)

    cluster_output_dir = output_path.replace(".png", "_cluster_csvs")
    os.makedirs(cluster_output_dir, exist_ok=True)

    for cluster_id in np.unique(sorted_clusters):
        cluster_df = df_ordered_scores[df_ordered_scores["Cluster"] == cluster_id]
        cluster_csv_path = os.path.join(cluster_output_dir, f"cluster_{cluster_id}_genes.csv")
        cluster_df.to_csv(cluster_csv_path)
        print(f"Saved cluster {cluster_id} gene scores to {cluster_csv_path}")

    # Save to CSV
    csv_output_path = output_path.replace(".png", "_ordered_scores.csv")
    df_ordered_scores.to_csv(csv_output_path)
    print(f"Ordered relevance score matrix with cluster info saved to {csv_output_path}")



    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50, wspace=0.0, hspace=0.0)
    ax_bar = fig.add_subplot(gs[0, 2:45])
    ax = fig.add_subplot(gs[1:13, 2:45])
    ax_curve = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[5:9, 49])

    # Top bar
    feature_means = sorted_scores.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min() + 1e-6)
    ax_bar.axis("off")
    ax_bar.set_xlim(0, len(feature_names))
    ax_bar.set_ylim(0, 1.1)

    for i, (val, color) in enumerate(zip(feature_means, feature_colors)):
        ax_bar.bar(
            x=i + 0.5, 
            height=val, 
            width=1.0,
            color=color, 
            edgecolor='black', 
            linewidth=0.5, 
            alpha=0.3 + 0.7 * val
        )

    # Heatmap
    sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={"label": "Relevance Score", "shrink": 0.1, "aspect": 12, "pad": 0.02, "orientation": "vertical", "location": "right"},
        ax=ax
    )
    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=18)
    ax_cbar.yaxis.label.set_size(18)

    # Cluster stripes
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle((-1.5, i), 1.5, 1, linewidth=0, facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')), clip_on=False))

    # Cluster size labels
    unique_clusters, cluster_sizes = np.unique(sorted_clusters, return_counts=True)
    start_idx = 0
    for cluster, size in zip(unique_clusters, cluster_sizes):
        center_y = start_idx + size / 2
        ax.text(-2.0, center_y, f"{size}", va='center', ha='right', fontsize=18, fontweight='bold')
        start_idx += size

    # X-axis labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels([f.split(": ")[1] for f in feature_names], rotation=90, fontsize=14)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, length=5)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    # Saliency curve
    saliency_sums = sorted_scores.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())
    ax_curve.fill_betweenx(
        np.arange(len(saliency_sums)), 
        0, 
        saliency_sums, 
        color='#a9cce3', 
        alpha=0.8, 
        linewidth=3)

    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.set_ylim(0, len(saliency_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.hlines(y=1.01, xmin=0, xmax=1, color='black', linewidth=1.5, transform=ax_curve.get_xaxis_transform())
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')

    # Omics group bars
    for omics in sorted_omics_order:
        start, end = new_omics_splits[omics]
        group_center = (start + end) / 2 + 0.5
        mean_val = sorted_scores[:, start:end+1].mean()
        norm_mean = (mean_val - np.min(feature_means)) / (np.max(feature_means) - np.min(feature_means) + 1e-6)
        ax.bar(
            x=group_center,
            height=0.15,
            width=end - start + 1,
            bottom=len(sorted_scores) + 1.5,
            color=omics_colors[omics],
            edgecolor='black',
            linewidth=1,
            alpha=min(1.0, 0.3 + 0.7 * norm_mean)
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 🔹 Optional: Cluster-wise contributions
    plot_bio_clusterwise_feature_contributions(
        args=args,
        relevance_scores=relevance_scores,
        row_labels=row_labels,
        feature_names=feature_names,
        per_cluster_feature_contributions_output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions_bio"),
        omics_colors=omics_colors
    )

def apply_full_spectral_biclustering_bio(
    graph, summary_bio_features, node_names_topk, omics_splits,
    predicted_cancer_genes,
    save_path, save_row_labels_path,
    save_total_genes_per_cluster_path, 
    save_predicted_counts_path,
    output_path_genes_clusters, 
    output_path_heatmap,
    output_dir,
    args,
    topk_node_indices=None
):
    # import torch
    # import numpy as np
    # from sklearn.metrics import mean_squared_error
    # from sklearn.cluster import SpectralBiclustering
    # from utils import (  # Replace with actual locations if needed
    #     save_graph_with_clusters, save_row_labels, compute_total_genes_per_cluster,
    #     save_total_genes_per_cluster, count_predicted_genes_per_cluster, save_predicted_counts
    # )
    # from plotting import (
    #     plot_bio_biclustering_heatmap_unsort,
    #     plot_bio_biclustering_clustermap,
    #     plot_predicted_genes_distribution
    # )
    # import os

    print("🧪 Running Spectral Biclustering with fixed (16, 10) clusters...")

    # === ✅ Step 1: Filter top-k nodes
    if topk_node_indices is None:
        raise ValueError("`topk_node_indices` must be provided")

    ##summary_bio_features_topk = summary_bio_features[topk_node_indices]
    summary_bio_features_topk = summary_bio_features  # Already top-k
    #node_names_topk = node_names
    #node_names_topk = [node_names[i] for i in topk_node_indices]

    assert summary_bio_features_topk.shape[1] == 64, f"Expected 64 summary features, got {summary_bio_features_topk.shape[1]}"

    # === ✅ Step 2: Run Biclustering
    n_clusters_row = 10
    n_clusters_col = 5

    best_model = None
    best_score = np.inf

    print("🔁 Running biclustering trials:")
    for i in range(10):
        model = SpectralBiclustering(n_clusters=(n_clusters_row, n_clusters_col), method='bistochastic',
                                     svd_method='randomized', random_state=i)
        
        model.fit(summary_bio_features_topk)

        reconstructed = summary_bio_features_topk[np.argsort(model.row_labels_)][:, np.argsort(model.column_labels_)]
        mse = mean_squared_error(summary_bio_features_topk, reconstructed)

        if mse < best_score:
            best_score = mse
            best_model = model

    bicluster = best_model
    row_labels = bicluster.row_labels_
    col_labels = bicluster.column_labels_

    # === ✅ Step 3: Assign row cluster labels back to graph
    row_labels_tensor = torch.full((graph.num_nodes(),), -1, dtype=torch.long)
    row_labels_tensor[topk_node_indices] = torch.tensor(row_labels, dtype=torch.long)
    graph.ndata['cluster_bio_summary'] = row_labels_tensor

    print("✅ Spectral Biclustering complete.")

    # === ✅ Step 4: Save clustering outputs
    save_graph_with_clusters(graph, save_path)
    save_row_labels(row_labels, save_row_labels_path)
    total_genes_per_cluster = compute_total_genes_per_cluster(row_labels, n_clusters_row)
    save_total_genes_per_cluster(total_genes_per_cluster, save_total_genes_per_cluster_path)

    pred_counts, predicted_indices = count_predicted_genes_per_cluster(
        row_labels, node_names_topk, predicted_cancer_genes, n_clusters_row
    )
    save_predicted_counts(pred_counts, save_predicted_counts_path)

    # === ✅ Step 5: Plot heatmaps and distributions
    plot_bio_biclustering_heatmap_unsort(
        args=args,
        relevance_scores=summary_bio_features_topk,
        omics_splits=omics_splits,
        output_path=os.path.join(output_dir, "heatmap_unsort.png"),
        row_labels=row_labels,
        col_labels=col_labels
    )

    plot_bio_biclustering_clustermap(
        args=args,
        relevance_scores=summary_bio_features_topk,
        omics_splits=omics_splits,
        output_path=os.path.join(output_dir, "clustermap.png"),
        row_labels=row_labels,
        col_labels=col_labels
    )

    plot_predicted_genes_distribution(
        pred_counts=pred_counts,
        output_path=os.path.join(output_dir, "predicted_genes_per_cluster.png")
    )

    return graph, row_labels, col_labels, total_genes_per_cluster, pred_counts

def plot_bio_biclustering_heatmap(
    args,
    relevance_scores,
    row_labels,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None,
    col_labels=None
):
    # Normalize relevance scores
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min()) * 10

    if omics_colors is None:
        omics_colors = {
            'cna': '#9370DB',
            'ge': '#228B22',
            'meth': '#00008B',
            'mf': '#b22222',
        }

    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'KidneyCC', 'KidneyPC',
        'Liver', 'LungAD', 'LungSC', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    omics_order = ['cna', 'ge', 'meth', 'mf']
    feature_names = [f"{omics.upper()}: {cancer}" for omics in omics_order for cancer in cancer_names]

    # Column sorting (omics and per-feature)
    feature_avgs = relevance_scores.mean(axis=0)
    omics_group_means = {}
    for omics in omics_order:
        start, end = omics_splits[omics]
        group_indices = list(range(start, end + 1))
        group_mean = feature_avgs[group_indices].mean()
        omics_group_means[omics] = group_mean
    sorted_omics_order = sorted(omics_order, key=lambda x: omics_group_means[x], reverse=True)

    sorted_col_indices = []
    sorted_feature_names = []
    sorted_feature_colors = []
    new_omics_splits = {}
    col_cursor = 0

    for omics in sorted_omics_order:
        start, end = omics_splits[omics]
        group_indices = list(range(start, end + 1))
        group_avgs = feature_avgs[group_indices]
        group_sorted = [i for _, i in sorted(zip(group_avgs, group_indices), reverse=True)]

        new_omics_splits[omics] = (col_cursor, col_cursor + len(group_sorted) - 1)
        col_cursor += len(group_sorted)

        sorted_col_indices.extend(group_sorted)
        sorted_feature_names.extend([feature_names[i] for i in group_sorted])
        sorted_feature_colors.extend([omics_colors[omics]] * len(group_sorted))

    relevance_scores = relevance_scores[:, sorted_col_indices]
    feature_names = sorted_feature_names
    feature_colors = sorted_feature_colors

    # Row sorting: by cluster, then by saliency within cluster
    cluster_ids = np.unique(row_labels)
    ordered_row_indices = []
    for cluster_id in np.sort(cluster_ids):
        cluster_mask = (row_labels == cluster_id)
        cluster_scores = relevance_scores[cluster_mask]
        saliency_sums = cluster_scores.sum(axis=1)
        intra_cluster_order = np.argsort(-saliency_sums)
        cluster_indices = np.where(cluster_mask)[0][intra_cluster_order]
        ordered_row_indices.extend(cluster_indices)

    sorted_scores = relevance_scores[ordered_row_indices]
    sorted_clusters = row_labels[ordered_row_indices]

    # Plotting
    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
    vmin, vmax = 0, np.percentile(sorted_scores, 99)



    # Construct row labels: use gene names if available, else fallback to generic
    row_gene_names = [gene_names[i] if gene_names is not None else f"Gene_{i}" for i in ordered_row_indices]

    # Build DataFrame for ordered scores
    df_ordered_scores = pd.DataFrame(sorted_scores, index=row_gene_names, columns=feature_names)

    # Add cluster information
    df_ordered_scores.insert(0, "Cluster", sorted_clusters)
    
    base_dir = os.path.dirname(output_path)
    cluster_output_dir = os.path.join(base_dir, "cluster_csvs")
    os.makedirs(cluster_output_dir, exist_ok=True)

    cluster_output_dir = output_path.replace(".png", "_cluster_csvs")
    os.makedirs(cluster_output_dir, exist_ok=True)

    for cluster_id in np.unique(sorted_clusters):
        cluster_df = df_ordered_scores[df_ordered_scores["Cluster"] == cluster_id]
        cluster_csv_path = os.path.join(cluster_output_dir, f"cluster_{cluster_id}_genes.csv")
        cluster_df.to_csv(cluster_csv_path)
        print(f"Saved cluster {cluster_id} gene scores to {cluster_csv_path}")

    # Save to CSV
    csv_output_path = output_path.replace(".png", "_ordered_scores.csv")
    df_ordered_scores.to_csv(csv_output_path)
    print(f"Ordered relevance score matrix with cluster info saved to {csv_output_path}")



    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50, wspace=0.0, hspace=0.0)
    ax_bar = fig.add_subplot(gs[0, 2:45])
    ax = fig.add_subplot(gs[1:13, 2:45])
    ax_curve = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[5:9, 49])

    # Top bar
    feature_means = sorted_scores.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min() + 1e-6)
    ax_bar.axis("off")
    ax_bar.set_xlim(0, len(feature_names))
    ax_bar.set_ylim(0, 1.1)

    for i, (val, color) in enumerate(zip(feature_means, feature_colors)):
        ax_bar.bar(
            x=i + 0.5, 
            height=val, 
            width=1.0,
            color=color, 
            edgecolor='black', 
            linewidth=0.5, 
            alpha=0.3 + 0.7 * val
        )

    # Heatmap
    sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={"label": "Relevance Score", "shrink": 0.1, "aspect": 12, "pad": 0.02, "orientation": "vertical", "location": "right"},
        ax=ax
    )
    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=18)
    ax_cbar.yaxis.label.set_size(18)

    # Cluster stripes
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle((-1.5, i), 1.5, 1, linewidth=0, facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')), clip_on=False))

    # Cluster size labels
    unique_clusters, cluster_sizes = np.unique(sorted_clusters, return_counts=True)
    start_idx = 0
    for cluster, size in zip(unique_clusters, cluster_sizes):
        center_y = start_idx + size / 2
        ax.text(-2.0, center_y, f"{size}", va='center', ha='right', fontsize=18, fontweight='bold')
        start_idx += size

    # X-axis labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels([f.split(": ")[1] for f in feature_names], rotation=90, fontsize=14)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, length=5)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    # Saliency curve
    saliency_sums = sorted_scores.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())
    ax_curve.fill_betweenx(
        np.arange(len(saliency_sums)), 
        0, 
        saliency_sums, 
        color='#a9cce3', 
        alpha=0.8, 
        linewidth=3)

    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.set_ylim(0, len(saliency_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.hlines(y=1.01, xmin=0, xmax=1, color='black', linewidth=1.5, transform=ax_curve.get_xaxis_transform())
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')

    # Omics group bars
    for omics in sorted_omics_order:
        start, end = new_omics_splits[omics]
        group_center = (start + end) / 2 + 0.5
        mean_val = sorted_scores[:, start:end+1].mean()
        norm_mean = (mean_val - np.min(feature_means)) / (np.max(feature_means) - np.min(feature_means) + 1e-6)
        ax.bar(
            x=group_center,
            height=0.15,
            width=end - start + 1,
            bottom=len(sorted_scores) + 1.5,
            color=omics_colors[omics],
            edgecolor='black',
            linewidth=1,
            alpha=min(1.0, 0.3 + 0.7 * norm_mean)
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 🔹 Optional: Cluster-wise contributions
    plot_bio_clusterwise_feature_contributions(
        args=args,
        relevance_scores=relevance_scores,
        row_labels=row_labels,
        feature_names=feature_names,
        per_cluster_feature_contributions_output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions_bio"),
        omics_colors=omics_colors
    )


def save_and_plot_enriched_pathways(enrichment_results, args, output_dir):
    # === Prepare Data ===
    heatmap_data = pd.DataFrame()

    for cluster_type in ['bio', 'topo']:
        for cid, df in enrichment_results[cluster_type].items():
            colname = f"{cluster_type.capitalize()}_{cid}"
            vals = {}
            for _, row in df.iterrows():
                p = row['p_value']
                name = row['name']
                if p < 0.05 and len(name) <= 60:
                    term = f"{name} ({row['source']})"
                    vals[term] = -np.log10(p)
            heatmap_data[colname] = pd.Series(vals)

    # Clean and filter
    heatmap_data = heatmap_data.fillna(0)
    heatmap_data = heatmap_data[heatmap_data.max(axis=1) > 1]

    # Save full enrichment data to CSV
    enrichment_csv_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_enrichment_matrix_epo{args.num_epochs}.csv"
    )
    heatmap_data.to_csv(enrichment_csv_path, index_label='Enriched Pathway')

    # === Save Topo Cluster → Top Enriched Terms to CSV ===
    topo_terms = []
    for cid, df in enrichment_results['topo'].items():
        for _, row in df.iterrows():
            if row['p_value'] < 0.05 and len(row['name']) <= 60:
                topo_terms.append({
                    "Cluster": f"Topo_{cid}",
                    "Term": row['name'],
                    "Source": row['source'],
                    "p_value": row['p_value'],
                    "-log10(p)": -np.log10(row['p_value']),
                })
    topo_terms_df = pd.DataFrame(topo_terms)
    topo_terms_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_topo_cluster_top_terms_epo{args.num_epochs}.csv"
    )
    topo_terms_df.to_csv(topo_terms_path, index=False)

    # === Save Bio Cluster → Top Enriched Terms to CSV ===
    bio_terms = []
    for cid, df in enrichment_results['bio'].items():
        for _, row in df.iterrows():
            if row['p_value'] < 0.05 and len(row['name']) <= 60:
                bio_terms.append({
                    "Cluster": f"Bio_{cid}",
                    "Term": row['name'],
                    "Source": row['source'],
                    "p_value": row['p_value'],
                    "-log10(p)": -np.log10(row['p_value']),
                })
    bio_terms_df = pd.DataFrame(bio_terms)
    bio_terms_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_bio_cluster_top_terms_epo{args.num_epochs}.csv"
    )
    bio_terms_df.to_csv(bio_terms_path, index=False)

    # === Select 50 evenly spaced rows for plotting ===
    if heatmap_data.shape[0] > 50:
        step = max(1, heatmap_data.shape[0] // 50)
        selected_indices = heatmap_data.index[::step][:50]
        heatmap_data = heatmap_data.loc[selected_indices]

    # === Normalize for color contrast ===
    norm_data = heatmap_data.copy()
    norm_data = norm_data / norm_data.max().replace(0, 1)

    # === Apply group-wise colormaps ===
    colormaps = {
        'bio': get_cmap('Blues'),
        'topo': get_cmap('YlOrRd'),
    }

    colors = np.zeros((heatmap_data.shape[0], heatmap_data.shape[1], 4))  # RGBA
    col_types = []

    for i, col in enumerate(norm_data.columns):
        group = 'bio' if col.lower().startswith("bio") else 'topo'
        col_types.append(group)
        cmap = colormaps[group]
        colors[:, i, :] = cmap(norm_data[col].values)

    # === Plot ===
    fig, ax = plt.subplots(figsize=(0.5 * len(norm_data.columns), 0.2 * len(norm_data)))

    ax.imshow(colors, aspect='auto')
    ax.set_xticks(np.arange(len(norm_data.columns)))
    ax.set_xticklabels(norm_data.columns, rotation=90, fontsize=12)
    ax.set_yticks(np.arange(len(norm_data.index)))
    ax.set_yticklabels(norm_data.index, fontsize=12)
    ax.set_ylabel("Enriched Pathway", fontsize=18, labelpad=20)

    # Color x-axis labels
    for xtick, col in zip(ax.get_xticklabels(), col_types):
        xtick.set_color('darkblue' if col == 'bio' else 'darkred')

    ax.set_title("Top Enriched Pathways per Cluster", fontsize=14, pad=16)
    ax.set_xlabel("Cluster", fontsize=14)

    # legend_patches = [
    #     Patch(color='cornflowerblue', label='Bio'),
    #     Patch(color='salmon', label='Topo')
    # ]
    # fig.legend(handles=legend_patches, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))

    sns.despine(ax=ax, trim=True)
    ax.tick_params(axis='both', which='both', length=0)
    plt.tight_layout(rect=[0, 0, 0.95, 0.93])

    # Save plot
    enriched_terms_heatmap_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_enriched_terms_heatmap_epo{args.num_epochs}.png"
    )
    plt.savefig(enriched_terms_heatmap_path, dpi=300)
    plt.show()

    # === Return DataFrames for downstream analysis ===
    return heatmap_data, topo_terms_df, bio_terms_df

def plot_gene_feature_contributions_bio(
    gene_name,
    relevance_vector,
    feature_names,
    score,
    cluster_id,
    base_output_dir
):
    assert len(relevance_vector) == 64, "Expected 64 feature contributions (4 omics × 16 cancers)."

    cluster_dir = os.path.join(base_output_dir, f"cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)

    output_path = os.path.join(cluster_dir, f"{gene_name}.png")
    barplot_path = output_path.replace(".png", "_omics_barplot.png")


    # Barplot of all 64 features
    df = pd.DataFrame({'Feature': feature_names, 'Relevance': relevance_vector})
    plot_omics_barplot_bio(df, str(barplot_path))
    
    # Barplot of all 64 features
    # df = pd.DataFrame({'Feature': feature_names, 'Relevance': relevance_vector})
    # barplot_path = output_path.replace(".png", "_omics_barplot.png") if output_path else None
    # plot_omics_barplot_bio(df, barplot_path)

    # Prepare for heatmap
    df[['Omics', 'Cancer']] = df['Feature'].str.split(':', expand=True)
    df['Omics'] = df['Omics'].str.lower()

    heatmap_data = df.pivot(index='Cancer', columns='Omics', values='Relevance')
    heatmap_data = heatmap_data[['cna', 'ge', 'meth', 'mf']]  # Ensure column order

    # Plot vertical heatmap (Cancers as rows)
    plt.figure(figsize=(2.0, 5.0))
    # Capitalize omics column labels
    heatmap_data.columns = [col.upper() for col in heatmap_data.columns]
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, cbar=False, linewidths=0.3, linecolor='gray')

    # Handle gene name and score
    if isinstance(score, np.ndarray):
        score = score.item()
    plt.title(f"{gene_name}", fontsize=12)

    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(rotation=90, ha='center', fontsize=10)
    plt.xlabel('')
    plt.ylabel('')

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_gene_feature_contributions_topo(
    gene_name,
    relevance_vector,
    feature_names,
    score,
    cluster_id,
    base_output_dir
):
    assert len(relevance_vector) == 64, "Expected 64 feature contributions (4 omics × 16 cancers)."

    cluster_dir = os.path.join(base_output_dir, f"cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)

    output_path = os.path.join(cluster_dir, f"{gene_name}.png")
    barplot_path = output_path.replace(".png", "_omics_barplot.png")


    # Barplot of all 64 features
    df = pd.DataFrame({'Feature': feature_names, 'Relevance': relevance_vector})
    plot_omics_barplot_topo(df, str(barplot_path))

    # Prepare for heatmap
    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)
    df['Omics'] = df['Omics'].str.lower()

    heatmap_data = df.pivot(index='Cancer', columns='Omics', values='Relevance')
    heatmap_data = heatmap_data[['cna', 'ge', 'meth', 'mf']]  # Ensure column order

    # Plot vertical heatmap (Cancers as rows)
    plt.figure(figsize=(2.0, 5.0))
    # Capitalize omics column labels
    heatmap_data.columns = [col.upper() for col in heatmap_data.columns]
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, cbar=False, linewidths=0.3, linecolor='gray')

    # Capitalize the first letter of each y-tick (cancer name)
    plt.yticks(
        ticks=plt.yticks()[0], 
        labels=[label.get_text().capitalize() for label in plt.gca().get_yticklabels()],
        rotation=0, fontsize=10
    )

    if isinstance(score, np.ndarray):
        score = score.item()
    plt.title(f"{gene_name}", fontsize=12)

    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(rotation=90, ha='center', fontsize=10)
    plt.xlabel('')
    plt.ylabel('')

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_bio_biclustering_heatmap_unsort(
    args,
    relevance_scores,
    row_labels,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None,
    col_labels=None
):  
    
    # 🔹 Extract and normalize relevance scores
    # relevance_scores = extract_summary_features_np_bio(relevance_scores)
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min()) * 20

    if omics_colors is None:
        omics_colors = {
            'cna': '#9370DB',    # purple
            'ge': '#228B22',     # dark green
            'meth': '#00008B',   # dark blue
            'mf': '#b22222',     # dark red
        }

    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'KidneyPap',
        'Liver', 'LungAd', 'LungSc', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    omics_order = ['cna', 'ge', 'meth', 'mf']
    feature_names = [f"{omics.upper()}: {cancer}" for omics in omics_order for cancer in cancer_names]

    # If col_labels provided, reorder columns accordingly
    # if col_labels is not None:
    #     sorted_order = np.argsort(col_labels)
    #     relevance_scores = relevance_scores[:, sorted_order]
    #     feature_names = [feature_names[i] for i in sorted_order]

    # Build feature color bar
    feature_colors = []
    for i in range(len(feature_names)):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")  # fallback color

    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50, wspace=0.0, hspace=0.0)

    ax_bar    = fig.add_subplot(gs[0, 2:45])      
    ax        = fig.add_subplot(gs[1:13, 2:45])    
    ax_curve  = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar   = fig.add_subplot(gs[5:9, 49])

    ax_bar.axis("off")
    ax_bar.set_xlim(0, len(feature_names))
    ax_bar.set_ylim(0, 1.1)

    # Normalize per-feature means
    feature_means = relevance_scores.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min() + 1e-6)

    for i, (mean_val, color) in enumerate(zip(feature_means, feature_colors)):
        ax_bar.bar(
            x=i + 0.5,
            height=mean_val,
            width=1.0,
            bottom=0,
            color=color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.3 + 0.7 * mean_val
        )

    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient",
        ["#F0F3F4", "#85929e"]
    )

    vmin = 0
    vmax = np.percentile(relevance_scores, 99)

    sns.heatmap(
        relevance_scores,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Relevance Score",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )

    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=18)
    ax_cbar.yaxis.label.set_size(18)

    # Sort by cluster only (no intra-cluster sorting)
    ##cluster_order = np.argsort(row_labels)
    cluster_order = np.argsort(row_labels)
    sorted_scores = relevance_scores[cluster_order]
    sorted_clusters = row_labels[cluster_order]

    # Cluster stripe
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle((-1.5, i), 1.5, 1, linewidth=0, facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')), clip_on=False))

    # Cluster size text
    unique_clusters, cluster_sizes = np.unique(sorted_clusters, return_counts=True)
    start_idx = 0
    for cluster, size in zip(unique_clusters, cluster_sizes):
        center_y = start_idx + size / 2
        ax.text(-2.0, center_y, f"{size}", va='center', ha='right', fontsize=18, fontweight='bold')
        start_idx += size

    # Add xtick labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels([c.split(": ")[1] for c in feature_names], rotation=90, fontsize=14)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # LRP curve
    saliency_sums = relevance_scores.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())
    y = np.arange(len(saliency_sums))
    ax_curve.fill_betweenx(
        y, 0, saliency_sums,
        color='#a9cce3',
        alpha=0.8,
        linewidth=3
    )

    # Omics bar below
    omics_means = {}
    for omics, (start, end) in omics_splits.items():
        group_scores = relevance_scores[:, start:end+1]
        omics_means[omics] = group_scores.mean()

    group_centers = {
        omics: (omics_splits[omics][0] + omics_splits[omics][1]) / 2 + 0.5
        for omics in omics_order
    }

    mean_vals = np.array([omics_means[om] for om in omics_order])
    min_mean, max_mean = mean_vals.min(), mean_vals.max()
    normalized_means = (mean_vals - min_mean) / (max_mean - min_mean + 1e-6)

    for i, omics in enumerate(omics_order):
        ax.bar(
            x=group_centers[omics],
            height=0.15,
            width=(omics_splits[omics][1] - omics_splits[omics][0] + 1),
            bottom=len(relevance_scores) + 1.5,
            color=omics_colors[omics],
            edgecolor='black',
            linewidth=1,
            alpha=0.3 + 0.7 * normalized_means[i]
        )

    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.hlines(
        y=1.01, xmin=0, xmax=1,
        color='black', linewidth=1.5, transform=ax_curve.get_xaxis_transform()
    )
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')
    ax_curve.set_ylim(0, len(saliency_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.set_yticks([])
    ax_curve.set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_bio_biclustering_heatmap_clusters_unsort(
    args,
    relevance_scores,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None,
    row_labels=None,
    col_labels=None,
):
    #relevance_scores = extract_summary_features_np_bio(relevance_scores)

    # Normalize to [0, 20]
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min()) * 10

    if omics_colors is None:
        omics_colors = {
            'cna': '#9370DB',  # purple
            'ge': '#228B22',   # green
            'meth': '#00008B', # blue
            'mf': '#b22222',   # red
        }

    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'KidneyPap',
        'Liver', 'LungAd', 'LungSc', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    omics_order = ['cna', 'ge', 'meth', 'mf']
    feature_names = [f"{omics.upper()}: {cancer}" for omics in omics_order for cancer in cancer_names]

    # Sort rows by cluster labels only
    ##cluster_order = np.argsort(row_labels)
    cluster_order = np.argsort(row_labels)
    sorted_scores = relevance_scores[cluster_order]
    sorted_clusters = row_labels[cluster_order]

    original_col_indices = list(range(relevance_scores.shape[1]))
    sorted_scores = sorted_scores[:, original_col_indices]
    feature_names = [feature_names[i] for i in original_col_indices]

    feature_colors = []
    for omics in omics_order:
        start, end = omics_splits[omics]
        feature_colors.extend([omics_colors[omics]] * (end - start + 1))

    # Colormap
    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient", 
        ["#F0F3F4", "#85929e"])
    vmin, vmax = 0, np.percentile(sorted_scores, 99)

    # Grid layout
    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50, wspace=0.0, hspace=0.0)

    ax_bar    = fig.add_subplot(gs[0, 2:45])       # top bar
    ax        = fig.add_subplot(gs[1:13, 2:45])     # main heatmap
    ax_curve  = fig.add_subplot(gs[1:13, 45:48], sharey=ax)  # saliency curve
    ax_cbar   = fig.add_subplot(gs[5:9, 49])       # colorbar

    # Top feature bar
    feature_means = sorted_scores.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min() + 1e-6)
    
    ax_bar.axis("off")
    ax_bar.set_xlim(0, len(feature_names))
    ax_bar.set_ylim(0, 1.1)
    
    for i, (val, color) in enumerate(zip(feature_means, feature_colors)):
        ax_bar.bar(
            x=i + 0.5, 
            height=val, 
            width=1.0,
            color=color, 
            edgecolor='black', 
            linewidth=0.5, 
            alpha=0.3 + 0.7 * val
        )

    # Heatmap
    sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={"label": "Relevance Score", "shrink": 0.1, "aspect": 12, "pad": 0.02, "orientation": "vertical", "location": "right"},
        ax=ax
    )
    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=18)
    ax_cbar.yaxis.label.set_size(18)

    # Cluster stripes
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle((-1.5, i), 1.5, 1, linewidth=0, facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')), clip_on=False))

    # Cluster size labels
    unique_clusters, cluster_sizes = np.unique(sorted_clusters, return_counts=True)
    start_idx = 0
    for cluster, size in zip(unique_clusters, cluster_sizes):
        center_y = start_idx + size / 2
        ax.text(-2.0, center_y, f"{size}", va='center', ha='right', fontsize=18, fontweight='bold')
        start_idx += size

    # X-axis labels with omics colors
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels([f.split(": ")[1] for f in feature_names], rotation=90, fontsize=14)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, length=5)
    
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)


    # Saliency curve 
    saliency_sums = sorted_scores.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())
    ax_curve.fill_betweenx(
        np.arange(len(saliency_sums)), 
        0, 
        saliency_sums, 
        color='#a9cce3', 
        alpha=0.8, 
        linewidth=3)
    
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.set_ylim(0, len(saliency_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.hlines(
        y=1.01, xmin=0, xmax=1, 
        color='black', linewidth=1.5, 
        transform=ax_curve.get_xaxis_transform())
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')

    # Omics group bars (right below feature bar, above heatmap)
    for omics in omics_order:
        start, end = omics_splits[omics]
        group_center = (start + end) / 2 + 0.5
        mean_val = sorted_scores[:, start:end+1].mean()
        norm_mean = (mean_val - np.min(feature_means)) / (np.max(feature_means) - np.min(feature_means) + 1e-6)
        ax.bar(
            x=group_center,
            height=0.15,
            width=end - start + 1,
            bottom=len(sorted_scores) + 1.5,
            color=omics_colors[omics],
            edgecolor='black',
            linewidth=1,
            alpha=min(1.0, 0.3 + 0.7 * norm_mean)
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_and_plot_confirmed_genes_bio(
    args,
    node_names_topk,
    node_scores_topk,
    summary_feature_relevance,
    output_dir,
    confirmed_genes_save_path,
    row_labels_topk,
    tag="bio",
    confirmed_gene_path="../acgnn/data/ncg_8886.txt"):
    """
    Finds confirmed cancer genes and plots their biological feature contributions.
    """

    
    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'KidneyCC', 'KidneyPC',
        'Liver', 'LungAD', 'LungSC', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]

    omics_order = ['cna', 'ge', 'meth', 'mf']
    feature_names = [f"{omics}:{cancer}" for omics in omics_order for cancer in cancer_names]

    with open(confirmed_gene_path) as f:
        known_cancer_genes = set(line.strip() for line in f if line.strip())

    confirmed_genes = [g for g in node_names_topk if g in known_cancer_genes]

    with open(confirmed_genes_save_path, "w") as f:
        for gene in confirmed_genes:
            f.write(f"{gene}\n")

    plot_dir = os.path.join(output_dir, f"{tag}_confirmed_feature_contributions")
    os.makedirs(plot_dir, exist_ok=True)

    def get_scalar_score(score):
        if isinstance(score, np.ndarray):
            return score.item() if score.size == 1 else score[0]
        return float(score)

    # for gene_name in confirmed_genes:
    #     idx = node_names_topk.index(gene_name)
    #     relevance_vector = summary_feature_relevance[idx]
    #     score = get_scalar_score(node_scores_topk[idx])
    #     cluster_id = row_labels_topk[idx].item()

    #     output_path = os.path.join(
    #         "results/gene_prediction/bio_confirmed_feature_contributions/",
    #         f"{args.model_type}_{args.net_type}_{gene}_bio_confirmed_feature_contributions_epo{args.num_epochs}.png"
    #     )

    #     plot_gene_feature_contributions_bio(
    #         gene_name=gene,
    #         relevance_vector=relevance_vector,
    #         feature_names=feature_names,
    #         score=score,
    #         cluster_id=cluster_id,
    #         output_path=output_path
    #     )
    for gene_name in confirmed_genes:
        idx = node_names_topk.index(gene_name)
        relevance_vector = summary_feature_relevance[idx]
        score = get_scalar_score(node_scores_topk[idx])
        cluster_id = row_labels_topk[idx].item()

        plot_gene_feature_contributions_bio(
            gene_name=gene_name,
            relevance_vector=relevance_vector,
            feature_names=feature_names,
            score=score,
            cluster_id=cluster_id,
            base_output_dir=os.path.join(
                "results/gene_prediction/bio_confirmed_feature_contributions",
                f"{args.model_type}_{args.net_type}_epo{args.num_epochs}"
            )
        )

def save_and_plot_novel_genes_bio(
    args,
    node_names_topk,
    node_scores_topk,
    summary_feature_relevance,
    output_dir,
    novel_genes_save_path,
    row_labels_topk,
    tag="bio",
    confirmed_gene_path="../acgnn/data/ncg_8886.txt"
):
    """
    Finds novel predicted cancer genes (not in known list) and plots their biological feature contributions.
    """

    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'KidneyCC', 'KidneyPC',
        'Liver', 'LungAD', 'LungSC', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    omics_order = ['cna', 'ge', 'meth', 'mf']
    feature_names = [f"{omics}:{cancer}" for omics in omics_order for cancer in cancer_names]

    # Load confirmed genes
    with open(confirmed_gene_path) as f:
        known_cancer_genes = set(line.strip() for line in f if line.strip())

    # Filter for novel genes
    novel_predicted_genes = [g for g in node_names_topk if g not in known_cancer_genes]

    # Save novel genes
    with open(novel_genes_save_path, "w") as f:
        for gene in novel_predicted_genes:
            f.write(f"{gene}\n")

    # Plot directory for NPCGs
    plot_dir = os.path.join(output_dir, f"{tag}_novel_feature_contributions")
    os.makedirs(plot_dir, exist_ok=True)

    def get_scalar_score(score):
        if isinstance(score, np.ndarray):
            return score.item() if score.size == 1 else score[0]
        return float(score)

    for gene_name in novel_predicted_genes:
        idx = node_names_topk.index(gene_name)
        relevance_vector = summary_feature_relevance[idx]
        score = get_scalar_score(node_scores_topk[idx])
        cluster_id = row_labels_topk[idx].item()

        plot_gene_feature_contributions_bio(
            gene_name=gene_name,
            relevance_vector=relevance_vector,
            feature_names=feature_names,
            score=score,
            cluster_id=cluster_id,
            base_output_dir=os.path.join(
                "results/gene_prediction/bio_novel_feature_contributions",
                f"{args.model_type}_{args.net_type}_epo{args.num_epochs}"
            )
        )

def plot_topo_biclustering_heatmap(
    args,
    relevance_scores,
    row_labels,
    output_path,
    gene_names=None,
    col_labels=None
    ):
    

    """
    Plots a spectral biclustering heatmap for topological embeddings (1024–2047),
    with within-cluster gene sorting and column sorting by global relevance.

    Args:
        args: CLI or config object with settings.
        relevance_scores (np.ndarray): shape [num_nodes, 2048], full embedding.
        row_labels (np.ndarray): shape [num_nodes], integer cluster assignments.
        output_path (str): Path to save the figure.
        gene_names (list of str, optional): Gene name labels for heatmap index.

    Returns:
        pd.DataFrame: heatmap matrix with genes as rows and topo features as columns.
    """

    # 🔹 Extract 64D summary of topological features
    #relevance_scores = extract_summary_features_np_topo(relevance_scores)
    # Normalize features-----------------------------------------------------------------------------------------------------------------
    # relevance_scores = StandardScaler().fit_transform(relevance_scores)*10
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min())
    
    # 🔹 Create topo feature names (01–64)
    feature_names = [f"{i+1:02d}" for i in range(relevance_scores.shape[1])]

    # 🔹 Sort columns (features) by total relevance across all genes
    col_sums = relevance_scores.sum(axis=0)
    col_order = np.argsort(-col_sums)
    relevance_scores = relevance_scores[:, col_order]
    feature_names = [feature_names[i] for i in col_order]
    if col_labels is not None:
        col_labels = np.array(col_labels)[col_order]

    # 🔹 Sort by cluster → then by gene-wise relevance within cluster
    ordered_row_indices = []
    row_labels = np.array(row_labels)
    unique_clusters = np.unique(row_labels)

    for cluster in unique_clusters:
        cluster_idx = np.where(row_labels == cluster)[0]
        cluster_scores = relevance_scores[cluster_idx]
        cluster_gene_sums = cluster_scores.sum(axis=1)
        sorted_cluster = cluster_idx[np.argsort(-cluster_gene_sums)]
        ordered_row_indices.extend(sorted_cluster)

    sorted_scores = relevance_scores[ordered_row_indices]
    sorted_clusters = row_labels[ordered_row_indices]
    if gene_names is not None:
        gene_names = [gene_names[i] for i in ordered_row_indices]

    # 🔹 Compute cluster boundaries and centers
    _, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Apply log transformation to enhance low-intensity features
    sorted_scores = np.log1p(sorted_scores)  # This will emphasize smaller values

    # 🔹 Normalize scores (optional but improves contrast)
    #sorted_scores = (sorted_scores - sorted_scores.min()) / (sorted_scores.max() - sorted_scores.min())
    
    # 🔹 Set colormap
    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient", ["#F0F3F4", "#85929e"]
    )

    
    # Construct row labels: use gene names if available, else fallback to generic
    row_gene_names = [gene_names[i] if gene_names is not None else f"Gene_{i}" for i in ordered_row_indices]

    # Build DataFrame for ordered scores
    df_ordered_scores = pd.DataFrame(sorted_scores, index=row_gene_names, columns=feature_names)

    # Add cluster information
    df_ordered_scores.insert(0, "Cluster", sorted_clusters)
    
    base_dir = os.path.dirname(output_path)
    cluster_output_dir = os.path.join(base_dir, "cluster_csvs")
    os.makedirs(cluster_output_dir, exist_ok=True)

    cluster_output_dir = output_path.replace(".png", "_cluster_csvs")
    os.makedirs(cluster_output_dir, exist_ok=True)

    for cluster_id in np.unique(sorted_clusters):
        cluster_df = df_ordered_scores[df_ordered_scores["Cluster"] == cluster_id]
        cluster_csv_path = os.path.join(cluster_output_dir, f"cluster_{cluster_id}_genes.csv")
        cluster_df.to_csv(cluster_csv_path)
        print(f"Saved cluster {cluster_id} gene scores to {cluster_csv_path}")

    # Save to CSV
    csv_output_path = output_path.replace(".png", "_ordered_scores.csv")
    df_ordered_scores.to_csv(csv_output_path)
    print(f"Ordered relevance score matrix with cluster info saved to {csv_output_path}")


    # 🔹 Setup figure layout
    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50, wspace=0.0, hspace=0.0)


    ax_bar = fig.add_subplot(gs[0, 2:45])
    ax = fig.add_subplot(gs[1:13, 2:45])
    ax_curve = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[5:9, 49])
    #ax_legend = fig.add_subplot(gs[14, 2:45])

    # 🔹 Compute dynamic vmax
    vmin = np.percentile(sorted_scores, 5)
    vmax = np.percentile(sorted_scores, 99)


    feature_means = sorted_scores.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min()) * 0.04

    ax_bar.bar(
        np.arange(len(feature_means)) + 0.5, 
        feature_means,
        width=1.0,
        color="#B0BEC5",
        linewidth=0,
        alpha=0.6
    )

    ax_bar.set_xticks([0, len(feature_means)])
    ax_bar.set_xticklabels(['0', '1'], fontsize=16)
    ax_bar.tick_params(axis='x', direction='out', pad=1)
        
    ax_bar.set_xlim(0, len(feature_means))  # align with heatmap width
    ax_bar.set_ylim(0, 0.04)
    ax_bar.set_yticks([])
    ax_bar.set_yticklabels([])
    ax_bar.tick_params(axis='y', length=0)  # removes tick marks
    ax_bar.set_xticks([])


    for spine in ['left', 'bottom', 'top', 'right']:
        ax_bar.spines[spine].set_visible(False)
    '''for spine in ['left', 'bottom']:
        ax_bar.spines[spine].set_visible(True)
        ax_bar.spines[spine].set_linewidth(1.0)
        ax_bar.spines[spine].set_color("black")'''


    # 🔹 Apply log transformation to enhance low-intensity features
    sorted_scores = np.log1p(sorted_scores)  # This will emphasize smaller values

    # 🔹 Normalize scores (optional but improves contrast)
    #sorted_scores = (sorted_scores - sorted_scores.min()) / (sorted_scores.max() - sorted_scores.min())

    # 🔹 Compute vmin and vmax dynamically
    vmin = 0#np.percentile(sorted_scores, 1)   # Stretch the color range from low values
    vmax = np.percentile(sorted_scores, 99)  # Cap extreme values

    # 🔹 Choose a perceptually clear colormap
    #colormap = "mako"  # or try "viridis", "plasma", "rocket", etc.

    # 🔹 Plot heatmap with new settings
    sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=True,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Log-Scaled Relevance",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )

    # 🔹 Plot heatmap
    '''sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=15,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Relevance Score",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )'''
    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=16)
    ax_cbar.yaxis.label.set_size(18)

    # 🔹 Add cluster color stripes
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # 🔹 Cluster size labels
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=18, fontweight='bold'
        )

    # 🔹 X-tick labels below heatmap
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=90, fontsize=16)
    ax.tick_params(axis='x', bottom=True, labelbottom=True)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Omics + LRP Legend
    '''ax_legend.axis("off")
    lrp_patch = Patch(facecolor='#a9cce3', alpha=0.8, label='Saliency Sum')
    ax_legend.legend(
        handles=[lrp_patch],
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=16,
        handleheight=1.5,
        handlelength=3
    )'''

    # 🔹 Saliency Sum curve
    saliency_sums = sorted_scores.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())
    y = np.arange(len(saliency_sums))

    ax_curve.fill_betweenx(
        y, 0, saliency_sums,
        color='#a9cce3',
        alpha=0.8,
        linewidth=3
    )

    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.hlines(
        y=1.05, xmin=0, xmax=1,
        color='black', linewidth=1.5, transform=ax_curve.get_xaxis_transform()
    )
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')
    ax_curve.set_ylim(0, len(saliency_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    # 🔹 Final layout + save
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

    # 🔹 Cluster-wise contribution breakdown
    plot_topo_clusterwise_feature_contributions(
        args=args,
        relevance_scores=relevance_scores,  # Not sorted for per-cluster breakdown
        row_labels=row_labels,
        feature_names=[f"{i+1:02d}" for i in range(relevance_scores.shape[1])],
        per_cluster_feature_contributions_output_dir=os.path.join(
            os.path.dirname(output_path), "per_cluster_feature_contributions_topo"
        )
    )

    return pd.DataFrame(sorted_scores, index=gene_names, columns=feature_names)

def plot_topo_biclustering_heatmap_unsorted(
    args,
    relevance_scores,
    row_labels,
    output_path,
    gene_names=None,
    col_labels=None
):
    """
    Unsorted topo heatmap with connected cluster bar and gene count labels.
    """
    from matplotlib.patches import Rectangle

    # 🔹 Normalize summary topo features (0-1)
    #relevance_scores = extract_summary_features_np_topo(relevance_scores)
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min())

    # 🔹 Generate feature names (01 to 64)
    feature_names = [f"{i+1:02d}" for i in range(relevance_scores.shape[1])]
    row_labels = np.array(row_labels)

    sorted_indices = np.argsort(row_labels)
    row_labels = row_labels[sorted_indices]
    
    # 🔹 Compute cluster stats
    _, counts = np.unique(row_labels, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Setup figure layout
    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50, wspace=0.0, hspace=0.0)

    ax_bar = fig.add_subplot(gs[0, 2:45])
    ax = fig.add_subplot(gs[1:13, 2:45])
    ax_curve = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[5:9, 49])

    # 🔹 Log-transform scores
    scores = np.log1p(relevance_scores)

    # 🔹 Color map
    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])

    # 🔹 Feature mean bar
    feature_means = scores.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min()) * 0.04
    ax_bar.bar(np.arange(len(feature_means)) + 0.5, feature_means, width=1.0, color="#B0BEC5", alpha=0.6)
    ax_bar.set_xticks([]), ax_bar.set_yticks([]), ax_bar.set_xlim(0, len(feature_means)), ax_bar.set_ylim(0, 0.04)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    # 🔹 Heatmap
    vmin, vmax = 0, np.percentile(scores, 99)
    sns.heatmap(
        scores, cmap=bluish_gray_gradient, vmin=vmin, vmax=vmax,
        xticklabels=False, yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Log-Scaled Relevance",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )

    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=16)
    ax_cbar.yaxis.label.set_size(18)

    # 🔹 Add cluster bar
    for i, cluster in enumerate(row_labels):
        ax.add_patch(Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # 🔹 Add cluster counts
    for cluster_id, center_y, count in zip(np.unique(row_labels), cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=18, fontweight='bold'
        )

    # 🔹 X-axis labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=90, fontsize=16)
    ax.tick_params(axis='x', bottom=True, labelbottom=True)
    ax.set_xlabel(""), ax.set_ylabel(""), ax.set_title("")

    # 🔹 Saliency sum curve
    saliency_sums = scores.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())
    y = np.arange(len(saliency_sums))

    ax_curve.fill_betweenx(y, 0, saliency_sums, color='#a9cce3', alpha=0.8, linewidth=3)
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.hlines(y=1.05, xmin=0, xmax=1,
                    color='black', linewidth=1.5, transform=ax_curve.get_xaxis_transform())
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')
    ax_curve.set_ylim(0, len(saliency_sums))
    for spine in ax_curve.spines.values():
        spine.set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    # 🔹 Save plot
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(rect=[0.02, 0.03, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved unsorted topo clustering heatmap to {output_path}")

    return pd.DataFrame(scores, index=gene_names, columns=feature_names)


def save_and_plot_confirmed_genes_topo(
    args,
    node_names_topk,
    node_scores_topk,
    summary_feature_relevance,
    output_dir,
    confirmed_genes_save_path,
    row_labels_topk,
    tag="topo",
    confirmed_gene_path="../acgnn/data/ncg_8886.txt"):
    """
    Finds confirmed cancer genes and plots their topological feature contributions.
    """

    cancer_names = [
        'BLADDER', 'BREAST', 'CERVIX', 'COLON', 'ESOPHAGUS', 'HEADNECK', 'KIDNEYCC', 'KIDNEYPC',
        'LIVER', 'LUNGAD', 'LUNGSC', 'PROSTATE', 'RECTUM', 'STOMACH', 'THYROID', 'UTERUS'
    ]
    omics_order = ['cna', 'ge', 'meth', 'mf']
    feature_names = [f"{cancer}_{omics}" for cancer in cancer_names for omics in omics_order]

    with open(confirmed_gene_path) as f:
        known_cancer_genes = set(line.strip() for line in f if line.strip())

    confirmed_genes = [g for g in node_names_topk if g in known_cancer_genes]

    with open(confirmed_genes_save_path, "w") as f:
        for gene in confirmed_genes:
            f.write(f"{gene}\n")

    ##summary_feature_relevance = extract_summary_features_np_topo(summary_feature_relevance)

    plot_dir = os.path.join(output_dir, f"{tag}_confirmed_feature_contributions")
    os.makedirs(plot_dir, exist_ok=True)

    def get_scalar_score(score):
        if isinstance(score, np.ndarray):
            return score.item() if score.size == 1 else score[0]
        return float(score)

    for gene_name in confirmed_genes:
        idx = node_names_topk.index(gene_name)
        relevance_vector = summary_feature_relevance[idx]
        score = get_scalar_score(node_scores_topk[idx])
        cluster_id = row_labels_topk[idx].item()

        plot_gene_feature_contributions_topo(
            gene_name=gene_name,
            relevance_vector=relevance_vector,
            feature_names=feature_names,
            score=score,
            cluster_id=cluster_id,
            base_output_dir=os.path.join(
                "results/gene_prediction/topo_confirmed_feature_contributions",
                f"{args.model_type}_{args.net_type}_epo{args.num_epochs}"
            )
        )
    
def plot_model_performance(args):
    """
    Generates and saves a scatter plot comparing AUROC and AUPRC values 
    for different models across multiple networks.

    Parameters:
    - models: List of model names.
    - networks: List of network names.
    - auroc: 2D list of AUROC scores (rows: models, cols: networks).
    - auprc: 2D list of AUPRC scores (rows: models, cols: networks).
    - args: Arguments containing model and training configuration.
    - output_dir: Directory to save the plot.
    """


    # Define models and networks
    models = ["GRAIL", "HGDC", "EMOGI", "MTGCN", "GCN", "GAT", "GraphSAGE", "GIN", "Chebnet"]
    networks = ["CPDB", "STRING", "HIPPIE"]

    # AUPRC values for ONGene and OncoKB for each model (rows: models, cols: networks)
    auroc = [
        [0.9652, 0.9578, 0.9297],  # ACGNN ACGNN & 0.9652 & 0.9783 & 0.9578 & 0.9738 & 0.9297 & 0.9597 \\
        [0.6776, 0.7133, 0.6525],  # HGDC
        [0.6735, 0.8184, 0.6672],  # EMOGI
        [0.6862, 0.7130, 0.6762],  # MTGCN
        [0.6915, 0.6688, 0.6708],  # GCN
        [0.6670, 0.8166, 0.6478],  # GAT
        [0.6664, 0.6166, 0.6571],  # GraphSAGE
        [0.5836, 0.5173, 0.5844],  # GIN
        [0.8017, 0.8777, 0.7409]   # Chebnet
    ]

    auprc = [
        [0.9783, 0.9738, 0.9597],  # ACGNN
        [0.7288, 0.7740, 0.7634],  # HGDC
        [0.7230, 0.8737, 0.7960],  # EMOGI
        [0.7712, 0.7878, 0.7785],  # MTGCN
        [0.7730, 0.7681, 0.7675],  # GCN
        [0.7086, 0.8791, 0.7496],  # GAT
        [0.7522, 0.7182, 0.7624],  # GraphSAGE
        [0.6405, 0.5918, 0.6791],  # GIN
        [0.8622, 0.9159, 0.8443]   # Chebnet
    ]

    # Compute averages for each model
    avg_auroc = np.mean(auroc, axis=1)
    avg_auprc = np.mean(auprc, axis=1)

    # Define colors for models and unique shapes for networks
    colors = ['red', 'grey', 'blue', 'green', 'purple', 'orange', 'cyan', 'brown', 'pink']
    network_markers = ['P', '^', 's']  # One shape for each network
    avg_marker = 'o'  # Marker for average points

    # Create the plot
    plt.figure(figsize=(8, 7))

    # Plot individual points for each model and network
    for i, model in enumerate(models):
        for j, network in enumerate(networks):
            plt.scatter(auprc[i][j], auroc[i][j], color=colors[i], 
                        marker=network_markers[j], s=90, alpha=0.6)

    # Add average points for each model
    for i, model in enumerate(models):
        plt.scatter(avg_auprc[i], avg_auroc[i], color=colors[i], marker=avg_marker, 
                    s=240, edgecolor='none', alpha=0.5)

    # Create legends for models (colors) and networks (shapes)
    model_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                            markersize=14, label=models[i], alpha=0.5) for i in range(len(models))]
    network_legend = [Line2D([0], [0], marker=network_markers[i], color='k', linestyle='None', 
                            markersize=8, label=networks[i]) for i in range(len(networks))]

    # Add legends
    network_legend_artist = plt.legend(handles=network_legend, loc='lower right', title="Networks", fontsize=12, title_fontsize=14, frameon=True)
    plt.gca().add_artist(network_legend_artist)
    plt.legend(handles=model_legend, loc='upper left', fontsize=12, frameon=True)

    # Labels and title
    plt.ylabel("AUPRC", fontsize=14)
    plt.xlabel("AUROC", fontsize=14)

    # Save the plot
    comp_output_path = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_comp_plot_epo{args.num_epochs}_infeats{args.in_feats}.jpeg')
    plt.savefig(comp_output_path, bbox_inches='tight')
    
    print(f"Comparison plot saved to {comp_output_path}")

    # Show plot
    plt.tight_layout()
    plt.close()


def plot_confirmed_neighbors_topo(
    args,
    graph,
    node_names,
    name_to_index,
    predicted_cancer_genes, #confirmed_genes,
    # scores,
    row_labels,
    total_clusters,
    relevance_scores):
    # Only top-k names are passed in node_names
    topk_name_to_index = {name: i for i, name in enumerate(node_names)}
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, predicted_cancer_genes)

    for gene in predicted_cancer_genes:
        if gene not in topk_name_to_index:
            print(f"⚠️ Gene {gene} not in top-k node list.")
            continue

        node_idx = topk_name_to_index[gene]
        if node_idx >= relevance_scores.shape[0]:
            print(f"⚠️ Skipping {gene}: index {node_idx} out of bounds for relevance_scores with shape {relevance_scores.shape}")
            continue

        gene_score = relevance_scores[node_idx].sum().item()


        # ✅ Get cluster from graph
        gene_cluster = graph.ndata["cluster_topo"][node_idx].item()
        print(f"{gene} → Node {node_idx} | Topo score: {gene_score:.4f} | Cluster: {gene_cluster}")

        neighbors = neighbors_dict.get(gene, [])
        neighbor_scores_dict = {}

        for n in neighbors:
            if n in topk_name_to_index:
                rel_idx = topk_name_to_index[n]
                if rel_idx < relevance_scores.shape[0]:  # bounds check
                    rel_score = relevance_scores[rel_idx].sum().item()
                    # if rel_score > 0.1:
                    neighbor_scores_dict[rel_idx] = rel_score

        if not neighbor_scores_dict:
            print(f"⚠️ No valid neighbors found for {gene}.")
            continue

        top_neighbors = dict(sorted(neighbor_scores_dict.items(), key=lambda x: -x[1])[:10])

        # plot_path = os.path.join(
        #     "results/gene_prediction/topo_neighbor_feature_contributions/",
        #     f"{args.model_type}_{args.net_type}_{gene}_topo_confirmed_neighbor_relevance_epo{args.num_epochs}.png"
        # )

        # plot_neighbor_relevance(
        #     neighbor_scores=top_neighbors,
        #     gene_name=f"{gene} (Cluster {gene_cluster})",
        #     node_id_to_name=node_id_to_name,
        #     output_path=plot_path,
        #     row_labels=row_labels,
        #     total_clusters=total_clusters,
        #     add_legend=False
        # )
        plot_path = os.path.join(
            "results/gene_prediction/topo_neighbor_feature_contributions/",
            f"{args.model_type}_{args.net_type}_{gene}_topo_confirmed_neighbor_relevance_epo{args.num_epochs}.png"
        )

        plot_neighbor_relevance(
            neighbor_scores=top_neighbors,
            gene_name=f"{gene} (Cluster {gene_cluster})",
            node_id_to_name=node_id_to_name,
            output_path=plot_path,
            row_labels=row_labels,
            total_clusters=total_clusters,
            add_legend=False
        )

def plot_confirmed_neighbor_relevance(
    args,
    graph,
    node_names,
    name_to_index,
    predicted_cancer_genes, #confirmed_genes,
    scores,
    relevance_scores,
    mode="bio"):  # or "topo"):
    """
    Plots top-10 neighbor relevance scores for confirmed genes.
    Mode can be 'bio' or 'topo'.
    """


    assert mode in ("bio", "topo"), "Mode must be 'bio' or 'topo'"

    node_id_to_name = {i: name for i, name in enumerate(node_names)}
    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, predicted_cancer_genes)

    for gene in predicted_cancer_genes:
        if gene not in name_to_index:
            print(f"⚠️ Gene {gene} not found in the graph.")
            continue

        node_idx = name_to_index[gene]
        gene_score = scores[node_idx]
        print(f"{gene} → Node {node_idx} | {mode.capitalize()} score: {gene_score:.4f}")

        neighbors = neighbors_dict.get(gene, [])
        neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]

        # Get relevance scores for neighbors (filtering by score > 0.1)
        neighbor_scores_dict = {
            i: relevance_scores[i].sum().item()
            for i in neighbor_indices
            if relevance_scores[i].sum().item() > 0.1
        }

        if not neighbor_scores_dict:
            print(f"⚠️ No valid neighbors found for {gene}.")
            continue

        # Sort and select top 10
        top_neighbors = dict(sorted(neighbor_scores_dict.items(), key=lambda x: -x[1])[:10])

        # plot_path = os.path.join(
        #     "results/gene_prediction/neighbor_feature_contributions/",
        #     f"{args.model_type}_{args.net_type}_{gene}_{mode}_neighbor_relevance_epo{args.num_epochs}.png"
        # )

        # plot_neighbor_relevance(
        #     neighbor_scores=top_neighbors,
        #     gene_name=f"{gene} (Cluster {gene_cluster})",
        #     node_id_to_name=node_id_to_name,
        #     output_path=plot_path,
        #     row_labels=row_labels,
        #     total_clusters=total_clusters,
        #     add_legend=False
        # )
        plot_path = os.path.join(
            "results/gene_prediction/bio_neighbor_feature_contributions/",
            f"{args.model_type}_{args.net_type}_{gene}_bio_confirmed_neighbor_relevance_epo{args.num_epochs}.png"
        )

        plot_neighbor_relevance(
            neighbor_scores=top_neighbors,
            gene_name=f"{gene} (Cluster {gene_cluster})",
            node_id_to_name=node_id_to_name,
            output_path=plot_path,
            row_labels=row_labels,
            total_clusters=total_clusters,
            add_legend=False
        )

def plot_topo_clusterwise_feature_contributions(
    args,
    relevance_scores,           # 2D array (samples x features)
    row_labels,             # 1D array of cluster assignments
    feature_names,              # List of feature names (e.g., TOPO: BRCA, ...)
    per_cluster_feature_contributions_output_dir):  # Output folder
    ##omics_colors                # Dict of omics type colors (e.g., 'topo': '#1F77B4')):
    os.makedirs(per_cluster_feature_contributions_output_dir, exist_ok=True)

    '''def get_omics_color(feature_name):
        prefix = feature_name.split(":")[0].lower()
        return omics_colors.get(prefix, "#AAAAAA")'''

    unique_clusters = np.unique(row_labels)

    for cluster_id in sorted(unique_clusters):
        indices = np.where(row_labels == cluster_id)[0]
        cluster_scores = relevance_scores[indices]
        avg_contribution = np.mean(cluster_scores, axis=0)
        total_score = np.sum(avg_contribution)

        fig, ax = plt.subplots(figsize=(10, 2.5))

        x = np.linspace(0, 1, len(feature_names))
        bar_width = 1 / len(feature_names) * 0.95

        bars = ax.bar(
            x,
            avg_contribution,
            width=bar_width,
            ##color=[get_omics_color(name) for name in feature_names],
            align='center'
        )

        ax.set_title(
            fr"Cluster {cluster_id} $\mathregular{{({len(indices)}\ genes,\ avg = {total_score:.2f})}}$",
            fontsize=14
        )

        clean_labels = [name.split(":")[1].strip() if ":" in name else name for name in feature_names]
        ax.set_xticks(x)
        ax.set_xticklabels(clean_labels, rotation=90)

        '''for label, feature_name in zip(ax.get_xticklabels(), feature_names):
            label.set_color(get_omics_color(feature_name))'''

        ax.tick_params(axis='x', labelsize=9)
        ax.set_xlim(-bar_width, 1 + bar_width)

        plt.tight_layout()
        save_path = os.path.join(
            per_cluster_feature_contributions_output_dir,
            f"{args.model_type}_{args.net_type}_TOPO_cluster_{cluster_id}_feature_contributions_epo{args.num_epochs}.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved TOPO feature contribution barplot for Cluster {cluster_id} to {save_path}")

def plot_feature_importance_topo(
    relevance_vector,
    feature_names,
    node_name=None,
    output_path="plots"):


    if len(relevance_vector) != len(feature_names):
        raise ValueError("Length mismatch between relevance vector and feature names.")

    ##pretty_labels = [f"emb_{i}" for i in range(len(relevance_vector))]
    pretty_labels = [f"{i}" for i in range(len(relevance_vector))]

    df = pd.DataFrame({
        "feature": pretty_labels,
        "relevance": relevance_vector
    })

    plt.figure(figsize=(24, 5))
    sns.set_style("white")
    bars = plt.bar(df["feature"], df["relevance"], color="#607D8B")  # bluish-gray

    # Title (no mean)
    if node_name:
        plt.title(f"{node_name}", fontsize=16)

    # Axis labels and formatting
    plt.xlabel("Topology Embedding Dimension", fontsize=14)
    plt.ylabel("Relevance", fontsize=14)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.margins(x=0)

    num_bars = len(df)
    margin = 0.75
    ax = plt.gca()
    ax.set_xlim(-margin, num_bars - 1 + margin)

    ##ax.set_xlim(-bar_width, 1 + bar_width) 

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved TOPO feature importance plot to {output_path}")

def plot_bio_topo_saliency(bio_scores, topo_scores, title="", save_path=None):
    """
    Plot saliency relevance for bio and topo features with mean values in a color-patched legend.

    Parameters:
        bio_scores (np.ndarray): Relevance scores for bio features (length 1024).
        topo_scores (np.ndarray): Relevance scores for topo features (length 1024).
        title (str): Plot title.
        save_path (str, optional): Path to save the figure.
    """

    # 🔹 Normalize
    bio_scores_norm = (bio_scores - bio_scores.min()) / (bio_scores.max() - bio_scores.min() + 1e-8)
    topo_scores_norm = (topo_scores - topo_scores.min()) / (topo_scores.max() - topo_scores.min() + 1e-8)
    #x = np.arange(0, 1024)
    x = np.arange(0, 64)

    # 🔹 Means
    mean_bio = bio_scores_norm.mean()
    mean_topo = topo_scores_norm.mean()

    # 🔹 Plot
    plt.figure(figsize=(12, 6))
    plt.fill_between(x, bio_scores_norm, alpha=0.4, color="royalblue")
    plt.fill_between(x, topo_scores_norm, alpha=0.4, color="darkorange")

    # 🔹 Mean lines
    plt.axhline(mean_bio, color="royalblue", linestyle="--", linewidth=1.5)
    plt.axhline(mean_topo, color="darkorange", linestyle="--", linewidth=1.5)

    # 🔹 Custom legend with mean values
    legend_handles = [
        Patch(facecolor='royalblue', label=f'Bio Mean: {mean_bio:.2f}'),
        Patch(facecolor='darkorange', label=f'Topo Mean: {mean_topo:.2f}')
    ]
    plt.legend(handles=legend_handles, loc='upper left', frameon=False, fontsize=10)

    # 🔹 Formatting
    #plt.xlim(0, 1023)
    plt.xlim(0, 63)
    plt.ylim(0, 1)
    plt.xlabel("Feature Index (0 - 63)", fontsize=12)
    plt.ylabel("Normalized Relevance Score", fontsize=12)
    plt.title(title, fontsize=14)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved bio-topo saliency plot to {save_path}")
    else:
        plt.close()

def plot_bio_topo_saliency_cuberoot(bio_scores, topo_scores, title="", save_path=None):
    """
    Plot saliency relevance for bio and topo features using cube root transformed normalized values.
    This enhances low scores and compresses high spikes.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import seaborn as sns

    # 🔹 Normalize to [0, 1]
    bio_scores_norm = (bio_scores - bio_scores.min()) / (bio_scores.max() - bio_scores.min() + 1e-8)
    topo_scores_norm = (topo_scores - topo_scores.min()) / (topo_scores.max() - topo_scores.min() + 1e-8)

    # 🔹 Apply cube root transform
    bio_scores_scaled = np.cbrt(bio_scores_norm)
    topo_scores_scaled = np.cbrt(topo_scores_norm)

    x = np.arange(0, len(bio_scores))

    # 🔹 Means (after transform)
    mean_bio = bio_scores_scaled.mean()
    mean_topo = topo_scores_scaled.mean()

    # 🔹 Plot
    plt.figure(figsize=(12, 6))
    plt.fill_between(x, bio_scores_scaled, alpha=0.4, color="royalblue")
    plt.fill_between(x, topo_scores_scaled, alpha=0.4, color="darkorange")

    # 🔹 Mean lines
    plt.axhline(mean_bio, color="royalblue", linestyle="--", linewidth=1.5)
    plt.axhline(mean_topo, color="darkorange", linestyle="--", linewidth=1.5)

    # 🔹 Legend
    legend_handles = [
        Patch(facecolor='royalblue', label=f'Bio Mean: {mean_bio:.2f}'),
        Patch(facecolor='darkorange', label=f'Topo Mean: {mean_topo:.2f}')
    ]
    plt.legend(handles=legend_handles, loc='upper left', frameon=False, fontsize=10)

    # 🔹 Formatting
    plt.xlim(0, len(bio_scores) - 1)
    plt.ylim(0, 1)
    plt.xlabel("Feature Index (0 – 63)", fontsize=12)
    plt.ylabel("Cube Root Transformed Score", fontsize=12)
    plt.title(title, fontsize=14)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved cube root bio-topo saliency plot to {save_path}")
    else:
        plt.close()

def plot_bio_topo_saliency_log(bio_scores, topo_scores, title="", save_path=None):
    """
    Plot saliency relevance for bio and topo features with log-transformed normalized values.
    Enhances small values and compresses high peaks.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import seaborn as sns

    # 🔹 Normalize
    bio_scores_norm = (bio_scores - bio_scores.min()) / (bio_scores.max() - bio_scores.min() + 1e-8)
    topo_scores_norm = (topo_scores - topo_scores.min()) / (topo_scores.max() - topo_scores.min() + 1e-8)

    # 🔹 Apply log transform
    bio_scores_scaled = np.log1p(bio_scores_norm) / np.log1p(1)  # log1p(x)/log1p(1) keeps range in [0,1]
    topo_scores_scaled = np.log1p(topo_scores_norm) / np.log1p(1)

    x = np.arange(0, len(bio_scores))

    # 🔹 Means (after transform)
    mean_bio = bio_scores_scaled.mean()
    mean_topo = topo_scores_scaled.mean()

    # 🔹 Plot
    plt.figure(figsize=(12, 6))
    plt.fill_between(x, bio_scores_scaled, alpha=0.4, color="royalblue")
    plt.fill_between(x, topo_scores_scaled, alpha=0.4, color="darkorange")

    # 🔹 Mean lines
    plt.axhline(mean_bio, color="royalblue", linestyle="--", linewidth=1.5)
    plt.axhline(mean_topo, color="darkorange", linestyle="--", linewidth=1.5)

    # 🔹 Legend
    legend_handles = [
        Patch(facecolor='royalblue', label=f'Bio Mean: {mean_bio:.2f}'),
        Patch(facecolor='darkorange', label=f'Topo Mean: {mean_topo:.2f}')
    ]
    plt.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=10)

    # 🔹 Formatting
    plt.xlim(0, len(bio_scores) - 1)
    plt.ylim(0, 1)
    plt.xlabel("Feature Index (0–1023)", fontsize=12)
    plt.ylabel("Transformed Relevance Score (log1p)", fontsize=12)
    plt.title(title, fontsize=14)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved log-transformed bio-topo saliency plot to {save_path}")
    else:
        plt.close()

def extract_bio_summary_features_np(features_np):
    num_nodes = features_np.shape[0]
    total_dim = features_np.shape[1]
    summary_features = []

    num_omics = 4
    num_cancers = 16
    features_per_pair = 16

    for o_idx in range(num_omics):
        for c_idx in range(num_cancers):
            base = o_idx * num_cancers * features_per_pair + c_idx * features_per_pair
            if base + features_per_pair > total_dim:
                continue
            group = features_np[:, base:base + features_per_pair]
            max_vals = group.max(axis=1, keepdims=True)
            summary_features.append(max_vals)

    return np.concatenate(summary_features, axis=1)  # ➜ (N, 64)

def extract_topo_summary_features_np(features_np):
    num_nodes = features_np.shape[0]
    total_dim = features_np.shape[1]
    summary_features = []

    num_topo_blocks = 4
    num_cancers = 16
    features_per_block = 16

    for t_idx in range(num_topo_blocks):
        for c_idx in range(num_cancers):
            base = t_idx * num_cancers * features_per_block + c_idx * features_per_block
            if base + features_per_block > total_dim:
                continue
            group = features_np[:, base:base + features_per_block]
            max_vals = group.max(axis=1, keepdims=True)
            summary_features.append(max_vals)

    return np.concatenate(summary_features, axis=1)  # ➜ (N, 64)

def extract_summary_features_np(features_np):
    """
    Extracts summary features by computing the max of 16-dimensional segments across all (omics, cancer) pairs.
    This version only works with the first 1024 biological features.

    Args:
        features_np (np.ndarray): shape [num_nodes, 2048]

    Returns:
        np.ndarray: shape [num_nodes, 64]
    """
    num_nodes, num_features = features_np.shape
    summary_features = []

    assert num_features == 2048, f"Expected 2048 features, got {num_features}"

    # First 1024 for biological features (omics and cancer)
    for o_idx in range(4):  # 4 omics types
        for c_idx in range(16):  # 16 cancer types
            base = o_idx * 16 * 16 + c_idx * 16
            group = features_np[:, base:base + 16]  # [num_nodes, 16]
            max_vals = group.max(axis=1, keepdims=True)  # [num_nodes, 1]
            summary_features.append(max_vals)

    return np.concatenate(summary_features, axis=1)  # shape: [num_nodes, 64]

def plot_neighbor_relevance_by_mode(
    gene,
    relevance_scores,
    mode,
    neighbor_scores,
    neighbors_dict,
    name_to_index,
    node_id_to_name,
    graph,
    row_labels,
    total_clusters,
    args,
    save_dir="results/gene_prediction/neighbor_feature_contributions/"):
    os.makedirs(save_dir, exist_ok=True)

    if gene not in name_to_index:
        print(f"⚠️ Gene {gene} not found in the graph.")
        return

    node_idx = name_to_index[gene]
    gene_score = neighbor_scores[node_idx]

    # ✅ Get gene cluster based on mode
    if mode == "bio":
        gene_cluster = graph.ndata["cluster_bio"][node_idx].item()
    elif mode == "topo":
        gene_cluster = graph.ndata["cluster_topo"][node_idx].item()
    else:
        gene_cluster = -1  # fallback if cluster type is unknown

    print(f"[{mode}] {gene} → Node {node_idx} | Predicted score: {gene_score:.4f} | Cluster: {gene_cluster}")

    neighbors = neighbors_dict.get(gene, [])
    neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]

    relevance_vals = [relevance_scores[i].sum().item() for i in neighbor_indices]
    scores_dict = dict(zip(neighbor_indices, relevance_vals))

    output_path = os.path.join(
        save_dir,
        f"{args.model_type}_{args.net_type}_{gene}_{mode}_neighbor_relevance_epo{args.num_epochs}.png"
    )

    plot_neighbor_relevance(
        neighbor_scores=scores_dict,
        gene_name=f"{gene} (Cluster {gene_cluster})",
        node_id_to_name=node_id_to_name,
        output_path=output_path,
        row_labels=row_labels,
        total_clusters=total_clusters,
        add_legend=False
    )

def plot_saliency_for_gene(
    gene,
    relevance_scores,
    node_idx,
    save_dir,
    args,
    bio_feat_names,
    topo_feat_names):
    bio_1024 = relevance_scores[node_idx]["bio"].cpu().numpy().reshape(1, -1)
    topo_1024 = relevance_scores[node_idx]["topo"].cpu().numpy().reshape(1, -1)

    bio_64 = extract_summary_features_np_skip(bio_1024).squeeze()
    topo_64 = extract_summary_features_np_skip(topo_1024).squeeze()

    # BIO plot
    plot_feature_importance_bio(
        relevance_vector=bio_64,
        feature_names=bio_feat_names,
        node_name=gene,
        output_path=os.path.join(
            save_dir,
            f"{args.model_type}_{args.net_type}_{gene}_bio_feature_importance_epo{args.num_epochs}.png"
        )
    )

    # TOPO plot
    plot_feature_importance_topo(
        relevance_vector=topo_64,
        feature_names=topo_feat_names,
        node_name=gene,
        output_path=os.path.join(
            save_dir,
            f"{args.model_type}_{args.net_type}_{gene}_topo_feature_importance_epo{args.num_epochs}.png"
        )
    )

def plot_topo_biclustering_heatmap_clusters_unsort(
    args,
    relevance_scores,
    row_labels,
    output_path,
    gene_names=None,
    col_labels=None
):
    #relevance_scores = extract_summary_features_np_topo(relevance_scores)

    # Normalize to [0, 20]
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min())# * 10

    # if topo_colors is None:
    #     topo_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] * 16  # Adjust to match 64 features

    # 🔹 Generate feature names (01 to 64)
    feature_names = [f"{i+1:02d}" for i in range(relevance_scores.shape[1])]
    row_labels = np.array(row_labels)
    
    # Sort rows by cluster labels only
    cluster_order = np.argsort(row_labels)
    sorted_scores = relevance_scores[cluster_order]
    sorted_clusters = row_labels[cluster_order]

    #feature_colors = topo_colors[:len(feature_names)]

    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
    vmin, vmax = 0, np.percentile(sorted_scores, 99)

    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50, wspace=0.0, hspace=0.0)
    
    ax_bar    = fig.add_subplot(gs[0, 2:45])
    ax        = fig.add_subplot(gs[1:13, 2:45])
    ax_curve  = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar   = fig.add_subplot(gs[5:9, 49])

    # 🔹 Log-transform scores
    scores = np.log1p(relevance_scores)

    # 🔹 Color map
    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])

    # 🔹 Feature mean bar
    feature_means = scores.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min()) * 0.04
    ax_bar.bar(np.arange(len(feature_means)) + 0.5, feature_means, width=1.0, color="#B0BEC5", alpha=0.6)
    ax_bar.set_xticks([]), ax_bar.set_yticks([]), ax_bar.set_xlim(0, len(feature_means)), ax_bar.set_ylim(0, 0.04)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    # 🔹 Heatmap
    vmin, vmax = 0, np.percentile(scores, 99)
    sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={"label": "Relevance Score", "shrink": 0.1, "aspect": 12, "pad": 0.02, "orientation": "vertical", "location": "right"},
        ax=ax
    )
    
    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=18)
    ax_cbar.yaxis.label.set_size(18)

    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1, 
            linewidth=0, 
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')), 
            clip_on=False
        ))

    unique_clusters, cluster_sizes = np.unique(sorted_clusters, return_counts=True)
    start_idx = 0
    for cluster, size in zip(unique_clusters, cluster_sizes):
        center_y = start_idx + size / 2
        ax.text(-2.0, center_y, f"{size}", va='center', ha='right', fontsize=18, fontweight='bold')
        start_idx += size
    
    # 🔹 X-axis labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=90, fontsize=16)
    ax.tick_params(axis='x', bottom=True, labelbottom=True)
    ax.set_xlabel(""), ax.set_ylabel(""), ax.set_title("")

    # 🔹 Saliency Sum Curve
    saliency_sums = sorted_scores.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())
    y = np.arange(len(saliency_sums))

    ax_curve.fill_betweenx(
        y, 
        
        0, 
        saliency_sums,
        color='#a9cce3', 
        alpha=0.8, 
        linewidth=3)
    
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.set_ylim(0, len(saliency_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.hlines(
        y=1.01, xmin=0, xmax=1, 
        color='black', linewidth=1.5, 
        transform=ax_curve.get_xaxis_transform())
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')

    # 🔹 Save figure
    plt.subplots_adjust(wspace=0, hspace=0)
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved unsorted topo clustering heatmap to {output_path}")

# def plot_neighbor_relevance(
#     neighbor_scores,
#     gene_name,
#     node_id_to_name,
#     output_path,
#     row_labels=None,
#     total_clusters=10,
#     add_legend=False
# ):
#     """
#     Plots the relevance scores of top neighbors and saves into a cluster-specific folder,
#     inferred from row_labels.
#     Always plots 10 bars, padding with zero-height bars if necessary.
#     Filters out the gene itself from being plotted as its own neighbor (by name).
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import matplotlib.patches as mpatches
#     from pathlib import Path

#     # Filter out self-links by comparing names
#     filtered = {
#         k: v for k, v in neighbor_scores.items()
#         if v > 0.0 and node_id_to_name.get(k, "") != gene_name
#     }

#     # Keep top 10
#     top_neighbors = dict(sorted(filtered.items(), key=lambda x: -x[1])[:10])

#     # Pad with dummy neighbors if fewer than 10
#     while len(top_neighbors) < 10:
#         dummy_id = f"dummy_{len(top_neighbors)}"
#         top_neighbors[dummy_id] = 0.0

#     neighbor_ids = list(top_neighbors.keys())
#     neighbor_names = [node_id_to_name.get(nid, f"{nid}") for nid in neighbor_ids]
#     raw_scores = list(top_neighbors.values())

#     # Normalize scores (skip if all are zero)
#     if np.max(raw_scores) - np.min(raw_scores) > 1e-8:
#         norm_scores = (np.array(raw_scores) - np.min(raw_scores)) / (np.max(raw_scores) - np.min(raw_scores) + 1e-8)
#         norm_scores = norm_scores * 0.95 + 0.025
#     else:
#         norm_scores = np.zeros_like(raw_scores)

#     # Assign cluster colors
#     colors = []
#     cluster_ids = []
#     for nid in neighbor_ids:
#         if row_labels is not None and not str(nid).startswith("dummy_"):
#             try:
#                 nid_int = int(nid)
#                 if nid_int < len(row_labels):
#                     cid = int(row_labels[nid_int])
#                     cluster_ids.append(cid)
#                     colors.append(CLUSTER_COLORS.get(cid % total_clusters, "gray"))
#                 else:
#                     print(f"⚠️ nid {nid_int} is out of bounds (row_labels size: {len(row_labels)}).")
#                     colors.append("gray")
#             except Exception as e:
#                 print(f"⚠️ Skipping nid={nid} due to error: {e}")
#                 colors.append("gray")
#         else:
#             colors.append("white")

#     # Inject cluster subfolder into output_path
#     output_path = Path(output_path)
#     if row_labels is not None:
#         real_neighbors = [nid for nid in neighbor_ids if not str(nid).startswith("dummy_")]
#         if real_neighbors:
#             try:
#                 # Try extracting cluster from gene_name string, if it's in format: "GENE (Cluster X)"
#                 import re
#                 match = re.search(r'\(Cluster (\d+)\)', gene_name)
#                 if match:
#                     main_cluster_id = int(match.group(1))
#                 else:
#                     main_cluster_id = 0

#             except Exception:
#                 main_cluster_id = 0
#         else:
#             main_cluster_id = 0

#         cluster_subdir = output_path.parent / f"cluster_{main_cluster_id}"
#         output_path = cluster_subdir / output_path.name
#         cluster_subdir.mkdir(parents=True, exist_ok=True)
#     else:
#         output_path.parent.mkdir(parents=True, exist_ok=True)

#     # Plot
#     plt.figure(figsize=(2.2, 2.2))
#     sns.set_style("white")
#     ax = sns.barplot(x=neighbor_names, y=norm_scores, palette=colors)

#     plt.title(f"{gene_name}", fontsize=12)
#     plt.ylabel("Relevance score", fontsize=10)
#     plt.xticks(rotation=90, fontsize=8)
#     plt.yticks(fontsize=8)

#     sns.despine(left=False, bottom=False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_linewidth(0.8)
#     ax.spines['bottom'].set_linewidth(0.8)
#     ax.tick_params(axis='x', length=0)
#     ax.tick_params(axis='y', length=0)

#     if add_legend and row_labels is not None:
#         unique_clusters = sorted(set(cluster_ids))
#         legend_handles = [
#             mpatches.Patch(color=CLUSTER_COLORS.get(cid % total_clusters, "gray"), label=f"Cluster {cid}")
#             for cid in unique_clusters
#         ]
#         plt.legend(handles=legend_handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
#     else:
#         plt.legend().remove()

#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.show()
#     print(f"✅ Saved neighbor relevance plot to {output_path}")

def plot_neighbor_relevance(
    neighbor_scores,
    gene_name,
    node_id_to_name,
    output_path,
    row_labels=None,
    total_clusters=10,
    add_legend=False
):
    """
    Plots the relevance scores of top neighbors and saves into a cluster-specific folder,
    inferred from row_labels.

    Always plots 10 bars, padding with zero-height bars if necessary.
    Filters out the gene itself from being plotted as its own neighbor (by name).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as mpatches
    from pathlib import Path

    # Filter out self-links by comparing names
    filtered = {
        k: v for k, v in neighbor_scores.items()
        if v > 0.0 and node_id_to_name.get(k, "") != gene_name
    }

    # Keep top 10
    top_neighbors = dict(
        sorted(filtered.items(), key=lambda x: -x[1])[:10]
    )

    # Pad with dummy neighbors if fewer than 10
    while len(top_neighbors) < 10:
        dummy_id = f"dummy_{len(top_neighbors)}"
        top_neighbors[dummy_id] = 0.0

    neighbor_ids = list(top_neighbors.keys())
    neighbor_names = [
        node_id_to_name.get(nid, f"{nid}")
        for nid in neighbor_ids
    ]
    raw_scores = list(top_neighbors.values())

    # Normalize scores (skip if all are zero)
    if np.max(raw_scores) - np.min(raw_scores) > 1e-8:
        norm_scores = (
            (np.array(raw_scores) - np.min(raw_scores))
            / (np.max(raw_scores) - np.min(raw_scores) + 1e-8)
        )

        norm_scores = norm_scores * 0.95 + 0.025

    else:
        norm_scores = np.zeros_like(raw_scores)

    # Assign cluster colors
    colors = []
    cluster_ids = []

    for nid in neighbor_ids:

        if row_labels is not None and not str(nid).startswith("dummy_"):

            try:
                nid_int = int(nid)

                if nid_int < len(row_labels):

                    cid = int(row_labels[nid_int])

                    cluster_ids.append(cid)

                    colors.append(
                        CLUSTER_COLORS.get(
                            cid % total_clusters,
                            "gray"
                        )
                    )

                else:

                    print(
                        f"⚠️ nid {nid_int} is out of bounds "
                        f"(row_labels size: {len(row_labels)})."
                    )

                    colors.append("gray")

            except Exception as e:

                print(
                    f"⚠️ Skipping nid={nid} due to error: {e}"
                )

                colors.append("gray")

        else:

            colors.append("white")

    # Inject cluster subfolder into output_path
    output_path = Path(output_path)

    if row_labels is not None:

        real_neighbors = [
            nid for nid in neighbor_ids
            if not str(nid).startswith("dummy_")
        ]

        if real_neighbors:

            try:

                # Try extracting cluster from gene_name string,
                # if it is in format: "GENE (Cluster X)"
                import re

                match = re.search(
                    r'\(Cluster (\d+)\)',
                    gene_name
                )

                if match:

                    main_cluster_id = int(match.group(1))

                else:

                    main_cluster_id = 0

            except Exception:

                main_cluster_id = 0

        else:

            main_cluster_id = 0

        cluster_subdir = (
            output_path.parent
            / f"cluster_{main_cluster_id}"
        )

        output_path = (
            cluster_subdir
            / output_path.name
        )

        cluster_subdir.mkdir(
            parents=True,
            exist_ok=True
        )

    else:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    # Plot
    plt.figure(figsize=(2.2, 2.2))

    sns.set_style("white")

    # Seaborn >= 0.14:
    # assign x to hue and disable the automatic legend
    ax = sns.barplot(
        x=neighbor_names,
        y=norm_scores,
        hue=neighbor_names,
        palette=colors,
        legend=False
    )

    plt.title(
        f"{gene_name}",
        fontsize=12
    )

    plt.ylabel(
        "Relevance score",
        fontsize=10
    )

    plt.xticks(
        rotation=90,
        fontsize=8
    )

    plt.yticks(
        fontsize=8
    )

    sns.despine(
        left=False,
        bottom=False
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    ax.tick_params(
        axis='x',
        length=0
    )

    ax.tick_params(
        axis='y',
        length=0
    )

    if add_legend and row_labels is not None:

        unique_clusters = sorted(
            set(cluster_ids)
        )

        legend_handles = [
            mpatches.Patch(
                color=CLUSTER_COLORS.get(
                    cid % total_clusters,
                    "gray"
                ),
                label=f"Cluster {cid}"
            )
            for cid in unique_clusters
        ]

        plt.legend(
            handles=legend_handles,
            title="Clusters",
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

    else:

        legend = ax.get_legend()

        if legend is not None:
            legend.remove()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"✅ Saved neighbor relevance plot to {output_path}"
    )


def perform_survival_analysis_cluster_assigned(
    survival_path: str,
    expression_matrix_path: str,
    cluster_labels: np.ndarray,
    top_gene_names: list,
    output_dir: str,
    min_samples_per_cluster=5,
    cancer_type="PAN",
):
    print("🔍 Running cluster-assigned survival analysis...")

    os.makedirs(output_dir, exist_ok=True)

    # Load survival data
    survival_df = pd.read_csv(survival_path, sep='\t')
    survival_df['sample'] = survival_df['sample'].str.strip().str[:15]
    survival_df = survival_df.dropna(subset=['OS', 'OS.time'])

    # Load expression
    expr_df = pd.read_csv(expression_matrix_path, sep='\t', index_col=0)
    expr_df.index = expr_df.index.str.upper()
    expr_df.columns = [col.strip().upper()[:15] for col in expr_df.columns]

    top_gene_names = [g.upper() for g in top_gene_names]
    expr_df = expr_df.loc[expr_df.index.intersection(top_gene_names)]

    if expr_df.empty:
        print("❌ No top-k genes found in expression matrix.")
        return

    # Map genes to clusters
    gene_to_cluster = {gene: cluster_labels[i] for i, gene in enumerate(top_gene_names)}
    expr_df['cluster'] = expr_df.index.map(gene_to_cluster)

    # Average per cluster
    cluster_expr_df = pd.DataFrame(index=expr_df.columns)
    for cluster_id in sorted(np.unique(cluster_labels)):
        genes = expr_df[expr_df['cluster'] == cluster_id].drop(columns='cluster').index
        if len(genes) == 0:
            continue
        cluster_expr_df[f'cluster_{cluster_id}'] = expr_df.loc[genes].mean(axis=0)

    # Assign each patient to cluster with max score
    assigned_clusters = (
        cluster_expr_df.idxmax(axis=1).str.extract(r'cluster_(\d+)')[0].astype(int)
    )
    cluster_expr_df['assigned_cluster'] = assigned_clusters

    # Merge with survival
    merged_df = survival_df.merge(cluster_expr_df, left_on='sample', right_index=True, how='inner')
    if merged_df.empty:
        print("❌ No overlap between survival and expression.")
        return

    T = merged_df['OS.time']
    E = merged_df['OS']
    groups = merged_df['assigned_cluster']

    fig, ax = plt.subplots(figsize=(10, 8))
    kmf = KaplanMeierFitter()
    ax.tick_params(axis='both', which='major', labelsize=20)

    handles = []
    # colors = plt.cm.tab10.colors
    curve_width = 2

    # KM per cluster
    unique_clusters = sorted(groups.unique())
    for idx, cluster_id in enumerate(unique_clusters):
        mask = (groups == cluster_id)
        if mask.sum() < min_samples_per_cluster:
            print(f"⚠️ Cluster {cluster_id} skipped: too few samples ({mask.sum()}).")
            continue

        color = CLUSTER_COLORS.get(cluster_id, "#333333")

        kmf.fit(T[mask], E[mask], label=f"Cluster {cluster_id}")
        kmf.plot(
            ax=ax,
            ci_show=False,
            show_censors=True,
            color=color,
            linewidth=curve_width,
            censor_styles={'marker': '|', 'ms': 8, 'mew': 1}
        )

        line_handle = Line2D(
            [], [],
            color=color,
            linestyle='-',
            lw=curve_width,
            marker='|',
            markersize=6,
            markeredgewidth=1,
            markerfacecolor='none',
            label=f"Cluster {cluster_id} (n={mask.sum()})"
        )
        handles.append(line_handle)

    # Logrank test
    result = multivariate_logrank_test(T, groups=groups, event_observed=E)
    pval = result.p_value

    # Cox PH
    cox_df = merged_df[['OS.time', 'OS', 'assigned_cluster']].copy()
    cox_df.columns = ['time', 'event', 'cluster']
    cox_df = cox_df.dropna()
    cox_df['cluster'] = cox_df['cluster'].astype('category')

    valid_clusters = []
    for cl in cox_df['cluster'].cat.categories:
        if cox_df.loc[cox_df['cluster'] == cl, 'event'].sum() > 0:
            valid_clusters.append(cl)
        else:
            print(f"⚠️ Cluster {cl} has no events. Dropping from Cox PH.")

    cox_df = cox_df[cox_df['cluster'].isin(valid_clusters)].copy()
    cox_df['cluster'] = cox_df['cluster'].astype('category')

    if len(valid_clusters) <= 1:
        print("❌ Not enough valid clusters for Cox PH.")
        representative_hr = np.nan
        representative_ci = np.nan
        cph_summary = None
    else:
        cph = CoxPHFitter()
        try:
            cph.fit(cox_df, duration_col='time', event_col='event')
            hr_values = cph.summary['exp(coef)'].values
            representative_hr = np.mean(hr_values)
            ci_widths = cph.summary['coef upper 95%'] - cph.summary['coef lower 95%']
            representative_ci = ci_widths.mean()
            cph.summary.to_csv(os.path.join(output_dir, f"cox_summary_{cancer_type}.csv"))
            cph_summary = cph.summary
        except Exception as e:
            print(f"❌ Cox PH fitting failed: {e}")
            representative_hr = np.nan
            representative_ci = np.nan
            cph_summary = None

    # Annotations
    ax.set_xlabel("Time (Days)", fontsize=28)
    ax.set_ylabel("Survival probability", fontsize=28)
    ax.set_title(f"{cancer_type} Survival (Cluster-Assigned)", fontsize=28)
    ax.tick_params(axis='both', labelsize=28)
    ax.grid(False)

    text_x = 0.03
    text_y_start = 0.15
    text_y_step = 0.085

    ax.text(
        text_x, text_y_start + text_y_step,
        f"CI = {representative_ci:.3f}" if not np.isnan(representative_ci) else "CI = N/A",
        transform=ax.transAxes, fontsize=28
    )
    ax.text(
        text_x, text_y_start,
        f"HR = {representative_hr:.2f}" if not np.isnan(representative_hr) else "HR = N/A",
        transform=ax.transAxes, fontsize=28
    )
    ax.text(
        text_x, text_y_start - text_y_step,
        f"p-value = {pval:.4g}" if pval >= 0.0001 else f"p-value = {pval:.1e}",
        transform=ax.transAxes, fontsize=28
    )

    ax.legend(handles=handles, fontsize=28, loc='upper right', frameon=False, handlelength=1.0)

    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"km_cluster_assigned_{cancer_type}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✅ Cluster-assigned survival plot saved: {save_path}")

    return cph_summary, merged_df

def survival_analysis_by_cluster_assigned(
    survival_path,
    expr_path,
    node_names_topk,
    row_labels,
    output_dir,
    cancer_type="PAN",
    min_samples_per_cluster=5
):
    os.makedirs(output_dir, exist_ok=True)

    survival_df = pd.read_csv(survival_path, sep="\t")
    expr_df = pd.read_csv(expr_path, sep="\t", index_col=0)

    survival_df['_PATIENT'] = survival_df['_PATIENT'].str.upper()
    expr_df.index = expr_df.index.str.upper()
    expr_df.columns = expr_df.columns.str.upper().str[:12]

    cluster_to_genes = defaultdict(list)
    for gene, label in zip(node_names_topk, row_labels):
        cluster_to_genes[label].append(gene.upper())

    patient_cluster_scores = pd.DataFrame(index=expr_df.columns)
    for cl, genes in cluster_to_genes.items():
        valid_genes = list(set(genes) & set(expr_df.index))
        if not valid_genes:
            print(f"\u26a0\ufe0f No valid genes for cluster {cl}. Skipping.")
            continue
        patient_cluster_scores[f'cluster {cl}'] = expr_df.loc[valid_genes].mean(axis=0)

    cluster_assignments = (
        patient_cluster_scores.idxmax(axis=1).str.extract(r'cluster (\d+)')[0].astype(int)
    )
    patient_cluster_scores["cluster_id"] = cluster_assignments

    merged = survival_df.merge(patient_cluster_scores, left_on="_PATIENT", right_index=True)
    if merged.empty:
        print("\u274c No overlap between survival and expression.")
        return None, None, None, None, cancer_type

    T = merged["OS.time"]
    E = merged["OS"]
    groups = merged["cluster_id"]

    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    unique_clusters = sorted(groups.unique())

    handles = []
    for cl in unique_clusters:
        mask = (groups == cl)
        if mask.sum() < min_samples_per_cluster:
            print(f"\u26a0\ufe0f Cluster {cl} skipped: too few patients.")
            continue

        kmf.fit(T[mask], E[mask], label=f"Cluster {cl}")
        kmf.plot(
            ax=ax,
            ci_show=False,
            show_censors=True,
            censor_styles={'marker': '|', 'ms': 8, 'mew': 1},
            color=CLUSTER_COLORS.get(cl, "#000000"),
            linewidth=2
        )

        curve_handle = Line2D(
            [], [],
            color=CLUSTER_COLORS.get(cl, "#000000"),
            linestyle='-',
            lw=2,
            marker='|',
            markersize=6,
            markeredgewidth=1,
            label=f"Cluster {cl} (n={mask.sum()})"
        )
        handles.append(curve_handle)

    result = multivariate_logrank_test(T, groups=groups, event_observed=E)
    pval = result.p_value

    cox_df = merged[["OS.time", "OS", "cluster_id"]].copy()
    cox_df.columns = ["time", "event", "cluster"]
    cox_df = cox_df.dropna()

    if cox_df["cluster"].nunique() <= 1:
        print("\u274c Not enough clusters for Cox regression.")
        return result.summary, merged, None, None, cancer_type

    cox_df["cluster"] = cox_df["cluster"].astype("category")
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="time", event_col="event")

    hr_values = cph.summary["exp(coef)"].values
    representative_hr = np.mean(hr_values) if len(hr_values) else np.nan

    ci_widths = cph.summary["coef upper 95%"] - cph.summary["coef lower 95%"]
    representative_ci = ci_widths.mean() if len(ci_widths) else np.nan

    ax.set_title(f"GRAIL", fontsize=28)
    ax.set_xlabel("Time (Days)", fontsize=28)
    ax.set_ylabel("Survival probability", fontsize=28)
    ax.tick_params(axis='both', labelsize=28)
    ax.grid(False)

    text_x = 0.03
    text_y_start = 0.15
    text_y_step = 0.085

    ax.text(
        text_x, text_y_start + text_y_step,
        f"CI = {representative_ci:.3f}" if not np.isnan(representative_ci) else "CI = N/A",
        transform=ax.transAxes, fontsize=28
    )
    ax.text(
        text_x, text_y_start,
        f"HR = {representative_hr:.2f}" if not np.isnan(representative_hr) else "HR = N/A",
        transform=ax.transAxes, fontsize=28
    )
    ax.text(
        text_x, text_y_start - text_y_step,
        f"p-value = {pval:.4g}" if pval >= 0.0001 else f"p-value = {pval:.1e}",
        transform=ax.transAxes, fontsize=28
    )

    ax.legend(handles=handles, loc='upper right', fontsize=28, frameon=False, handlelength=1.0)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"survival_assigned_{cancer_type}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\u2705 KM plot saved: {plot_path}")

    cox_summary_path = os.path.join(output_dir, f"cox_summary_{cancer_type}.csv")
    cph.summary.to_csv(cox_summary_path)

    # return result.summary, merged, cox_summary_path, cph.summary, cancer_type
    return result.summary, merged, cox_summary_path, cph.summary, cluster_to_genes


def plot_cox_forest_with_pvalues_styled(
    cox_summary,
    output_dir,
    title="Cox PH Forest Plot",
    # title="Cox PH Forest Plot",
    cancer_type="PAN"
):
    # Use white background without grid
    sns.set_style("white")

    df = cox_summary.copy().reset_index().rename(columns={'index': 'covariate'})
    df = df[['covariate', 'exp(coef)', 'coef lower 95%', 'coef upper 95%', 'p']]
    df = df.sort_values('exp(coef)', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, len(df) * 0.8 + 2))
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    for spine in ['top', 'right']:#, 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
        
    for i, row in df.iterrows():
        hr = row['exp(coef)']
        lower = row['coef lower 95%']
        upper = row['coef upper 95%']
        pval = row['p']

        lower_error = max(hr - lower, 0)
        upper_error = max(upper - hr, 0)

        color = 'red' if hr > 1 else 'blue'

        ax.errorbar(
            hr, i,
            xerr=[[lower_error], [upper_error]],
            fmt='o',
            color='black',
            ecolor=color,
            elinewidth=3,
            capsize=7
        )

        # P-value closer to dot
        p_text = f"p = {pval:.1e}" if pval < 0.0001 else f"p = {pval:.3g}"

        ax.text(
            hr, i + 0.15,  # tight offset
            p_text,
            ha='center',
            va='bottom',
            fontsize=26,
            color='black',
            # weight='bold'
        )


    # Reference line at HR = 1
    ax.axvline(1, color='red', linestyle='--', linewidth=2)

    # Y labels
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['covariate'], fontsize=26)

    # Axis labels
    ax.set_xlabel("Hazard Ratio (HR)", fontsize=28)
    ax.set_title(title, fontsize=32, pad=20)

    ax.tick_params(axis='x', labelsize=26)
    ax.grid(False)

    plot_path = os.path.join(output_dir, f"cox_forest_plot_styled_{cancer_type}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"✅ Saved styled forest plot with large fonts: {plot_path}")

def plot_sankey_gene_cluster_pathway(
    cluster_dict: dict,
    enrichment_terms: dict,
    cancer_type: str,
    output_path: str,
    top_n: int = 10
):
    """
    Plots a Sankey diagram showing gene → cluster → enriched pathway relationships.

    Args:
        cluster_dict: dict of cluster_id → list of genes.
        enrichment_terms: dict of cluster_id → list of enriched pathway names.
        cancer_type: cancer type label for the plot.
        output_path: path to save the plot (HTML or image).
        top_n: maximum number of enriched terms to show per cluster.
    """

    nodes = []
    links = {"source": [], "target": [], "value": []}
    node_to_id = {}
    current_id = 0

    # Step 1: Add gene → cluster links
    for cluster_id, gene_list in cluster_dict.items():
        cluster_label = f"Cluster {cluster_id}"

        # Add cluster node
        if cluster_label not in node_to_id:
            node_to_id[cluster_label] = current_id
            nodes.append(cluster_label)
            current_id += 1

        for gene in gene_list:
            if gene not in node_to_id:
                node_to_id[gene] = current_id
                nodes.append(gene)
                current_id += 1

            # gene → cluster
            links["source"].append(node_to_id[gene])
            links["target"].append(node_to_id[cluster_label])
            links["value"].append(1)

    # Step 2: Add cluster → pathway links
    for cluster_id, term_list in enrichment_terms.items():
        cluster_label = f"Cluster {cluster_id}"
        if cluster_label not in node_to_id:
            continue

        for term in term_list[:top_n]:
            if term not in node_to_id:
                node_to_id[term] = current_id
                nodes.append(term)
                current_id += 1

            # cluster → pathway
            links["source"].append(node_to_id[cluster_label])
            links["target"].append(node_to_id[term])
            links["value"].append(1)

    # Plot
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="lightblue"
        ),
        link=dict(
            source=links["source"],
            target=links["target"],
            value=links["value"]
        )
    )])

    fig.update_layout(title_text=f"{cancer_type} Gene–Cluster–Pathway Mapping", font_size=14)
    fig.write_html(output_path)
    print(f"✅ Sankey plot saved to {output_path}")

def plot_sankey_after_survival(
    cluster_to_genes,
    cancer_type,
    tag,
    output_dir,
    enrichment_source="REAC",
    top_n=20
):
    """
    Generates a Cluster → Gene → Pathway Sankey plot after survival analysis.

    Args:
        cluster_to_genes: dict[int → list of genes]
        cancer_type: str, e.g., "PAN"
        tag: str, identifier used in enrichment files (e.g., "bio")
        output_dir: str, where to save the Sankey plot
        enrichment_source: str, one of ["REAC", "KEGG", "GO:BP", "HP"]
        top_n: int, how many top enriched pathways to include
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load enrichment terms
    enrichment_terms = collect_enrichment_with_ratios(
        cancer_type=cancer_type,
        tag=tag,
        source=enrichment_source,
        top_n=top_n
    )

    # Build mapping: cluster → pathways
    cluster_to_pathways = defaultdict(list)
    for term in enrichment_terms:
        cluster = int(term['cluster'])
        pathway = term['term_name']
        cluster_to_pathways[cluster].append(pathway)

    # Prepare node labels
    all_clusters = sorted(cluster_to_genes.keys())
    all_genes = list({gene for genes in cluster_to_genes.values() for gene in genes})
    all_pathways = list({p for paths in cluster_to_pathways.values() for p in paths})

    cluster_labels = [f"Cluster {c}" for c in all_clusters]
    node_labels = cluster_labels + all_genes + all_pathways
    node_indices = {label: i for i, label in enumerate(node_labels)}

    # Assign colors
    node_colors = []
    for label in node_labels:
        if label.startswith("Cluster"):
            cl = int(label.split()[-1])
            node_colors.append(CLUSTER_COLORS.get(cl, "#999999"))
        else:
            node_colors.append("#DDDDDD")  # Light grey for genes/pathways

    sources, targets, values = [], [], []

    # Cluster → Gene
    for c in all_clusters:
        cl_label = f"Cluster {c}"
        for gene in cluster_to_genes[c]:
            sources.append(node_indices[cl_label])
            targets.append(node_indices[gene])
            values.append(1)

    # Gene → Pathway (via cluster-based pathways)
    for c in all_clusters:
        if c not in cluster_to_pathways:
            continue
        for gene in cluster_to_genes[c]:
            for pathway in cluster_to_pathways[c]:
                sources.append(node_indices[gene])
                targets.append(node_indices[pathway])
                values.append(1)

    # Sankey
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.4),
            label=node_labels,
            color=node_colors
        ),
        link=dict(source=sources, target=targets, value=values)
    )])

    fig.update_layout(
        title_text=f"Sankey: Cluster–Gene–Pathway Mapping ({cancer_type}, {enrichment_source})",
        font_size=12,
        margin=dict(t=40, l=20, r=20, b=20)
    )

    sankey_path = os.path.join(output_dir, f"sankey_{cancer_type}_{enrichment_source}.png")
    fig.write_image(sankey_path, width=1400, height=900, scale=2)
    print(f"✅ Sankey plot saved: {sankey_path}")

def shorten_pathway(term_name, max_len=30):
    return term_name if len(term_name) <= max_len else term_name[:max_len] + "…"

def plot_sankey_after_survival(
    cluster_to_genes,
    cancer_type,
    tag,
    output_dir,
    enrichment_source="REAC",
    top_n=20
):
    """
    Plots Sankey diagram showing Cluster–Gene–Pathway mapping after survival analysis.
    """
    print(f"🧬 Generating Sankey plot for {cancer_type} ({enrichment_source})...")

    enrichment_path = f"results/enrichment/{cancer_type}_{tag}/{enrichment_source}.json"
    if not os.path.exists(enrichment_path):
        print(f"❌ Enrichment file not found: {enrichment_path}")
        return

    # Load enrichment results
    with open(enrichment_path, 'r') as f:
        enrichment_terms = json.load(f)

    # Build mapping: cluster → genes → pathways
    cluster_to_pathways = defaultdict(set)
    for cl, genes in cluster_to_genes.items():
        for term in enrichment_terms:
            if set(genes) & set(term.get("intersect_genes", [])):
                cluster_to_pathways[cl].add(term["name"])

    # Build Sankey components
    node_labels = []
    node_colors = []
    node_index = {}
    links = {"source": [], "target": [], "value": [], "color": []}
    node_id = 0

    # CLUSTER_COLORS should be globally defined or passed in
    for cl, genes in cluster_to_genes.items():
        cluster_name = f"Cluster {cl}"
        if cluster_name not in node_index:
            node_index[cluster_name] = node_id
            node_labels.append(cluster_name)
            node_colors.append(CLUSTER_COLORS.get(cl, "#999999"))
            node_id += 1

        for gene in genes:
            if gene not in node_index:
                node_index[gene] = node_id
                node_labels.append(gene)
                node_colors.append("#BBBBBB")
                node_id += 1

            links["source"].append(node_index[cluster_name])
            links["target"].append(node_index[gene])
            links["value"].append(1)
            links["color"].append("rgba(100,100,100,0.4)")

    for cl, pathways in cluster_to_pathways.items():
        for pathway in list(pathways)[:top_n]:
            if pathway not in node_index:
                node_index[pathway] = node_id
                node_labels.append(pathway)
                node_colors.append("black")
                node_id += 1

            for gene in cluster_to_genes[cl]:
                if gene in node_index:
                    links["source"].append(node_index[gene])
                    links["target"].append(node_index[pathway])
                    links["value"].append(1)
                    links["color"].append("rgba(160,160,160,0.3)")

    # Build the Sankey plot
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=links["source"],
            target=links["target"],
            value=links["value"],
            color=links["color"]
        )
    )])

    fig.update_layout(
        title_text=f"Sankey: Cluster–Gene–Pathway Mapping ({cancer_type}, {enrichment_source})",
        font_size=10
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"sankey_cluster_gene_pathway_{cancer_type}_{enrichment_source}.html")
    fig.write_html(output_path)
    print(f"✅ Sankey saved to: {output_path}")

def plot_sankey_cluster_gene_pathway(
    cluster_to_genes,
    enrichment_terms,
    cancer_type,
    tag,
    enrichment_source,
    output_dir,
    top_n=20,
    top_k_genes=10,
    truncate_pathway_name=40
):
    import plotly.graph_objects as go
    from collections import defaultdict
    import os

    # Optional: color maps
    CLUSTER_COLORS = {
        0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728", 4: "#9467bd",
        5: "#8c564b", 6: "#e377c2", 7: "#7f7f7f", 8: "#bcbd22", 9: "#17becf"
    }
    PATHWAY_SOURCE_COLORS = {
        "REAC": "#1f77b4", "KEGG": "#ff7f0e", "GO:BP": "#2ca02c", "HP": "#d62728"
    }

    cluster_labels = {i: f"Cluster {i}" for i in cluster_to_genes}

    # Step 1: Build gene to pathway mapping
    gene_to_pathways = defaultdict(list)
    pathway_sources = {}
    pathway_scores = {}
    pathway_name_map = {}
    for term in enrichment_terms:
        name = term["name"]
        source = term.get("source", "UNKNOWN")
        score = term.get("score", 1.0)
        short_name = name[:truncate_pathway_name] + "..." if len(name) > truncate_pathway_name else name
        pathway_name_map[name] = short_name
        pathway_sources[short_name] = source
        pathway_scores[short_name] = score
        intersect_genes = term.get("intersect_genes", [])
        if not isinstance(intersect_genes, list):
            continue
        for g in intersect_genes:
            gene_to_pathways[g].append(short_name)

    # Step 2: Build full node list
    cluster_nodes = list(cluster_labels.values())
    gene_nodes = sorted({gene for genes in cluster_to_genes.values() for gene in genes})
    pathway_nodes = sorted({p for g in gene_nodes for p in gene_to_pathways.get(g, [])})
    all_nodes = cluster_nodes + gene_nodes + pathway_nodes
    node_idx = {name: i for i, name in enumerate(all_nodes)}

    # Step 3: Sankey links
    source_indices, target_indices, values, link_colors = [], [], [], []

    for cl, genes in cluster_to_genes.items():
        cluster_label = cluster_labels[cl]
        for gene in genes[:top_k_genes]:
            if gene not in node_idx:
                continue
            source_indices.append(node_idx[cluster_label])
            target_indices.append(node_idx[gene])
            values.append(1)
            link_colors.append(CLUSTER_COLORS.get(cl, "#999999"))

    for gene in gene_nodes:
        for pathway in gene_to_pathways.get(gene, []):
            if pathway not in node_idx:
                continue
            source_indices.append(node_idx[gene])
            target_indices.append(node_idx[pathway])
            values.append(pathway_scores.get(pathway, 1.0))
            link_colors.append("lightgray")

    node_colors = (
        ["#cccccc"] * len(cluster_nodes) +
        ["#dddddd"] * len(gene_nodes) +
        [PATHWAY_SOURCE_COLORS.get(pathway_sources.get(p, "UNKNOWN"), "#eeeeee") for p in pathway_nodes]
    )

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors,
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors
        )
    )])

    fig.update_layout(
        title_text=f"Sankey: Cluster–Gene–Pathway Mapping ({cancer_type}, {enrichment_source})",
        font_size=16,
        title_font_size=22
    )

    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"sankey_{tag}_{cancer_type}_{enrichment_source}.html")
    png_path = html_path.replace(".html", ".png")
    fig.write_html(html_path)

    try:
        fig.write_image(png_path)
        print(f"✅ Sankey saved to: {html_path} and {png_path}")
    except Exception as e:
        print(f"⚠️ Could not save PNG: {e}\nSaved HTML to: {html_path}")

def plot_sankey_cluster_gene_pathway_limited(
    cluster_to_genes,
    enrichment_terms,
    cancer_type,
    tag,
    enrichment_source,
    output_dir,
    top_n_genes_per_cluster=10,
    max_pathway_name_len=30,
    export_png=True
):
    # Step 1: Trim and group
    cluster_labels = {i: f"Cluster {i}" for i in cluster_to_genes}
    gene_to_pathways = defaultdict(list)
    pathway_colors = {}
    
    for term in enrichment_terms:
        intersect_genes = term.get("intersect_genes", [])
        if not isinstance(intersect_genes, list):
            continue
        source = term.get("source", "Other")
        color = {
            "REAC": "#FFD700",  # gold
            "KEGG": "#FF7F0E",  # orange
            "GO:BP": "#2CA02C",  # green
            "HP": "#17BECF"     # cyan
        }.get(source, "#DDDDDD")

        truncated_name = f"{source}: {term['name'][:max_pathway_name_len]}..."
        for g in intersect_genes:
            gene_to_pathways[g].append((truncated_name, term.get("p_value", 0.01)))
            pathway_colors[truncated_name] = color

    # Step 2: Limit genes and pathways
    cluster_nodes = list(cluster_labels.values())
    gene_nodes = sorted({g for genes in cluster_to_genes.values() for g in genes})
    gene_nodes_limited = []
    for cl, genes in cluster_to_genes.items():
        gene_nodes_limited.extend(genes[:top_n_genes_per_cluster])
    gene_nodes = sorted(set(gene_nodes_limited))

    pathway_nodes = sorted({p for g in gene_nodes for (p, _) in gene_to_pathways.get(g, [])})
    all_nodes = cluster_nodes + gene_nodes + pathway_nodes
    node_idx = {name: i for i, name in enumerate(all_nodes)}

    # Step 3: Links
    source_indices = []
    target_indices = []
    values = []
    link_colors = []

    # Cluster → Gene
    for cl, genes in cluster_to_genes.items():
        cluster_label = cluster_labels[cl]
        for gene in genes[:top_n_genes_per_cluster]:
            if gene not in node_idx:
                continue
            source_indices.append(node_idx[cluster_label])
            target_indices.append(node_idx[gene])
            values.append(1)
            link_colors.append(CLUSTER_COLORS.get(cl, "#999999"))

    # Gene → Pathway
    for gene in gene_nodes:
        for pathway, pval in gene_to_pathways.get(gene, []):
            if pathway not in node_idx:
                continue
            source_indices.append(node_idx[gene])
            target_indices.append(node_idx[pathway])
            values.append(max(1e-5, -np.log10(pval)))  # p-value as weight
            link_colors.append(pathway_colors.get(pathway, "lightgray"))

    # Step 4: Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=["#cccccc"] * len(cluster_nodes) +
                  ["#dddddd"] * len(gene_nodes) +
                  [pathway_colors.get(n, "#eeeeee") for n in pathway_nodes],
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors
        )
    )])

    fig.update_layout(
        title_text=f"Sankey: Cluster–Gene–Pathway Mapping ({cancer_type}, {enrichment_source})",
        font_size=14,
        title_font_size=20
    )

    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"sankey_{tag}_{cancer_type}_{enrichment_source}.html")
    png_path = os.path.join(output_dir, f"sankey_{tag}_{cancer_type}_{enrichment_source}.png")
    
    fig.write_html(html_path)
    if export_png:
        fig.write_image(png_path, scale=2)
    
    return html_path, png_path


# Define the updated sankey plotting function
def plot_sankey_cluster_gene_pathway_limited(
    cluster_to_genes,
    enrichment_terms,
    cancer_type,
    tag,
    enrichment_source,
    output_dir,
    top_n_genes_per_cluster=20,
    max_pathway_name_len=40
):
    os.makedirs(output_dir, exist_ok=True)

    cluster_labels = {i: f"Cluster {i}" for i in cluster_to_genes}
    gene_to_pathways = defaultdict(list)
    pathway_to_source = {}

    for term in enrichment_terms:
        intersect_genes = term.get("intersect_genes", [])
        if not isinstance(intersect_genes, list):
            continue
        for g in intersect_genes:
            gene_to_pathways[g].append(term["name"])
        pathway_to_source[term["name"]] = term.get("source", "OTHER")

    pathway_name_map = {}
    for name in pathway_to_source:
        short = name[:max_pathway_name_len] + "..." if len(name) > max_pathway_name_len else name
        pathway_name_map[name] = short

    cluster_nodes = list(cluster_labels.values())
    gene_nodes = sorted({
        gene for genes in cluster_to_genes.values() for gene in genes[:top_n_genes_per_cluster]
    })
    pathway_nodes = sorted({
        pathway_name_map[p] for g in gene_nodes for p in gene_to_pathways.get(g, [])
    })

    all_nodes = cluster_nodes + gene_nodes + pathway_nodes
    node_idx = {name: i for i, name in enumerate(all_nodes)}

    source_indices = []
    target_indices = []
    values = []
    link_colors = []

    for cl, genes in cluster_to_genes.items():
        cluster_label = cluster_labels[cl]
        for gene in genes[:top_n_genes_per_cluster]:
            if gene not in node_idx:
                continue
            source_indices.append(node_idx[cluster_label])
            target_indices.append(node_idx[gene])
            values.append(1)
            link_colors.append(CLUSTER_COLORS.get(cl, "#999999"))

    for gene in gene_nodes:
        for p in gene_to_pathways.get(gene, []):
            short_p = pathway_name_map[p]
            if short_p not in node_idx:
                continue
            source_indices.append(node_idx[gene])
            target_indices.append(node_idx[short_p])
            term = next((t for t in enrichment_terms if t["name"] == p), None)
            weight = -np.log10(term["p_value"]) if term and term.get("p_value") else 1.0
            values.append(weight)
            source = pathway_to_source.get(p, "OTHER")
            link_colors.append(SOURCE_COLORS.get(source, "#cccccc"))

    node_colors = (
        [CLUSTER_COLORS.get(i, "#cccccc") for i in cluster_to_genes] +
        ["#dddddd"] * len(gene_nodes) +
        [SOURCE_COLORS.get(pathway_to_source.get(k, "OTHER"), "#eeeeee")
         for k in pathway_name_map if pathway_name_map[k] in pathway_nodes]
    )

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors
        )
    )])

    fig.update_layout(
        title_text=f"Sankey: Cluster–Gene–Pathway Mapping ({cancer_type}, {enrichment_source})",
        font_size=14,
        title_font_size=22
    )

    base = f"sankey_{tag}_{cancer_type}_{str(enrichment_source).replace(':', '')}"
    fig.write_html(os.path.join(output_dir, base + ".html"))
    fig.write_image(os.path.join(output_dir, base + ".png"))

def classify_pam50(expression_df: pd.DataFrame, pam50_centroids_path: str):
    """
    Classify BRCA samples into PAM50 subtypes using centroid correlation.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Gene expression matrix: rows = genes, columns = samples.
    pam50_centroids_path : str
        Path to PAM50 centroid table (CSV).

    Returns
    -------
    pd.Series : subtype calls for each sample.
    pd.DataFrame : correlation matrix (samples x subtypes).
    """
    centroids = pd.read_csv(pam50_centroids_path, index_col=0)
    common_genes = centroids.index.intersection(expression_df.index)
    if len(common_genes) < 40:
        raise ValueError(f"Too few PAM50 genes found: {len(common_genes)}")

    expr_pam50 = expression_df.loc[common_genes]
    expr_pam50_z = expr_pam50.apply(zscore, axis=1, result_type='broadcast')
    centroids_pam50 = centroids.loc[common_genes]
    centroids_pam50_z = centroids_pam50.apply(zscore, axis=0)

    corr_matrix = pd.DataFrame(
        np.dot(expr_pam50_z.T, centroids_pam50_z) / len(common_genes),
        index=expr_pam50_z.columns,
        columns=centroids_pam50_z.columns
    )
    subtype_calls = corr_matrix.idxmax(axis=1)
    return subtype_calls, corr_matrix

def load_and_merge_survival_with_subtypes(survival_path: str, subtype_path: str, save_path: str = None):
    """
    Merge survival data with PAM50 subtype calls.

    Parameters
    ----------
    survival_path : str
        Path to survival data TSV.
    subtype_path : str
        Path to PAM50 subtype calls CSV.
    save_path : str, optional
        If provided, path to save the merged dataframe as CSV.

    Returns
    -------
    pd.DataFrame
        Merged dataframe with survival and subtype info.
    """
    import pandas as pd

    survival = pd.read_csv(survival_path, sep="\t")
    subtypes = pd.read_csv(subtype_path, index_col=0)

    survival['_PATIENT'] = survival['_PATIENT'].str.upper().str[:12]
    subtypes.index = subtypes.index.str.upper().str[:12]

    merged = survival.merge(subtypes, left_on="_PATIENT", right_index=True)
    merged = merged.rename(columns={merged.columns[-1]: "PAM50"})

    if save_path:
        merged.to_csv(save_path, index=False)
        print(f"✅ Merged dataframe saved to: {save_path}")

    return merged

def plot_km_by_pam50_small_fontsize(df: pd.DataFrame, output_path: str = "tcga_brca_km_survival.png"):
    """
    Plot Kaplan-Meier survival curves grouped by PAM50 subtype, including HR, CI, p-value.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'OS.time', 'OS', and 'PAM50'.
    output_path : str
        Path to save KM plot image.
    """
    CLUSTER_COLORS = {
        0: '#0077B6',  # Basal
        1: '#0000FF',  # Her2
        2: '#00B4D8',  # LumA
        3: '#48EAC4',  # LumB
    }

    SUBTYPE_TO_CLUSTER = {
        'Basal': 0,
        'Her2': 1,
        'LumA': 2,
        'LumB': 3
    }

    if 'OS.time' not in df.columns or 'OS' not in df.columns:
        raise KeyError("❌ Could not find survival time columns ('OS.time' and 'OS').")

    df = df.copy()
    df['time'] = pd.to_numeric(df['OS.time'], errors='coerce')
    df['event'] = df['OS']
    df = df.dropna(subset=['time', 'event', 'PAM50'])
    df = df[df['PAM50'].str.lower() != 'normal']

    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    handles = []

    for subtype in sorted(SUBTYPE_TO_CLUSTER.keys()):
        mask = df['PAM50'] == subtype
        if not mask.any():
            continue
        cluster_id = SUBTYPE_TO_CLUSTER[subtype]
        color = CLUSTER_COLORS.get(cluster_id, '#000000')
        kmf.fit(df.loc[mask, 'time'], event_observed=df.loc[mask, 'event'], label=subtype)
        kmf.plot(
            ax=ax,
            ci_show=False,
            show_censors=True,
            censor_styles={'marker': '|', 'ms': 8, 'mew': 1},
            color=color,
            linewidth=2
        )
        handles.append(Line2D([], [], color=color, linestyle='-', lw=2,
                              marker='|', markersize=6, markeredgewidth=1,
                              label=f"{subtype} (n={mask.sum()})"))

    # --- Cox Regression Analysis ---
    df_cox = df[['time', 'event', 'PAM50']].copy()
    df_cox['PAM50'] = df_cox['PAM50'].astype("category")
    if df_cox['PAM50'].nunique() <= 1:
        print("❌ Not enough subtypes for Cox regression.")
        representative_hr = np.nan
        representative_ci = np.nan
        pval = np.nan
    else:
        cph = CoxPHFitter()
        cph.fit(df_cox, duration_col="time", event_col="event")
        cox_summary = cph.summary

        hr_values = cox_summary["exp(coef)"].values
        representative_hr = np.mean(hr_values) if len(hr_values) else np.nan

        ci_widths = cox_summary["coef upper 95%"] - cox_summary["coef lower 95%"]
        representative_ci = ci_widths.mean() if len(ci_widths) else np.nan

        pval = cph._compute_p_values()["PAM50"].values[0] if "PAM50" in cph._compute_p_values().index else np.nan

        # Save Cox summary CSV
        cox_summary_path = os.path.splitext(output_path)[0] + "_cox_summary.csv"
        cox_summary.to_csv(cox_summary_path)
        print(f"📄 Cox summary saved: {cox_summary_path}")

    # --- Annotations ---
    ax.set_title("Survival by PAM50 Subtype", fontsize=24)
    ax.set_xlabel("Time (Days)", fontsize=20)
    ax.set_ylabel("Survival probability", fontsize=20)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(handles=handles, loc='upper right', fontsize=14, frameon=False)

    text_x = 0.03
    text_y_start = 0.15
    text_y_step = 0.08

    ax.text(
        text_x, text_y_start + text_y_step,
        f"CI = {representative_ci:.3f}" if not np.isnan(representative_ci) else "CI = N/A",
        transform=ax.transAxes, fontsize=16
    )
    ax.text(
        text_x, text_y_start,
        f"HR = {representative_hr:.2f}" if not np.isnan(representative_hr) else "HR = N/A",
        transform=ax.transAxes, fontsize=16
    )
    ax.text(
        text_x, text_y_start - text_y_step,
        f"p-value = {pval:.4g}" if pval >= 0.0001 else f"p-value = {pval:.1e}",
        transform=ax.transAxes, fontsize=16
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ KM plot saved to: {output_path}")

def plot_km_by_pam50_cox(df: pd.DataFrame, output_path: str = "tcga_brca_km_survival.png"):
    """
    Plot Kaplan-Meier survival curves for patients grouped by PAM50 subtype.
    Also runs Cox Proportional Hazards model with subtype as predictor.

    Parameters
    ----------
    df : pd.DataFrame
        Merged survival and PAM50 dataframe.
    output_path : str
        Path to save KM plot image.
    """
    CLUSTER_COLORS = {
        0: '#0077B6',
        1: '#0000FF',
        2: '#00B4D8',
        3: '#48EAC4'
    }
    SUBTYPE_TO_CLUSTER = {
        'Basal': 0,
        'Her2': 1,
        'LumA': 2,
        'LumB': 3
    }

    # Handle survival time/event formats
    if 'OS.time' in df.columns and 'OS' in df.columns:
        df['time'] = df['OS.time']
        df['event'] = df['OS']
    elif 'days_to_death' in df.columns and 'days_to_last_follow_up' in df.columns:
        df['time'] = df['days_to_death'].fillna(df['days_to_last_follow_up'])
        df['event'] = df['vital_status'].apply(lambda x: 1 if str(x).lower() == 'dead' else 0)
    elif 'demographic.days_to_death' in df.columns and 'diagnoses.days_to_last_follow_up' in df.columns:
        df['time'] = df['demographic.days_to_death'].fillna(df['diagnoses.days_to_last_follow_up'])
        df['event'] = df['demographic.vital_status'].apply(lambda x: 1 if str(x).lower() == 'dead' else 0)
    else:
        raise KeyError("❌ Could not find survival time columns.")

    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time', 'PAM50'])
    df = df[df['PAM50'].str.lower() != 'normal']

    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    handles = []

    for subtype in sorted(SUBTYPE_TO_CLUSTER.keys()):
        mask = df['PAM50'] == subtype
        if not mask.any():
            continue
        cluster_id = SUBTYPE_TO_CLUSTER[subtype]
        color = CLUSTER_COLORS.get(cluster_id, '#000000')
        kmf.fit(df.loc[mask, 'time'], event_observed=df.loc[mask, 'event'], label=subtype)
        kmf.plot(
            ax=ax,
            ci_show=False,
            show_censors=True,
            censor_styles={'marker': '|', 'ms': 8, 'mew': 1},
            color=color,
            linewidth=2
        )
        handles.append(Line2D([], [], color=color, linestyle='-', lw=2,
                              marker='|', markersize=6, markeredgewidth=1,
                              label=f"{subtype} (n={mask.sum()})"))

    # Cox model
    df_cox = df[['time', 'event', 'PAM50']].copy()
    df_cox = pd.get_dummies(df_cox, columns=['PAM50'], drop_first=True)

    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col="time", event_col="event")

    hr_values = cph.summary["exp(coef)"].values
    representative_hr = hr_values.mean() if len(hr_values) else float('nan')

    ci_widths = cph.summary["coef upper 95%"] - cph.summary["coef lower 95%"]
    representative_ci = ci_widths.mean() if len(ci_widths) else float('nan')

    # Plot formatting (match fontsize 28)
    ax.set_title("Survival by PAM50 Subtype", fontsize=28)
    ax.set_xlabel("Time (Days)", fontsize=28)
    ax.set_ylabel("Survival probability", fontsize=28)
    ax.tick_params(axis='both', labelsize=28)
    ax.grid(False)

    # Optional: Display HR, CI, and p-value from Cox
    text_x = 0.03
    text_y_start = 0.15
    text_y_step = 0.085

    ax.text(
        text_x, text_y_start + text_y_step,
        f"CI = {representative_ci:.3f}" if not pd.isna(representative_ci) else "CI = N/A",
        transform=ax.transAxes, fontsize=28
    )
    ax.text(
        text_x, text_y_start,
        f"HR = {representative_hr:.2f}" if not pd.isna(representative_hr) else "HR = N/A",
        transform=ax.transAxes, fontsize=28
    )
    pval = cph.summary["p"].min()
    ax.text(
        text_x, text_y_start - text_y_step,
        f"p-value = {pval:.4g}" if pval >= 0.0001 else f"p-value = {pval:.1e}",
        transform=ax.transAxes, fontsize=28
    )

    ax.legend(handles=handles, loc='upper right', fontsize=28, frameon=False, handlelength=1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ KM plot saved to: {output_path}")
    print("✅ Cox regression summary:")
    print(cph.summary)

def plot_km_by_pam50_log_rank_test(df: pd.DataFrame, output_path: str = "tcga_brca_km_survival.png"):
    """
    Plot Kaplan-Meier survival curves for patients grouped by PAM50 subtype.
    Uses log-rank test to assess survival difference.

    Parameters
    ----------
    df : pd.DataFrame
        Merged survival and PAM50 dataframe.
    output_path : str
        Path to save KM plot image.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    CLUSTER_COLORS = {
        0: '#0077B6',
        1: '#0000FF',
        2: '#00B4D8',
        3: '#48EAC4'
    }
    SUBTYPE_TO_CLUSTER = {
        'Basal': 0,
        'Her2': 1,
        'LumA': 2,
        'LumB': 3
    }

    # Handle survival time/event formats
    if 'OS.time' in df.columns and 'OS' in df.columns:
        df['time'] = df['OS.time']
        df['event'] = df['OS']
    elif 'days_to_death' in df.columns and 'days_to_last_follow_up' in df.columns:
        df['time'] = df['days_to_death'].fillna(df['days_to_last_follow_up'])
        df['event'] = df['vital_status'].apply(lambda x: 1 if str(x).lower() == 'dead' else 0)
    elif 'demographic.days_to_death' in df.columns and 'diagnoses.days_to_last_follow_up' in df.columns:
        df['time'] = df['demographic.days_to_death'].fillna(df['diagnoses.days_to_last_follow_up'])
        df['event'] = df['demographic.vital_status'].apply(lambda x: 1 if str(x).lower() == 'dead' else 0)
    else:
        raise KeyError("❌ Could not find survival time columns.")

    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time', 'PAM50'])
    df = df[df['PAM50'].str.lower() != 'normal']

    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    handles = []

    for subtype in sorted(SUBTYPE_TO_CLUSTER.keys()):
        mask = df['PAM50'] == subtype
        if not mask.any():
            continue
        cluster_id = SUBTYPE_TO_CLUSTER[subtype]
        color = CLUSTER_COLORS.get(cluster_id, '#000000')
        kmf.fit(df.loc[mask, 'time'], event_observed=df.loc[mask, 'event'], label=subtype)
        kmf.plot(
            ax=ax,
            ci_show=False,
            show_censors=True,
            censor_styles={'marker': '|', 'ms': 8, 'mew': 1},
            color=color,
            linewidth=2
        )
        handles.append(Line2D([], [], color=color, linestyle='-', lw=2,
                              marker='|', markersize=6, markeredgewidth=1,
                              label=f"{subtype} (n={mask.sum()})"))

    # 🔑 Run log-rank test instead of Cox
    result = multivariate_logrank_test(df['time'], df['PAM50'], df['event'])
    pval = result.p_value

    # Plot formatting
    ax.set_title("Survival by PAM50 Subtype", fontsize=28)
    ax.set_xlabel("Time (Days)", fontsize=28)
    ax.set_ylabel("Survival probability", fontsize=28)
    ax.tick_params(axis='both', labelsize=28)
    ax.grid(False)

    text_x = 0.03
    text_y_start = 0.10

    ax.text(
        text_x, text_y_start,
        f"p-value = {pval:.4g}" if pval >= 0.0001 else f"p-value = {pval:.1e}",
        transform=ax.transAxes, fontsize=28
    )

    ax.legend(handles=handles, loc='upper right', fontsize=28, frameon=False, handlelength=1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ KM plot saved to: {output_path}")
    print(f"✅ Log-rank test p-value: {pval:.4g}")
    print(result.summary)

def plot_km_by_pam50(df: pd.DataFrame, output_path: str = "tcga_brca_km_survival.png"):
    """
    Plot KM survival curves for patients grouped by PAM50 subtype, showing vertically stacked CI width, HR, and p-value.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe with survival and PAM50 labels.
    output_path : str
        Where to save the KM plot image.

    Returns
    -------
    result : lifelines.statistics.StatisticalResult
        Result from the multivariate log-rank test.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import multivariate_logrank_test

    CLUSTER_COLORS = {
        0: '#0077B6',
        1: '#0000FF',
        2: '#00B4D8',
        3: '#48EAC4'
    }
    SUBTYPE_TO_CLUSTER = {
        'Basal': 0,
        'Her2': 1,
        'LumA': 2,
        'LumB': 3
    }

    # Handle survival columns
    if 'OS.time' in df.columns and 'OS' in df.columns:
        df['time'] = df['OS.time']
        df['event'] = df['OS']
    elif 'days_to_death' in df.columns and 'days_to_last_follow_up' in df.columns:
        df['time'] = df['days_to_death'].fillna(df['days_to_last_follow_up'])
        df['event'] = df['vital_status'].apply(lambda x: 1 if str(x).lower() == 'dead' else 0)
    elif 'demographic.days_to_death' in df.columns and 'diagnoses.days_to_last_follow_up' in df.columns:
        df['time'] = df['demographic.days_to_death'].fillna(df['diagnoses.days_to_last_follow_up'])
        df['event'] = df['demographic.vital_status'].apply(lambda x: 1 if str(x).lower() == 'dead' else 0)
    else:
        raise KeyError("❌ Could not find survival time columns.")

    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time', 'event', 'PAM50'])
    df = df[df['PAM50'].str.lower() != 'normal']
    df = df[df['PAM50'].isin(SUBTYPE_TO_CLUSTER.keys())]

    # Binary encoding for Cox: reference vs rest
    unique_subtypes = sorted(df['PAM50'].unique())
    ref_subtype = unique_subtypes[0]
    df['Subtype_binary'] = (df['PAM50'] != ref_subtype).astype(int)

    # Cox regression
    cph_df = df[['time', 'event', 'Subtype_binary']].copy()
    cph = CoxPHFitter()
    cph.fit(cph_df, duration_col='time', event_col='event')

    hr = np.exp(cph.params_['Subtype_binary'])
    ci_lower, ci_upper = np.exp(cph.confidence_intervals_.loc['Subtype_binary'])
    ci_width = ci_upper - ci_lower

    # Log-rank test
    result = multivariate_logrank_test(df['time'], df['PAM50'], df['event'])
    pval = result.p_value

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    handles = []

    for subtype in unique_subtypes:
        mask = df['PAM50'] == subtype
        if not mask.any():
            continue
        cluster_id = SUBTYPE_TO_CLUSTER.get(subtype, 0)
        color = CLUSTER_COLORS.get(cluster_id, '#000000')
        kmf.fit(df.loc[mask, 'time'], event_observed=df.loc[mask, 'event'], label=subtype)
        kmf.plot(
            ax=ax,
            ci_show=False,
            show_censors=True,
            censor_styles={'marker': '|', 'ms': 8, 'mew': 1},
            color=color,
            linewidth=2
        )
        handles.append(Line2D([], [], color=color, linestyle='-', lw=2,
                              marker='|', markersize=6, markeredgewidth=1,
                              label=f"{subtype} (n={mask.sum()})"))

    # Plot formatting
    ax.set_title("PAM50", fontsize=28)
    ax.set_xlabel("Time (Days)", fontsize=28)
    ax.set_ylabel("Survival probability", fontsize=28)
    ax.tick_params(axis='both', labelsize=28)
    ax.grid(False)

    # Stats: vertical alignment
    ax.text(0.03, 0.15 + 0.085, f"CI = {ci_width:.3f}", transform=ax.transAxes, fontsize=28)
    ax.text(0.03, 0.15, f"HR = {hr:.2f}", transform=ax.transAxes, fontsize=28)
    ax.text(0.03, 0.15 - 0.085,
            f"p-value = {pval:.4g}" if pval >= 0.0001 else f"p-value = {pval:.1e}",
            transform=ax.transAxes, fontsize=28)

    ax.legend(handles=handles, loc='upper right', fontsize=28, frameon=False, handlelength=1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ KM plot saved to: {output_path}")
    print(f"✅ Log-rank p = {pval:.4g}")
    print(result.summary)

    return result

def run_nmf_clustering(expr_df, n_clusters=4):
    from sklearn.decomposition import NMF
    import numpy as np
    import pandas as pd

    df = expr_df.copy()

    # If log transform needed:
    df = np.log2(df + 1)

    # Shift to non-negative
    df = df - df.min().min()

    X = df.T  # samples × genes

    model = NMF(n_components=n_clusters, random_state=42)
    W = model.fit_transform(X)

    clusters = W.argmax(axis=1)
    cluster_df = pd.DataFrame({
        "SampleID": X.index.str.upper().str[:12],
        "NMF_Cluster": clusters
    })

    return cluster_df

def plot_km_by_cluster(df, cluster_col='cluster_id', output_path="km_clustered.png"):

    # Prep data
    df = df.dropna(subset=['time', 'event', cluster_col])
    df[cluster_col] = df[cluster_col].astype("category")

    T = df["time"]
    E = df["event"]
    groups = df[cluster_col]

    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    unique_clusters = sorted(groups.unique())

    handles = []
    for cl in unique_clusters:
        mask = (groups == cl)
        kmf.fit(T[mask], E[mask], label=f"Cluster {cl}")
        kmf.plot(
            ax=ax,
            ci_show=False,
            show_censors=True,
            censor_styles={'marker': '|', 'ms': 8, 'mew': 1},
            color=CLUSTER_COLORS.get(int(cl), "#000000"),
            linewidth=2
        )
        handles.append(Line2D([], [], color=CLUSTER_COLORS.get(int(cl), "#000000"),
                              linestyle='-', lw=2, marker='|', markersize=6,
                              markeredgewidth=1, label=f"Cluster {cl} (n={mask.sum()})"))

    # Log-rank
    result = multivariate_logrank_test(T, groups=groups, event_observed=E)
    logrank_pval = result.p_value

    # Cox regression
    cox_df = df[["time", "event", cluster_col]].copy()
    cox_df.columns = ["time", "event", "cluster"]
    cox_df["cluster"] = cox_df["cluster"].astype("category")

    cph = CoxPHFitter()
    if cox_df["cluster"].nunique() > 1:
        cph.fit(cox_df, duration_col="time", event_col="event")
        hr_values = cph.summary["exp(coef)"].values
        representative_hr = np.mean(hr_values)
        ci_widths = cph.summary["coef upper 95%"] - cph.summary["coef lower 95%"]
        representative_ci = ci_widths.mean()
    else:
        representative_hr = representative_ci = np.nan

    # Annotate
    ax.set_title("NMF", fontsize=28)
    ax.set_xlabel("Time (Days)", fontsize=28)
    ax.set_ylabel("Survival probability", fontsize=28)
    ax.tick_params(axis='both', labelsize=28)
    ax.grid(False)

    ax.text(0.03, 0.15 + 0.085, f"CI = {representative_ci:.3f}" if not np.isnan(representative_ci) else "CI = N/A",
            transform=ax.transAxes, fontsize=28)
    ax.text(0.03, 0.15, f"HR = {representative_hr:.2f}" if not np.isnan(representative_hr) else "HR = N/A",
            transform=ax.transAxes, fontsize=28)
    ax.text(0.03, 0.15 - 0.085, f"p-value = {logrank_pval:.4g}" if logrank_pval >= 0.0001 else f"p-value = {logrank_pval:.1e}",
            transform=ax.transAxes, fontsize=28)

    ax.legend(handles=handles, loc='upper right', fontsize=28, frameon=False, handlelength=1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ KM plot saved: {output_path}")
    return result

def nmf_survival(expr_path, survival_path, n_clusters=4, output_path="km_nmf_survival.png", min_samples_per_cluster=5):
    import pandas as pd

    # Load expression
    expr_df = pd.read_csv(expr_path, sep="\t", index_col=0)
    clusters_df = run_nmf_clustering(expr_df, n_clusters=n_clusters)

    # Load survival
    survival_df = pd.read_csv(survival_path, sep="\t")
    survival_df['_PATIENT'] = survival_df['_PATIENT'].str.upper()
    clusters_df['SampleID'] = clusters_df['SampleID'].str.upper().str[:12]

    # Merge expression clusters and survival
    merged = survival_df.merge(clusters_df, left_on="_PATIENT", right_on="SampleID")
    merged['time'] = merged['OS.time']
    merged['event'] = merged['OS']

    # Filter clusters with too few samples
    cluster_counts = merged['NMF_Cluster'].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= min_samples_per_cluster].index
    merged = merged[merged['NMF_Cluster'].isin(valid_clusters)]

    if merged['NMF_Cluster'].nunique() <= 1:
        print("⚠️ Not enough clusters with sufficient samples for survival analysis.")
        return None, merged

    result = plot_km_by_cluster(merged, cluster_col='NMF_Cluster', output_path=output_path)
    return result, merged

def run_survival_analysis(
    survival_path,
    expr_path,
    node_names_topk,
    row_labels,
    output_dir,
    cluster_threshold="median",
    clusters_to_plot=[0, 1, 2, 3],
    cancer_type="PAN",
):
    from matplotlib.ticker import PercentFormatter

    os.makedirs(output_dir, exist_ok=True)

    # ==========================================================
    # Load survival data
    # ==========================================================
    survival_df = pd.read_csv(
        survival_path,
        sep="\t"
    )

    survival_df["_PATIENT"] = (
        survival_df["_PATIENT"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ==========================================================
    # Load expression data
    # ==========================================================
    expr_df = pd.read_csv(
        expr_path,
        sep="\t",
        index_col=0
    )

    expr_df.index = (
        expr_df.index
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ==========================================================
    # Normalize expression sample IDs to TCGA patient IDs
    #
    # Examples:
    #   19-1787-01   -> TCGA-19-1787
    #   S9-A7J2-01   -> TCGA-S9-A7J2
    #   G3-A3CH-11   -> TCGA-G3-A3CH
    #
    # Standard TCGA:
    #   TCGA-19-1787-01A -> TCGA-19-1787
    # ==========================================================
    def normalize_tcga_patient_id(sample_id):

        sample_id = str(sample_id).strip().upper()

        parts = sample_id.split("-")

        # Already has TCGA prefix
        if parts[0] == "TCGA":

            if len(parts) >= 3:
                return (
                    f"TCGA-{parts[1]}-{parts[2]}"
                )

            return sample_id

        # TCGA prefix omitted
        if len(parts) >= 2:

            return (
                f"TCGA-{parts[0]}-{parts[1]}"
            )

        return sample_id

    expr_df.columns = [
        normalize_tcga_patient_id(col)
        for col in expr_df.columns
    ]

    # ==========================================================
    # Check patient ID matching
    # ==========================================================
    common_patients = sorted(
        set(expr_df.columns)
        & set(survival_df["_PATIENT"])
    )

    print(
        f"Expression patients: {len(expr_df.columns)}"
    )

    print(
        f"Survival patients:   {len(survival_df)}"
    )

    print(
        f"Matched patients:    {len(common_patients)}"
    )

    print(
        "Example expression IDs:",
        expr_df.columns[:5].tolist()
    )

    print(
        "Example survival IDs:",
        survival_df["_PATIENT"].head().tolist()
    )

    print(
        "Example matched IDs:",
        common_patients[:5]
    )

    # ==========================================================
    # Cluster-gene mapping
    # ==========================================================
    cluster_to_genes = defaultdict(list)

    for gene, label in zip(
        node_names_topk,
        row_labels
    ):
        cluster_to_genes[label].append(
            gene.upper()
        )

    # ==========================================================
    # Compute cluster scores
    # ==========================================================
    patient_cluster_scores = pd.DataFrame(
        index=expr_df.columns
    )

    for cl, genes in cluster_to_genes.items():

        valid_genes = list(
            set(genes)
            & set(expr_df.index)
        )

        if not valid_genes:

            print(
                f"⚠️ No valid genes for cluster {cl}"
            )

            continue

        patient_cluster_scores[
            f"cluster {cl}"
        ] = expr_df.loc[
            valid_genes
        ].mean(axis=0)

    # ==========================================================
    # Merge survival + cluster scores
    # ==========================================================
    merged = survival_df.merge(
        patient_cluster_scores,
        left_on="_PATIENT",
        right_index=True,
        how="inner"
    )

    print(
        f"Matched survival-expression rows: "
        f"{len(merged)}"
    )

    # ==========================================================
    # Clean survival data
    # ==========================================================
    merged = merged.dropna(
        subset=["OS.time", "OS"]
    )

    merged = merged[
        merged["OS.time"] > 0
    ]

    merged["OS"] = (
        merged["OS"]
        .astype(int)
    )

    if merged.empty:

        print(
            "❌ No matching patients after cleaning."
        )

        return

    print(
        f"✅ Merged shape after clean: "
        f"{merged.shape}"
    )

    # === Plot KM ===
    for cl in clusters_to_plot:
        col = f"cluster {cl}"
        if col not in merged.columns:
            print(f"⚠️ Cluster {col} not found. Skipping.")
            continue

        T = merged["OS.time"]
        E = merged["OS"]

        threshold = merged[col].median() if cluster_threshold == "median" else merged[col].mean()
        group_col = f"group {cl}"
        merged[group_col] = (merged[col] >= threshold).astype(int)

        kmf = KaplanMeierFitter()
        fig, ax = plt.subplots(figsize=(4, 3.5))

        colors = {0: "#1f77b4", 1: "#d62728"}
        legend_handles = []

        for group in [0, 1]:
            mask = merged[group_col] == group
            if mask.sum() == 0:
                continue
            label = f"{'Low' if group == 0 else 'High'} {col} (n={mask.sum()})"
            kmf.fit(T[mask], E[mask], label=label)
            kmf.plot_survival_function(
                ax=ax,
                ci_show=False,
                show_censors=True,
                color=colors[group],
                linewidth=2,
                censor_styles={'marker': '|', 'ms': 8, 'mew': 1}
            )
            legend_handles.append(Line2D([], [], color=colors[group], lw=2,
                                         marker='|', markersize=6, markeredgewidth=1, label=label))

        try:
            result = logrank_test(
                T[merged[group_col] == 1],
                T[merged[group_col] == 0],
                event_observed_A=E[merged[group_col] == 1],
                event_observed_B=E[merged[group_col] == 0]
            )
            pval = result.p_value
        except Exception as e:
            print(f"⚠️ Log-rank failed: {e}")
            pval = np.nan

        cph_df = merged[["OS.time", "OS", col]].copy()
        cph_df.columns = ["time", "event", "score"]
        cph_single = CoxPHFitter()
        try:
            cph_single.fit(cph_df, duration_col="time", event_col="event")
            hr = cph_single.hazard_ratios_["score"]
        except Exception as e:
            print(f"⚠️ Cox PH failed: {e}")
            hr = np.nan

        ax.set_title(f"Cluster {cl}", fontsize=26)
        ax.set_xlabel("Time (Days)", fontsize=16)
        ax.set_ylabel("Survival probability", fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_ylim(0, 1.05)
        # yticks = ax.get_yticks()
        # ax.set_yticklabels([f"{y:.0%}" for y in yticks])

        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        ax.text(0.05, 0.12, f"HR = {hr:.2f}" if not np.isnan(hr) else "HR=N/A", transform=ax.transAxes, fontsize=10)
        ax.text(0.05, 0.05, f"p-value = {pval:.3g}" if not np.isnan(pval) else "p-value = N/A", transform=ax.transAxes, fontsize=10)
        ax.legend(handles=legend_handles, loc="upper right", fontsize=10, frameon=False, handlelength=1.0)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"km_cluster_{cl}_{cancer_type}.png")
        plt.savefig(out_path, dpi=300)
        plt.show()
        print(f"✅ Saved plot: {out_path}")

    # === Multivariate Cox ===
    cox_cols = [c for c in patient_cluster_scores.columns if c in merged.columns]
    cox_df_all = merged[["OS.time", "OS"] + cox_cols].copy()
    cox_df_all.columns = ["time", "event"] + cox_cols
    cph = CoxPHFitter()
    try:
        cph.fit(cox_df_all, duration_col="time", event_col="event")
        cph.summary.to_csv(os.path.join(output_dir, f"cox_summary_{cancer_type}.csv"))
        print(f"✅ Saved Cox summary: {os.path.join(output_dir, f'cox_summary_{cancer_type}.csv')}")
    except Exception as e:
        print(f"⚠️ Multivariate Cox failed: {e}")

    return cph.summary, merged, patient_cluster_scores

def run_survival_analysis_cox(
    survival_path,
    expr_path,
    node_names_topk,
    row_labels,
    output_dir,
    cluster_threshold="median",
    cancer_type="PAN",
):
    """
    Safe version: Global cluster survival analysis.
    1) Average cluster scores.
    2) Split patients into High vs Low.
    3) KM plot with censor ticks + logrank + Cox.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 18
    })

    # ==========================================================
    # Load survival data
    # ==========================================================
    survival_df = pd.read_csv(
        survival_path,
        sep="\t"
    )

    survival_df["_PATIENT"] = (
        survival_df["_PATIENT"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ==========================================================
    # Load expression data
    # ==========================================================
    expr_df = pd.read_csv(
        expr_path,
        sep="\t",
        index_col=0
    )

    expr_df.index = (
        expr_df.index
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ==========================================================
    # Normalize expression sample IDs to TCGA patient IDs
    #
    # Examples:
    #   19-1787-01   -> TCGA-19-1787
    #   S9-A7J2-01   -> TCGA-S9-A7J2
    #   G3-A3CH-11   -> TCGA-G3-A3CH
    #
    # Standard TCGA:
    #   TCGA-19-1787-01A -> TCGA-19-1787
    # ==========================================================
    def normalize_tcga_patient_id(sample_id):

        sample_id = str(sample_id).strip().upper()

        parts = sample_id.split("-")

        # Already has TCGA prefix
        if parts[0] == "TCGA":

            if len(parts) >= 3:
                return (
                    f"TCGA-{parts[1]}-{parts[2]}"
                )

            return sample_id

        # TCGA prefix omitted
        if len(parts) >= 2:

            return (
                f"TCGA-{parts[0]}-{parts[1]}"
            )

        return sample_id

    expr_df.columns = [
        normalize_tcga_patient_id(col)
        for col in expr_df.columns
    ]

    # ==========================================================
    # Check patient ID matching
    # ==========================================================
    common_patients = sorted(
        set(expr_df.columns)
        & set(survival_df["_PATIENT"])
    )

    print(
        f"Expression patients: {len(expr_df.columns)}"
    )

    print(
        f"Survival patients:   {len(survival_df)}"
    )

    print(
        f"Matched patients:    {len(common_patients)}"
    )

    print(
        "Example expression IDs:",
        expr_df.columns[:5].tolist()
    )

    print(
        "Example survival IDs:",
        survival_df["_PATIENT"].head().tolist()
    )

    print(
        "Example matched IDs:",
        common_patients[:5]
    )

    # ==========================================================
    # Cluster-gene mapping
    # ==========================================================
    cluster_to_genes = defaultdict(list)

    for gene, label in zip(
        node_names_topk,
        row_labels
    ):
        cluster_to_genes[label].append(
            gene.upper()
        )

    # ==========================================================
    # Compute cluster scores
    # ==========================================================
    patient_cluster_scores = pd.DataFrame(
        index=expr_df.columns
    )

    for cl, genes in cluster_to_genes.items():

        valid_genes = list(
            set(genes)
            & set(expr_df.index)
        )

        if not valid_genes:

            print(
                f"⚠️ No valid genes for cluster {cl}"
            )

            continue

        patient_cluster_scores[
            f"cluster {cl}"
        ] = expr_df.loc[
            valid_genes
        ].mean(axis=0)

    # ==========================================================
    # Merge survival + cluster scores
    # ==========================================================
    merged = survival_df.merge(
        patient_cluster_scores,
        left_on="_PATIENT",
        right_index=True,
        how="inner"
    )

    print(
        f"Matched survival-expression rows: "
        f"{len(merged)}"
    )

    # ==========================================================
    # Clean survival data
    # ==========================================================
    merged = merged.dropna(
        subset=["OS.time", "OS"]
    )

    merged = merged[
        merged["OS.time"] > 0
    ]

    merged["OS"] = (
        merged["OS"]
        .astype(int)
    )

    if merged.empty:

        print(
            "❌ No matching patients after cleaning."
        )

        return

    print(
        f"✅ Merged shape after clean: "
        f"{merged.shape}"
    )

    # ✅ Drop NaNs, fix OS.time & OS
    merged = merged.dropna(subset=["OS.time", "OS"])
    merged = merged[merged["OS.time"] > 0]
    merged["OS"] = merged["OS"].fillna(0).astype(int)

    print(f"✅ Merged shape after cleaning: {merged.shape}")

    # -------------------------------
    # Compute global cluster score
    cluster_cols = [col for col in patient_cluster_scores.columns if col in merged.columns]
    if not cluster_cols:
        print("❌ No valid cluster columns.")
        return

    merged["global_cluster_score"] = merged[cluster_cols].mean(axis=1)
    T = merged["OS.time"]
    E = merged["OS"]

    threshold = merged["global_cluster_score"].median() if cluster_threshold == "median" else merged["global_cluster_score"].mean()
    merged["global_group"] = (merged["global_cluster_score"] >= threshold).astype(int)

    mask_high = merged["global_group"] == 1
    mask_low = merged["global_group"] == 0

    # ✅ Guard: check if both groups exist
    if mask_high.sum() == 0 or mask_low.sum() == 0:
        print("❌ One group is empty after thresholding.")
        return

    T_high, E_high = T[mask_high], E[mask_high]
    T_low, E_low = T[mask_low], E[mask_low]

    # -------------------------------
    # Kaplan–Meier
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(4, 3.5))

    kmf_high.fit(T_high, E_high, label="High")
    kmf_low.fit(T_low, E_low, label="Low")

    kmf_high.plot_survival_function(
        ax=ax,
        ci_show=False,
        show_censors=True,
        color="#d62728",
        linewidth=2,
        censor_styles={'marker': '|', 'ms': 8, 'mew': 1}
    )
    kmf_low.plot_survival_function(
        ax=ax,
        ci_show=False,
        show_censors=True,
        color="#17becf",
        linewidth=2,
        censor_styles={'marker': '|', 'ms': 8, 'mew': 1}
    )

    # --- Custom handles for nice legend
    line_high = Line2D([], [], color="#d62728", lw=2, linestyle='-',
                       marker='|', markersize=6, markeredgewidth=1,
                       label=f"High (n={mask_high.sum()})")
    line_low = Line2D([], [], color="#17becf", lw=2, linestyle='-',
                      marker='|', markersize=6, markeredgewidth=1,
                      label=f"Low (n={mask_low.sum()})")

    # Logrank test
    result = logrank_test(T_high, T_low, event_observed_A=E_high, event_observed_B=E_low)
    pval = result.p_value

    # ax.set_title("Survival by global cluster expression", fontsize=24)

    # ax.set_title("Survival by global cluster expression", fontsize=24)
    ax.set_xlabel("Time (Days)", fontsize=14)
    ax.set_ylabel("Overall survival probability", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(handles=[line_high, line_low], fontsize=12, loc='upper right', frameon=False, handlelength=1.0)
    ax.text(0.05, 0.05, f"p-value = {pval:.4g}", transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"km_global_clusters_{cancer_type}.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"✅ KM plot saved: {plot_path}")

    # -------------------------------
    # Cox PH model
    cox_df = merged[["OS.time", "OS"] + cluster_cols].copy()
    cox_df.columns = ["time", "event"] + cluster_cols

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="time", event_col="event")
    cox_path = os.path.join(output_dir, f"cox_summary_{cancer_type}.csv")
    cph.summary.to_csv(cox_path)
    print(f"📄 Saved Cox summary: {cox_path}")

    return cph.summary, merged, patient_cluster_scores

def get_top_neighbors(
    gene,
    neighbors,
    name_to_index,
    topk_name_to_index,
    relevance_scores,
    node_id_to_name,
    output_dir,
    k=5
):
    """
    Returns a dict of top-k neighbor indices and scores.
    - Only includes neighbors in topk.
    - No self-loops.
    - If there are fewer than k valid topk neighbors: returns as many as available.
    """
    neighbor_scores = {}

    for n in neighbors:
        if n == gene:
            continue  # skip self
        if n not in topk_name_to_index:
            continue  # skip if not in topk

        rel_idx = topk_name_to_index[n]
        if rel_idx < relevance_scores.shape[0]:
            rel_score = relevance_scores[rel_idx].sum().item()
            neighbor_scores[rel_idx] = rel_score

    # Strict: keep only top k by score
    top_k = dict(sorted(neighbor_scores.items(), key=lambda x: -x[1])[:k])

    # Optional strict: ensure we got exactly k? If not, return empty.
    if len(top_k) < k:
        print(f"⚠️ Not enough topk neighbors for {gene}. Found {len(top_k)}, needed {k}. Returning empty.")
        return {}

    return top_k

def survival_analysis_by_cluster_assigned(
    survival_path,
    expr_path,
    node_names_topk,
    row_labels,
    output_dir,
    cancer_type="PAN",
    min_samples_per_cluster=5
):
    os.makedirs(output_dir, exist_ok=True)

    survival_df = pd.read_csv(survival_path, sep="\t")
    expr_df = pd.read_csv(expr_path, sep="\t", index_col=0)

    survival_df['_PATIENT'] = survival_df['_PATIENT'].str.upper()
    expr_df.index = expr_df.index.str.upper()
    expr_df.columns = expr_df.columns.str.upper().str[:12]

    cluster_to_genes = defaultdict(list)
    for gene, label in zip(node_names_topk, row_labels):
        cluster_to_genes[label].append(gene.upper())

    patient_cluster_scores = pd.DataFrame(index=expr_df.columns)
    for cl, genes in cluster_to_genes.items():
        valid_genes = list(set(genes) & set(expr_df.index))
        if not valid_genes:
            print(f"⚠️ No valid genes for cluster {cl}. Skipping.")
            continue
        patient_cluster_scores[f'cluster {cl}'] = expr_df.loc[valid_genes].mean(axis=0)

    cluster_assignments = (
        patient_cluster_scores.idxmax(axis=1).str.extract(r'cluster (\d+)')[0].astype(int)
    )
    patient_cluster_scores["cluster_id"] = cluster_assignments

    merged = survival_df.merge(patient_cluster_scores, left_on="_PATIENT", right_index=True)
    if merged.empty:
        print("❌ No overlap between survival and expression.")
        return None, None, None, None, cancer_type

    # ✅ Drop rows with NaNs in any key columns for KM
    merged_clean = merged.dropna(subset=["OS.time", "OS", "cluster_id"])
    print(f"Original merged shape: {merged.shape} ➜ After dropna: {merged_clean.shape}")

    T = merged_clean["OS.time"]
    E = merged_clean["OS"]
    groups = merged_clean["cluster_id"]

    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    unique_clusters = sorted(groups.unique())

    handles = []
    for cl in unique_clusters:
        mask = (groups == cl)
        if mask.sum() < min_samples_per_cluster:
            print(f"⚠️ Cluster {cl} skipped: too few patients.")
            continue

        kmf.fit(T[mask], E[mask], label=f"Cluster {cl}")
        kmf.plot(
            ax=ax,
            ci_show=False,
            show_censors=True,
            censor_styles={'marker': '|', 'ms': 8, 'mew': 1},
            color=CLUSTER_COLORS.get(cl, "#000000"),
            linewidth=2
        )

        curve_handle = Line2D(
            [], [],
            color=CLUSTER_COLORS.get(cl, "#000000"),
            linestyle='-',
            lw=2,
            marker='|',
            markersize=6,
            markeredgewidth=1,
            label=f"Cluster {cl} (n={mask.sum()})"
        )
        handles.append(curve_handle)

    result = multivariate_logrank_test(T, groups=groups, event_observed=E)
    pval = result.p_value

    cox_df = merged_clean[["OS.time", "OS", "cluster_id"]].copy()
    cox_df.columns = ["time", "event", "cluster"]

    if cox_df["cluster"].nunique() <= 1:
        print("❌ Not enough clusters for Cox regression.")
        return result.summary, merged_clean, None, None, cancer_type

    cox_df["cluster"] = cox_df["cluster"].astype("category")
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="time", event_col="event")

    hr_values = cph.summary["exp(coef)"].values
    representative_hr = np.mean(hr_values) if len(hr_values) else np.nan

    ci_widths = cph.summary["coef upper 95%"] - cph.summary["coef lower 95%"]
    representative_ci = ci_widths.mean() if len(ci_widths) else np.nan

    ax.set_title(f"GRAIL", fontsize=28)
    ax.set_xlabel("Time (Days)", fontsize=28)
    ax.set_ylabel("Survival probability", fontsize=28)
    ax.tick_params(axis='both', labelsize=28)
    ax.grid(False)

    text_x = 0.03
    text_y_start = 0.15
    text_y_step = 0.085

    ax.text(
        text_x, text_y_start + text_y_step,
        f"CI = {representative_ci:.3f}" if not np.isnan(representative_ci) else "CI = N/A",
        transform=ax.transAxes, fontsize=28
    )
    ax.text(
        text_x, text_y_start,
        f"HR = {representative_hr:.2f}" if not np.isnan(representative_hr) else "HR = N/A",
        transform=ax.transAxes, fontsize=28
    )
    ax.text(
        text_x, text_y_start - text_y_step,
        f"p-value = {pval:.4g}" if pval >= 0.0001 else f"p-value = {pval:.1e}",
        transform=ax.transAxes, fontsize=28
    )

    ax.legend(handles=handles, loc='upper right', fontsize=28, frameon=False, handlelength=1.0)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"survival_assigned_{cancer_type}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ KM plot saved: {plot_path}")

    cox_summary_path = os.path.join(output_dir, f"cox_summary_{cancer_type}.csv")
    cph.summary.to_csv(cox_summary_path)

    return result.summary, merged_clean, cox_summary_path, cph.summary, cluster_to_genes

def apply_full_spectral_biclustering_bio(
    graph, summary_bio_features, node_names_topk, omics_splits,
    predicted_cancer_genes,
    save_path, save_row_labels_path,
    save_total_genes_per_cluster_path, 
    save_predicted_counts_path,
    output_path_genes_clusters, 
    output_path_heatmap,
    best_k,
    output_dir,
    args,
    topk_node_indices=None
):
    # import torch
    # import numpy as np
    # from sklearn.metrics import mean_squared_error
    # from sklearn.cluster import SpectralBiclustering
    # from utils import (  # Replace with actual locations if needed
    #     save_graph_with_clusters, save_row_labels, compute_total_genes_per_cluster,
    #     save_total_genes_per_cluster, count_predicted_genes_per_cluster, save_predicted_counts
    # )
    # from plotting import (
    #     plot_bio_biclustering_heatmap_unsort,
    #     plot_bio_biclustering_clustermap,
    #     plot_predicted_genes_distribution
    # )
    # import os

    print("🧪 Running Spectral Biclustering with fixed (16, 10) clusters...")

    # === ✅ Step 1: Filter top-k nodes
    if topk_node_indices is None:
        raise ValueError("`topk_node_indices` must be provided")

    ##summary_bio_features_topk = summary_bio_features[topk_node_indices]
    summary_bio_features_topk = summary_bio_features  # Already top-k
    #node_names_topk = node_names
    #node_names_topk = [node_names[i] for i in topk_node_indices]

    assert summary_bio_features_topk.shape[1] == 64, f"Expected 64 summary features, got {summary_bio_features_topk.shape[1]}"

    # === ✅ Step 2: Run Biclustering
    n_clusters_row = best_k
    n_clusters_col = best_k

    best_model = None
    best_score = np.inf

    print("🔁 Running biclustering trials:")
    for i in range(10):
        model = SpectralBiclustering(n_clusters=(n_clusters_row, n_clusters_col), method='bistochastic',
                                     svd_method='randomized', random_state=i)
        
        model.fit(summary_bio_features_topk)

        reconstructed = summary_bio_features_topk[np.argsort(model.row_labels_)][:, np.argsort(model.column_labels_)]
        mse = mean_squared_error(summary_bio_features_topk, reconstructed)

        if mse < best_score:
            best_score = mse
            best_model = model

    bicluster = best_model
    row_labels = bicluster.row_labels_
    col_labels = bicluster.column_labels_

    # === ✅ Step 3: Assign row cluster labels back to graph
    row_labels_tensor = torch.full((graph.num_nodes(),), -1, dtype=torch.long)
    row_labels_tensor[topk_node_indices] = torch.tensor(row_labels, dtype=torch.long)
    graph.ndata['cluster_bio_summary'] = row_labels_tensor

    print("✅ Spectral Biclustering complete.")

    # === ✅ Step 4: Save clustering outputs
    save_graph_with_clusters(graph, save_path)
    save_row_labels(row_labels, save_row_labels_path)
    total_genes_per_cluster = compute_total_genes_per_cluster(row_labels, n_clusters_row)
    save_total_genes_per_cluster(total_genes_per_cluster, save_total_genes_per_cluster_path)

    pred_counts, predicted_indices = count_predicted_genes_per_cluster(
        row_labels, node_names_topk, predicted_cancer_genes, n_clusters_row
    )
    save_predicted_counts(pred_counts, save_predicted_counts_path)

    # === ✅ Step 5: Plot heatmaps and distributions
    plot_bio_biclustering_heatmap_unsort(
        args=args,
        relevance_scores=summary_bio_features_topk,
        omics_splits=omics_splits,
        output_path=os.path.join(output_dir, "heatmap_unsort.png"),
        row_labels=row_labels,
        col_labels=col_labels
    )

    plot_bio_biclustering_clustermap(
        args=args,
        relevance_scores=summary_bio_features_topk,
        omics_splits=omics_splits,
        output_path=os.path.join(output_dir, "clustermap.png"),
        row_labels=row_labels,
        col_labels=col_labels
    )

    plot_predicted_genes_distribution(
        pred_counts=pred_counts,
        output_path=os.path.join(output_dir, "predicted_genes_per_cluster.png")
    )

    return graph, row_labels, col_labels, total_genes_per_cluster, pred_counts

def plot_bio_biclustering_clustermap(
    args,
    relevance_scores,
    omics_splits,
    output_path,
    # cluster_colors,
    omics_colors=None,
    row_labels=None,
    col_labels=None,
    gene_names=None,
    top_k=10
):
    # === Normalize
    relevance_scores = normalize(relevance_scores, axis=1)
    saliency_matrix_norm = relevance_scores

    cluster_colors = {
        0: '#0077B6',   1: '#0000FF',   2: '#00B4D8',   3: '#48EAC4',
        4: '#F1C0E8',   5: '#B9FBC0',   6: '#32CD32',   7: '#bee1e6',
        8: '#8A2BE2',   9: '#E377C2',  10: '#8EECF5',  11: '#A3C4F3',
        12: '#FFB347', 13: '#FFD700',  14: '#FF69B4',  15: '#CD5C5C',
        16: '#7FFFD4', 17: '#FF7F50',  18: '#C71585',  19: '#20B2AA',
        20: '#6A5ACD', 21: '#40E0D0',  22: '#FF8C00',  23: '#DC143C',
        24: '#9ACD32'
    }
    omics_colors = {
        'CNA': '#9370DB', 'GE': '#228B22', 'METH': '#00008B', 'MF': '#b22222'
    }


    # === Feature names
    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'KidneyPap',
        'Liver', 'LungAd', 'LungSc', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    omics_order = ['CNA', 'GE', 'METH', 'MF']
    feature_names = [f"{omics}: {cancer}" for omics in omics_order for cancer in cancer_names]
    feature_names = feature_names[:relevance_scores.shape[1]]

    # === Reorder rows/columns
    if row_labels is not None:
        row_order = np.argsort(row_labels)
        relevance_scores = relevance_scores[row_order]
        row_labels = row_labels[row_order]
        if gene_names is not None:
            gene_names = [gene_names[i] for i in row_order]
    if col_labels is not None:
        col_order = np.argsort(col_labels)
        relevance_scores = relevance_scores[:, col_order]
        feature_names = [feature_names[i] for i in col_order]
        col_labels = col_labels[col_order]

    reordered_omics_labels = [f.split(":")[0].strip().upper() for f in feature_names]
    reordered_feature_labels = [f.split(":")[1].strip() for f in feature_names]

    # === Save top-k features
    if gene_names is not None:
        topk_dir = os.path.join(os.path.dirname(output_path), "topk_features_per_gene")
        os.makedirs(topk_dir, exist_ok=True)
        for i, gene in enumerate(gene_names):
            topk_indices = np.argsort(-relevance_scores[i])[:top_k]
            topk_features = [feature_names[j] for j in topk_indices]
            topk_scores = relevance_scores[i][topk_indices]
            df_gene = pd.DataFrame({"Feature": topk_features, "Score": topk_scores})
            df_gene.to_csv(os.path.join(topk_dir, f"{gene}_top{top_k}_features.csv"), index=False)

    # === Setup figure with 3 rows: omics bar, cluster bar, heatmap
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.5, 19], hspace=0.0)
    ##gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 18], hspace=0.05)


    # === Top bar: column cluster stripe
    ax_col_cluster = fig.add_subplot(gs[0])
    col_cluster_colors = [CLUSTER_COLORS.get(c, '#FFFFFF') for c in col_labels]
    col_cluster_rgb = np.array([[to_rgb(c) for c in col_cluster_colors]])
    ax_col_cluster.imshow(col_cluster_rgb, aspect='auto', extent=[0, len(col_cluster_colors), 0, 1])
    ax_col_cluster.set_xlim([0, len(col_cluster_colors)])
    ax_col_cluster.set_xticks([])
    ax_col_cluster.set_yticks([])
    ax_col_cluster.set_frame_on(False)

    # === Main heatmap
    ax = fig.add_subplot(gs[1])

    ###################################


    # === Main heatmap
    #ax = fig.add_subplot(gs[2])
    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
    vmin, vmax = 0, np.percentile(relevance_scores, 99)

    sns.heatmap(
        relevance_scores,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax
    )

    # === Row cluster color bars
    if row_labels is not None:
        for i, cluster in enumerate(row_labels):
            ax.add_patch(plt.Rectangle(
                (-1.5, i), 1.5, 1,
                linewidth=0,
                facecolor=to_rgba(cluster_colors.get(cluster, '#FFFFFF')),
                clip_on=False
            ))

        unique_clusters, cluster_sizes = np.unique(row_labels, return_counts=True)
        start_idx = 0
        for cluster, size in zip(unique_clusters, cluster_sizes):
            center_y = start_idx + size / 2
            ax.text(-2.0, center_y, f"{size}", va='center', ha='right', fontsize=22)#, fontweight='bold')
            start_idx += size

    # === Feature x-labels, color-coded by omics
    ax.set_xticks(np.arange(len(reordered_feature_labels)) + 0.5)
    ax.set_xticklabels(reordered_feature_labels, rotation=90, fontsize=24)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, length=5)
    for label, omics in zip(ax.get_xticklabels(), reordered_omics_labels):
        label.set_color(omics_colors.get(omics, 'black'))

    # === Finalize
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.subplots_adjust(hspace=0.0)
    plt.close()
    print(f"[Saved] Clustermap with top omics bar → {output_path}")

def plot_pcg_cancer_genes(
    clusters,
    pcg_count,
    total_genes_per_cluster,
    node_names,
    row_labels,
    output_path
):
    cluster_ids = sorted(clusters)

    # PCG count for each cluster
    pcgs = [
        pcg_count.get(c, 0)
        for c in cluster_ids
    ]

    # Total number of PCGs across all clusters
    total_pcgs = sum(pcgs)

    # Percentage of all PCGs located in each cluster
    proportions = [
        100 * p / total_pcgs if total_pcgs > 0 else 0
        for p in pcgs
    ]

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        cluster_ids,
        proportions,
        color=[
            CLUSTER_COLORS.get(c, '#333333')
            for c in cluster_ids
        ],
        edgecolor='black'
    )

    # Annotate each bar with the raw PCG count
    for bar, cluster_id in zip(bars, cluster_ids):
        height = bar.get_height()
        count = pcg_count.get(cluster_id, 0)

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(count),
            ha='center',
            va='bottom',
            fontsize=16,
            fontweight='bold'
        )

    ax = plt.gca()
    num_clusters = len(cluster_ids)

    ax.set_xlim(-0.55, num_clusters - 0.65)

    # Formatting
    plt.ylabel("Percent of PCGs", fontsize=20)
    plt.xlabel("")

    plt.xticks(
        cluster_ids,
        fontsize=16
    )

    plt.yticks(fontsize=16)

    plt.ylim(
        0,
        max(proportions) * 1.15
    )

    sns.despine()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
