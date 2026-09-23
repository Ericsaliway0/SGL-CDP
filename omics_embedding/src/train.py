import copy
import json
import os
import csv
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import dataset
import model, utils, network
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from py2neo import Graph, Node, Relationship
from neo4j import GraphDatabase
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import dgl



CLUSTER_COLORS = {
    0: '#0077B6',   1: '#0000FF',   2: '#00B4D8',   3: '#48EAC4',
    4: '#F1C0E8',   5: '#B9FBC0',   6: '#32CD32',   7: '#bee1e6',
    8: '#8A2BE2',   9: '#E377C2',  10: '#8EECF5',  11: '#A3C4F3',
    12: '#FFB347', 13: '#FFD700',  14: '#FF69B4',  15: '#CD5C5C',
    16: '#7FFFD4', 17: '#FF7F50',  18: '#C71585',  19: '#20B2AA',
    20: '#6A5ACD', 21: '#40E0D0',  22: '#FF8C00',  23: '#DC143C',
    24: '#9ACD32'
}



CLUSTER_COLORS_OMICS = {
    0: '#D62728',  1: '#1F77B4',  2: '#2CA02C',  3: '#9467BD'
}



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure the input and target have the same shape
        if inputs.dim() > targets.dim():
            inputs = inputs.squeeze(dim=-1)
        elif targets.dim() > inputs.dim():
            targets = targets.squeeze(dim=-1)

        # Check if the shapes match after squeezing
        if inputs.size() != targets.size():
            raise ValueError(f"Target size ({targets.size()}) must be the same as input size ({inputs.size()})")

        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def train_ori(hyperparams=None, data_path='../data/omics/', plot=True, omics='cna', cancer='BLCA'):
    num_epochs = hyperparams['num_epochs']
    ##feat_drop = hyperparams['feat_drop']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']
    
    model_path = os.path.join(data_path, omics, cancer, 'emb/models')
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(model_path, f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth')
    
    '''omics_types = ['cna', 'ge', 'meth', 'mf']
    cancer_types = ['KIRC', 'BRCA', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']

    for omics in omics_types:
        for cancer in cancer_types:    
            ##data_path = os.path.join(data_path, 'processed', omics, cancer)'''
            
    data_path_ = os.path.join(data_path, omics, cancer)##, 'emb/processed')
    ds = dataset.Dataset(data_path_)
    graph, name = ds[0]

    print("\n========== SIGNIFICANCE DEBUG ==========")
    print("dtype:", graph.ndata['significance'].dtype)
    print("shape:", graph.ndata['significance'].shape)
    print(
        "unique:",
        torch.unique(
            graph.ndata['significance'],
            return_counts=True
        )
    )
    print("first 20:", graph.ndata['significance'][:20])
    print("========================================\n")

    ds_train = [ds[0]]
    ds_valid = [ds[1]]
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)
    
    # Create the TAGCN model instance
    net = model.TAGCNModel(dim_latent=out_feats, num_layers=num_layers, do_train=True).to(device)

    # Set up the optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Save the best model
    best_model = model.TAGCNModel(dim_latent=out_feats, num_layers=num_layers, do_train=True).to(device)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    loss_per_epoch_train, loss_per_epoch_valid = [], []
    f1_per_epoch_train, f1_per_epoch_valid = [], []

    criterion = FocalLoss(
        alpha=0.25,
        gamma=2.0,
        reduction='mean'
    )

    loss = criterion(logits, labels)
    ##criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    # weight = torch.tensor([0.00001, 0.99999]).to(device)
    # # Temporarily:
    # weight = torch.tensor([1.0, 1.0]).to(device)

    best_train_loss, best_valid_loss = float('inf'), float('inf')
    best_f1_score = 0.0

    max_f1_scores_train = []
    max_f1_scores_valid = []
    
    results_path = os.path.abspath(os.path.join('results', 'multiomics_meth', 'node_embeddings', omics, cancer))
    os.makedirs(results_path, exist_ok=True)

    all_embeddings_initial, cluster_labels_initial = calculate_cluster_labels(best_model, dl_train, device)
    ##print('all_embeddings_initial---------------------------------\n', all_embeddings_initial)
    all_embeddings_initial = all_embeddings_initial.reshape(all_embeddings_initial.shape[0], -1)  # Flatten 
    save_path_heatmap_initial= os.path.join(results_path, f'embeddings_heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_matrix_initial= os.path.join(results_path, f'embeddings_matrix_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_pca_initial = os.path.join(results_path, f'embeddings_pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_t_SNE_initial = os.path.join(results_path, f'embeddings_t-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
        
    for data in dl_train:
        graph, _ = data
        node_embeddings_initial= best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, omics, cancer, 'emb/raw', 'emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        assert len(cluster_labels_initial) == len(nx_graph.nodes), "Cluster labels and number of nodes must match"
        node_to_index_initial = {node: idx for idx, node in enumerate(nx_graph.nodes)}
        first_node_stId_in_cluster_initial= {}
        first_node_embedding_in_cluster_initial= {}

        stid_dic_initial= {}

        # Populate stid_dic with node stIds mapped to embeddings
        for node in nx_graph.nodes:
            if 'stId' in nx_graph.nodes[node]:
                stId = nx_graph.nodes[node]['stId']
                stid_dic_initial[nx_graph.nodes[node]['stId']] = node_embeddings_initial[node_to_index_initial[node]]

        # Convert stid_dic_initial to a DataFrame
        stid_df_initial = pd.DataFrame.from_dict(stid_dic_initial, orient='index')

        # Save to CSV
        ##csv_save_path = 'gat/data/gene_embeddings_initial_sage.csv'
        csv_save_path_initial = os.path.join(results_path, f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.csv')
        stid_df_initial.to_csv(csv_save_path_initial, index_label='stId')
                
        ##print('stid_dic_initial=======================\n',stid_dic_initial) 
        
        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if 'stId' in nx_graph.nodes[node]:
                if cluster not in first_node_stId_in_cluster_initial:
                    first_node_stId_in_cluster_initial[cluster] = nx_graph.nodes[node]['stId']
                    first_node_embedding_in_cluster_initial[cluster] = node_embeddings_initial[node_to_index_initial[node]]

        print('first_node_stId_in_cluster_initial-------------------------------\n', first_node_stId_in_cluster_initial)
        cluster_stid_map = first_node_stId_in_cluster_initial

        stid_list = [
            cluster_stid_map[c]
            for c in sorted(cluster_stid_map)
        ]
        embedding_list_initial = list(first_node_embedding_in_cluster_initial.values())

        # Use the cluster representatives generated from the current embedding
        cluster_stid_map_initial = first_node_stId_in_cluster_initial

        # Sort clusters explicitly so that stIds and embeddings have
        # exactly the same cluster ordering
        sorted_clusters_initial = sorted(cluster_stid_map_initial.keys())

        stid_list_initial = [
            first_node_stId_in_cluster_initial[c]
            for c in sorted_clusters_initial
        ]

        embedding_list_initial = [
            first_node_embedding_in_cluster_initial[c]
            for c in sorted_clusters_initial
        ]

        # Convert embeddings to 1-D vectors
        embedding_list_initial = [
            np.asarray(emb).reshape(-1)
            for emb in embedding_list_initial
        ]

        heatmap_data_initial = pd.DataFrame(
            embedding_list_initial,
            index=stid_list_initial
        )

        print(f"Number of clusters: {len(sorted_clusters_initial)}")
        print(f"Embedding matrix shape: {heatmap_data_initial.shape}")




        create_heatmap_with_stid(embedding_list_initial, stid_list, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list_initial, stid_list, save_path_matrix_initial)

        break

    visualize_embeddings_tsne(all_embeddings_initial, cluster_labels_initial, stid_list, save_path_t_SNE_initial)
    # visualize_embeddings_pca(all_embeddings_initial, cluster_labels_initial, stid_list, save_path_pca_initial)
    visualize_embeddings_pca(
        all_embeddings_initial,
        cluster_labels_initial,
        first_node_stId_in_cluster_initial,
        save_path_pca_initial
    )

    silhouette_avg_ = silhouette_score(all_embeddings_initial, cluster_labels_initial)
    davies_bouldin_ = davies_bouldin_score(all_embeddings_initial, cluster_labels_initial)
    summary_  = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_= os.path.join(results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt')
    with open(save_file_, 'w') as f:
        f.write(summary_)

    # Start training  
    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=False) as pbar:
        for epoch in range(num_epochs):
            loss_per_graph = []
            f1_per_graph = [] 
            net.train()
            for data in dl_train:
                graph, name = data
                name = name[0]
                logits = net(graph)
                labels = graph.ndata['significance'].unsqueeze(-1)
                weight_ = weight[labels.data.view(-1).long()].view_as(labels)

                loss = criterion(logits, labels)
                loss_weighted = loss * weight_
                loss_weighted = loss_weighted.mean()

                # Update parameters
                optimizer.zero_grad()
                loss_weighted.backward()
                optimizer.step()
                
                # Append output metrics
                loss_per_graph.append(loss_weighted.item())
                ##preds = (logits.sigmoid() > 0.5).squeeze(1).int()
                # preds = (logits.sigmoid() > 0.5).int()
                # labels = labels.squeeze(1).int()
                # f1 = metrics.f1_score(labels, preds)
                # f1_per_graph.append(f1)

                probs = torch.sigmoid(logits).detach()
                labels = labels.squeeze(1).int()

                print("========================================")
                print("LABEL CHECK")
                print("labels shape:", labels.shape)
                print("labels dtype:", labels.dtype)
                print("unique labels:", torch.unique(labels, return_counts=True))
                print("========================================")

                print("PROB CHECK")
                print("prob min:", probs.min().item())
                print("prob max:", probs.max().item())
                print("prob mean:", probs.mean().item())
                print("========================================")

                # Test different thresholds
                # for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                #     preds_tmp = (probs > threshold).int()

                #     f1_tmp = metrics.f1_score(
                #         labels.detach().cpu().numpy(),
                #         preds_tmp.detach().cpu().numpy(),
                #         zero_division=0
                #     )

                #     print(
                #         f"threshold={threshold:.1f}, "
                #         f"predicted_positive={(preds_tmp == 1).sum().item()}, "
                #         f"F1={f1_tmp:.4f}"
                #     )
                for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:

                    preds_tmp = (probs > threshold).int()

                    print(
                        f"threshold={threshold:.1f} | "
                        f"true positives={(labels == 1).sum().item()} | "
                        f"true negatives={(labels == 0).sum().item()} | "
                        f"predicted positives={(preds_tmp == 1).sum().item()} | "
                        f"predicted negatives={(preds_tmp == 0).sum().item()}"
                    )

                    f1_tmp = metrics.f1_score(
                        labels.cpu().numpy(),
                        preds_tmp.cpu().numpy(),
                        zero_division=0
                    )

                    print(f"F1={f1_tmp:.4f}")
                    
                # Keep 0.5 as your normal F1 for now
                preds = (probs > 0.5).int()

                f1 = metrics.f1_score(
                    labels.detach().cpu().numpy(),
                    preds.detach().cpu().numpy(),
                    zero_division=0
                )

                f1_per_graph.append(f1)

            running_loss = np.array(loss_per_graph).mean()
            running_f1_train = np.array(f1_per_graph).mean()
            loss_per_epoch_train.append(running_loss)
            f1_per_epoch_train.append(running_f1_train)

            # Validation iteration
            with torch.no_grad():
                loss_per_graph = []
                f1_per_graph = []
                net.eval()
                for data in dl_valid:
                    graph, name = data
                    name = name[0]
                    logits = net(graph)
                    labels = graph.ndata['significance'].unsqueeze(-1)
                    weight_ = weight[labels.data.view(-1).long()].view_as(labels)
                    loss = criterion(logits, labels)
                    loss_weighted = loss * weight_
                    loss_weighted = loss_weighted.mean()
                    loss_per_graph.append(loss_weighted.item())
                    ##preds = (logits.sigmoid() > 0.5).squeeze(1).int()
                    # preds = (logits.sigmoid() > 0.5).int()
                    # labels = labels.squeeze(1).int()
                    # f1 = metrics.f1_score(labels, preds)
                    # f1_per_graph.append(f1)

                    probs = torch.sigmoid(logits).detach()
                    labels = labels.squeeze(1).int()

                    # Test different thresholds
                    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                        preds_tmp = (probs > threshold).int()

                        f1_tmp = metrics.f1_score(
                            labels.detach().cpu().numpy(),
                            preds_tmp.detach().cpu().numpy(),
                            zero_division=0
                        )

                        print(
                            f"threshold={threshold:.1f}, "
                            f"predicted_positive={(preds_tmp == 1).sum().item()}, "
                            f"F1={f1_tmp:.4f}"
                        )

                    # Keep 0.5 as your normal F1 for now
                    preds = (probs > 0.5).int()

                    f1 = metrics.f1_score(
                        labels.detach().cpu().numpy(),
                        preds.detach().cpu().numpy(),
                        zero_division=0
                    )

                    f1_per_graph.append(f1)

                running_loss = np.array(loss_per_graph).mean()
                running_f1_val = np.array(f1_per_graph).mean()
                loss_per_epoch_valid.append(running_loss)
                f1_per_epoch_valid.append(running_f1_val)
                
                max_f1_train = max(f1_per_epoch_train)
                max_f1_valid = max(f1_per_epoch_valid)
                max_f1_scores_train.append(max_f1_train)
                max_f1_scores_valid.append(max_f1_valid)

                if running_loss < best_valid_loss:
                    best_train_loss = running_loss
                    best_valid_loss = running_loss
                    best_f1_score = running_f1_val
                    best_model.load_state_dict(copy.deepcopy(net.state_dict()))
                    print(f"Best F1 Validation Score: {best_f1_score}")

            pbar.update(1)
            print(f"Epoch {epoch + 1} - F1 Train: {running_f1_train}, F1 Valid: {running_f1_val}")
            ## print(f"Epoch {epoch + 1} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}")

    all_embeddings, cluster_labels = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)  # Flatten 
    ##print('cluster_labels=========================\n', cluster_labels)

    cos_sim = np.dot(all_embeddings, all_embeddings.T)
    norms = np.linalg.norm(all_embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    if plot:
        loss_path = os.path.join(results_path, f'embeddings_loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        f1_path = os.path.join(results_path, f'embeddings_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        max_f1_path = os.path.join(results_path, f'embeddings_max_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        matrix_path = os.path.join(results_path, f'embeddings_matrix_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_max_f1_plot(max_f1_scores_train, max_f1_scores_valid, max_f1_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

    torch.save(best_model.state_dict(), model_path)

    save_path_pca = os.path.join(results_path, f'embeddings_pca_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png')
    save_path_t_SNE = os.path.join(results_path, f'embeddings_t-SNE_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png')
    save_path_heatmap_= os.path.join(results_path, f'embeddings_heatmap_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png')
    save_path_matrix = os.path.join(results_path, f'embeddings_matrix_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png')
    
    cluster_stId_dict = {}  # Dictionary to store clusters and corresponding stIds
    significant_stIds = []  # List to store significant stIds
    clusters_with_significant_stId = {}  # Dictionary to store clusters and corresponding significant stIds
    clusters_node_info = {}  # Dictionary to store node info for each cluster
    
    for data in dl_train:
        graph, _ = data
        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, omics, cancer, 'emb/raw', 'emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        assert len(cluster_labels) == len(nx_graph.nodes), "Cluster labels and number of nodes must match"
        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}
        first_node_stId_in_cluster = {}
        first_node_embedding_in_cluster = {}

        stid_dic = {}

        # Populate stid_dic with node stIds mapped to embeddings
        for node in nx_graph.nodes:
            if 'stId' in nx_graph.nodes[node]:
                stid = nx_graph.nodes[node]['stId']
                stid_dic[nx_graph.nodes[node]['stId']] = node_embeddings[node_to_index[node]]
                # Check if the node's significance is 'significant' and add its stId to the list
                if graph.ndata['significance'][node_to_index[node]].item() == 'significant':
                    significant_stIds.append(nx_graph.nodes[node]['stId'])

        # Convert stid_dic_initial to a DataFrame
        stid_df_final = pd.DataFrame.from_dict(stid_dic, orient='index')

        # Save to CSV
        ##csv_save_path = 'gat/data/gene_embeddings_final_sage.csv'csv_save_path_final = os.path.join(results_path, f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.csv')
        csv_save_path_final = os.path.join(results_path, f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.csv')
                        
        # csv_save_path_final = os.path.join('data/', omics, cancer, f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.csv')
        stid_df_final.to_csv(csv_save_path_final, index_label='stId')
                
        for node, cluster in zip(nx_graph.nodes, cluster_labels):
            if 'stId' in nx_graph.nodes[node]:
                stid = nx_graph.nodes[node]['stId']
                if cluster not in first_node_stId_in_cluster:
                    first_node_stId_in_cluster[cluster] = nx_graph.nodes[node]['stId']
                    first_node_embedding_in_cluster[cluster] = node_embeddings[node_to_index[node]]
                    
                # Populate cluster_stId_dict
                if cluster not in cluster_stId_dict:
                    cluster_stId_dict[cluster] = []
                cluster_stId_dict[cluster].append(nx_graph.nodes[node]['stId'])

                # Populate clusters_with_significant_stId
                if cluster not in clusters_with_significant_stId:
                    clusters_with_significant_stId[cluster] = []
                if nx_graph.nodes[node]['stId'] in significant_stIds:
                    clusters_with_significant_stId[cluster].append(nx_graph.nodes[node]['stId'])
                
                # Populate clusters_node_info with node information for each cluster
                if cluster not in clusters_node_info:
                    clusters_node_info[cluster] = []
                node_info = {
                    'stId': nx_graph.nodes[node]['stId'],
                    'significance': graph.ndata['significance'][node_to_index[node]].item(),
                    'other_info': nx_graph.nodes[node]  # Add other relevant info if necessary
                }
                clusters_node_info[cluster].append(node_info)
            
        print(first_node_stId_in_cluster)
        # stid_list = list(first_node_stId_in_cluster.values())
        # cluster_stid_map_final = first_node_stId_in_cluster

        # stid_list = [
        #     cluster_stid_map[c]
        #     for c in sorted(cluster_stid_map_final)
        # ]


        # embedding_list = list(first_node_embedding_in_cluster.values())
        # heatmap_data = pd.DataFrame(embedding_list, index=stid_list)

        # Use the cluster representatives generated from the current embedding
        cluster_stid_map_final = first_node_stId_in_cluster

        # Sort clusters explicitly so that stIds and embeddings have
        # exactly the same cluster ordering
        sorted_clusters = sorted(cluster_stid_map_final.keys())

        stid_list = [
            first_node_stId_in_cluster[c]
            for c in sorted_clusters
        ]

        embedding_list = [
            first_node_embedding_in_cluster[c]
            for c in sorted_clusters
        ]

        # Convert embeddings to 1-D vectors
        embedding_list = [
            np.asarray(emb).reshape(-1)
            for emb in embedding_list
        ]

        heatmap_data = pd.DataFrame(
            embedding_list,
            index=stid_list
        )

        print(f"Number of clusters: {len(sorted_clusters)}")
        print(f"Embedding matrix shape: {heatmap_data.shape}")


        create_heatmap_with_stid(embedding_list, stid_list, save_path_heatmap_)
        # Call the function to plot cosine similarity matrix for cluster representatives with similarity values
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list, stid_list, save_path_matrix)

        break

    visualize_embeddings_tsne(all_embeddings, cluster_labels, stid_list, save_path_t_SNE)
    # visualize_embeddings_pca(all_embeddings, cluster_labels, stid_list, save_path_pca)
    visualize_embeddings_pca(
        all_embeddings,
        cluster_labels,
        first_node_stId_in_cluster,
        save_path_pca
    )

    silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
    davies_bouldin = davies_bouldin_score(all_embeddings, cluster_labels)

    print(f"Silhouette Score%%%%%%%%%%%%###########################: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    summary = f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {max_f1_train}\n"
    summary += f"Silhouette Score: {silhouette_avg}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin}\n"

    save_file = os.path.join(results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt')
    with open(save_file, 'w') as f:
        f.write(summary)
    return model_path

def train_ori_pas(
    hyperparams=None,
    data_path='../data/omics/',
    plot=True,
    omics='cna',
    cancer='BLCA'
):
    """
    Stage-1 training:
        Predict generic biological significance from the graph.

    IMPORTANT:
        This stage does NOT use cancer-driver labels.
        'significance' is the Stage-1 pretraining target.

    Expected significance values:
        0 = not significant
        1 = significant

    If -1 is present:
        -1 = unlabeled / ignored during loss and F1 calculation
    """

    import os
    import copy
    import pickle

    import numpy as np
    import pandas as pd
    import torch
    import torch.optim as optim
    from tqdm import tqdm

    from sklearn import metrics
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score
    )

    # ============================================================
    # 1. Hyperparameters
    # ============================================================

    if hyperparams is None:
        raise ValueError("hyperparams must be provided.")

    num_epochs = hyperparams['num_epochs']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']

    print("\n" + "=" * 70)
    print("START STAGE-1 SIGNIFICANCE PRETRAINING")
    print("=" * 70)
    print(f"Omics       : {omics}")
    print(f"Cancer      : {cancer}")
    print(f"Device      : {device}")
    print(f"Epochs      : {num_epochs}")
    print(f"LR          : {learning_rate}")
    print(f"Batch size  : {batch_size}")
    print(f"Out feats   : {out_feats}")
    print(f"Layers      : {num_layers}")
    print("=" * 70 + "\n")

    # ============================================================
    # 2. Model save path
    # ============================================================

    model_path = os.path.join(
        data_path,
        omics,
        cancer,
        'emb',
        'models'
    )

    os.makedirs(model_path, exist_ok=True)

    model_path = os.path.join(
        model_path,
        f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth'
    )

    # ============================================================
    # 3. Load dataset
    # ============================================================

    # data_path_ = os.path.join(
    #     data_path,
    #     omics,
    #     cancer
    # )

    # ds = dataset.Dataset(data_path_)

    # graph, name = ds[0]

    # ============================================================
    # 3. Load dataset
    # ============================================================

    data_path_ = os.path.join(
        data_path,
        omics,
        cancer
    )

    ds = dataset.Dataset(data_path_)

    # Loop through both the training (0) and validation (1) graphs 
    # to add node degree features to both datasets

    # ============================================================
    # 3. Load dataset
    # ============================================================

    data_path_ = os.path.join(
        data_path,
        omics,
        cancer
    )

    ds = dataset.Dataset(data_path_)

    # Loop through both the training (0) and validation (1) graphs 
    # to add node degree features to both datasets
    for idx in [0, 1]:
        g, _ = ds[idx]
        
        # 1. Extract the node structural degrees and force to 2D matrix [nodes, 1]
        node_degrees = g.in_degrees().float().unsqueeze(-1)
        node_degrees_norm = (node_degrees - node_degrees.mean()) / (node_degrees.std() + 1e-5)
        
        # 2. Safely find existing node features (checking 'weight' or 'feat')
        if 'weight' in g.ndata:
            existing_feats = g.ndata['weight']
        elif 'feat' in g.ndata:
            existing_feats = g.ndata['feat']
        else:
            # Fallback if the graph has absolutely no node feature metrics initialized yet
            existing_feats = torch.ones((g.num_nodes(), 1), dtype=torch.float32)
            
        # --- FIXED LAYER: Force existing features to be a 2D matrix [nodes, features] ---
        if existing_feats.dim() == 1:
            existing_feats = existing_feats.unsqueeze(-1)
        # ---------------------------------------------------------------------------------
            
        # 3. Concatenate structural degrees to existing features (now both are 2D tensors)
        updated_feats = torch.cat([existing_feats, node_degrees_norm], dim=-1)
        
        # 4. Set both attributes so downstream layers or validation loops won't crash
        g.ndata['feat'] = updated_feats
        g.ndata['weight'] = updated_feats


    # --- FIXED LAYER: Explicitly request index 0 instead of unpacking the full object ---
    graph, name = ds[0]


    # for idx in [0, 1]:
    #     g, _ = ds[idx]
        
    #     # 1. Extract the node structural degrees
    #     node_degrees = g.in_degrees().float().unsqueeze(-1)
    #     node_degrees_norm = (node_degrees - node_degrees.mean()) / (node_degrees.std() + 1e-5)
        
    #     # 2. Safely find existing node features (checking 'weight' or 'feat')
    #     if 'weight' in g.ndata:
    #         existing_feats = g.ndata['weight']
    #     elif 'feat' in g.ndata:
    #         existing_feats = g.ndata['feat']
    #     else:
    #         # Fallback if the graph has absolutely no node feature metrics initialized yet
    #         existing_feats = torch.ones((g.num_nodes(), 1), dtype=torch.float32)
            
    #     # 3. Concatenate structural degrees to existing features
    #     updated_feats = torch.cat([existing_feats, node_degrees_norm], dim=-1)
        
    #     # 4. Set both attributes so downstream layers or validation loops won't crash
    #     g.ndata['feat'] = updated_feats
    #     g.ndata['weight'] = updated_feats

    # # Pull out the primary graph reference for the diagnostic checks below
    # graph, name = ds[0]


    # # Extract the structural degree array of all nodes in DGL graph
    # node_degrees = graph.in_degrees().float().unsqueeze(-1)

    # # Normalize the values to prevent massive gradient spikes
    # node_degrees_norm = (node_degrees - node_degrees.mean()) / (node_degrees.std() + 1e-5)

    # # Append this new structural layout directly onto your node data tensors
    # graph.ndata['feat'] = torch.cat([graph.ndata['feat'], node_degrees_norm], dim=-1)


    # ============================================================
    # 4. SIGNIFICANCE LABEL DEBUG
    # ============================================================

    if 'significance' not in graph.ndata:
        raise KeyError(
            "graph.ndata['significance'] does not exist. "
            "Check your preprocessing and DGL graph conversion."
        )

    significance = graph.ndata['significance']

    print("\n" + "=" * 70)
    print("SIGNIFICANCE DEBUG")
    print("=" * 70)

    print("dtype :", significance.dtype)
    print("shape :", significance.shape)

    unique_labels, label_counts = torch.unique(
        significance,
        return_counts=True
    )

    print("unique labels:")

    for label, count in zip(unique_labels, label_counts):
        print(
            f"    label={label.item():>4} "
            f"count={count.item():>6}"
        )

    print("first 20:", significance[:20])
    print("=" * 70)

    # ------------------------------------------------------------
    # Check positive and negative labels
    # ------------------------------------------------------------

    valid_labels = significance[
        (significance == 0) |
        (significance == 1)
    ]

    if valid_labels.numel() == 0:
        raise ValueError(
            "No valid significance labels (0/1) were found."
        )

    num_positive = (valid_labels == 1).sum().item()
    num_negative = (valid_labels == 0).sum().item()
    num_unlabeled = (significance == -1).sum().item()

    print("\nLABEL DISTRIBUTION")
    print(f"Positive / significant : {num_positive}")
    print(f"Negative / non-signif. : {num_negative}")
    print(f"Unlabeled (-1)         : {num_unlabeled}")

    if num_positive == 0:
        raise ValueError(
            "There are ZERO positive significance labels. "
            "F1 will necessarily be 0."
        )

    if num_negative == 0:
        raise ValueError(
            "There are ZERO negative significance labels. "
            "Check your preprocessing."
        )

    positive_ratio = num_positive / valid_labels.numel()

    print(f"Positive ratio          : {positive_ratio:.4%}")

    # ============================================================
    # 5. Train / validation graphs
    # ============================================================

    ds_train = [ds[0]]
    ds_valid = [ds[1]]

    dl_train = GraphDataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True
    )

    dl_valid = GraphDataLoader(
        ds_valid,
        batch_size=batch_size,
        shuffle=False
    )
    

    # ============================================================
    # 6. Create model
    # ============================================================

    net = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)

    # ============================================================
    # 7. Optimizer
    # ============================================================

    optimizer = optim.Adam(
        net.parameters(),
        lr=learning_rate
    )

    # ============================================================
    # 8. Best model
    # ============================================================

    best_model = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)

    best_model.load_state_dict(
        copy.deepcopy(net.state_dict())
    )

    # # ============================================================
    # # 9. Focal Loss
    # # ============================================================

    # criterion = FocalLoss(
    #     alpha=0.25,
    #     gamma=2.0,
    #     reduction='mean'
    # )

    # ============================================================
    # 9. Focal Loss (With Dynamic Class Balancing)
    # ============================================================
    
    # Dynamically compute balance: alpha should favor the rare positive class
    # Calculate ratio of negatives to total valid samples
    total_valid = num_positive + num_negative
    dynamic_alpha = num_negative / total_valid # e.g., 1333 / 1607 ≈ 0.83
    
    print(f"Configuring Focal Loss with Dynamic Alpha: {dynamic_alpha:.4f} (favoring positive class)")

    criterion = FocalLoss(
        alpha=dynamic_alpha, # <-- Replaced hardcoded 0.25 with balanced inverse ratio
        gamma=2.0,
        reduction='mean'
    )


    # IMPORTANT:
    #
    # Do NOT use:
    #
    # weight = torch.tensor([0.00001, 0.99999])
    #
    # That almost completely suppresses class 0.
    #
    # Also, because FocalLoss(reduction='mean') already returns
    # a scalar, multiplying it by node-level weights is incorrect.
    #
    # Therefore, we use FocalLoss directly here.

    # ============================================================
    # 10. Training history
    # ============================================================

    loss_per_epoch_train = []
    loss_per_epoch_valid = []

    f1_per_epoch_train = []
    f1_per_epoch_valid = []

    max_f1_scores_train = []
    max_f1_scores_valid = []

    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_f1_score = 0.0
    best_epoch = 0

    # ============================================================
    # 11. Results path
    # ============================================================

    results_path = os.path.abspath(
        os.path.join(
            'results',
            'multiomics_meth',
            'node_embeddings',
            omics,
            cancer
        )
    )

    os.makedirs(
        results_path,
        exist_ok=True
    )

    # ============================================================
    # 12. INITIAL EMBEDDINGS
    # ============================================================

    print("\n" + "=" * 70)
    print("CALCULATING INITIAL EMBEDDINGS")
    print("=" * 70)

    all_embeddings_initial, cluster_labels_initial = (
        calculate_cluster_labels(
            best_model,
            dl_train,
            device
        )
    )

    all_embeddings_initial = (
        all_embeddings_initial.reshape(
            all_embeddings_initial.shape[0],
            -1
        )
    )

    save_path_heatmap_initial = os.path.join(
        results_path,
        f'embeddings_heatmap_stId_dim{out_feats}_'
        f'lay{num_layers}_epo{num_epochs}_initial.png'
    )

    save_path_matrix_initial = os.path.join(
        results_path,
        f'embeddings_matrix_stId_dim{out_feats}_'
        f'lay{num_layers}_epo{num_epochs}_initial.png'
    )

    save_path_pca_initial = os.path.join(
        results_path,
        f'embeddings_pca_dim{out_feats}_'
        f'lay{num_layers}_epo{num_epochs}_initial.png'
    )

    save_path_t_SNE_initial = os.path.join(
        results_path,
        f'embeddings_t-SNE_dim{out_feats}_'
        f'lay{num_layers}_epo{num_epochs}_initial.png'
    )

    # ============================================================
    # 13. INITIAL EMBEDDING INFORMATION
    # ============================================================

    for data in dl_train:

        graph, _ = data

        graph = graph.to(device)

        node_embeddings_initial = (
            best_model
            .get_node_embeddings(graph)
            .detach()
            .cpu()
            .numpy()
        )

        graph_path = os.path.join(
            data_path,
            omics,
            cancer,
            'emb',
            'raw',
            'emb_train.pkl'
        )

        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert (
            len(cluster_labels_initial)
            == len(nx_graph.nodes)
        ), (
            "Cluster labels and number of nodes must match"
        )

        node_to_index_initial = {
            node: idx
            for idx, node in enumerate(nx_graph.nodes)
        }

        first_node_stId_in_cluster_initial = {}
        first_node_embedding_in_cluster_initial = {}

        stid_dic_initial = {}

        for node in nx_graph.nodes:

            if 'stId' in nx_graph.nodes[node]:

                stId = nx_graph.nodes[node]['stId']

                stid_dic_initial[stId] = (
                    node_embeddings_initial[
                        node_to_index_initial[node]
                    ]
                )

        stid_df_initial = pd.DataFrame.from_dict(
            stid_dic_initial,
            orient='index'
        )

        csv_save_path_initial = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_'
            f'dim{out_feats}_lay{num_layers}_'
            f'epo{num_epochs}_initial.csv'
        )

        stid_df_initial.to_csv(
            csv_save_path_initial,
            index_label='stId'
        )

        for node, cluster in zip(
            nx_graph.nodes,
            cluster_labels_initial
        ):

            if 'stId' in nx_graph.nodes[node]:

                if cluster not in (
                    first_node_stId_in_cluster_initial
                ):

                    first_node_stId_in_cluster_initial[
                        cluster
                    ] = nx_graph.nodes[node]['stId']

                    first_node_embedding_in_cluster_initial[
                        cluster
                    ] = node_embeddings_initial[
                        node_to_index_initial[node]
                    ]

        print(
            "first_node_stId_in_cluster_initial:"
        )
        print(
            first_node_stId_in_cluster_initial
        )

        sorted_clusters_initial = sorted(
            first_node_stId_in_cluster_initial.keys()
        )

        stid_list_initial = [
            first_node_stId_in_cluster_initial[c]
            for c in sorted_clusters_initial
        ]

        embedding_list_initial = [
            first_node_embedding_in_cluster_initial[c]
            for c in sorted_clusters_initial
        ]

        embedding_list_initial = [
            np.asarray(emb).reshape(-1)
            for emb in embedding_list_initial
        ]

        heatmap_data_initial = pd.DataFrame(
            embedding_list_initial,
            index=stid_list_initial
        )

        print(
            f"Number of clusters: "
            f"{len(sorted_clusters_initial)}"
        )

        print(
            f"Embedding matrix shape: "
            f"{heatmap_data_initial.shape}"
        )

        create_heatmap_with_stid(
            embedding_list_initial,
            stid_list_initial,
            save_path_heatmap_initial
        )

        plot_cosine_similarity_matrix_for_clusters_with_values(
            embedding_list_initial,
            stid_list_initial,
            save_path_matrix_initial
        )

        break

    # ============================================================
    # 14. INITIAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(
        all_embeddings_initial,
        cluster_labels_initial,
        stid_list_initial,
        save_path_t_SNE_initial
    )

    visualize_embeddings_pca(
        all_embeddings_initial,
        cluster_labels_initial,
        first_node_stId_in_cluster_initial,
        save_path_pca_initial
    )

    # ------------------------------------------------------------
    # Initial clustering metrics
    # ------------------------------------------------------------

    if len(np.unique(cluster_labels_initial)) > 1:

        silhouette_avg_ = silhouette_score(
            all_embeddings_initial,
            cluster_labels_initial
        )

        davies_bouldin_ = davies_bouldin_score(
            all_embeddings_initial,
            cluster_labels_initial
        )

    else:

        silhouette_avg_ = np.nan
        davies_bouldin_ = np.nan

    summary_ = (
        f"Silhouette Score: {silhouette_avg_}\n"
    )

    summary_ += (
        f"Davies-Bouldin Index: "
        f"{davies_bouldin_}\n"
    )

    save_file_ = os.path.join(
        results_path,
        f'embeddings_head{num_heads}_'
        f'dim{out_feats}_lay{num_layers}_'
        f'epo{num_epochs}_initial.txt'
    )

    with open(save_file_, 'w') as f:
        f.write(summary_)

    # ============================================================
    # 15. TRAINING
    # ============================================================

    print("\n" + "=" * 70)
    print("START TRAINING")
    print("=" * 70)

    with tqdm(
        total=num_epochs,
        desc="Training",
        unit="epoch",
        leave=True
    ) as pbar:

        for epoch in range(num_epochs):

            # ====================================================
            # TRAIN
            # ====================================================

            net.train()

            loss_per_graph = []
            f1_per_graph = []

            for data in dl_train:

                graph, name = data

                graph = graph.to(device)

                # ------------------------------------------------
                # Forward
                # ------------------------------------------------

                logits = net(graph)

                labels = (
                    graph.ndata['significance']
                    .float()
                    .view(-1)
                )

                logits = logits.view(-1)

                # ------------------------------------------------
                # Valid labels
                #
                # 0 = negative
                # 1 = positive
                # -1 = ignored
                # ------------------------------------------------

                valid_mask = (
                    (labels == 0) |
                    (labels == 1)
                )

                if valid_mask.sum().item() == 0:
                    raise ValueError(
                        "Training graph contains no valid "
                        "significance labels."
                    )

                valid_logits = logits[valid_mask]
                valid_labels = labels[valid_mask]

                # ------------------------------------------------
                # Loss
                # ------------------------------------------------

                loss = criterion(
                    valid_logits,
                    valid_labels
                )

                # ------------------------------------------------
                # Backpropagation
                # ------------------------------------------------

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                loss_per_graph.append(
                    loss.item()
                )

                # ------------------------------------------------
                # Predictions
                # ------------------------------------------------

                with torch.no_grad():

                    probs = torch.sigmoid(
                        valid_logits
                    )

                    preds = (
                        probs > 0.5
                    ).int()

                    labels_int = (
                        valid_labels
                        .int()
                    )

                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(),
                        preds.cpu().numpy(),
                        zero_division=0
                    )

                    f1_per_graph.append(f1)

            running_loss_train = float(
                np.mean(loss_per_graph)
            )

            running_f1_train = float(
                np.mean(f1_per_graph)
            )

            loss_per_epoch_train.append(
                running_loss_train
            )

            f1_per_epoch_train.append(
                running_f1_train
            )

            # ====================================================
            # VALIDATION
            # ====================================================

            net.eval()

            loss_per_graph = []
            f1_per_graph = []

            with torch.no_grad():

                for data in dl_valid:

                    graph, name = data

                    graph = graph.to(device)

                    logits = net(graph)

                    labels = (
                        graph.ndata['significance']
                        .float()
                        .view(-1)
                    )

                    logits = logits.view(-1)

                    valid_mask = (
                        (labels == 0) |
                        (labels == 1)
                    )

                    if valid_mask.sum().item() == 0:
                        continue

                    valid_logits = logits[
                        valid_mask
                    ]

                    valid_labels = labels[
                        valid_mask
                    ]

                    # ------------------------------------------------
                    # Validation loss
                    # ------------------------------------------------

                    loss = criterion(
                        valid_logits,
                        valid_labels
                    )

                    loss_per_graph.append(
                        loss.item()
                    )

                    # ------------------------------------------------
                    # Validation predictions
                    # ------------------------------------------------

                    probs = torch.sigmoid(
                        valid_logits
                    )

                    preds = (
                        probs > 0.5
                    ).int()

                    labels_int = (
                        valid_labels
                        .int()
                    )

                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(),
                        preds.cpu().numpy(),
                        zero_division=0
                    )

                    f1_per_graph.append(f1)

            if len(loss_per_graph) == 0:

                raise ValueError(
                    "Validation graph contains no valid "
                    "significance labels."
                )

            running_loss_valid = float(
                np.mean(loss_per_graph)
            )

            running_f1_val = float(
                np.mean(f1_per_graph)
            )

            loss_per_epoch_valid.append(
                running_loss_valid
            )

            f1_per_epoch_valid.append(
                running_f1_val
            )

            # ====================================================
            # MAX F1
            # ====================================================

            max_f1_train = max(
                f1_per_epoch_train
            )

            max_f1_valid = max(
                f1_per_epoch_valid
            )

            max_f1_scores_train.append(
                max_f1_train
            )

            max_f1_scores_valid.append(
                max_f1_valid
            )

            # ====================================================
            # BEST MODEL
            # ====================================================

            if running_loss_valid < best_valid_loss:

                best_train_loss = (
                    running_loss_train
                )

                best_valid_loss = (
                    running_loss_valid
                )

                best_f1_score = (
                    running_f1_val
                )

                best_epoch = epoch + 1

                # best_model.load_state_dict(
                #     copy.deepcopy(
                #         net.state_dict()
                #     )
                # )
                # --- TRACK THE MAX VALIDATION F1 CROSS-THRESHOLD ---
                best_f1_this_epoch = 0.0
                best_threshold_this_epoch = 0.5

                for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
                    preds_tmp = (probs > threshold).int()
                    f1_tmp = metrics.f1_score(labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0)
                    
                    if f1_tmp > best_f1_this_epoch:
                        best_f1_this_epoch = f1_tmp
                        best_threshold_this_epoch = threshold

                # When deciding whether to save a checkpoint at the end of the epoch:
                if best_f1_this_epoch > best_f1_score:
                    best_f1_score = best_f1_this_epoch
                    best_epoch = epoch
                    best_model.load_state_dict(copy.deepcopy(net.state_dict()))
                    print(f"*** BEST MODEL UPDATED AT THRESHOLD {best_threshold_this_epoch:.2f}! Best F1: {best_f1_score:.4f} ***")


                print(
                    f"\n*** BEST MODEL UPDATED ***"
                )

                print(
                    f"Epoch       : {best_epoch}"
                )

                print(
                    f"Train Loss  : "
                    f"{best_train_loss:.6f}"
                )

                print(
                    f"Valid Loss  : "
                    f"{best_valid_loss:.6f}"
                )

                print(
                    f"Valid F1    : "
                    f"{best_f1_score:.6f}"
                )

            # ====================================================
            # DETAILED DIAGNOSTIC EVERY EPOCH
            # ====================================================

            if epoch == 0 or (
                (epoch + 1) % 10 == 0
            ):

                # Get predictions from validation graph
                for data in dl_valid:

                    graph, _ = data

                    graph = graph.to(device)

                    logits = net(graph)

                    labels = (
                        graph.ndata[
                            'significance'
                        ]
                        .float()
                        .view(-1)
                    )

                    logits = logits.view(-1)

                    valid_mask = (
                        (labels == 0) |
                        (labels == 1)
                    )

                    valid_logits = logits[
                        valid_mask
                    ]

                    valid_labels = labels[
                        valid_mask
                    ]

                    probs = torch.sigmoid(
                        valid_logits
                    )

                    preds = (
                        probs > 0.5
                    ).int()

                    labels_int = (
                        valid_labels.int()
                    )

                    tp = (
                        (labels_int == 1) &
                        (preds == 1)
                    ).sum().item()

                    fp = (
                        (labels_int == 0) &
                        (preds == 1)
                    ).sum().item()

                    fn = (
                        (labels_int == 1) &
                        (preds == 0)
                    ).sum().item()

                    tn = (
                        (labels_int == 0) &
                        (preds == 0)
                    ).sum().item()

                    print("\n" + "-" * 70)
                    print(
                        f"EPOCH {epoch + 1} DIAGNOSTIC"
                    )
                    print("-" * 70)

                    print(
                        f"Train Loss : "
                        f"{running_loss_train:.6f}"
                    )

                    print(
                        f"Valid Loss : "
                        f"{running_loss_valid:.6f}"
                    )

                    print(
                        f"Train F1   : "
                        f"{running_f1_train:.6f}"
                    )

                    print(
                        f"Valid F1   : "
                        f"{running_f1_val:.6f}"
                    )

                    print(
                        f"TP={tp} | "
                        f"FP={fp} | "
                        f"FN={fn} | "
                        f"TN={tn}"
                    )

                    print(
                        f"Probability min  : "
                        f"{probs.min().item():.6f}"
                    )

                    print(
                        f"Probability max  : "
                        f"{probs.max().item():.6f}"
                    )

                    print(
                        f"Probability mean : "
                        f"{probs.mean().item():.6f}"
                    )

                    print("-" * 70)

                    # # ------------------------------------------------
                    # # Threshold diagnostic
                    # # ------------------------------------------------

                    # for threshold in [
                    #     0.1,
                    #     0.2,
                    #     0.3,
                    #     0.4,
                    #     0.5
                    # ]:

                    #     preds_tmp = (
                    #         probs > threshold
                    #     ).int()

                    #     f1_tmp = (
                    #         metrics.f1_score(
                    #             labels_int.cpu().numpy(),
                    #             preds_tmp.cpu().numpy(),
                    #             zero_division=0
                    #         )
                    #     )

                    #     print(
                    #         f"threshold={threshold:.1f} | "
                    #         f"true_pos="
                    #         f"{(labels_int == 1).sum().item()} | "
                    #         f"true_neg="
                    #         f"{(labels_int == 0).sum().item()} | "
                    #         f"pred_pos="
                    #         f"{(preds_tmp == 1).sum().item()} | "
                    #         f"pred_neg="
                    #         f"{(preds_tmp == 0).sum().item()} | "
                    #         f"F1={f1_tmp:.4f}"
                    #     )

                    # break

                    # ------------------------------------------------
                    # Threshold diagnostic (Expanded Resolution)
                    # ------------------------------------------------

                    # Expanding to look at higher decision barriers matching your prob max/mean
                    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:

                        preds_tmp = (
                            probs > threshold
                        ).int()

                        f1_tmp = (
                            metrics.f1_score(
                                labels_int.cpu().numpy(),
                                preds_tmp.cpu().numpy(),
                                zero_division=0
                            )
                        )

                        print(
                            f"threshold={threshold:.2f} | "  # Changed to .2f to show 0.55, 0.65, etc.
                            f"true_pos="
                            f"{(labels_int == 1).sum().item()} | "
                            f"true_neg="
                            f"{(labels_int == 0).sum().item()} | "
                            f"pred_pos="
                            f"{(preds_tmp == 1).sum().item()} | "
                            f"pred_neg="
                            f"{(preds_tmp == 0).sum().item()} | "
                            f"F1={f1_tmp:.4f}"
                        )

                    break


            pbar.update(1)

            pbar.set_postfix({
                'train_loss':
                    f'{running_loss_train:.4f}',
                'valid_loss':
                    f'{running_loss_valid:.4f}',
                'train_f1':
                    f'{running_f1_train:.4f}',
                'valid_f1':
                    f'{running_f1_val:.4f}'
            })

    # ============================================================
    # 16. FINAL BEST MODEL
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print(
        f"Best epoch       : {best_epoch}"
    )

    print(
        f"Best train loss  : "
        f"{best_train_loss:.6f}"
    )

    print(
        f"Best valid loss  : "
        f"{best_valid_loss:.6f}"
    )

    print(
        f"Best valid F1    : "
        f"{best_f1_score:.6f}"
    )

    # ============================================================
    # 17. FINAL EMBEDDINGS
    # ============================================================

    all_embeddings, cluster_labels = (
        calculate_cluster_labels(
            best_model,
            dl_train,
            device
        )
    )

    all_embeddings = (
        all_embeddings.reshape(
            all_embeddings.shape[0],
            -1
        )
    )

    # ============================================================
    # 18. COSINE SIMILARITY
    # ============================================================

    norms = np.linalg.norm(
        all_embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1e-12

    normalized_embeddings = (
        all_embeddings / norms
    )

    cos_sim = np.dot(
        normalized_embeddings,
        normalized_embeddings.T
    )

    # ============================================================
    # 19. PLOTS
    # ============================================================

    if plot:

        loss_path = os.path.join(
            results_path,
            f'embeddings_loss_head{num_heads}_'
            f'dim{out_feats}_lay{num_layers}_'
            f'epo{num_epochs}.png'
        )

        f1_path = os.path.join(
            results_path,
            f'embeddings_f1_head{num_heads}_'
            f'dim{out_feats}_lay{num_layers}_'
            f'epo{num_epochs}.png'
        )

        max_f1_path = os.path.join(
            results_path,
            f'embeddings_max_f1_head{num_heads}_'
            f'dim{out_feats}_lay{num_layers}_'
            f'epo{num_epochs}.png'
        )

        matrix_path = os.path.join(
            results_path,
            f'embeddings_matrix_head{num_heads}_'
            f'dim{out_feats}_lay{num_layers}_'
            f'epo{num_epochs}.png'
        )

        draw_loss_plot(
            loss_per_epoch_train,
            loss_per_epoch_valid,
            loss_path
        )

        draw_max_f1_plot(
            max_f1_scores_train,
            max_f1_scores_valid,
            max_f1_path
        )

        draw_f1_plot(
            f1_per_epoch_train,
            f1_per_epoch_valid,
            f1_path
        )

    # ============================================================
    # 20. SAVE BEST MODEL
    # ============================================================

    torch.save(
        best_model.state_dict(),
        model_path
    )

    print(
        f"\nBest model saved to:\n{model_path}"
    )

    # ============================================================
    # 21. FINAL VISUALIZATION PATHS
    # ============================================================

    save_path_pca = os.path.join(
        results_path,
        f'embeddings_pca_head{num_heads}_'
        f'dim{out_feats}_lay{num_layers}_'
        f'epo{num_epochs}_final.png'
    )

    save_path_t_SNE = os.path.join(
        results_path,
        f'embeddings_t-SNE_head{num_heads}_'
        f'dim{out_feats}_lay{num_layers}_'
        f'epo{num_epochs}_final.png'
    )

    save_path_heatmap_ = os.path.join(
        results_path,
        f'embeddings_heatmap_stId_head{num_heads}_'
        f'dim{out_feats}_lay{num_layers}_'
        f'epo{num_epochs}_final.png'
    )

    save_path_matrix = os.path.join(
        results_path,
        f'embeddings_matrix_stId_head{num_heads}_'
        f'dim{out_feats}_lay{num_layers}_'
        f'epo{num_epochs}_final.png'
    )

    # ============================================================
    # 22. FINAL NODE / CLUSTER INFORMATION
    # ============================================================

    cluster_stId_dict = {}

    significant_stIds = []

    clusters_with_significant_stId = {}

    clusters_node_info = {}

    # ------------------------------------------------------------
    # Process training graph
    # ------------------------------------------------------------

    for data in dl_train:

        graph, _ = data

        graph = graph.to(device)

        node_embeddings = (
            best_model
            .get_node_embeddings(graph)
            .detach()
            .cpu()
            .numpy()
        )

        graph_path = os.path.join(
            data_path,
            omics,
            cancer,
            'emb',
            'raw',
            'emb_train.pkl'
        )

        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert (
            len(cluster_labels)
            == len(nx_graph.nodes)
        ), (
            "Cluster labels and number of nodes "
            "must match"
        )

        node_to_index = {
            node: idx
            for idx, node in enumerate(nx_graph.nodes)
        }

        first_node_stId_in_cluster = {}

        first_node_embedding_in_cluster = {}

        stid_dic = {}

        # --------------------------------------------------------
        # Save embeddings and significant genes
        # --------------------------------------------------------

        for node in nx_graph.nodes:

            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']

            idx = node_to_index[node]

            stid_dic[stid] = (
                node_embeddings[idx]
            )

            # ----------------------------------------------------
            # FIXED:
            #
            # Old code:
            #     == 'significant'
            #
            # Correct:
            #     == 1
            # ----------------------------------------------------

            significance_value = (
                graph.ndata[
                    'significance'
                ][idx]
                .item()
            )

            if significance_value == 1:

                significant_stIds.append(
                    stid
                )

        # --------------------------------------------------------
        # Save final embeddings
        # --------------------------------------------------------

        stid_df_final = pd.DataFrame.from_dict(
            stid_dic,
            orient='index'
        )

        csv_save_path_final = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_'
            f'dim{out_feats}_lay{num_layers}_'
            f'epo{num_epochs}_final.csv'
        )

        stid_df_final.to_csv(
            csv_save_path_final,
            index_label='stId'
        )

        # --------------------------------------------------------
        # Cluster information
        # --------------------------------------------------------

        for node, cluster in zip(
            nx_graph.nodes,
            cluster_labels
        ):

            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']

            idx = node_to_index[node]

            significance_value = (
                graph.ndata[
                    'significance'
                ][idx]
                .item()
            )

            # First representative
            if cluster not in (
                first_node_stId_in_cluster
            ):

                first_node_stId_in_cluster[
                    cluster
                ] = stid

                first_node_embedding_in_cluster[
                    cluster
                ] = node_embeddings[idx]

            # ----------------------------------------------------
            # cluster_stId_dict
            # ----------------------------------------------------

            if cluster not in cluster_stId_dict:

                cluster_stId_dict[
                    cluster
                ] = []

            cluster_stId_dict[
                cluster
            ].append(stid)

            # ----------------------------------------------------
            # clusters_with_significant_stId
            # ----------------------------------------------------

            if cluster not in (
                clusters_with_significant_stId
            ):

                clusters_with_significant_stId[
                    cluster
                ] = []

            if significance_value == 1:

                clusters_with_significant_stId[
                    cluster
                ].append(stid)

            # ----------------------------------------------------
            # clusters_node_info
            # ----------------------------------------------------

            if cluster not in clusters_node_info:

                clusters_node_info[
                    cluster
                ] = []

            node_info = {
                'stId': stid,
                'significance':
                    significance_value,
                'other_info':
                    nx_graph.nodes[node]
            }

            clusters_node_info[
                cluster
            ].append(node_info)

        print(
            "\nFirst node in each cluster:"
        )

        print(
            first_node_stId_in_cluster
        )

        # --------------------------------------------------------
        # Cluster representatives
        # --------------------------------------------------------

        cluster_stid_map_final = (
            first_node_stId_in_cluster
        )

        sorted_clusters = sorted(
            cluster_stid_map_final.keys()
        )

        stid_list = [
            first_node_stId_in_cluster[c]
            for c in sorted_clusters
        ]

        embedding_list = [
            first_node_embedding_in_cluster[c]
            for c in sorted_clusters
        ]

        embedding_list = [
            np.asarray(emb).reshape(-1)
            for emb in embedding_list
        ]

        heatmap_data = pd.DataFrame(
            embedding_list,
            index=stid_list
        )

        print(
            f"Number of clusters: "
            f"{len(sorted_clusters)}"
        )

        print(
            f"Embedding matrix shape: "
            f"{heatmap_data.shape}"
        )

        # --------------------------------------------------------
        # Cluster plots
        # --------------------------------------------------------

        create_heatmap_with_stid(
            embedding_list,
            stid_list,
            save_path_heatmap_
        )

        plot_cosine_similarity_matrix_for_clusters_with_values(
            embedding_list,
            stid_list,
            save_path_matrix
        )

        break

    # ============================================================
    # 23. FINAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(
        all_embeddings,
        cluster_labels,
        stid_list,
        save_path_t_SNE
    )

    visualize_embeddings_pca(
        all_embeddings,
        cluster_labels,
        first_node_stId_in_cluster,
        save_path_pca
    )

    # ============================================================
    # 24. FINAL CLUSTERING METRICS
    # ============================================================

    if len(np.unique(cluster_labels)) > 1:

        silhouette_avg = silhouette_score(
            all_embeddings,
            cluster_labels
        )

        davies_bouldin = davies_bouldin_score(
            all_embeddings,
            cluster_labels
        )

    else:

        silhouette_avg = np.nan
        davies_bouldin = np.nan

    print(
        f"\nSilhouette Score: "
        f"{silhouette_avg}"
    )

    print(
        f"Davies-Bouldin Index: "
        f"{davies_bouldin}"
    )

    # ============================================================
    # 25. FINAL SUMMARY
    # ============================================================

    summary = (
        f"Epoch {num_epochs} - "
        f"Max F1 Train: {max_f1_train}, "
        f"Max F1 Valid: {max_f1_valid}\n"
    )

    summary += (
        f"Best Epoch: {best_epoch}\n"
    )

    summary += (
        f"Best Train Loss: "
        f"{best_train_loss}\n"
    )

    summary += (
        f"Best Validation Loss: "
        f"{best_valid_loss}\n"
    )

    # FIXED:
    # Previously this incorrectly used max_f1_train.
    summary += (
        f"Best F1 Score: "
        f"{best_f1_score}\n"
    )

    summary += (
        f"Silhouette Score: "
        f"{silhouette_avg}\n"
    )

    summary += (
        f"Davies-Bouldin Index: "
        f"{davies_bouldin}\n"
    )

    summary += (
        f"Number of significant genes: "
        f"{len(significant_stIds)}\n"
    )

    save_file = os.path.join(
        results_path,
        f'embeddings_head{num_heads}_'
        f'dim{out_feats}_lay{num_layers}_'
        f'epo{num_epochs}.txt'
    )

    with open(save_file, 'w') as f:
        f.write(summary)

    # ============================================================
    # 26. FINAL INFORMATION
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL STAGE-1 RESULTS")
    print("=" * 70)

    print(
        f"Best epoch            : {best_epoch}"
    )

    print(
        f"Best validation loss  : "
        f"{best_valid_loss:.6f}"
    )

    print(
        f"Best validation F1    : "
        f"{best_f1_score:.6f}"
    )

    print(
        f"Max training F1       : "
        f"{max_f1_train:.6f}"
    )

    print(
        f"Max validation F1     : "
        f"{max_f1_valid:.6f}"
    )

    print(
        f"Significant genes     : "
        f"{len(significant_stIds)}"
    )

    print(
        f"Model path:\n{model_path}"
    )

    print("=" * 70)

    return model_path

def train_(
    hyperparams=None,
    data_path='../data/omics/',
    plot=True,
    omics='cna',
    cancer='BLCA'
):
    """
    Stage-1 training:
        Predict generic biological significance from the graph.

    IMPORTANT:
        This stage does NOT use cancer-driver labels.
        'significance' is the Stage-1 pretraining target.

    Expected significance values:
        0 = not significant
        1 = significant

    If -1 is present:
        -1 = unlabeled / ignored during loss and metric calculation.

    Major fixes:
        1. Explicitly identify train/test graphs by filename.
        2. Add normalized in-degree to the graphs actually used by
           GraphDataLoader.
        3. Aggregate validation predictions across all validation graphs.
        4. Track PR-AUC and ROC-AUC.
        5. Select the best model using validation PR-AUC.
        6. Use threshold-optimized F1 only as a diagnostic.
        7. Track the best F1 threshold separately.
        8. Avoid the previous best-model logic that could select an
           all-positive prediction as the best model.
    """

    # ============================================================
    # 0. Imports
    # ============================================================

    import os
    import copy
    import pickle

    import numpy as np
    import pandas as pd

    import torch
    import torch.optim as optim

    from tqdm import tqdm

    from sklearn import metrics

    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        average_precision_score,
        roc_auc_score
    )

    # ============================================================
    # 1. Validate hyperparameters
    # ============================================================

    if hyperparams is None:
        raise ValueError(
            "hyperparams must be provided."
        )

    required_keys = [
        'num_epochs',
        'in_feats',
        'out_feats',
        'num_layers',
        'num_heads',
        'lr',
        'batch_size',
        'device'
    ]

    missing_keys = [
        key for key in required_keys
        if key not in hyperparams
    ]

    if missing_keys:
        raise KeyError(
            f"Missing hyperparameters: {missing_keys}"
        )

    num_epochs = hyperparams['num_epochs']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']

    print("\n" + "=" * 70)
    print("START STAGE-1 SIGNIFICANCE PRETRAINING")
    print("=" * 70)

    print(f"Omics       : {omics}")
    print(f"Cancer      : {cancer}")
    print(f"Device      : {device}")
    print(f"Epochs      : {num_epochs}")
    print(f"LR          : {learning_rate}")
    print(f"Batch size  : {batch_size}")
    print(f"Input feats : {in_feats}")
    print(f"Out feats   : {out_feats}")
    print(f"Layers      : {num_layers}")
    print("=" * 70 + "\n")

    # ============================================================
    # 2. Dataset paths
    # ============================================================

    data_path_ = os.path.join(
        data_path,
        omics,
        cancer
    )

    processed_path = os.path.join(
        data_path_,
        'emb',
        'processed'
    )

    raw_path = os.path.join(
        data_path_,
        'emb',
        'raw'
    )

    model_dir = os.path.join(
        data_path_,
        'emb',
        'models'
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    # ============================================================
    # 3. Model save path
    # ============================================================

    model_path = os.path.join(
        model_dir,
        f'model_dim{out_feats}_'
        f'lay{num_layers}_'
        f'epo{num_epochs}.pth'
    )

    # ============================================================
    # 4. Load Dataset
    # ============================================================

    ds = dataset.Dataset(data_path_)

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT assume:
    #
    #     ds[0] = train
    #     ds[1] = validation
    #
    # because Dataset.__getitem__() sorts filenames.
    #
    # Explicitly identify train/test files.
    # ------------------------------------------------------------

    processed_files = sorted(
        [
            f
            for f in os.listdir(processed_path)
            if f.endswith('.dgl')
        ]
    )

    print("\nPROCESSED GRAPH FILES")
    print("-" * 70)

    for filename in processed_files:
        print(filename)

    if len(processed_files) < 2:
        raise ValueError(
            f"Expected at least two processed DGL graphs in "
            f"{processed_path}, found: {processed_files}"
        )

    train_candidates = [
        f for f in processed_files
        if 'train' in f.lower()
    ]

    valid_candidates = [
        f for f in processed_files
        if (
            'valid' in f.lower()
            or 'val' in f.lower()
            or 'test' in f.lower()
        )
    ]

    if not train_candidates:
        raise ValueError(
            "Could not identify a training graph. "
            f"Available files: {processed_files}"
        )

    if not valid_candidates:
        raise ValueError(
            "Could not identify a validation/test graph. "
            f"Available files: {processed_files}"
        )

    train_filename = sorted(
        train_candidates
    )[0]

    valid_filename = sorted(
        valid_candidates
    )[0]

    print(
        f"\nTraining graph   : {train_filename}"
    )

    print(
        f"Validation graph : {valid_filename}"
    )

    # ------------------------------------------------------------
    # Load graphs directly from the known filenames.
    # ------------------------------------------------------------

    train_graph_path = os.path.join(
        processed_path,
        train_filename
    )

    valid_graph_path = os.path.join(
        processed_path,
        valid_filename
    )

    train_graphs, _ = dgl.load_graphs(
        train_graph_path
    )

    valid_graphs, _ = dgl.load_graphs(
        valid_graph_path
    )

    if len(train_graphs) == 0:
        raise ValueError(
            f"No graph found in {train_graph_path}"
        )

    if len(valid_graphs) == 0:
        raise ValueError(
            f"No graph found in {valid_graph_path}"
        )

    graph_train = train_graphs[0]
    graph_valid = valid_graphs[0]

    name_train = train_filename
    name_valid = valid_filename

    print("\nGRAPH INFORMATION")
    print("-" * 70)

    print(
        f"Train nodes : {graph_train.num_nodes():,}"
    )

    print(
        f"Train edges : {graph_train.num_edges():,}"
    )

    print(
        f"Valid nodes : {graph_valid.num_nodes():,}"
    )

    print(
        f"Valid edges : {graph_valid.num_edges():,}"
    )

    # ============================================================
    # 5. Add normalized in-degree feature
    # ============================================================

    def add_degree_feature(graph, graph_name):

        # --------------------------------------------------------
        # In-degree
        # --------------------------------------------------------

        node_degrees = (
            graph.in_degrees()
            .float()
            .unsqueeze(-1)
        )

        degree_mean = node_degrees.mean()

        degree_std = node_degrees.std()

        node_degrees_norm = (
            node_degrees - degree_mean
        ) / (
            degree_std + 1e-5
        )

        # --------------------------------------------------------
        # Existing features
        # --------------------------------------------------------

        if 'weight' in graph.ndata:

            existing_feats = (
                graph.ndata['weight']
            )

        elif 'feat' in graph.ndata:

            existing_feats = (
                graph.ndata['feat']
            )

        else:

            existing_feats = torch.ones(
                (
                    graph.num_nodes(),
                    1
                ),
                dtype=torch.float32
            )

        # --------------------------------------------------------
        # Force [N, F]
        # --------------------------------------------------------

        if existing_feats.dim() == 1:

            existing_feats = (
                existing_feats.unsqueeze(-1)
            )

        existing_feats = (
            existing_feats.float()
        )

        # --------------------------------------------------------
        # Concatenate degree
        # --------------------------------------------------------

        updated_feats = torch.cat(
            [
                existing_feats,
                node_degrees_norm
            ],
            dim=-1
        )

        # --------------------------------------------------------
        # Store both names
        # --------------------------------------------------------

        graph.ndata['feat'] = (
            updated_feats
        )

        graph.ndata['weight'] = (
            updated_feats
        )

        print(
            f"\n{graph_name} FEATURE INFORMATION"
        )

        print(
            f"Original feature dimension : "
            f"{existing_feats.shape[1]}"
        )

        print(
            f"Degree feature dimension   : 1"
        )

        print(
            f"Final feature dimension    : "
            f"{updated_feats.shape[1]}"
        )

        print(
            f"Feature shape              : "
            f"{tuple(updated_feats.shape)}"
        )

        return graph

    graph_train = add_degree_feature(
        graph_train,
        "TRAIN"
    )

    graph_valid = add_degree_feature(
        graph_valid,
        "VALIDATION"
    )

    actual_train_feats = (
        graph_train.ndata['weight'].shape[1]
    )

    actual_valid_feats = (
        graph_valid.ndata['weight'].shape[1]
    )

    if actual_train_feats != actual_valid_feats:

        raise ValueError(
            "Train and validation feature dimensions differ: "
            f"{actual_train_feats} vs "
            f"{actual_valid_feats}"
        )

    actual_in_feats = actual_train_feats

    print(
        f"\nActual input feature dimension: "
        f"{actual_in_feats}"
    )

    if in_feats != actual_in_feats:

        print(
            "\nWARNING:"
        )

        print(
            f"hyperparams['in_feats'] = {in_feats}"
        )

        print(
            f"actual graph feature dimension = "
            f"{actual_in_feats}"
        )

        print(
            "Because normalized in-degree was appended, "
            "the input dimension increased by one."
        )

        print(
            "Update hyperparams['in_feats'] if your "
            "TAGCNModel uses the configured input dimension."
        )

    # ============================================================
    # 6. SIGNIFICANCE LABEL DEBUG
    # ============================================================

    def inspect_labels(graph, graph_name):

        if 'significance' not in graph.ndata:

            raise KeyError(
                f"{graph_name} graph does not contain "
                "'significance'."
            )

        significance = (
            graph.ndata['significance']
            .float()
            .view(-1)
        )

        print("\n" + "=" * 70)
        print(
            f"{graph_name.upper()} SIGNIFICANCE DEBUG"
        )
        print("=" * 70)

        print(
            f"dtype : {significance.dtype}"
        )

        print(
            f"shape : {tuple(significance.shape)}"
        )

        unique_labels, label_counts = (
            torch.unique(
                significance,
                return_counts=True
            )
        )

        print("unique labels:")

        for label, count in zip(
            unique_labels,
            label_counts
        ):

            print(
                f"    label={label.item():>4} "
                f"count={count.item():>7}"
            )

        valid_mask = (
            (significance == 0)
            |
            (significance == 1)
        )

        valid_labels = (
            significance[valid_mask]
        )

        if valid_labels.numel() == 0:

            raise ValueError(
                f"{graph_name} contains no valid "
                "0/1 significance labels."
            )

        num_positive = (
            (valid_labels == 1)
            .sum()
            .item()
        )

        num_negative = (
            (valid_labels == 0)
            .sum()
            .item()
        )

        num_unlabeled = (
            (significance == -1)
            .sum()
            .item()
        )

        print(
            f"Positive / significant : "
            f"{num_positive:,}"
        )

        print(
            f"Negative / non-signif. : "
            f"{num_negative:,}"
        )

        print(
            f"Unlabeled (-1)         : "
            f"{num_unlabeled:,}"
        )

        positive_ratio = (
            num_positive /
            valid_labels.numel()
        )

        print(
            f"Positive ratio          : "
            f"{positive_ratio:.4%}"
        )

        return {
            'positive': num_positive,
            'negative': num_negative,
            'unlabeled': num_unlabeled,
            'positive_ratio': positive_ratio
        }

    train_label_stats = inspect_labels(
        graph_train,
        "train"
    )

    valid_label_stats = inspect_labels(
        graph_valid,
        "validation"
    )

    # ============================================================
    # 7. DataLoaders
    # ============================================================

    # IMPORTANT:
    #
    # These are the SAME in-memory graph objects whose degree
    # features were just added.
    # ============================================================

    ds_train = [
        (
            graph_train,
            name_train
        )
    ]

    ds_valid = [
        (
            graph_valid,
            name_valid
        )
    ]

    dl_train = GraphDataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True
    )

    dl_valid = GraphDataLoader(
        ds_valid,
        batch_size=batch_size,
        shuffle=False
    )

    # ============================================================
    # 8. Create model
    # ============================================================

    net = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)

    # ============================================================
    # 9. Optimizer
    # ============================================================

    optimizer = optim.Adam(
        net.parameters(),
        lr=learning_rate
    )

    # ============================================================
    # 10. Best model
    # ============================================================

    best_model = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)

    best_model.load_state_dict(
        copy.deepcopy(
            net.state_dict()
        )
    )

    # ============================================================
    # 11. Focal Loss
    # ============================================================

    num_positive = (
        train_label_stats['positive']
    )

    num_negative = (
        train_label_stats['negative']
    )

    total_valid = (
        num_positive +
        num_negative
    )

    # Positive class weight.
    #
    # This preserves your original dynamic-alpha idea.
    dynamic_alpha = (
        num_negative /
        total_valid
    )

    print(
        "\nFOCAL LOSS"
    )

    print(
        f"Dynamic alpha : "
        f"{dynamic_alpha:.6f}"
    )

    print(
        f"Gamma         : 2.0"
    )

    criterion = FocalLoss(
        alpha=dynamic_alpha,
        gamma=2.0,
        reduction='mean'
    )

    # ============================================================
    # 12. Training history
    # ============================================================

    loss_per_epoch_train = []
    loss_per_epoch_valid = []

    f1_per_epoch_train = []
    f1_per_epoch_valid = []

    pr_auc_per_epoch_train = []
    pr_auc_per_epoch_valid = []

    roc_auc_per_epoch_train = []
    roc_auc_per_epoch_valid = []

    max_f1_scores_train = []
    max_f1_scores_valid = []

    best_train_loss = float('inf')
    best_valid_loss = float('inf')

    best_f1_score = 0.0
    best_f1_threshold = 0.5

    best_valid_pr_auc = -float('inf')
    best_valid_roc_auc = -float('inf')

    best_epoch = 0

    # ============================================================
    # 13. Results path
    # ============================================================

    results_path = os.path.abspath(
        os.path.join(
            'results',
            'multiomics_meth',
            'node_embeddings',
            omics,
            cancer
        )
    )

    os.makedirs(
        results_path,
        exist_ok=True
    )

    # ============================================================
    # 14. Initial embeddings
    # ============================================================

    print("\n" + "=" * 70)
    print("CALCULATING INITIAL EMBEDDINGS")
    print("=" * 70)

    all_embeddings_initial, cluster_labels_initial = (
        calculate_cluster_labels(
            best_model,
            dl_train,
            device
        )
    )

    all_embeddings_initial = (
        all_embeddings_initial.reshape(
            all_embeddings_initial.shape[0],
            -1
        )
    )

    save_path_heatmap_initial = os.path.join(
        results_path,
        f'embeddings_heatmap_stId_dim{out_feats}_'
        f'lay{num_layers}_epo{num_epochs}_initial.png'
    )

    save_path_matrix_initial = os.path.join(
        results_path,
        f'embeddings_matrix_stId_dim{out_feats}_'
        f'lay{num_layers}_epo{num_epochs}_initial.png'
    )

    save_path_pca_initial = os.path.join(
        results_path,
        f'embeddings_pca_dim{out_feats}_'
        f'lay{num_layers}_epo{num_epochs}_initial.png'
    )

    save_path_t_SNE_initial = os.path.join(
        results_path,
        f'embeddings_t-SNE_dim{out_feats}_'
        f'lay{num_layers}_epo{num_epochs}_initial.png'
    )

    # ============================================================
    # 15. Initial embedding information
    # ============================================================

    first_node_stId_in_cluster_initial = {}
    first_node_embedding_in_cluster_initial = {}

    stid_list_initial = []

    for data in dl_train:

        graph, _ = data

        graph = graph.to(device)

        node_embeddings_initial = (
            best_model
            .get_node_embeddings(graph)
            .detach()
            .cpu()
            .numpy()
        )

        graph_path = os.path.join(
            raw_path,
            'emb_train.pkl'
        )

        with open(
            graph_path,
            'rb'
        ) as f:

            nx_graph = pickle.load(f)

        if (
            len(cluster_labels_initial)
            != len(nx_graph.nodes)
        ):

            raise ValueError(
                "Initial cluster labels and graph nodes "
                "do not match: "
                f"{len(cluster_labels_initial)} vs "
                f"{len(nx_graph.nodes)}"
            )

        node_to_index_initial = {
            node: idx
            for idx, node in enumerate(
                nx_graph.nodes
            )
        }

        stid_dic_initial = {}

        for node in nx_graph.nodes:

            if (
                'stId'
                not in nx_graph.nodes[node]
            ):
                continue

            stId = (
                nx_graph.nodes[node]['stId']
            )

            idx = (
                node_to_index_initial[node]
            )

            stid_dic_initial[stId] = (
                node_embeddings_initial[idx]
            )

        stid_df_initial = (
            pd.DataFrame.from_dict(
                stid_dic_initial,
                orient='index'
            )
        )

        csv_save_path_initial = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_'
            f'dim{out_feats}_lay{num_layers}_'
            f'epo{num_epochs}_initial.csv'
        )

        stid_df_initial.to_csv(
            csv_save_path_initial,
            index_label='stId'
        )

        for node, cluster in zip(
            nx_graph.nodes,
            cluster_labels_initial
        ):

            if (
                'stId'
                not in nx_graph.nodes[node]
            ):
                continue

            stid = (
                nx_graph.nodes[node]['stId']
            )

            idx = (
                node_to_index_initial[node]
            )

            if cluster not in (
                first_node_stId_in_cluster_initial
            ):

                first_node_stId_in_cluster_initial[
                    cluster
                ] = stid

                first_node_embedding_in_cluster_initial[
                    cluster
                ] = (
                    node_embeddings_initial[idx]
                )

        sorted_clusters_initial = sorted(
            first_node_stId_in_cluster_initial
            .keys()
        )

        stid_list_initial = [
            first_node_stId_in_cluster_initial[c]
            for c in sorted_clusters_initial
        ]

        embedding_list_initial = [
            first_node_embedding_in_cluster_initial[c]
            for c in sorted_clusters_initial
        ]

        embedding_list_initial = [
            np.asarray(emb).reshape(-1)
            for emb in embedding_list_initial
        ]

        heatmap_data_initial = pd.DataFrame(
            embedding_list_initial,
            index=stid_list_initial
        )

        print(
            f"Initial clusters: "
            f"{len(sorted_clusters_initial)}"
        )

        print(
            f"Initial embedding matrix: "
            f"{heatmap_data_initial.shape}"
        )

        create_heatmap_with_stid(
            embedding_list_initial,
            stid_list_initial,
            save_path_heatmap_initial
        )

        plot_cosine_similarity_matrix_for_clusters_with_values(
            embedding_list_initial,
            stid_list_initial,
            save_path_matrix_initial
        )

        break

    # ============================================================
    # 16. Initial embedding visualization
    # ============================================================

    if (
        len(all_embeddings_initial) > 1
        and len(
            np.unique(
                cluster_labels_initial
            )
        ) > 1
    ):

        visualize_embeddings_tsne(
            all_embeddings_initial,
            cluster_labels_initial,
            stid_list_initial,
            save_path_t_SNE_initial
        )

        visualize_embeddings_pca(
            all_embeddings_initial,
            cluster_labels_initial,
            first_node_stId_in_cluster_initial,
            save_path_pca_initial
        )

        silhouette_avg_initial = (
            silhouette_score(
                all_embeddings_initial,
                cluster_labels_initial
            )
        )

        davies_bouldin_initial = (
            davies_bouldin_score(
                all_embeddings_initial,
                cluster_labels_initial
            )
        )

    else:

        silhouette_avg_initial = np.nan
        davies_bouldin_initial = np.nan

    summary_initial = (
        f"Silhouette Score: "
        f"{silhouette_avg_initial}\n"
    )

    summary_initial += (
        f"Davies-Bouldin Index: "
        f"{davies_bouldin_initial}\n"
    )

    save_file_initial = os.path.join(
        results_path,
        f'embeddings_head{num_heads}_'
        f'dim{out_feats}_lay{num_layers}_'
        f'epo{num_epochs}_initial.txt'
    )

    with open(
        save_file_initial,
        'w'
    ) as f:

        f.write(
            summary_initial
        )

    # ============================================================
    # 17. Helper: collect graph predictions
    # ============================================================

    def collect_predictions(
        dataloader,
        current_model,
        calculate_loss=True
    ):

        current_model.eval()

        losses = []

        all_probs = []
        all_labels = []

        with torch.no_grad():

            for data in dataloader:

                graph, name = data

                graph = graph.to(device)

                logits = current_model(graph)

                logits = (
                    logits
                    .float()
                    .view(-1)
                )

                labels = (
                    graph.ndata['significance']
                    .float()
                    .view(-1)
                )

                valid_mask = (
                    (labels == 0)
                    |
                    (labels == 1)
                )

                if valid_mask.sum().item() == 0:
                    continue

                valid_logits = (
                    logits[valid_mask]
                )

                valid_labels = (
                    labels[valid_mask]
                )

                if calculate_loss:

                    loss = criterion(
                        valid_logits,
                        valid_labels
                    )

                    losses.append(
                        loss.item()
                    )

                probs = torch.sigmoid(
                    valid_logits
                )

                all_probs.append(
                    probs.detach().cpu()
                )

                all_labels.append(
                    valid_labels
                    .detach()
                    .cpu()
                    .int()
                )

        if not all_probs:

            raise ValueError(
                "No valid predictions were collected."
            )

        probs = torch.cat(
            all_probs
        ).numpy()

        labels = torch.cat(
            all_labels
        ).numpy()

        mean_loss = (
            float(np.mean(losses))
            if losses
            else np.nan
        )

        return (
            mean_loss,
            probs,
            labels
        )

    # ============================================================
    # 18. Helper: metrics
    # ============================================================

    def calculate_metrics(
        probs,
        labels
    ):

        # --------------------------------------------------------
        # Standard threshold
        # --------------------------------------------------------

        preds_05 = (
            probs >= 0.5
        ).astype(np.int64)

        f1_05 = metrics.f1_score(
            labels,
            preds_05,
            zero_division=0
        )

        # --------------------------------------------------------
        # PR-AUC
        # --------------------------------------------------------

        try:

            pr_auc = (
                average_precision_score(
                    labels,
                    probs
                )
            )

        except ValueError:

            pr_auc = np.nan

        # --------------------------------------------------------
        # ROC-AUC
        # --------------------------------------------------------

        try:

            roc_auc = (
                roc_auc_score(
                    labels,
                    probs
                )
            )

        except ValueError:

            roc_auc = np.nan

        # --------------------------------------------------------
        # Threshold sweep
        # --------------------------------------------------------

        thresholds = [
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95
        ]

        best_threshold = 0.5
        best_f1 = -1.0

        threshold_results = []

        for threshold in thresholds:

            preds = (
                probs >= threshold
            ).astype(np.int64)

            f1 = metrics.f1_score(
                labels,
                preds,
                zero_division=0
            )

            threshold_results.append(
                (
                    threshold,
                    f1,
                    preds.sum(),
                    len(preds) - preds.sum()
                )
            )

            if f1 > best_f1:

                best_f1 = f1
                best_threshold = threshold

        return {
            'f1_05': f1_05,
            'best_f1': best_f1,
            'best_threshold': best_threshold,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'threshold_results': threshold_results
        }

    # ============================================================
    # 19. TRAINING
    # ============================================================

    print("\n" + "=" * 70)
    print("START TRAINING")
    print("=" * 70)

    with tqdm(
        total=num_epochs,
        desc="Training",
        unit="epoch",
        leave=True
    ) as pbar:

        for epoch in range(num_epochs):

            # ====================================================
            # TRAIN
            # ====================================================

            net.train()

            train_losses = []

            train_probs_all = []
            train_labels_all = []

            for data in dl_train:

                graph, name = data

                graph = graph.to(device)

                logits = net(graph)

                logits = (
                    logits
                    .float()
                    .view(-1)
                )

                labels = (
                    graph.ndata['significance']
                    .float()
                    .view(-1)
                )

                valid_mask = (
                    (labels == 0)
                    |
                    (labels == 1)
                )

                if valid_mask.sum().item() == 0:

                    raise ValueError(
                        "Training graph contains no valid "
                        "0/1 significance labels."
                    )

                valid_logits = (
                    logits[valid_mask]
                )

                valid_labels = (
                    labels[valid_mask]
                )

                loss = criterion(
                    valid_logits,
                    valid_labels
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                train_losses.append(
                    loss.item()
                )

                with torch.no_grad():

                    probs = torch.sigmoid(
                        valid_logits
                    )

                    train_probs_all.append(
                        probs.detach().cpu()
                    )

                    train_labels_all.append(
                        valid_labels
                        .detach()
                        .cpu()
                        .int()
                    )

            running_loss_train = float(
                np.mean(train_losses)
            )

            train_probs = torch.cat(
                train_probs_all
            ).numpy()

            train_labels = torch.cat(
                train_labels_all
            ).numpy()

            train_metric = calculate_metrics(
                train_probs,
                train_labels
            )

            running_f1_train = (
                train_metric['f1_05']
            )

            train_pr_auc = (
                train_metric['pr_auc']
            )

            train_roc_auc = (
                train_metric['roc_auc']
            )

            # ====================================================
            # VALIDATION
            # ====================================================

            (
                running_loss_valid,
                valid_probs,
                valid_labels
            ) = collect_predictions(
                dl_valid,
                net,
                calculate_loss=True
            )

            valid_metric = calculate_metrics(
                valid_probs,
                valid_labels
            )

            running_f1_valid = (
                valid_metric['f1_05']
            )

            valid_pr_auc = (
                valid_metric['pr_auc']
            )

            valid_roc_auc = (
                valid_metric['roc_auc']
            )

            best_f1_this_epoch = (
                valid_metric['best_f1']
            )

            best_threshold_this_epoch = (
                valid_metric['best_threshold']
            )

            # ====================================================
            # History
            # ====================================================

            loss_per_epoch_train.append(
                running_loss_train
            )

            loss_per_epoch_valid.append(
                running_loss_valid
            )

            f1_per_epoch_train.append(
                running_f1_train
            )

            f1_per_epoch_valid.append(
                running_f1_valid
            )

            pr_auc_per_epoch_train.append(
                train_pr_auc
            )

            pr_auc_per_epoch_valid.append(
                valid_pr_auc
            )

            roc_auc_per_epoch_train.append(
                train_roc_auc
            )

            roc_auc_per_epoch_valid.append(
                valid_roc_auc
            )

            max_f1_train = max(
                f1_per_epoch_train
            )

            max_f1_valid = max(
                f1_per_epoch_valid
            )

            max_f1_scores_train.append(
                max_f1_train
            )

            max_f1_scores_valid.append(
                max_f1_valid
            )

            # ====================================================
            # BEST MODEL
            #
            # Select checkpoint by validation PR-AUC.
            #
            # This prevents "predict everything positive"
            # from automatically becoming the best model.
            # ====================================================

            is_best_model = False

            if (
                not np.isnan(valid_pr_auc)
                and valid_pr_auc > best_valid_pr_auc
            ):

                is_best_model = True

                best_valid_pr_auc = (
                    valid_pr_auc
                )

                best_valid_roc_auc = (
                    valid_roc_auc
                )

                best_train_loss = (
                    running_loss_train
                )

                best_valid_loss = (
                    running_loss_valid
                )

                best_f1_score = (
                    best_f1_this_epoch
                )

                best_f1_threshold = (
                    best_threshold_this_epoch
                )

                best_epoch = (
                    epoch + 1
                )

                best_model.load_state_dict(
                    copy.deepcopy(
                        net.state_dict()
                    )
                )

            # ----------------------------------------------------
            # Diagnostic output
            # ----------------------------------------------------

            if is_best_model:

                print(
                    "\n*** BEST MODEL UPDATED ***"
                )

                print(
                    f"Epoch             : "
                    f"{best_epoch}"
                )

                print(
                    f"Train Loss        : "
                    f"{best_train_loss:.6f}"
                )

                print(
                    f"Valid Loss        : "
                    f"{best_valid_loss:.6f}"
                )

                print(
                    f"Valid PR-AUC      : "
                    f"{best_valid_pr_auc:.6f}"
                )

                print(
                    f"Valid ROC-AUC     : "
                    f"{best_valid_roc_auc:.6f}"
                )

                print(
                    f"Best Valid F1     : "
                    f"{best_f1_score:.6f}"
                )

                print(
                    f"Best F1 Threshold : "
                    f"{best_f1_threshold:.2f}"
                )

            # ====================================================
            # Detailed diagnostic
            # ====================================================

            if (
                epoch == 0
                or (epoch + 1) % 10 == 0
                or is_best_model
            ):

                preds_05 = (
                    valid_probs >= 0.5
                ).astype(np.int64)

                tp = (
                    (valid_labels == 1)
                    &
                    (preds_05 == 1)
                ).sum()

                fp = (
                    (valid_labels == 0)
                    &
                    (preds_05 == 1)
                ).sum()

                fn = (
                    (valid_labels == 1)
                    &
                    (preds_05 == 0)
                ).sum()

                tn = (
                    (valid_labels == 0)
                    &
                    (preds_05 == 0)
                ).sum()

                pos_probs = (
                    valid_probs[
                        valid_labels == 1
                    ]
                )

                neg_probs = (
                    valid_probs[
                        valid_labels == 0
                    ]
                )

                print("\n" + "-" * 70)
                print(
                    f"EPOCH {epoch + 1} DIAGNOSTIC"
                )
                print("-" * 70)

                print(
                    f"Train Loss       : "
                    f"{running_loss_train:.6f}"
                )

                print(
                    f"Valid Loss       : "
                    f"{running_loss_valid:.6f}"
                )

                print(
                    f"Train F1@0.50    : "
                    f"{running_f1_train:.6f}"
                )

                print(
                    f"Valid F1@0.50    : "
                    f"{running_f1_valid:.6f}"
                )

                print(
                    f"Train PR-AUC     : "
                    f"{train_pr_auc:.6f}"
                )

                print(
                    f"Valid PR-AUC     : "
                    f"{valid_pr_auc:.6f}"
                )

                print(
                    f"Train ROC-AUC    : "
                    f"{train_roc_auc:.6f}"
                )

                print(
                    f"Valid ROC-AUC    : "
                    f"{valid_roc_auc:.6f}"
                )

                print(
                    f"Best F1          : "
                    f"{best_f1_this_epoch:.6f}"
                )

                print(
                    f"Best F1 threshold: "
                    f"{best_threshold_this_epoch:.2f}"
                )

                print(
                    f"TP={tp} | "
                    f"FP={fp} | "
                    f"FN={fn} | "
                    f"TN={tn}"
                )

                print(
                    f"Probability min   : "
                    f"{valid_probs.min():.6f}"
                )

                print(
                    f"Probability max   : "
                    f"{valid_probs.max():.6f}"
                )

                print(
                    f"Probability mean  : "
                    f"{valid_probs.mean():.6f}"
                )

                # ------------------------------------------------
                # CRITICAL DIAGNOSTIC
                # ------------------------------------------------

                if len(pos_probs) > 0:

                    print(
                        f"Positive prob mean: "
                        f"{pos_probs.mean():.6f}"
                    )

                    print(
                        f"Positive prob std : "
                        f"{pos_probs.std():.6f}"
                    )

                if len(neg_probs) > 0:

                    print(
                        f"Negative prob mean: "
                        f"{neg_probs.mean():.6f}"
                    )

                    print(
                        f"Negative prob std : "
                        f"{neg_probs.std():.6f}"
                    )

                if (
                    len(pos_probs) > 0
                    and len(neg_probs) > 0
                ):

                    print(
                        f"Probability "
                        f"separation      : "
                        f"{pos_probs.mean() - neg_probs.mean():.6f}"
                    )

                print("-" * 70)

                # ------------------------------------------------
                # Threshold diagnostics
                # ------------------------------------------------

                for (
                    threshold,
                    f1_tmp,
                    pred_pos,
                    pred_neg
                ) in valid_metric[
                    'threshold_results'
                ]:

                    print(
                        f"threshold={threshold:.2f} | "
                        f"true_pos="
                        f"{(valid_labels == 1).sum()} | "
                        f"true_neg="
                        f"{(valid_labels == 0).sum()} | "
                        f"pred_pos="
                        f"{pred_pos} | "
                        f"pred_neg="
                        f"{pred_neg} | "
                        f"F1={f1_tmp:.4f}"
                    )

                print("-" * 70)

            # ====================================================
            # Progress bar
            # ====================================================

            pbar.update(1)

            pbar.set_postfix(
                {
                    'train_loss':
                        f'{running_loss_train:.4f}',

                    'valid_loss':
                        f'{running_loss_valid:.4f}',

                    'train_f1':
                        f'{running_f1_train:.4f}',

                    'valid_f1':
                        f'{running_f1_valid:.4f}',

                    'valid_pr':
                        f'{valid_pr_auc:.4f}'
                }
            )

    # ============================================================
    # 20. Training finished
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print(
        f"Best epoch            : "
        f"{best_epoch}"
    )

    print(
        f"Best train loss       : "
        f"{best_train_loss:.6f}"
    )

    print(
        f"Best validation loss  : "
        f"{best_valid_loss:.6f}"
    )

    print(
        f"Best validation "
        f"PR-AUC                : "
        f"{best_valid_pr_auc:.6f}"
    )

    print(
        f"Best validation "
        f"ROC-AUC               : "
        f"{best_valid_roc_auc:.6f}"
    )

    print(
        f"Best validation F1    : "
        f"{best_f1_score:.6f}"
    )

    print(
        f"Best F1 threshold     : "
        f"{best_f1_threshold:.2f}"
    )

    # ============================================================
    # 21. Save best model
    # ============================================================

    torch.save(
        best_model.state_dict(),
        model_path
    )

    print(
        f"\nBest model saved to:\n"
        f"{model_path}"
    )

    # ============================================================
    # 22. Final embeddings from BEST MODEL
    # ============================================================

    all_embeddings, cluster_labels = (
        calculate_cluster_labels(
            best_model,
            dl_train,
            device
        )
    )

    all_embeddings = (
        all_embeddings.reshape(
            all_embeddings.shape[0],
            -1
        )
    )

    # ============================================================
    # 23. Cosine similarity
    # ============================================================

    norms = np.linalg.norm(
        all_embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1e-12

    normalized_embeddings = (
        all_embeddings /
        norms
    )

    cos_sim = np.dot(
        normalized_embeddings,
        normalized_embeddings.T
    )

    # Prevent unused-variable warnings in some environments.
    _ = cos_sim

    # ============================================================
    # 24. Plot training curves
    # ============================================================

    if plot:

        loss_path = os.path.join(
            results_path,
            f'embeddings_loss_head{num_heads}_'
            f'dim{out_feats}_lay{num_layers}_'
            f'epo{num_epochs}.png'
        )

        f1_path = os.path.join(
            results_path,
            f'embeddings_f1_head{num_heads}_'
            f'dim{out_feats}_lay{num_layers}_'
            f'epo{num_epochs}.png'
        )

        max_f1_path = os.path.join(
            results_path,
            f'embeddings_max_f1_head{num_heads}_'
            f'dim{out_feats}_lay{num_layers}_'
            f'epo{num_epochs}.png'
        )

        matrix_path = os.path.join(
            results_path,
            f'embeddings_matrix_head{num_heads}_'
            f'dim{out_feats}_lay{num_layers}_'
            f'epo{num_epochs}.png'
        )

        draw_loss_plot(
            loss_per_epoch_train,
            loss_per_epoch_valid,
            loss_path
        )

        draw_max_f1_plot(
            max_f1_scores_train,
            max_f1_scores_valid,
            max_f1_path
        )

        draw_f1_plot(
            f1_per_epoch_train,
            f1_per_epoch_valid,
            f1_path
        )

    # ============================================================
    # 25. Final visualization paths
    # ============================================================

    save_path_pca = os.path.join(
        results_path,
        f'embeddings_pca_head{num_heads}_'
        f'dim{out_feats}_lay{num_layers}_'
        f'epo{num_epochs}_final.png'
    )

    save_path_t_SNE = os.path.join(
        results_path,
        f'embeddings_t-SNE_head{num_heads}_'
        f'dim{out_feats}_lay{num_layers}_'
        f'epo{num_epochs}_final.png'
    )

    save_path_heatmap_ = os.path.join(
        results_path,
        f'embeddings_heatmap_stId_head{num_heads}_'
        f'dim{out_feats}_lay{num_layers}_'
        f'epo{num_epochs}_final.png'
    )

    save_path_matrix = os.path.join(
        results_path,
        f'embeddings_matrix_stId_head{num_heads}_'
        f'dim{out_feats}_lay{num_layers}_'
        f'epo{num_epochs}_final.png'
    )

    # ============================================================
    # 26. Final node / cluster information
    # ============================================================

    cluster_stId_dict = {}

    significant_stIds = []

    clusters_with_significant_stId = {}

    clusters_node_info = {}

    first_node_stId_in_cluster = {}
    first_node_embedding_in_cluster = {}

    stid_list = []

    # ------------------------------------------------------------
    # Process training graph
    # ------------------------------------------------------------

    for data in dl_train:

        graph, _ = data

        graph = graph.to(device)

        node_embeddings = (
            best_model
            .get_node_embeddings(graph)
            .detach()
            .cpu()
            .numpy()
        )

        graph_path = os.path.join(
            raw_path,
            'emb_train.pkl'
        )

        if not os.path.exists(
            graph_path
        ):

            raise FileNotFoundError(
                f"Could not find raw training graph: "
                f"{graph_path}"
            )

        with open(
            graph_path,
            'rb'
        ) as f:

            nx_graph = pickle.load(f)

        if (
            len(cluster_labels)
            != len(nx_graph.nodes)
        ):

            raise ValueError(
                "Final cluster labels and number of "
                "graph nodes do not match: "
                f"{len(cluster_labels)} vs "
                f"{len(nx_graph.nodes)}"
            )

        node_to_index = {
            node: idx
            for idx, node in enumerate(
                nx_graph.nodes
            )
        }

        stid_dic = {}

        # --------------------------------------------------------
        # Save embeddings and significant genes
        # --------------------------------------------------------

        for node in nx_graph.nodes:

            if (
                'stId'
                not in nx_graph.nodes[node]
            ):
                continue

            stid = (
                nx_graph.nodes[node]['stId']
            )

            idx = node_to_index[node]

            stid_dic[stid] = (
                node_embeddings[idx]
            )

            significance_value = (
                graph.ndata[
                    'significance'
                ][idx]
                .item()
            )

            if significance_value == 1:

                significant_stIds.append(
                    stid
                )

        # --------------------------------------------------------
        # Save final embeddings
        # --------------------------------------------------------

        stid_df_final = (
            pd.DataFrame.from_dict(
                stid_dic,
                orient='index'
            )
        )

        csv_save_path_final = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_'
            f'dim{out_feats}_lay{num_layers}_'
            f'epo{num_epochs}_final.csv'
        )

        stid_df_final.to_csv(
            csv_save_path_final,
            index_label='stId'
        )

        # --------------------------------------------------------
        # Cluster information
        # --------------------------------------------------------

        for node, cluster in zip(
            nx_graph.nodes,
            cluster_labels
        ):

            if (
                'stId'
                not in nx_graph.nodes[node]
            ):
                continue

            stid = (
                nx_graph.nodes[node]['stId']
            )

            idx = (
                node_to_index[node]
            )

            significance_value = (
                graph.ndata[
                    'significance'
                ][idx]
                .item()
            )

            # ----------------------------------------------------
            # First representative
            # ----------------------------------------------------

            if cluster not in (
                first_node_stId_in_cluster
            ):

                first_node_stId_in_cluster[
                    cluster
                ] = stid

                first_node_embedding_in_cluster[
                    cluster
                ] = node_embeddings[idx]

            # ----------------------------------------------------
            # All stIds by cluster
            # ----------------------------------------------------

            if cluster not in (
                cluster_stId_dict
            ):

                cluster_stId_dict[
                    cluster
                ] = []

            cluster_stId_dict[
                cluster
            ].append(stid)

            # ----------------------------------------------------
            # Significant stIds by cluster
            # ----------------------------------------------------

            if cluster not in (
                clusters_with_significant_stId
            ):

                clusters_with_significant_stId[
                    cluster
                ] = []

            if significance_value == 1:

                clusters_with_significant_stId[
                    cluster
                ].append(stid)

            # ----------------------------------------------------
            # Node information
            # ----------------------------------------------------

            if cluster not in (
                clusters_node_info
            ):

                clusters_node_info[
                    cluster
                ] = []

            node_info = {
                'stId': stid,
                'significance':
                    significance_value,
                'other_info':
                    nx_graph.nodes[node]
            }

            clusters_node_info[
                cluster
            ].append(node_info)

        print(
            "\nFirst node in each cluster:"
        )

        print(
            first_node_stId_in_cluster
        )

        # --------------------------------------------------------
        # Cluster representatives
        # --------------------------------------------------------

        sorted_clusters = sorted(
            first_node_stId_in_cluster.keys()
        )

        stid_list = [
            first_node_stId_in_cluster[c]
            for c in sorted_clusters
        ]

        embedding_list = [
            first_node_embedding_in_cluster[c]
            for c in sorted_clusters
        ]

        embedding_list = [
            np.asarray(emb).reshape(-1)
            for emb in embedding_list
        ]

        heatmap_data = pd.DataFrame(
            embedding_list,
            index=stid_list
        )

        print(
            f"Number of clusters: "
            f"{len(sorted_clusters)}"
        )

        print(
            f"Embedding matrix shape: "
            f"{heatmap_data.shape}"
        )

        # --------------------------------------------------------
        # Cluster plots
        # --------------------------------------------------------

        create_heatmap_with_stid(
            embedding_list,
            stid_list,
            save_path_heatmap_
        )

        plot_cosine_similarity_matrix_for_clusters_with_values(
            embedding_list,
            stid_list,
            save_path_matrix
        )

        break

    # ============================================================
    # 27. Final embedding visualization
    # ============================================================

    if (
        len(all_embeddings) > 1
        and len(
            np.unique(
                cluster_labels
            )
        ) > 1
    ):

        visualize_embeddings_tsne(
            all_embeddings,
            cluster_labels,
            stid_list,
            save_path_t_SNE
        )

        visualize_embeddings_pca(
            all_embeddings,
            cluster_labels,
            first_node_stId_in_cluster,
            save_path_pca
        )

    # ============================================================
    # 28. Final clustering metrics
    # ============================================================

    if (
        len(all_embeddings) > 1
        and len(
            np.unique(
                cluster_labels
            )
        ) > 1
    ):

        silhouette_avg = (
            silhouette_score(
                all_embeddings,
                cluster_labels
            )
        )

        davies_bouldin = (
            davies_bouldin_score(
                all_embeddings,
                cluster_labels
            )
        )

    else:

        silhouette_avg = np.nan
        davies_bouldin = np.nan

    print(
        f"\nSilhouette Score: "
        f"{silhouette_avg}"
    )

    print(
        f"Davies-Bouldin Index: "
        f"{davies_bouldin}"
    )

    # ============================================================
    # 29. Final summary
    # ============================================================

    summary = (
        f"Epochs: {num_epochs}\n"
    )

    summary += (
        f"Best Epoch: "
        f"{best_epoch}\n"
    )

    summary += (
        f"Best Train Loss: "
        f"{best_train_loss}\n"
    )

    summary += (
        f"Best Validation Loss: "
        f"{best_valid_loss}\n"
    )

    summary += (
        f"Best Validation PR-AUC: "
        f"{best_valid_pr_auc}\n"
    )

    summary += (
        f"Best Validation ROC-AUC: "
        f"{best_valid_roc_auc}\n"
    )

    summary += (
        f"Best F1 Score: "
        f"{best_f1_score}\n"
    )

    summary += (
        f"Best F1 Threshold: "
        f"{best_f1_threshold}\n"
    )

    summary += (
        f"Final Max F1 Train: "
        f"{max_f1_train}\n"
    )

    summary += (
        f"Final Max F1 Valid: "
        f"{max_f1_valid}\n"
    )

    summary += (
        f"Silhouette Score: "
        f"{silhouette_avg}\n"
    )

    summary += (
        f"Davies-Bouldin Index: "
        f"{davies_bouldin}\n"
    )

    summary += (
        f"Number of significant genes: "
        f"{len(significant_stIds)}\n"
    )

    summary += (
        f"Input feature dimension: "
        f"{actual_in_feats}\n"
    )

    summary += (
        f"Model path: "
        f"{model_path}\n"
    )

    save_file = os.path.join(
        results_path,
        f'embeddings_head{num_heads}_'
        f'dim{out_feats}_lay{num_layers}_'
        f'epo{num_epochs}.txt'
    )

    with open(
        save_file,
        'w'
    ) as f:

        f.write(summary)

    # ============================================================
    # 30. Final information
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL STAGE-1 RESULTS")
    print("=" * 70)

    print(
        f"Best epoch            : "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss  : "
        f"{best_valid_loss:.6f}"
    )

    print(
        f"Best validation PR-AUC: "
        f"{best_valid_pr_auc:.6f}"
    )

    print(
        f"Best validation ROC-AUC: "
        f"{best_valid_roc_auc:.6f}"
    )

    print(
        f"Best validation F1    : "
        f"{best_f1_score:.6f}"
    )

    print(
        f"Best F1 threshold     : "
        f"{best_f1_threshold:.2f}"
    )

    print(
        f"Max training F1       : "
        f"{max_f1_train:.6f}"
    )

    print(
        f"Max validation F1     : "
        f"{max_f1_valid:.6f}"
    )

    print(
        f"Significant genes     : "
        f"{len(significant_stIds)}"
    )

    print(
        f"Input feature dim     : "
        f"{actual_in_feats}"
    )

    print(
        f"Model path:\n"
        f"{model_path}"
    )

    print("=" * 70)

    return model_path

def train__pas(
    hyperparams=None,
    data_path='../data/omics/',
    plot=True,
    omics='cna',
    cancer='BLCA'
):
    """
    Stage-1 training:
        Predict generic biological significance from the graph.

    IMPORTANT:
        This stage does NOT use cancer-driver labels.
        'significance' is the Stage-1 pretraining target.

    Expected significance values:
        0 = not significant
        1 = significant

    If -1 is present:
        -1 = unlabeled / ignored during loss and F1 calculation
    """

    import os
    import copy
    import pickle

    import numpy as np
    import pandas as pd
    import torch
    import torch.optim as optim
    from tqdm import tqdm

    from sklearn import metrics
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score
    )

    # ============================================================
    # 1. Hyperparameters
    # ============================================================

    if hyperparams is None:
        raise ValueError("hyperparams must be provided.")

    num_epochs = hyperparams['num_epochs']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']

    print("\n" + "=" * 70)
    print("START STAGE-1 SIGNIFICANCE PRETRAINING")
    print("=" * 70)
    print(f"Omics       : {omics}")
    print(f"Cancer      : {cancer}")
    print(f"Device      : {device}")
    print(f"Epochs      : {num_epochs}")
    print(f"LR          : {learning_rate}")
    print(f"Batch size  : {batch_size}")
    print(f"Out feats   : {out_feats}")
    print(f"Layers      : {num_layers}")
    print("=" * 70 + "\n")

    # ============================================================
    # 2. Model save path
    # ============================================================

    model_path = os.path.join(data_path, omics, cancer, 'emb', 'models')
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(
        model_path,
        f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth'
    )

    # ============================================================
    # 3. Load dataset and add node-degree structural features
    # ============================================================

    data_path_ = os.path.join(data_path, omics, cancer)
    ds = dataset.Dataset(data_path_)

    # Add node in-degree (normalized) as an extra structural feature to
    # both the training (0) and validation (1) graphs.
    for idx in [0, 1]:
        g, _ = ds[idx]

        node_degrees = g.in_degrees().float().unsqueeze(-1)
        node_degrees_norm = (
            (node_degrees - node_degrees.mean())
            / (node_degrees.std() + 1e-5)
        )

        if 'weight' in g.ndata:
            existing_feats = g.ndata['weight']
        elif 'feat' in g.ndata:
            existing_feats = g.ndata['feat']
        else:
            # No node features initialized yet — fall back to a constant.
            existing_feats = torch.ones((g.num_nodes(), 1), dtype=torch.float32)

        if existing_feats.dim() == 1:
            existing_feats = existing_feats.unsqueeze(-1)

        updated_feats = torch.cat([existing_feats, node_degrees_norm], dim=-1)

        # Keep both keys in sync so downstream code reading either won't break.
        g.ndata['feat'] = updated_feats
        g.ndata['weight'] = updated_feats

    graph, name = ds[0]

    # ============================================================
    # 4. SIGNIFICANCE LABEL DEBUG
    # ============================================================

    if 'significance' not in graph.ndata:
        raise KeyError(
            "graph.ndata['significance'] does not exist. "
            "Check your preprocessing and DGL graph conversion."
        )

    significance = graph.ndata['significance']

    print("\n" + "=" * 70)
    print("SIGNIFICANCE DEBUG")
    print("=" * 70)
    print("dtype :", significance.dtype)
    print("shape :", significance.shape)

    unique_labels, label_counts = torch.unique(significance, return_counts=True)
    print("unique labels:")
    for label, count in zip(unique_labels, label_counts):
        print(f"    label={label.item():>4} count={count.item():>6}")
    print("first 20:", significance[:20])
    print("=" * 70)

    valid_labels = significance[(significance == 0) | (significance == 1)]

    if valid_labels.numel() == 0:
        raise ValueError("No valid significance labels (0/1) were found.")

    num_positive = (valid_labels == 1).sum().item()
    num_negative = (valid_labels == 0).sum().item()
    num_unlabeled = (significance == -1).sum().item()

    print("\nLABEL DISTRIBUTION")
    print(f"Positive / significant : {num_positive}")
    print(f"Negative / non-signif. : {num_negative}")
    print(f"Unlabeled (-1)         : {num_unlabeled}")

    if num_positive == 0:
        raise ValueError("There are ZERO positive significance labels. F1 will necessarily be 0.")
    if num_negative == 0:
        raise ValueError("There are ZERO negative significance labels. Check your preprocessing.")

    positive_ratio = num_positive / valid_labels.numel()
    print(f"Positive ratio          : {positive_ratio:.4%}")

    # ============================================================
    # 5. Train / validation graphs
    # ============================================================

    ds_train = [ds[0]]
    ds_valid = [ds[1]]

    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # ============================================================
    # 6. Model / optimizer / best-model tracker
    # ============================================================

    net = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    # ============================================================
    # 7. Focal Loss (dynamic class balancing)
    # ============================================================

    # alpha favors the rare positive class: weight ~ (negatives / total)
    total_valid = num_positive + num_negative
    dynamic_alpha = num_negative / total_valid

    print(f"Configuring Focal Loss with Dynamic Alpha: {dynamic_alpha:.4f} (favoring positive class)")

    criterion = FocalLoss(alpha=dynamic_alpha, gamma=2.0, reduction='mean')

    # NOTE: do NOT use a hardcoded weight tensor like [0.00001, 0.99999] —
    # that nearly fully suppresses class 0. And since FocalLoss(reduction='mean')
    # already returns a scalar, multiplying it by node-level weights afterwards
    # would be incorrect, so alpha is passed straight into FocalLoss instead.

    # ============================================================
    # 8. Training history
    # ============================================================

    loss_per_epoch_train = []
    loss_per_epoch_valid = []

    f1_per_epoch_train = []
    f1_per_epoch_valid = []

    max_f1_scores_train = []
    max_f1_scores_valid = []

    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_f1_score = 0.0
    best_epoch = 0

    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    # ============================================================
    # 9. Results path
    # ============================================================

    results_path = os.path.abspath(
        os.path.join('results', 'multiomics_meth', 'node_embeddings', omics, cancer)
    )
    os.makedirs(results_path, exist_ok=True)

    # ============================================================
    # 10. INITIAL EMBEDDINGS
    # ============================================================

    print("\n" + "=" * 70)
    print("CALCULATING INITIAL EMBEDDINGS")
    print("=" * 70)

    all_embeddings_initial, cluster_labels_initial = calculate_cluster_labels(
        best_model, dl_train, device
    )
    all_embeddings_initial = all_embeddings_initial.reshape(all_embeddings_initial.shape[0], -1)

    save_path_heatmap_initial = os.path.join(
        results_path, f'embeddings_heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_matrix_initial = os.path.join(
        results_path, f'embeddings_matrix_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_pca_initial = os.path.join(
        results_path, f'embeddings_pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_t_SNE_initial = os.path.join(
        results_path, f'embeddings_t-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )

    # ============================================================
    # 11. INITIAL EMBEDDING INFORMATION
    # ============================================================

    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings_initial = (
            best_model.get_node_embeddings(graph).detach().cpu().numpy()
        )

        graph_path = os.path.join(data_path, omics, cancer, 'emb', 'raw', 'emb_train.pkl')
        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert len(cluster_labels_initial) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index_initial = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster_initial = {}
        first_node_embedding_in_cluster_initial = {}
        stid_dic_initial = {}

        for node in nx_graph.nodes:
            if 'stId' in nx_graph.nodes[node]:
                stId = nx_graph.nodes[node]['stId']
                stid_dic_initial[stId] = node_embeddings_initial[node_to_index_initial[node]]

        stid_df_initial = pd.DataFrame.from_dict(stid_dic_initial, orient='index')

        csv_save_path_initial = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.csv'
        )
        stid_df_initial.to_csv(csv_save_path_initial, index_label='stId')

        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if 'stId' in nx_graph.nodes[node]:
                if cluster not in first_node_stId_in_cluster_initial:
                    first_node_stId_in_cluster_initial[cluster] = nx_graph.nodes[node]['stId']
                    first_node_embedding_in_cluster_initial[cluster] = (
                        node_embeddings_initial[node_to_index_initial[node]]
                    )

        print("first_node_stId_in_cluster_initial:")
        print(first_node_stId_in_cluster_initial)

        sorted_clusters_initial = sorted(first_node_stId_in_cluster_initial.keys())
        stid_list_initial = [first_node_stId_in_cluster_initial[c] for c in sorted_clusters_initial]
        embedding_list_initial = [
            np.asarray(first_node_embedding_in_cluster_initial[c]).reshape(-1)
            for c in sorted_clusters_initial
        ]

        heatmap_data_initial = pd.DataFrame(embedding_list_initial, index=stid_list_initial)

        print(f"Number of clusters: {len(sorted_clusters_initial)}")
        print(f"Embedding matrix shape: {heatmap_data_initial.shape}")

        create_heatmap_with_stid(embedding_list_initial, stid_list_initial, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(
            embedding_list_initial, stid_list_initial, save_path_matrix_initial
        )

        break

    # ============================================================
    # 12. INITIAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(
        all_embeddings_initial, cluster_labels_initial, stid_list_initial, save_path_t_SNE_initial
    )
    visualize_embeddings_pca(
        all_embeddings_initial, cluster_labels_initial,
        first_node_stId_in_cluster_initial, save_path_pca_initial
    )

    if len(np.unique(cluster_labels_initial)) > 1:
        silhouette_avg_ = silhouette_score(all_embeddings_initial, cluster_labels_initial)
        davies_bouldin_ = davies_bouldin_score(all_embeddings_initial, cluster_labels_initial)
    else:
        silhouette_avg_ = np.nan
        davies_bouldin_ = np.nan

    summary_ = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_ = os.path.join(
        results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt'
    )
    with open(save_file_, 'w') as f:
        f.write(summary_)

    # ============================================================
    # 13. TRAINING
    # ============================================================

    print("\n" + "=" * 70)
    print("START TRAINING")
    print("=" * 70)

    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=True) as pbar:

        for epoch in range(num_epochs):

            # ---------------- TRAIN ----------------
            net.train()
            loss_per_graph = []
            f1_per_graph = []

            for data in dl_train:
                graph, name = data
                graph = graph.to(device)

                logits = net(graph).view(-1)
                labels = graph.ndata['significance'].float().view(-1)

                valid_mask = (labels == 0) | (labels == 1)
                if valid_mask.sum().item() == 0:
                    raise ValueError("Training graph contains no valid significance labels.")

                valid_logits = logits[valid_mask]
                valid_labels = labels[valid_mask]

                loss = criterion(valid_logits, valid_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_per_graph.append(loss.item())

                with torch.no_grad():
                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()
                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0
                    )
                    f1_per_graph.append(f1)

            running_loss_train = float(np.mean(loss_per_graph))
            running_f1_train = float(np.mean(f1_per_graph))
            loss_per_epoch_train.append(running_loss_train)
            f1_per_epoch_train.append(running_f1_train)

            # ---------------- VALIDATION ----------------
            net.eval()
            loss_per_graph = []
            f1_per_graph = []

            # Keep the last batch's probs/labels around for the threshold
            # diagnostic and best-model selection below.
            val_probs, val_labels_int = None, None

            with torch.no_grad():
                for data in dl_valid:
                    graph, name = data
                    graph = graph.to(device)

                    logits = net(graph).view(-1)
                    labels = graph.ndata['significance'].float().view(-1)

                    valid_mask = (labels == 0) | (labels == 1)
                    if valid_mask.sum().item() == 0:
                        continue

                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]

                    loss = criterion(valid_logits, valid_labels)
                    loss_per_graph.append(loss.item())

                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()

                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0
                    )
                    f1_per_graph.append(f1)

                    val_probs, val_labels_int = probs, labels_int

            if len(loss_per_graph) == 0:
                raise ValueError("Validation graph contains no valid significance labels.")

            running_loss_valid = float(np.mean(loss_per_graph))
            running_f1_val = float(np.mean(f1_per_graph))
            loss_per_epoch_valid.append(running_loss_valid)
            f1_per_epoch_valid.append(running_f1_val)

            max_f1_train = max(f1_per_epoch_train)
            max_f1_valid = max(f1_per_epoch_valid)
            max_f1_scores_train.append(max_f1_train)
            max_f1_scores_valid.append(max_f1_valid)

            # ---------------- BEST MODEL SELECTION ----------------
            # Checkpoint whenever validation loss improves. This is decoupled
            # from the threshold scan below (which is a diagnostic only) so
            # the checkpoint save can never be silently skipped.
            if running_loss_valid < best_valid_loss:

                best_train_loss = running_loss_train
                best_valid_loss = running_loss_valid
                best_f1_score = running_f1_val
                best_epoch = epoch + 1

                best_model.load_state_dict(copy.deepcopy(net.state_dict()))

                # Diagnostic only: report the best F1 achievable this epoch
                # across thresholds, without changing whether we save.
                best_f1_this_epoch = 0.0
                best_threshold_this_epoch = 0.2
                if val_probs is not None:
                    for threshold in THRESHOLDS:
                        preds_tmp = (val_probs > threshold).int()
                        f1_tmp = metrics.f1_score(
                            val_labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0
                        )
                        if f1_tmp > best_f1_this_epoch:
                            best_f1_this_epoch = f1_tmp
                            best_threshold_this_epoch = threshold

                print("\n*** BEST MODEL UPDATED ***")
                print(f"Epoch                : {best_epoch}")
                print(f"Train Loss           : {best_train_loss:.6f}")
                print(f"Valid Loss           : {best_valid_loss:.6f}")
                print(f"Valid F1 (thr=0.2)   : {best_f1_score:.6f}")
                print(f"Best F1 across thresh: {best_f1_this_epoch:.6f} (thr={best_threshold_this_epoch:.2f})")

            # ---------------- DETAILED DIAGNOSTIC EVERY 10 EPOCHS ----------------
            if epoch == 0 or (epoch + 1) % 10 == 0:

                for data in dl_valid:
                    graph, _ = data
                    graph = graph.to(device)

                    logits = net(graph).view(-1)
                    labels = graph.ndata['significance'].float().view(-1)

                    valid_mask = (labels == 0) | (labels == 1)
                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]

                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()

                    tp = ((labels_int == 1) & (preds == 1)).sum().item()
                    fp = ((labels_int == 0) & (preds == 1)).sum().item()
                    fn = ((labels_int == 1) & (preds == 0)).sum().item()
                    tn = ((labels_int == 0) & (preds == 0)).sum().item()

                    print("\n" + "-" * 70)
                    print(f"EPOCH {epoch + 1} DIAGNOSTIC")
                    print("-" * 70)
                    print(f"Train Loss : {running_loss_train:.6f}")
                    print(f"Valid Loss : {running_loss_valid:.6f}")
                    print(f"Train F1   : {running_f1_train:.6f}")
                    print(f"Valid F1   : {running_f1_val:.6f}")
                    print(f"TP={tp} | FP={fp} | FN={fn} | TN={tn}")
                    print(f"Probability min  : {probs.min().item():.6f}")
                    print(f"Probability max  : {probs.max().item():.6f}")
                    print(f"Probability mean : {probs.mean().item():.6f}")
                    print("-" * 70)

                    for threshold in THRESHOLDS:
                        preds_tmp = (probs > threshold).int()
                        f1_tmp = metrics.f1_score(
                            labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0
                        )
                        print(
                            f"threshold={threshold:.2f} | "
                            f"true_pos={(labels_int == 1).sum().item()} | "
                            f"true_neg={(labels_int == 0).sum().item()} | "
                            f"pred_pos={(preds_tmp == 1).sum().item()} | "
                            f"pred_neg={(preds_tmp == 0).sum().item()} | "
                            f"F1={f1_tmp:.4f}"
                        )

                    break

            pbar.update(1)
            pbar.set_postfix({
                'train_loss': f'{running_loss_train:.4f}',
                'valid_loss': f'{running_loss_valid:.4f}',
                'train_f1': f'{running_f1_train:.4f}',
                'valid_f1': f'{running_f1_val:.4f}'
            })

    # ============================================================
    # 14. FINAL BEST MODEL
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Best epoch       : {best_epoch}")
    print(f"Best train loss  : {best_train_loss:.6f}")
    print(f"Best valid loss  : {best_valid_loss:.6f}")
    print(f"Best valid F1    : {best_f1_score:.6f}")

    # ============================================================
    # 15. FINAL EMBEDDINGS
    # ============================================================

    all_embeddings, cluster_labels = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)

    # ============================================================
    # 16. COSINE SIMILARITY
    # ============================================================

    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized_embeddings = all_embeddings / norms
    cos_sim = np.dot(normalized_embeddings, normalized_embeddings.T)

    # ============================================================
    # 17. PLOTS
    # ============================================================

    if plot:
        loss_path = os.path.join(
            results_path, f'embeddings_loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        f1_path = os.path.join(
            results_path, f'embeddings_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        max_f1_path = os.path.join(
            results_path, f'embeddings_max_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_max_f1_plot(max_f1_scores_train, max_f1_scores_valid, max_f1_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

    # ============================================================
    # 18. SAVE BEST MODEL
    # ============================================================

    torch.save(best_model.state_dict(), model_path)
    print(f"\nBest model saved to:\n{model_path}")

    # ============================================================
    # 19. FINAL VISUALIZATION PATHS
    # ============================================================

    save_path_pca = os.path.join(
        results_path, f'embeddings_pca_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_t_SNE = os.path.join(
        results_path, f'embeddings_t-SNE_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_heatmap_ = os.path.join(
        results_path, f'embeddings_heatmap_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_matrix = os.path.join(
        results_path, f'embeddings_matrix_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )

    # ============================================================
    # 20. FINAL NODE / CLUSTER INFORMATION
    # ============================================================

    cluster_stId_dict = {}
    significant_stIds = []
    clusters_with_significant_stId = {}
    clusters_node_info = {}

    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()

        graph_path = os.path.join(data_path, omics, cancer, 'emb', 'raw', 'emb_train.pkl')
        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert len(cluster_labels) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster = {}
        first_node_embedding_in_cluster = {}
        stid_dic = {}

        for node in nx_graph.nodes:
            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]
            stid_dic[stid] = node_embeddings[idx]

            significance_value = graph.ndata['significance'][idx].item()
            if significance_value == 1:
                significant_stIds.append(stid)

        stid_df_final = pd.DataFrame.from_dict(stid_dic, orient='index')
        csv_save_path_final = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.csv'
        )
        stid_df_final.to_csv(csv_save_path_final, index_label='stId')

        for node, cluster in zip(nx_graph.nodes, cluster_labels):
            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]
            significance_value = graph.ndata['significance'][idx].item()

            if cluster not in first_node_stId_in_cluster:
                first_node_stId_in_cluster[cluster] = stid
                first_node_embedding_in_cluster[cluster] = node_embeddings[idx]

            cluster_stId_dict.setdefault(cluster, []).append(stid)

            clusters_with_significant_stId.setdefault(cluster, [])
            if significance_value == 1:
                clusters_with_significant_stId[cluster].append(stid)

            clusters_node_info.setdefault(cluster, []).append({
                'stId': stid,
                'significance': significance_value,
                'other_info': nx_graph.nodes[node]
            })

        print("\nFirst node in each cluster:")
        print(first_node_stId_in_cluster)

        sorted_clusters = sorted(first_node_stId_in_cluster.keys())
        stid_list = [first_node_stId_in_cluster[c] for c in sorted_clusters]
        embedding_list = [
            np.asarray(first_node_embedding_in_cluster[c]).reshape(-1)
            for c in sorted_clusters
        ]

        heatmap_data = pd.DataFrame(embedding_list, index=stid_list)

        print(f"Number of clusters: {len(sorted_clusters)}")
        print(f"Embedding matrix shape: {heatmap_data.shape}")

        create_heatmap_with_stid(embedding_list, stid_list, save_path_heatmap_)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list, stid_list, save_path_matrix)

        break

    # ============================================================
    # 21. FINAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(all_embeddings, cluster_labels, stid_list, save_path_t_SNE)
    visualize_embeddings_pca(all_embeddings, cluster_labels, first_node_stId_in_cluster, save_path_pca)

    # ============================================================
    # 22. FINAL CLUSTERING METRICS
    # ============================================================

    if len(np.unique(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(all_embeddings, cluster_labels)
    else:
        silhouette_avg = np.nan
        davies_bouldin = np.nan

    print(f"\nSilhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    # ============================================================
    # 23. FINAL SUMMARY
    # ============================================================

    summary = (
        f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    )
    summary += f"Best Epoch: {best_epoch}\n"
    summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {best_f1_score}\n"
    summary += f"Silhouette Score: {silhouette_avg}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin}\n"
    summary += f"Number of significant genes: {len(significant_stIds)}\n"

    save_file = os.path.join(
        results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt'
    )
    with open(save_file, 'w') as f:
        f.write(summary)

    # ============================================================
    # 24. FINAL INFORMATION
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL STAGE-1 RESULTS")
    print("=" * 70)
    print(f"Best epoch            : {best_epoch}")
    print(f"Best validation loss  : {best_valid_loss:.6f}")
    print(f"Best validation F1    : {best_f1_score:.6f}")
    print(f"Max training F1       : {max_f1_train:.6f}")
    print(f"Max validation F1     : {max_f1_valid:.6f}")
    print(f"Significant genes     : {len(significant_stIds)}")
    print(f"Model path:\n{model_path}")
    print("=" * 70)

    return model_path


def train_1(
    hyperparams=None,
    data_path='../data/omics/',
    plot=True,
    omics='cna',
    cancer='BLCA'
):
    """
    Stage-1 training:
        Predict generic biological significance from the graph.

    IMPORTANT:
        This stage does NOT use cancer-driver labels.
        'significance' is the Stage-1 pretraining target.

    Expected significance values:
        0 = not significant
        1 = significant

    If -1 is present:
        -1 = unlabeled / ignored during loss and F1 calculation
    """

    import os
    import copy
    import pickle

    import numpy as np
    import pandas as pd
    import torch
    import torch.optim as optim
    from tqdm import tqdm

    from sklearn import metrics
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score
    )

    # ============================================================
    # 1. Hyperparameters
    # ============================================================

    if hyperparams is None:
        raise ValueError("hyperparams must be provided.")

    num_epochs = hyperparams['num_epochs']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']

    print("\n" + "=" * 70)
    print("START STAGE-1 SIGNIFICANCE PRETRAINING")
    print("=" * 70)
    print(f"Omics       : {omics}")
    print(f"Cancer      : {cancer}")
    print(f"Device      : {device}")
    print(f"Epochs      : {num_epochs}")
    print(f"LR          : {learning_rate}")
    print(f"Batch size  : {batch_size}")
    print(f"Out feats   : {out_feats}")
    print(f"Layers      : {num_layers}")
    print(f"Loss type   : {hyperparams.get('loss_type', 'focal')}")
    print("=" * 70 + "\n")

    # ============================================================
    # 2. Model save path
    # ============================================================

    model_path = os.path.join(data_path, omics, cancer, 'emb', 'models')
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(
        model_path,
        f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth'
    )

    # ============================================================
    # 3. Load dataset and add node-degree structural features
    # ============================================================

    data_path_ = os.path.join(data_path, omics, cancer)
    ds = dataset.Dataset(data_path_)

    # Add node in-degree (normalized) as an extra structural feature to
    # both the training (0) and validation (1) graphs.
    for idx in [0, 1]:
        g, _ = ds[idx]

        node_degrees = g.in_degrees().float().unsqueeze(-1)
        node_degrees_norm = (
            (node_degrees - node_degrees.mean())
            / (node_degrees.std() + 1e-5)
        )

        if 'weight' in g.ndata:
            existing_feats = g.ndata['weight']
        elif 'feat' in g.ndata:
            existing_feats = g.ndata['feat']
        else:
            # No node features initialized yet — fall back to a constant.
            existing_feats = torch.ones((g.num_nodes(), 1), dtype=torch.float32)

        if existing_feats.dim() == 1:
            existing_feats = existing_feats.unsqueeze(-1)

        updated_feats = torch.cat([existing_feats, node_degrees_norm], dim=-1)

        # Keep both keys in sync so downstream code reading either won't break.
        g.ndata['feat'] = updated_feats
        g.ndata['weight'] = updated_feats

    graph, name = ds[0]

    # ============================================================
    # 4. SIGNIFICANCE LABEL DEBUG
    # ============================================================

    if 'significance' not in graph.ndata:
        raise KeyError(
            "graph.ndata['significance'] does not exist. "
            "Check your preprocessing and DGL graph conversion."
        )

    significance = graph.ndata['significance']

    print("\n" + "=" * 70)
    print("SIGNIFICANCE DEBUG")
    print("=" * 70)
    print("dtype :", significance.dtype)
    print("shape :", significance.shape)

    unique_labels, label_counts = torch.unique(significance, return_counts=True)
    print("unique labels:")
    for label, count in zip(unique_labels, label_counts):
        print(f"    label={label.item():>4} count={count.item():>6}")
    print("first 20:", significance[:20])
    print("=" * 70)

    valid_labels = significance[(significance == 0) | (significance == 1)]

    if valid_labels.numel() == 0:
        raise ValueError("No valid significance labels (0/1) were found.")

    num_positive = (valid_labels == 1).sum().item()
    num_negative = (valid_labels == 0).sum().item()
    num_unlabeled = (significance == -1).sum().item()

    print("\nLABEL DISTRIBUTION")
    print(f"Positive / significant : {num_positive}")
    print(f"Negative / non-signif. : {num_negative}")
    print(f"Unlabeled (-1)         : {num_unlabeled}")

    if num_positive == 0:
        raise ValueError("There are ZERO positive significance labels. F1 will necessarily be 0.")
    if num_negative == 0:
        raise ValueError("There are ZERO negative significance labels. Check your preprocessing.")

    positive_ratio = num_positive / valid_labels.numel()
    print(f"Positive ratio          : {positive_ratio:.4%}")

    # ============================================================
    # 5. Train / validation graphs
    # ============================================================

    ds_train = [ds[0]]
    ds_valid = [ds[1]]

    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # ============================================================
    # 6. Model / optimizer / best-model tracker
    # ============================================================

    net = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    # ============================================================
    # 7. Loss function
    # ============================================================

    # loss_type lets you A/B test FocalLoss against plain BCEWithLogitsLoss
    # without touching the rest of the training loop. Pass e.g.
    # hyperparams['loss_type'] = 'bce' to use the baseline.
    loss_type = hyperparams.get('loss_type', 'focal')

    total_valid = num_positive + num_negative

    if loss_type == 'bce':
        # Standard BCE class-balancing convention: pos_weight scales the
        # loss contribution of positive examples. Since positives are the
        # minority class here, pos_weight > 1 upweights them.
        pos_weight_value = num_negative / max(num_positive, 1)
        pos_weight = torch.tensor(pos_weight_value, device=device)

        print(
            f"Configuring BCEWithLogitsLoss (baseline) with "
            f"pos_weight={pos_weight_value:.4f}"
        )

        bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
        criterion = bce_criterion

    else:
        # alpha favors the rare positive class: weight ~ (negatives / total)
        dynamic_alpha = num_negative / total_valid

        print(f"Configuring Focal Loss with Dynamic Alpha: {dynamic_alpha:.4f} (favoring positive class)")

        criterion = FocalLoss(alpha=dynamic_alpha, gamma=2.0, reduction='mean')

        # NOTE: do NOT use a hardcoded weight tensor like [0.00001, 0.99999] —
        # that nearly fully suppresses class 0. And since FocalLoss(reduction='mean')
        # already returns a scalar, multiplying it by node-level weights afterwards
        # would be incorrect, so alpha is passed straight into FocalLoss instead.

    # ============================================================
    # 8. Training history
    # ============================================================

    loss_per_epoch_train = []
    loss_per_epoch_valid = []

    f1_per_epoch_train = []
    f1_per_epoch_valid = []

    max_f1_scores_train = []
    max_f1_scores_valid = []

    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_f1_score = 0.0
    best_epoch = 0

    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    # ============================================================
    # 9. Results path
    # ============================================================

    results_path = os.path.abspath(
        os.path.join('results', 'multiomics_meth', 'node_embeddings', omics, cancer)
    )
    os.makedirs(results_path, exist_ok=True)

    # ============================================================
    # 10. INITIAL EMBEDDINGS
    # ============================================================

    print("\n" + "=" * 70)
    print("CALCULATING INITIAL EMBEDDINGS")
    print("=" * 70)

    all_embeddings_initial, cluster_labels_initial = calculate_cluster_labels(
        best_model, dl_train, device
    )
    all_embeddings_initial = all_embeddings_initial.reshape(all_embeddings_initial.shape[0], -1)

    save_path_heatmap_initial = os.path.join(
        results_path, f'embeddings_heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_matrix_initial = os.path.join(
        results_path, f'embeddings_matrix_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_pca_initial = os.path.join(
        results_path, f'embeddings_pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_t_SNE_initial = os.path.join(
        results_path, f'embeddings_t-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )

    # ============================================================
    # 11. INITIAL EMBEDDING INFORMATION
    # ============================================================

    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings_initial = (
            best_model.get_node_embeddings(graph).detach().cpu().numpy()
        )

        graph_path = os.path.join(data_path, omics, cancer, 'emb', 'raw', 'emb_train.pkl')
        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert len(cluster_labels_initial) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index_initial = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster_initial = {}
        first_node_embedding_in_cluster_initial = {}
        stid_dic_initial = {}

        for node in nx_graph.nodes:
            if 'stId' in nx_graph.nodes[node]:
                stId = nx_graph.nodes[node]['stId']
                stid_dic_initial[stId] = node_embeddings_initial[node_to_index_initial[node]]

        stid_df_initial = pd.DataFrame.from_dict(stid_dic_initial, orient='index')

        csv_save_path_initial = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.csv'
        )
        stid_df_initial.to_csv(csv_save_path_initial, index_label='stId')

        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if 'stId' in nx_graph.nodes[node]:
                if cluster not in first_node_stId_in_cluster_initial:
                    first_node_stId_in_cluster_initial[cluster] = nx_graph.nodes[node]['stId']
                    first_node_embedding_in_cluster_initial[cluster] = (
                        node_embeddings_initial[node_to_index_initial[node]]
                    )

        print("first_node_stId_in_cluster_initial:")
        print(first_node_stId_in_cluster_initial)

        sorted_clusters_initial = sorted(first_node_stId_in_cluster_initial.keys())
        stid_list_initial = [first_node_stId_in_cluster_initial[c] for c in sorted_clusters_initial]
        embedding_list_initial = [
            np.asarray(first_node_embedding_in_cluster_initial[c]).reshape(-1)
            for c in sorted_clusters_initial
        ]

        heatmap_data_initial = pd.DataFrame(embedding_list_initial, index=stid_list_initial)

        print(f"Number of clusters: {len(sorted_clusters_initial)}")
        print(f"Embedding matrix shape: {heatmap_data_initial.shape}")

        create_heatmap_with_stid(embedding_list_initial, stid_list_initial, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(
            embedding_list_initial, stid_list_initial, save_path_matrix_initial
        )

        break

    # ============================================================
    # 12. INITIAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(
        all_embeddings_initial, cluster_labels_initial, stid_list_initial, save_path_t_SNE_initial
    )
    visualize_embeddings_pca(
        all_embeddings_initial, cluster_labels_initial,
        first_node_stId_in_cluster_initial, save_path_pca_initial
    )

    if len(np.unique(cluster_labels_initial)) > 1:
        silhouette_avg_ = silhouette_score(all_embeddings_initial, cluster_labels_initial)
        davies_bouldin_ = davies_bouldin_score(all_embeddings_initial, cluster_labels_initial)
    else:
        silhouette_avg_ = np.nan
        davies_bouldin_ = np.nan

    summary_ = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_ = os.path.join(
        results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt'
    )
    with open(save_file_, 'w') as f:
        f.write(summary_)

    # ============================================================
    # 13. TRAINING
    # ============================================================

    print("\n" + "=" * 70)
    print("START TRAINING")
    print("=" * 70)

    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=True) as pbar:

        for epoch in range(num_epochs):

            # ---------------- TRAIN ----------------
            net.train()
            loss_per_graph = []
            f1_per_graph = []

            for data in dl_train:
                graph, name = data
                graph = graph.to(device)

                logits = net(graph).view(-1)
                labels = graph.ndata['significance'].float().view(-1)

                valid_mask = (labels == 0) | (labels == 1)
                if valid_mask.sum().item() == 0:
                    raise ValueError("Training graph contains no valid significance labels.")

                valid_logits = logits[valid_mask]
                valid_labels = labels[valid_mask]

                loss = criterion(valid_logits, valid_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_per_graph.append(loss.item())

                with torch.no_grad():
                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()
                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0
                    )
                    f1_per_graph.append(f1)

            running_loss_train = float(np.mean(loss_per_graph))
            running_f1_train = float(np.mean(f1_per_graph))
            loss_per_epoch_train.append(running_loss_train)
            f1_per_epoch_train.append(running_f1_train)

            # ---------------- VALIDATION ----------------
            net.eval()
            loss_per_graph = []
            f1_per_graph = []

            # Keep the last batch's probs/labels around for the threshold
            # diagnostic and best-model selection below.
            val_probs, val_labels_int = None, None

            with torch.no_grad():
                for data in dl_valid:
                    graph, name = data
                    graph = graph.to(device)

                    logits = net(graph).view(-1)
                    labels = graph.ndata['significance'].float().view(-1)

                    valid_mask = (labels == 0) | (labels == 1)
                    if valid_mask.sum().item() == 0:
                        continue

                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]

                    loss = criterion(valid_logits, valid_labels)
                    loss_per_graph.append(loss.item())

                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()

                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0
                    )
                    f1_per_graph.append(f1)

                    val_probs, val_labels_int = probs, labels_int

            if len(loss_per_graph) == 0:
                raise ValueError("Validation graph contains no valid significance labels.")

            running_loss_valid = float(np.mean(loss_per_graph))
            running_f1_val = float(np.mean(f1_per_graph))
            loss_per_epoch_valid.append(running_loss_valid)
            f1_per_epoch_valid.append(running_f1_val)

            max_f1_train = max(f1_per_epoch_train)
            max_f1_valid = max(f1_per_epoch_valid)
            max_f1_scores_train.append(max_f1_train)
            max_f1_scores_valid.append(max_f1_valid)

            # ---------------- BEST MODEL SELECTION ----------------
            # Checkpoint whenever validation loss improves. This is decoupled
            # from the threshold scan below (which is a diagnostic only) so
            # the checkpoint save can never be silently skipped.
            if running_loss_valid < best_valid_loss:

                best_train_loss = running_loss_train
                best_valid_loss = running_loss_valid
                best_f1_score = running_f1_val
                best_epoch = epoch + 1

                best_model.load_state_dict(copy.deepcopy(net.state_dict()))

                # Diagnostic only: report the best F1 achievable this epoch
                # across thresholds, without changing whether we save.
                best_f1_this_epoch = 0.0
                best_threshold_this_epoch = 0.5
                if val_probs is not None:
                    for threshold in THRESHOLDS:
                        preds_tmp = (val_probs > threshold).int()
                        f1_tmp = metrics.f1_score(
                            val_labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0
                        )
                        if f1_tmp > best_f1_this_epoch:
                            best_f1_this_epoch = f1_tmp
                            best_threshold_this_epoch = threshold

                print("\n*** BEST MODEL UPDATED ***")
                print(f"Epoch                : {best_epoch}")
                print(f"Train Loss           : {best_train_loss:.6f}")
                print(f"Valid Loss           : {best_valid_loss:.6f}")
                print(f"Valid F1 (thr=0.2)   : {best_f1_score:.6f}")
                print(f"Best F1 across thresh: {best_f1_this_epoch:.6f} (thr={best_threshold_this_epoch:.2f})")

            # ---------------- DETAILED DIAGNOSTIC EVERY 10 EPOCHS ----------------
            if epoch == 0 or (epoch + 1) % 10 == 0:

                for data in dl_valid:
                    graph, _ = data
                    graph = graph.to(device)

                    logits = net(graph).view(-1)
                    labels = graph.ndata['significance'].float().view(-1)

                    valid_mask = (labels == 0) | (labels == 1)
                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]

                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()

                    tp = ((labels_int == 1) & (preds == 1)).sum().item()
                    fp = ((labels_int == 0) & (preds == 1)).sum().item()
                    fn = ((labels_int == 1) & (preds == 0)).sum().item()
                    tn = ((labels_int == 0) & (preds == 0)).sum().item()

                    print("\n" + "-" * 70)
                    print(f"EPOCH {epoch + 1} DIAGNOSTIC")
                    print("-" * 70)
                    print(f"Train Loss : {running_loss_train:.6f}")
                    print(f"Valid Loss : {running_loss_valid:.6f}")
                    print(f"Train F1   : {running_f1_train:.6f}")
                    print(f"Valid F1   : {running_f1_val:.6f}")
                    print(f"TP={tp} | FP={fp} | FN={fn} | TN={tn}")
                    print(f"Probability min  : {probs.min().item():.6f}")
                    print(f"Probability max  : {probs.max().item():.6f}")
                    print(f"Probability mean : {probs.mean().item():.6f}")
                    print("-" * 70)

                    for threshold in THRESHOLDS:
                        preds_tmp = (probs > threshold).int()
                        f1_tmp = metrics.f1_score(
                            labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0
                        )
                        print(
                            f"threshold={threshold:.2f} | "
                            f"true_pos={(labels_int == 1).sum().item()} | "
                            f"true_neg={(labels_int == 0).sum().item()} | "
                            f"pred_pos={(preds_tmp == 1).sum().item()} | "
                            f"pred_neg={(preds_tmp == 0).sum().item()} | "
                            f"F1={f1_tmp:.4f}"
                        )

                    break

            pbar.update(1)
            pbar.set_postfix({
                'train_loss': f'{running_loss_train:.4f}',
                'valid_loss': f'{running_loss_valid:.4f}',
                'train_f1': f'{running_f1_train:.4f}',
                'valid_f1': f'{running_f1_val:.4f}'
            })

    # ============================================================
    # 14. FINAL BEST MODEL
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Best epoch       : {best_epoch}")
    print(f"Best train loss  : {best_train_loss:.6f}")
    print(f"Best valid loss  : {best_valid_loss:.6f}")
    print(f"Best valid F1    : {best_f1_score:.6f}")

    # ============================================================
    # 15. FINAL EMBEDDINGS
    # ============================================================

    all_embeddings, cluster_labels = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)

    # ============================================================
    # 16. COSINE SIMILARITY
    # ============================================================

    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized_embeddings = all_embeddings / norms
    cos_sim = np.dot(normalized_embeddings, normalized_embeddings.T)

    # ============================================================
    # 17. PLOTS
    # ============================================================

    if plot:
        loss_path = os.path.join(
            results_path, f'embeddings_loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        f1_path = os.path.join(
            results_path, f'embeddings_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        max_f1_path = os.path.join(
            results_path, f'embeddings_max_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_max_f1_plot(max_f1_scores_train, max_f1_scores_valid, max_f1_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

    # ============================================================
    # 18. SAVE BEST MODEL
    # ============================================================

    torch.save(best_model.state_dict(), model_path)
    print(f"\nBest model saved to:\n{model_path}")

    # ============================================================
    # 19. FINAL VISUALIZATION PATHS
    # ============================================================

    save_path_pca = os.path.join(
        results_path, f'embeddings_pca_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_t_SNE = os.path.join(
        results_path, f'embeddings_t-SNE_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_heatmap_ = os.path.join(
        results_path, f'embeddings_heatmap_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_matrix = os.path.join(
        results_path, f'embeddings_matrix_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )

    # ============================================================
    # 20. FINAL NODE / CLUSTER INFORMATION
    # ============================================================

    cluster_stId_dict = {}
    significant_stIds = []
    clusters_with_significant_stId = {}
    clusters_node_info = {}

    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()

        graph_path = os.path.join(data_path, omics, cancer, 'emb', 'raw', 'emb_train.pkl')
        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert len(cluster_labels) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster = {}
        first_node_embedding_in_cluster = {}
        stid_dic = {}

        for node in nx_graph.nodes:
            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]
            stid_dic[stid] = node_embeddings[idx]

            significance_value = graph.ndata['significance'][idx].item()
            if significance_value == 1:
                significant_stIds.append(stid)

        stid_df_final = pd.DataFrame.from_dict(stid_dic, orient='index')
        csv_save_path_final = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.csv'
        )
        stid_df_final.to_csv(csv_save_path_final, index_label='stId')

        for node, cluster in zip(nx_graph.nodes, cluster_labels):
            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]
            significance_value = graph.ndata['significance'][idx].item()

            if cluster not in first_node_stId_in_cluster:
                first_node_stId_in_cluster[cluster] = stid
                first_node_embedding_in_cluster[cluster] = node_embeddings[idx]

            cluster_stId_dict.setdefault(cluster, []).append(stid)

            clusters_with_significant_stId.setdefault(cluster, [])
            if significance_value == 1:
                clusters_with_significant_stId[cluster].append(stid)

            clusters_node_info.setdefault(cluster, []).append({
                'stId': stid,
                'significance': significance_value,
                'other_info': nx_graph.nodes[node]
            })

        print("\nFirst node in each cluster:")
        print(first_node_stId_in_cluster)

        sorted_clusters = sorted(first_node_stId_in_cluster.keys())
        stid_list = [first_node_stId_in_cluster[c] for c in sorted_clusters]
        embedding_list = [
            np.asarray(first_node_embedding_in_cluster[c]).reshape(-1)
            for c in sorted_clusters
        ]

        heatmap_data = pd.DataFrame(embedding_list, index=stid_list)

        print(f"Number of clusters: {len(sorted_clusters)}")
        print(f"Embedding matrix shape: {heatmap_data.shape}")

        create_heatmap_with_stid(embedding_list, stid_list, save_path_heatmap_)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list, stid_list, save_path_matrix)

        break

    # ============================================================
    # 21. FINAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(all_embeddings, cluster_labels, stid_list, save_path_t_SNE)
    visualize_embeddings_pca(all_embeddings, cluster_labels, first_node_stId_in_cluster, save_path_pca)

    # ============================================================
    # 22. FINAL CLUSTERING METRICS
    # ============================================================

    if len(np.unique(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(all_embeddings, cluster_labels)
    else:
        silhouette_avg = np.nan
        davies_bouldin = np.nan

    print(f"\nSilhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    # ============================================================
    # 23. FINAL SUMMARY
    # ============================================================

    summary = (
        f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    )
    summary += f"Best Epoch: {best_epoch}\n"
    summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {best_f1_score}\n"
    summary += f"Silhouette Score: {silhouette_avg}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin}\n"
    summary += f"Number of significant genes: {len(significant_stIds)}\n"

    save_file = os.path.join(
        results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt'
    )
    with open(save_file, 'w') as f:
        f.write(summary)

    # ============================================================
    # 24. FINAL INFORMATION
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL STAGE-1 RESULTS")
    print("=" * 70)
    print(f"Best epoch            : {best_epoch}")
    print(f"Best validation loss  : {best_valid_loss:.6f}")
    print(f"Best validation F1    : {best_f1_score:.6f}")
    print(f"Max training F1       : {max_f1_train:.6f}")
    print(f"Max validation F1     : {max_f1_valid:.6f}")
    print(f"Significant genes     : {len(significant_stIds)}")
    print(f"Model path:\n{model_path}")
    print("=" * 70)

    return model_path

def train_2(
    hyperparams=None,
    data_path='../data/omics/',
    plot=True,
    omics='cna',
    cancer='BLCA'
):
    """
    Stage-1 training:
        Predict generic biological significance from the graph.

    IMPORTANT:
        This stage does NOT use cancer-driver labels.
        'significance' is the Stage-1 pretraining target.

    Expected significance values:
        0 = not significant
        1 = significant

    If -1 is present:
        -1 = unlabeled / ignored during loss and F1 calculation
    """

    import os
    import copy
    import pickle

    import numpy as np
    import pandas as pd
    import torch
    import torch.optim as optim
    from tqdm import tqdm

    from sklearn import metrics
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score
    )

    # ============================================================
    # 1. Hyperparameters
    # ============================================================

    if hyperparams is None:
        raise ValueError("hyperparams must be provided.")

    num_epochs = hyperparams['num_epochs']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']

    print("\n" + "=" * 70)
    print("START STAGE-1 SIGNIFICANCE PRETRAINING")
    print("=" * 70)
    print(f"Omics       : {omics}")
    print(f"Cancer      : {cancer}")
    print(f"Device      : {device}")
    print(f"Epochs      : {num_epochs}")
    print(f"LR          : {learning_rate}")
    print(f"Batch size  : {batch_size}")
    print(f"Out feats   : {out_feats}")
    print(f"Layers      : {num_layers}")
    print(f"Loss type   : {hyperparams.get('loss_type', 'focal')}")
    print("=" * 70 + "\n")

    # ============================================================
    # 2. Model save path
    # ============================================================

    model_path = os.path.join(data_path, omics, cancer, 'emb', 'models')
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(
        model_path,
        f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth'
    )

    # ============================================================
    # 3. Load dataset and add node-degree structural features
    # ============================================================

    data_path_ = os.path.join(data_path, omics, cancer)
    ds = dataset.Dataset(data_path_)

    # Add node in-degree (normalized) as an extra structural feature to
    # both the training (0) and validation (1) graphs.
    for idx in [0, 1]:
        g, _ = ds[idx]

        node_degrees = g.in_degrees().float().unsqueeze(-1)
        node_degrees_norm = (
            (node_degrees - node_degrees.mean())
            / (node_degrees.std() + 1e-5)
        )

        if 'weight' in g.ndata:
            existing_feats = g.ndata['weight']
        elif 'feat' in g.ndata:
            existing_feats = g.ndata['feat']
        else:
            # No node features initialized yet — fall back to a constant.
            existing_feats = torch.ones((g.num_nodes(), 1), dtype=torch.float32)

        if existing_feats.dim() == 1:
            existing_feats = existing_feats.unsqueeze(-1)

        updated_feats = torch.cat([existing_feats, node_degrees_norm], dim=-1)

        # Keep both keys in sync so downstream code reading either won't break.
        g.ndata['feat'] = updated_feats
        g.ndata['weight'] = updated_feats

    graph, name = ds[0]

    # ============================================================
    # 4. SIGNIFICANCE LABEL DEBUG
    # ============================================================

    if 'significance' not in graph.ndata:
        raise KeyError(
            "graph.ndata['significance'] does not exist. "
            "Check your preprocessing and DGL graph conversion."
        )

    significance = graph.ndata['significance']

    print("\n" + "=" * 70)
    print("SIGNIFICANCE DEBUG")
    print("=" * 70)
    print("dtype :", significance.dtype)
    print("shape :", significance.shape)

    unique_labels, label_counts = torch.unique(significance, return_counts=True)
    print("unique labels:")
    for label, count in zip(unique_labels, label_counts):
        print(f"    label={label.item():>4} count={count.item():>6}")
    print("first 20:", significance[:20])
    print("=" * 70)

    valid_labels = significance[(significance == 0) | (significance == 1)]

    if valid_labels.numel() == 0:
        raise ValueError("No valid significance labels (0/1) were found.")

    num_positive = (valid_labels == 1).sum().item()
    num_negative = (valid_labels == 0).sum().item()
    num_unlabeled = (significance == -1).sum().item()

    print("\nLABEL DISTRIBUTION")
    print(f"Positive / significant : {num_positive}")
    print(f"Negative / non-signif. : {num_negative}")
    print(f"Unlabeled (-1)         : {num_unlabeled}")

    if num_positive == 0:
        raise ValueError("There are ZERO positive significance labels. F1 will necessarily be 0.")
    if num_negative == 0:
        raise ValueError("There are ZERO negative significance labels. Check your preprocessing.")

    positive_ratio = num_positive / valid_labels.numel()
    print(f"Positive ratio          : {positive_ratio:.4%}")

    # ============================================================
    # 5. Train / validation graphs
    # ============================================================

    ds_train = [ds[0]]
    ds_valid = [ds[1]]

    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # ============================================================
    # 6. Model / optimizer / best-model tracker
    # ============================================================

    net = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    # ============================================================
    # 7. Loss function
    # ============================================================

    # loss_type lets you A/B test FocalLoss against plain BCEWithLogitsLoss
    # without touching the rest of the training loop. Pass e.g.
    # hyperparams['loss_type'] = 'bce' to use the baseline.
    loss_type = hyperparams.get('loss_type', 'focal')

    total_valid = num_positive + num_negative

    if loss_type == 'bce':
        # Standard BCE class-balancing convention: pos_weight scales the
        # loss contribution of positive examples. Since positives are the
        # minority class here, pos_weight > 1 upweights them.
        #
        # NOTE: pos_weight = neg/pos makes the *weighted* cost of a false
        # negative and a false positive roughly equal at the population
        # level. That means a constant output of p* ~ 0.5 for every node
        # (ignoring the graph entirely) sits very close to loss-optimal —
        # a flat, easy-to-find attractor that gradient descent can settle
        # into instead of learning a genuinely discriminative solution.
        # Nudging pos_weight away from that exact tie point (via
        # pos_weight_multiplier) breaks the tie so the degenerate constant
        # solution is no longer near-optimal.
        pos_weight_multiplier = hyperparams.get('pos_weight_multiplier', 1.3)
        pos_weight_value = (num_negative / max(num_positive, 1)) * pos_weight_multiplier
        pos_weight = torch.tensor(pos_weight_value, device=device)

        print(
            f"Configuring BCEWithLogitsLoss (baseline) with "
            f"pos_weight={pos_weight_value:.4f} "
            f"(base={num_negative / max(num_positive, 1):.4f} x multiplier={pos_weight_multiplier:.2f})"
        )

        bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
        criterion = bce_criterion

    else:
        # alpha favors the rare positive class: weight ~ (negatives / total)
        dynamic_alpha = num_negative / total_valid

        print(f"Configuring Focal Loss with Dynamic Alpha: {dynamic_alpha:.4f} (favoring positive class)")

        criterion = FocalLoss(alpha=dynamic_alpha, gamma=2.0, reduction='mean')

        # NOTE: do NOT use a hardcoded weight tensor like [0.00001, 0.99999] —
        # that nearly fully suppresses class 0. And since FocalLoss(reduction='mean')
        # already returns a scalar, multiplying it by node-level weights afterwards
        # would be incorrect, so alpha is passed straight into FocalLoss instead.

    # ============================================================
    # 8. Training history
    # ============================================================

    loss_per_epoch_train = []
    loss_per_epoch_valid = []

    f1_per_epoch_train = []
    f1_per_epoch_valid = []

    max_f1_scores_train = []
    max_f1_scores_valid = []

    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_f1_score = 0.0
    best_epoch = 0
    best_threshold = 0.5

    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    # ============================================================
    # 9. Results path
    # ============================================================

    results_path = os.path.abspath(
        os.path.join('results', 'multiomics_meth', 'node_embeddings', omics, cancer)
    )
    os.makedirs(results_path, exist_ok=True)

    # ============================================================
    # 10. INITIAL EMBEDDINGS
    # ============================================================

    print("\n" + "=" * 70)
    print("CALCULATING INITIAL EMBEDDINGS")
    print("=" * 70)

    all_embeddings_initial, cluster_labels_initial = calculate_cluster_labels(
        best_model, dl_train, device
    )
    all_embeddings_initial = all_embeddings_initial.reshape(all_embeddings_initial.shape[0], -1)

    save_path_heatmap_initial = os.path.join(
        results_path, f'embeddings_heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_matrix_initial = os.path.join(
        results_path, f'embeddings_matrix_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_pca_initial = os.path.join(
        results_path, f'embeddings_pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_t_SNE_initial = os.path.join(
        results_path, f'embeddings_t-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )

    # ============================================================
    # 11. INITIAL EMBEDDING INFORMATION
    # ============================================================

    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings_initial = (
            best_model.get_node_embeddings(graph).detach().cpu().numpy()
        )

        graph_path = os.path.join(data_path, omics, cancer, 'emb', 'raw', 'emb_train.pkl')
        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert len(cluster_labels_initial) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index_initial = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster_initial = {}
        first_node_embedding_in_cluster_initial = {}
        stid_dic_initial = {}

        for node in nx_graph.nodes:
            if 'stId' in nx_graph.nodes[node]:
                stId = nx_graph.nodes[node]['stId']
                stid_dic_initial[stId] = node_embeddings_initial[node_to_index_initial[node]]

        stid_df_initial = pd.DataFrame.from_dict(stid_dic_initial, orient='index')

        csv_save_path_initial = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.csv'
        )
        stid_df_initial.to_csv(csv_save_path_initial, index_label='stId')

        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if 'stId' in nx_graph.nodes[node]:
                if cluster not in first_node_stId_in_cluster_initial:
                    first_node_stId_in_cluster_initial[cluster] = nx_graph.nodes[node]['stId']
                    first_node_embedding_in_cluster_initial[cluster] = (
                        node_embeddings_initial[node_to_index_initial[node]]
                    )

        print("first_node_stId_in_cluster_initial:")
        print(first_node_stId_in_cluster_initial)

        sorted_clusters_initial = sorted(first_node_stId_in_cluster_initial.keys())
        stid_list_initial = [first_node_stId_in_cluster_initial[c] for c in sorted_clusters_initial]
        embedding_list_initial = [
            np.asarray(first_node_embedding_in_cluster_initial[c]).reshape(-1)
            for c in sorted_clusters_initial
        ]

        heatmap_data_initial = pd.DataFrame(embedding_list_initial, index=stid_list_initial)

        print(f"Number of clusters: {len(sorted_clusters_initial)}")
        print(f"Embedding matrix shape: {heatmap_data_initial.shape}")

        create_heatmap_with_stid(embedding_list_initial, stid_list_initial, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(
            embedding_list_initial, stid_list_initial, save_path_matrix_initial
        )

        break

    # ============================================================
    # 12. INITIAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(
        all_embeddings_initial, cluster_labels_initial, stid_list_initial, save_path_t_SNE_initial
    )
    visualize_embeddings_pca(
        all_embeddings_initial, cluster_labels_initial,
        first_node_stId_in_cluster_initial, save_path_pca_initial
    )

    if len(np.unique(cluster_labels_initial)) > 1:
        silhouette_avg_ = silhouette_score(all_embeddings_initial, cluster_labels_initial)
        davies_bouldin_ = davies_bouldin_score(all_embeddings_initial, cluster_labels_initial)
    else:
        silhouette_avg_ = np.nan
        davies_bouldin_ = np.nan

    summary_ = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_ = os.path.join(
        results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt'
    )
    with open(save_file_, 'w') as f:
        f.write(summary_)

    # ============================================================
    # 13. TRAINING
    # ============================================================

    print("\n" + "=" * 70)
    print("START TRAINING")
    print("=" * 70)

    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=True) as pbar:

        for epoch in range(num_epochs):

            # ---------------- TRAIN ----------------
            net.train()
            loss_per_graph = []
            f1_per_graph = []

            for data in dl_train:
                graph, name = data
                graph = graph.to(device)

                logits = net(graph).view(-1)
                labels = graph.ndata['significance'].float().view(-1)

                valid_mask = (labels == 0) | (labels == 1)
                if valid_mask.sum().item() == 0:
                    raise ValueError("Training graph contains no valid significance labels.")

                valid_logits = logits[valid_mask]
                valid_labels = labels[valid_mask]

                loss = criterion(valid_logits, valid_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_per_graph.append(loss.item())

                with torch.no_grad():
                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()
                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0
                    )
                    f1_per_graph.append(f1)

            running_loss_train = float(np.mean(loss_per_graph))
            running_f1_train = float(np.mean(f1_per_graph))
            loss_per_epoch_train.append(running_loss_train)
            f1_per_epoch_train.append(running_f1_train)

            # ---------------- VALIDATION ----------------
            net.eval()
            loss_per_graph = []
            f1_per_graph = []

            # Keep the last batch's probs/labels around for the threshold
            # diagnostic and best-model selection below.
            val_probs, val_labels_int = None, None

            with torch.no_grad():
                for data in dl_valid:
                    graph, name = data
                    graph = graph.to(device)

                    logits = net(graph).view(-1)
                    labels = graph.ndata['significance'].float().view(-1)

                    valid_mask = (labels == 0) | (labels == 1)
                    if valid_mask.sum().item() == 0:
                        continue

                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]

                    loss = criterion(valid_logits, valid_labels)
                    loss_per_graph.append(loss.item())

                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()

                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0
                    )
                    f1_per_graph.append(f1)

                    val_probs, val_labels_int = probs, labels_int

            if len(loss_per_graph) == 0:
                raise ValueError("Validation graph contains no valid significance labels.")

            running_loss_valid = float(np.mean(loss_per_graph))
            running_f1_val = float(np.mean(f1_per_graph))
            loss_per_epoch_valid.append(running_loss_valid)
            f1_per_epoch_valid.append(running_f1_val)

            max_f1_train = max(f1_per_epoch_train)
            max_f1_valid = max(f1_per_epoch_valid)
            max_f1_scores_train.append(max_f1_train)
            max_f1_scores_valid.append(max_f1_valid)

            # ---------------- BEST-THRESHOLD F1 (every epoch) ----------------
            # A fixed thr=0.2 F1 is a bad lens once probabilities drift off
            # center (e.g. the whole distribution sliding below 0.5): the
            # ranking underneath can still be fine or improving while
            # thr=0.2 F1 reads a flat zero. Sweeping thresholds every epoch
            # (not just when loss improves) avoids that blind spot and is
            # what actually gates the checkpoint below.
            best_f1_this_epoch = 0.0
            best_threshold_this_epoch = 0.5
            if val_probs is not None:
                for threshold in THRESHOLDS:
                    preds_tmp = (val_probs > threshold).int()
                    f1_tmp = metrics.f1_score(
                        val_labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0
                    )
                    if f1_tmp > best_f1_this_epoch:
                        best_f1_this_epoch = f1_tmp
                        best_threshold_this_epoch = threshold

            # ---------------- BEST MODEL SELECTION ----------------
            # Primary criterion: best-across-threshold validation F1 must
            # improve. This is what actually reflects the model's ability
            # to separate classes, unlike raw loss (which a degenerate
            # constant-output solution can minimize) or thr=0.2 F1 (which
            # can read 0 even when the underlying ranking is fine). Loss is
            # used only as a tie-breaker when F1 is unchanged, so we still
            # prefer the more confident/better-calibrated model among ties.
            f1_improved = best_f1_this_epoch > best_f1_score
            f1_tied_but_loss_improved = (
                best_f1_this_epoch == best_f1_score and running_loss_valid < best_valid_loss
            )

            if f1_improved or f1_tied_but_loss_improved:

                best_train_loss = running_loss_train
                best_valid_loss = running_loss_valid
                best_f1_score = best_f1_this_epoch
                best_epoch = epoch + 1
                best_threshold = best_threshold_this_epoch

                best_model.load_state_dict(copy.deepcopy(net.state_dict()))

                print("\n*** BEST MODEL UPDATED ***")
                print(f"Epoch                : {best_epoch}")
                print(f"Train Loss           : {best_train_loss:.6f}")
                print(f"Valid Loss           : {best_valid_loss:.6f}")
                print(f"Valid F1 (thr=0.2)   : {running_f1_val:.6f}")
                print(f"Best F1 across thresh: {best_f1_score:.6f} (thr={best_threshold:.2f})")

            # ---------------- DETAILED DIAGNOSTIC ----------------
            # Every epoch for the first 15 (early training is where sudden
            # collapses/drifts like the epoch-7 case tend to happen and are
            # otherwise invisible between the epoch-1 and epoch-10 prints),
            # then every 10th epoch after that.
            if epoch < 15 or (epoch + 1) % 10 == 0:

                for data in dl_valid:
                    graph, _ = data
                    graph = graph.to(device)

                    logits = net(graph).view(-1)
                    labels = graph.ndata['significance'].float().view(-1)

                    valid_mask = (labels == 0) | (labels == 1)
                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]

                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()

                    tp = ((labels_int == 1) & (preds == 1)).sum().item()
                    fp = ((labels_int == 0) & (preds == 1)).sum().item()
                    fn = ((labels_int == 1) & (preds == 0)).sum().item()
                    tn = ((labels_int == 0) & (preds == 0)).sum().item()

                    print("\n" + "-" * 70)
                    print(f"EPOCH {epoch + 1} DIAGNOSTIC")
                    print("-" * 70)
                    print(f"Train Loss : {running_loss_train:.6f}")
                    print(f"Valid Loss : {running_loss_valid:.6f}")
                    print(f"Train F1   : {running_f1_train:.6f}")
                    print(f"Valid F1   : {running_f1_val:.6f}")
                    print(f"TP={tp} | FP={fp} | FN={fn} | TN={tn}")
                    print(f"Probability min  : {probs.min().item():.6f}")
                    print(f"Probability max  : {probs.max().item():.6f}")
                    print(f"Probability mean : {probs.mean().item():.6f}")
                    print("-" * 70)

                    for threshold in THRESHOLDS:
                        preds_tmp = (probs > threshold).int()
                        f1_tmp = metrics.f1_score(
                            labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0
                        )
                        print(
                            f"threshold={threshold:.2f} | "
                            f"true_pos={(labels_int == 1).sum().item()} | "
                            f"true_neg={(labels_int == 0).sum().item()} | "
                            f"pred_pos={(preds_tmp == 1).sum().item()} | "
                            f"pred_neg={(preds_tmp == 0).sum().item()} | "
                            f"F1={f1_tmp:.4f}"
                        )

                    break

            pbar.update(1)
            pbar.set_postfix({
                'train_loss': f'{running_loss_train:.4f}',
                'valid_loss': f'{running_loss_valid:.4f}',
                'train_f1': f'{running_f1_train:.4f}',
                'valid_f1': f'{running_f1_val:.4f}'
            })

    # ============================================================
    # 14. FINAL BEST MODEL
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Best epoch       : {best_epoch}")
    print(f"Best train loss  : {best_train_loss:.6f}")
    print(f"Best valid loss  : {best_valid_loss:.6f}")
    print(f"Best valid F1    : {best_f1_score:.6f} (thr={best_threshold:.2f})")

    # ============================================================
    # 15. FINAL EMBEDDINGS
    # ============================================================

    all_embeddings, cluster_labels = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)

    # ============================================================
    # 16. COSINE SIMILARITY
    # ============================================================

    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized_embeddings = all_embeddings / norms
    cos_sim = np.dot(normalized_embeddings, normalized_embeddings.T)

    # ============================================================
    # 17. PLOTS
    # ============================================================

    if plot:
        loss_path = os.path.join(
            results_path, f'embeddings_loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        f1_path = os.path.join(
            results_path, f'embeddings_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        max_f1_path = os.path.join(
            results_path, f'embeddings_max_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_max_f1_plot(max_f1_scores_train, max_f1_scores_valid, max_f1_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

    # ============================================================
    # 18. SAVE BEST MODEL
    # ============================================================

    torch.save(best_model.state_dict(), model_path)
    print(f"\nBest model saved to:\n{model_path}")

    # ============================================================
    # 19. FINAL VISUALIZATION PATHS
    # ============================================================

    save_path_pca = os.path.join(
        results_path, f'embeddings_pca_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_t_SNE = os.path.join(
        results_path, f'embeddings_t-SNE_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_heatmap_ = os.path.join(
        results_path, f'embeddings_heatmap_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_matrix = os.path.join(
        results_path, f'embeddings_matrix_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )

    # ============================================================
    # 20. FINAL NODE / CLUSTER INFORMATION
    # ============================================================

    cluster_stId_dict = {}
    significant_stIds = []
    clusters_with_significant_stId = {}
    clusters_node_info = {}

    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()

        graph_path = os.path.join(data_path, omics, cancer, 'emb', 'raw', 'emb_train.pkl')
        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert len(cluster_labels) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster = {}
        first_node_embedding_in_cluster = {}
        stid_dic = {}

        for node in nx_graph.nodes:
            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]
            stid_dic[stid] = node_embeddings[idx]

            significance_value = graph.ndata['significance'][idx].item()
            if significance_value == 1:
                significant_stIds.append(stid)

        stid_df_final = pd.DataFrame.from_dict(stid_dic, orient='index')
        csv_save_path_final = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.csv'
        )
        stid_df_final.to_csv(csv_save_path_final, index_label='stId')

        for node, cluster in zip(nx_graph.nodes, cluster_labels):
            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]
            significance_value = graph.ndata['significance'][idx].item()

            if cluster not in first_node_stId_in_cluster:
                first_node_stId_in_cluster[cluster] = stid
                first_node_embedding_in_cluster[cluster] = node_embeddings[idx]

            cluster_stId_dict.setdefault(cluster, []).append(stid)

            clusters_with_significant_stId.setdefault(cluster, [])
            if significance_value == 1:
                clusters_with_significant_stId[cluster].append(stid)

            clusters_node_info.setdefault(cluster, []).append({
                'stId': stid,
                'significance': significance_value,
                'other_info': nx_graph.nodes[node]
            })

        print("\nFirst node in each cluster:")
        print(first_node_stId_in_cluster)

        sorted_clusters = sorted(first_node_stId_in_cluster.keys())
        stid_list = [first_node_stId_in_cluster[c] for c in sorted_clusters]
        embedding_list = [
            np.asarray(first_node_embedding_in_cluster[c]).reshape(-1)
            for c in sorted_clusters
        ]

        heatmap_data = pd.DataFrame(embedding_list, index=stid_list)

        print(f"Number of clusters: {len(sorted_clusters)}")
        print(f"Embedding matrix shape: {heatmap_data.shape}")

        create_heatmap_with_stid(embedding_list, stid_list, save_path_heatmap_)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list, stid_list, save_path_matrix)

        break

    # ============================================================
    # 21. FINAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(all_embeddings, cluster_labels, stid_list, save_path_t_SNE)
    visualize_embeddings_pca(all_embeddings, cluster_labels, first_node_stId_in_cluster, save_path_pca)

    # ============================================================
    # 22. FINAL CLUSTERING METRICS
    # ============================================================

    if len(np.unique(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(all_embeddings, cluster_labels)
    else:
        silhouette_avg = np.nan
        davies_bouldin = np.nan

    print(f"\nSilhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    # ============================================================
    # 23. FINAL SUMMARY
    # ============================================================

    summary = (
        f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    )
    summary += f"Best Epoch: {best_epoch}\n"
    summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {best_f1_score} (thr={best_threshold})\n"
    summary += f"Silhouette Score: {silhouette_avg}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin}\n"
    summary += f"Number of significant genes: {len(significant_stIds)}\n"

    save_file = os.path.join(
        results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt'
    )
    with open(save_file, 'w') as f:
        f.write(summary)

    # ============================================================
    # 24. FINAL INFORMATION
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL STAGE-1 RESULTS")
    print("=" * 70)
    print(f"Best epoch            : {best_epoch}")
    print(f"Best validation loss  : {best_valid_loss:.6f}")
    print(f"Best validation F1    : {best_f1_score:.6f} (thr={best_threshold:.2f})")
    print(f"Max training F1       : {max_f1_train:.6f}")
    print(f"Max validation F1     : {max_f1_valid:.6f}")
    print(f"Significant genes     : {len(significant_stIds)}")
    print(f"Model path:\n{model_path}")
    print("=" * 70)

    return model_path

def train_3(
    hyperparams=None,
    data_path='../data/omics/',
    plot=True,
    omics='cna',
    cancer='BLCA'
):
    """
    Stage-1 training:
        Predict generic biological significance from the graph.

    IMPORTANT:
        This stage does NOT use cancer-driver labels.
        'significance' is the Stage-1 pretraining target.

    Expected significance values:
        0 = not significant
        1 = significant

    If -1 is present:
        -1 = unlabeled / ignored during loss and F1 calculation
    """

    import os
    import copy
    import pickle

    import numpy as np
    import pandas as pd
    import torch
    import torch.optim as optim
    from tqdm import tqdm

    from sklearn import metrics
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score
    )

    # ============================================================
    # 1. Hyperparameters
    # ============================================================

    if hyperparams is None:
        raise ValueError("hyperparams must be provided.")

    num_epochs = hyperparams['num_epochs']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']

    print("\n" + "=" * 70)
    print("START STAGE-1 SIGNIFICANCE PRETRAINING")
    print("=" * 70)
    print(f"Omics       : {omics}")
    print(f"Cancer      : {cancer}")
    print(f"Device      : {device}")
    print(f"Epochs      : {num_epochs}")
    print(f"LR          : {learning_rate}")
    print(f"Batch size  : {batch_size}")
    print(f"Out feats   : {out_feats}")
    print(f"Layers      : {num_layers}")
    print(f"Loss type   : {hyperparams.get('loss_type', 'focal')}")
    print("=" * 70 + "\n")

    # ============================================================
    # 2. Model save path
    # ============================================================

    model_path = os.path.join(data_path, omics, cancer, 'emb', 'models')
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(
        model_path,
        f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth'
    )

    # ============================================================
    # 3. Load dataset and add node-degree structural features
    # ============================================================

    data_path_ = os.path.join(data_path, omics, cancer)
    ds = dataset.Dataset(data_path_)

    # Add node in-degree (normalized) as an extra structural feature to
    # both the training (0) and validation (1) graphs.
    for idx in [0, 1]:
        g, _ = ds[idx]

        node_degrees = g.in_degrees().float().unsqueeze(-1)
        node_degrees_norm = (
            (node_degrees - node_degrees.mean())
            / (node_degrees.std() + 1e-5)
        )

        if 'weight' in g.ndata:
            existing_feats = g.ndata['weight']
        elif 'feat' in g.ndata:
            existing_feats = g.ndata['feat']
        else:
            # No node features initialized yet — fall back to a constant.
            existing_feats = torch.ones((g.num_nodes(), 1), dtype=torch.float32)

        if existing_feats.dim() == 1:
            existing_feats = existing_feats.unsqueeze(-1)

        updated_feats = torch.cat([existing_feats, node_degrees_norm], dim=-1)

        # Keep both keys in sync so downstream code reading either won't break.
        g.ndata['feat'] = updated_feats
        g.ndata['weight'] = updated_feats

    graph, name = ds[0]

    # ============================================================
    # 4. SIGNIFICANCE LABEL DEBUG
    # ============================================================

    if 'significance' not in graph.ndata:
        raise KeyError(
            "graph.ndata['significance'] does not exist. "
            "Check your preprocessing and DGL graph conversion."
        )

    significance = graph.ndata['significance']

    print("\n" + "=" * 70)
    print("SIGNIFICANCE DEBUG")
    print("=" * 70)
    print("dtype :", significance.dtype)
    print("shape :", significance.shape)

    unique_labels, label_counts = torch.unique(significance, return_counts=True)
    print("unique labels:")
    for label, count in zip(unique_labels, label_counts):
        print(f"    label={label.item():>4} count={count.item():>6}")
    print("first 20:", significance[:20])
    print("=" * 70)

    valid_labels = significance[(significance == 0) | (significance == 1)]

    if valid_labels.numel() == 0:
        raise ValueError("No valid significance labels (0/1) were found.")

    num_positive = (valid_labels == 1).sum().item()
    num_negative = (valid_labels == 0).sum().item()
    num_unlabeled = (significance == -1).sum().item()

    print("\nLABEL DISTRIBUTION")
    print(f"Positive / significant : {num_positive}")
    print(f"Negative / non-signif. : {num_negative}")
    print(f"Unlabeled (-1)         : {num_unlabeled}")

    if num_positive == 0:
        raise ValueError("There are ZERO positive significance labels. F1 will necessarily be 0.")
    if num_negative == 0:
        raise ValueError("There are ZERO negative significance labels. Check your preprocessing.")

    positive_ratio = num_positive / valid_labels.numel()
    print(f"Positive ratio          : {positive_ratio:.4%}")

    # ============================================================
    # 5. Train / validation graphs
    # ============================================================

    ds_train = [ds[0]]
    ds_valid = [ds[1]]

    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # ============================================================
    # 6. Model / optimizer / best-model tracker
    # ============================================================

    net = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    # ============================================================
    # 7. Loss function
    # ============================================================

    # loss_type lets you A/B test FocalLoss against plain BCEWithLogitsLoss
    # without touching the rest of the training loop. Pass e.g.
    # hyperparams['loss_type'] = 'bce' to use the baseline.
    loss_type = hyperparams.get('loss_type', 'focal')

    total_valid = num_positive + num_negative

    if loss_type == 'bce':
        # Standard BCE class-balancing convention: pos_weight scales the
        # loss contribution of positive examples. Since positives are the
        # minority class here, pos_weight > 1 upweights them.
        #
        # NOTE: pos_weight = neg/pos makes the *weighted* cost of a false
        # negative and a false positive roughly equal at the population
        # level. That means a constant output of p* ~ 0.5 for every node
        # (ignoring the graph entirely) sits very close to loss-optimal —
        # a flat, easy-to-find attractor that gradient descent can settle
        # into instead of learning a genuinely discriminative solution.
        # Nudging pos_weight away from that exact tie point (via
        # pos_weight_multiplier) breaks the tie so the degenerate constant
        # solution is no longer near-optimal.
        pos_weight_multiplier = hyperparams.get('pos_weight_multiplier', 1.3)
        pos_weight_value = (num_negative / max(num_positive, 1)) * pos_weight_multiplier
        pos_weight = torch.tensor(pos_weight_value, device=device)

        print(
            f"Configuring BCEWithLogitsLoss (baseline) with "
            f"pos_weight={pos_weight_value:.4f} "
            f"(base={num_negative / max(num_positive, 1):.4f} x multiplier={pos_weight_multiplier:.2f})"
        )

        bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
        criterion = bce_criterion

    else:
        # alpha favors the rare positive class: weight ~ (negatives / total)
        dynamic_alpha = num_negative / total_valid

        print(f"Configuring Focal Loss with Dynamic Alpha: {dynamic_alpha:.4f} (favoring positive class)")

        criterion = FocalLoss(alpha=dynamic_alpha, gamma=2.0, reduction='mean')

        # NOTE: do NOT use a hardcoded weight tensor like [0.00001, 0.99999] —
        # that nearly fully suppresses class 0. And since FocalLoss(reduction='mean')
        # already returns a scalar, multiplying it by node-level weights afterwards
        # would be incorrect, so alpha is passed straight into FocalLoss instead.

    # ============================================================
    # 8. Training history
    # ============================================================

    loss_per_epoch_train = []
    loss_per_epoch_valid = []

    f1_per_epoch_train = []
    f1_per_epoch_valid = []
    f1_best_thresh_per_epoch_valid = []

    max_f1_scores_train = []
    max_f1_scores_valid = []

    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_f1_score = 0.0
    best_epoch = 0
    best_threshold = 0.5

    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    # ============================================================
    # 9. Results path
    # ============================================================

    results_path = os.path.abspath(
        os.path.join('results', 'multiomics_meth', 'node_embeddings', omics, cancer)
    )
    os.makedirs(results_path, exist_ok=True)

    # ============================================================
    # 10. INITIAL EMBEDDINGS
    # ============================================================

    print("\n" + "=" * 70)
    print("CALCULATING INITIAL EMBEDDINGS")
    print("=" * 70)

    all_embeddings_initial, cluster_labels_initial = calculate_cluster_labels(
        best_model, dl_train, device
    )
    all_embeddings_initial = all_embeddings_initial.reshape(all_embeddings_initial.shape[0], -1)

    save_path_heatmap_initial = os.path.join(
        results_path, f'embeddings_heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_matrix_initial = os.path.join(
        results_path, f'embeddings_matrix_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_pca_initial = os.path.join(
        results_path, f'embeddings_pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_t_SNE_initial = os.path.join(
        results_path, f'embeddings_t-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )

    # ============================================================
    # 11. INITIAL EMBEDDING INFORMATION
    # ============================================================

    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings_initial = (
            best_model.get_node_embeddings(graph).detach().cpu().numpy()
        )

        graph_path = os.path.join(data_path, omics, cancer, 'emb', 'raw', 'emb_train.pkl')
        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert len(cluster_labels_initial) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index_initial = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster_initial = {}
        first_node_embedding_in_cluster_initial = {}
        stid_dic_initial = {}

        for node in nx_graph.nodes:
            if 'stId' in nx_graph.nodes[node]:
                stId = nx_graph.nodes[node]['stId']
                stid_dic_initial[stId] = node_embeddings_initial[node_to_index_initial[node]]

        stid_df_initial = pd.DataFrame.from_dict(stid_dic_initial, orient='index')

        csv_save_path_initial = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.csv'
        )
        stid_df_initial.to_csv(csv_save_path_initial, index_label='stId')

        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if 'stId' in nx_graph.nodes[node]:
                if cluster not in first_node_stId_in_cluster_initial:
                    first_node_stId_in_cluster_initial[cluster] = nx_graph.nodes[node]['stId']
                    first_node_embedding_in_cluster_initial[cluster] = (
                        node_embeddings_initial[node_to_index_initial[node]]
                    )

        print("first_node_stId_in_cluster_initial:")
        print(first_node_stId_in_cluster_initial)

        sorted_clusters_initial = sorted(first_node_stId_in_cluster_initial.keys())
        stid_list_initial = [first_node_stId_in_cluster_initial[c] for c in sorted_clusters_initial]
        embedding_list_initial = [
            np.asarray(first_node_embedding_in_cluster_initial[c]).reshape(-1)
            for c in sorted_clusters_initial
        ]

        heatmap_data_initial = pd.DataFrame(embedding_list_initial, index=stid_list_initial)

        print(f"Number of clusters: {len(sorted_clusters_initial)}")
        print(f"Embedding matrix shape: {heatmap_data_initial.shape}")

        create_heatmap_with_stid(embedding_list_initial, stid_list_initial, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(
            embedding_list_initial, stid_list_initial, save_path_matrix_initial
        )

        break

    # ============================================================
    # 12. INITIAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(
        all_embeddings_initial, cluster_labels_initial, stid_list_initial, save_path_t_SNE_initial
    )
    visualize_embeddings_pca(
        all_embeddings_initial, cluster_labels_initial,
        first_node_stId_in_cluster_initial, save_path_pca_initial
    )

    if len(np.unique(cluster_labels_initial)) > 1:
        silhouette_avg_ = silhouette_score(all_embeddings_initial, cluster_labels_initial)
        davies_bouldin_ = davies_bouldin_score(all_embeddings_initial, cluster_labels_initial)
    else:
        silhouette_avg_ = np.nan
        davies_bouldin_ = np.nan

    summary_ = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_ = os.path.join(
        results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt'
    )
    with open(save_file_, 'w') as f:
        f.write(summary_)

    # ============================================================
    # 13. TRAINING
    # ============================================================

    print("\n" + "=" * 70)
    print("START TRAINING")
    print("=" * 70)

    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=True) as pbar:

        for epoch in range(num_epochs):

            # ---------------- TRAIN ----------------
            net.train()
            loss_per_graph = []
            f1_per_graph = []

            for data in dl_train:
                graph, name = data
                graph = graph.to(device)

                logits = net(graph).view(-1)
                labels = graph.ndata['significance'].float().view(-1)

                valid_mask = (labels == 0) | (labels == 1)
                if valid_mask.sum().item() == 0:
                    raise ValueError("Training graph contains no valid significance labels.")

                valid_logits = logits[valid_mask]
                valid_labels = labels[valid_mask]

                loss = criterion(valid_logits, valid_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_per_graph.append(loss.item())

                with torch.no_grad():
                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()
                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0
                    )
                    f1_per_graph.append(f1)

            running_loss_train = float(np.mean(loss_per_graph))
            running_f1_train = float(np.mean(f1_per_graph))
            loss_per_epoch_train.append(running_loss_train)
            f1_per_epoch_train.append(running_f1_train)

            # ---------------- VALIDATION ----------------
            net.eval()
            loss_per_graph = []
            f1_per_graph = []

            # Keep the last batch's probs/labels around for the threshold
            # diagnostic and best-model selection below.
            val_probs, val_labels_int = None, None

            with torch.no_grad():
                for data in dl_valid:
                    graph, name = data
                    graph = graph.to(device)

                    logits = net(graph).view(-1)
                    labels = graph.ndata['significance'].float().view(-1)

                    valid_mask = (labels == 0) | (labels == 1)
                    if valid_mask.sum().item() == 0:
                        continue

                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]

                    loss = criterion(valid_logits, valid_labels)
                    loss_per_graph.append(loss.item())

                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()

                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0
                    )
                    f1_per_graph.append(f1)

                    val_probs, val_labels_int = probs, labels_int

            if len(loss_per_graph) == 0:
                raise ValueError("Validation graph contains no valid significance labels.")

            running_loss_valid = float(np.mean(loss_per_graph))
            running_f1_val = float(np.mean(f1_per_graph))
            loss_per_epoch_valid.append(running_loss_valid)
            f1_per_epoch_valid.append(running_f1_val)

            max_f1_train = max(f1_per_epoch_train)
            max_f1_valid = max(f1_per_epoch_valid)
            max_f1_scores_train.append(max_f1_train)
            max_f1_scores_valid.append(max_f1_valid)

            # ---------------- BEST-THRESHOLD F1 (every epoch) ----------------
            # A fixed thr=0.2 F1 is a bad lens once probabilities drift off
            # center (e.g. the whole distribution sliding below 0.5): the
            # ranking underneath can still be fine or improving while
            # thr=0.2 F1 reads a flat zero. Sweeping thresholds every epoch
            # (not just when loss improves) avoids that blind spot and is
            # what actually gates the checkpoint below.
            best_f1_this_epoch = 0.0
            best_threshold_this_epoch = 0.5
            if val_probs is not None:
                for threshold in THRESHOLDS:
                    preds_tmp = (val_probs > threshold).int()
                    f1_tmp = metrics.f1_score(
                        val_labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0
                    )
                    if f1_tmp > best_f1_this_epoch:
                        best_f1_this_epoch = f1_tmp
                        best_threshold_this_epoch = threshold

            f1_best_thresh_per_epoch_valid.append(best_f1_this_epoch)

            # ---------------- BEST MODEL SELECTION ----------------
            # Primary criterion: best-across-threshold validation F1 must
            # improve. This is what actually reflects the model's ability
            # to separate classes, unlike raw loss (which a degenerate
            # constant-output solution can minimize) or thr=0.2 F1 (which
            # can read 0 even when the underlying ranking is fine). Loss is
            # used only as a tie-breaker when F1 is unchanged, so we still
            # prefer the more confident/better-calibrated model among ties.
            f1_improved = best_f1_this_epoch > best_f1_score
            f1_tied_but_loss_improved = (
                best_f1_this_epoch == best_f1_score and running_loss_valid < best_valid_loss
            )

            if f1_improved or f1_tied_but_loss_improved:

                best_train_loss = running_loss_train
                best_valid_loss = running_loss_valid
                best_f1_score = best_f1_this_epoch
                best_epoch = epoch + 1
                best_threshold = best_threshold_this_epoch

                best_model.load_state_dict(copy.deepcopy(net.state_dict()))

                print("\n*** BEST MODEL UPDATED ***")
                print(f"Epoch                : {best_epoch}")
                print(f"Train Loss           : {best_train_loss:.6f}")
                print(f"Valid Loss           : {best_valid_loss:.6f}")
                print(f"Valid F1 (thr=0.2)   : {running_f1_val:.6f}")
                print(f"Best F1 across thresh: {best_f1_score:.6f} (thr={best_threshold:.2f})")

            # ---------------- DETAILED DIAGNOSTIC ----------------
            # Every epoch for the first 15 (early training is where sudden
            # collapses/drifts like the epoch-7 case tend to happen and are
            # otherwise invisible between the epoch-1 and epoch-10 prints),
            # then every 10th epoch after that.
            if epoch < 15 or (epoch + 1) % 10 == 0:

                for data in dl_valid:
                    graph, _ = data
                    graph = graph.to(device)

                    logits = net(graph).view(-1)
                    labels = graph.ndata['significance'].float().view(-1)

                    valid_mask = (labels == 0) | (labels == 1)
                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]

                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()

                    tp = ((labels_int == 1) & (preds == 1)).sum().item()
                    fp = ((labels_int == 0) & (preds == 1)).sum().item()
                    fn = ((labels_int == 1) & (preds == 0)).sum().item()
                    tn = ((labels_int == 0) & (preds == 0)).sum().item()

                    print("\n" + "-" * 70)
                    print(f"EPOCH {epoch + 1} DIAGNOSTIC")
                    print("-" * 70)
                    print(f"Train Loss : {running_loss_train:.6f}")
                    print(f"Valid Loss : {running_loss_valid:.6f}")
                    print(f"Train F1   : {running_f1_train:.6f}")
                    print(f"Valid F1   : {running_f1_val:.6f}")
                    print(f"TP={tp} | FP={fp} | FN={fn} | TN={tn}")
                    print(f"Probability min  : {probs.min().item():.6f}")
                    print(f"Probability max  : {probs.max().item():.6f}")
                    print(f"Probability mean : {probs.mean().item():.6f}")
                    print("-" * 70)

                    for threshold in THRESHOLDS:
                        preds_tmp = (probs > threshold).int()
                        f1_tmp = metrics.f1_score(
                            labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0
                        )
                        print(
                            f"threshold={threshold:.2f} | "
                            f"true_pos={(labels_int == 1).sum().item()} | "
                            f"true_neg={(labels_int == 0).sum().item()} | "
                            f"pred_pos={(preds_tmp == 1).sum().item()} | "
                            f"pred_neg={(preds_tmp == 0).sum().item()} | "
                            f"F1={f1_tmp:.4f}"
                        )

                    break

            pbar.update(1)
            pbar.set_postfix({
                'train_loss': f'{running_loss_train:.4f}',
                'valid_loss': f'{running_loss_valid:.4f}',
                'train_f1': f'{running_f1_train:.4f}',
                'valid_f1': f'{running_f1_val:.4f}'
            })

    # ============================================================
    # 14. FINAL BEST MODEL
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Best epoch       : {best_epoch}")
    print(f"Best train loss  : {best_train_loss:.6f}")
    print(f"Best valid loss  : {best_valid_loss:.6f}")
    print(f"Best valid F1    : {best_f1_score:.6f} (thr={best_threshold:.2f})")

    # ============================================================
    # 15. FINAL EMBEDDINGS
    # ============================================================

    all_embeddings, cluster_labels = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)

    # ============================================================
    # 16. COSINE SIMILARITY
    # ============================================================

    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized_embeddings = all_embeddings / norms
    cos_sim = np.dot(normalized_embeddings, normalized_embeddings.T)

    # ============================================================
    # 17. PLOTS
    # ============================================================

    if plot:
        loss_path = os.path.join(
            results_path, f'embeddings_loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        f1_path = os.path.join(
            results_path, f'embeddings_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        max_f1_path = os.path.join(
            results_path, f'embeddings_max_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        f1_best_thresh_path = os.path.join(
            results_path,
            f'embeddings_f1_best_thresh_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_max_f1_plot(max_f1_scores_train, max_f1_scores_valid, max_f1_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

        # This plot shows what the checkpoint selection actually sees, unlike
        # the fixed-thr=0.2 plot above: it can look completely flat at zero
        # (e.g. once probabilities drift below 0.5) even while this line
        # shows the model still improving underneath that blind spot.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(f1_per_epoch_valid, label='validation (thr=0.2)')
        plt.plot(f1_best_thresh_per_epoch_valid, label='validation (best threshold)')
        plt.xlabel('Epoch')
        plt.ylabel('F1-score')
        plt.title('Validation F1: fixed threshold vs. best threshold')
        plt.legend()
        plt.savefig(f1_best_thresh_path)
        plt.close()

    # ============================================================
    # 18. SAVE BEST MODEL
    # ============================================================

    torch.save(best_model.state_dict(), model_path)
    print(f"\nBest model saved to:\n{model_path}")

    # ============================================================
    # 19. FINAL VISUALIZATION PATHS
    # ============================================================

    save_path_pca = os.path.join(
        results_path, f'embeddings_pca_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_t_SNE = os.path.join(
        results_path, f'embeddings_t-SNE_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_heatmap_ = os.path.join(
        results_path, f'embeddings_heatmap_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_matrix = os.path.join(
        results_path, f'embeddings_matrix_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )

    # ============================================================
    # 20. FINAL NODE / CLUSTER INFORMATION
    # ============================================================

    cluster_stId_dict = {}
    significant_stIds = []
    clusters_with_significant_stId = {}
    clusters_node_info = {}

    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()

        graph_path = os.path.join(data_path, omics, cancer, 'emb', 'raw', 'emb_train.pkl')
        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert len(cluster_labels) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster = {}
        first_node_embedding_in_cluster = {}
        stid_dic = {}

        for node in nx_graph.nodes:
            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]
            stid_dic[stid] = node_embeddings[idx]

            significance_value = graph.ndata['significance'][idx].item()
            if significance_value == 1:
                significant_stIds.append(stid)

        stid_df_final = pd.DataFrame.from_dict(stid_dic, orient='index')
        csv_save_path_final = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.csv'
        )
        stid_df_final.to_csv(csv_save_path_final, index_label='stId')

        for node, cluster in zip(nx_graph.nodes, cluster_labels):
            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]
            significance_value = graph.ndata['significance'][idx].item()

            if cluster not in first_node_stId_in_cluster:
                first_node_stId_in_cluster[cluster] = stid
                first_node_embedding_in_cluster[cluster] = node_embeddings[idx]

            cluster_stId_dict.setdefault(cluster, []).append(stid)

            clusters_with_significant_stId.setdefault(cluster, [])
            if significance_value == 1:
                clusters_with_significant_stId[cluster].append(stid)

            clusters_node_info.setdefault(cluster, []).append({
                'stId': stid,
                'significance': significance_value,
                'other_info': nx_graph.nodes[node]
            })

        print("\nFirst node in each cluster:")
        print(first_node_stId_in_cluster)

        sorted_clusters = sorted(first_node_stId_in_cluster.keys())
        stid_list = [first_node_stId_in_cluster[c] for c in sorted_clusters]
        embedding_list = [
            np.asarray(first_node_embedding_in_cluster[c]).reshape(-1)
            for c in sorted_clusters
        ]

        heatmap_data = pd.DataFrame(embedding_list, index=stid_list)

        print(f"Number of clusters: {len(sorted_clusters)}")
        print(f"Embedding matrix shape: {heatmap_data.shape}")

        create_heatmap_with_stid(embedding_list, stid_list, save_path_heatmap_)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list, stid_list, save_path_matrix)

        break

    # ============================================================
    # 21. FINAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(all_embeddings, cluster_labels, stid_list, save_path_t_SNE)
    visualize_embeddings_pca(all_embeddings, cluster_labels, first_node_stId_in_cluster, save_path_pca)

    # ============================================================
    # 22. FINAL CLUSTERING METRICS
    # ============================================================

    if len(np.unique(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(all_embeddings, cluster_labels)
    else:
        silhouette_avg = np.nan
        davies_bouldin = np.nan

    print(f"\nSilhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    # ============================================================
    # 23. FINAL SUMMARY
    # ============================================================

    summary = (
        f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    )
    summary += f"Best Epoch: {best_epoch}\n"
    summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {best_f1_score} (thr={best_threshold})\n"
    summary += f"Final-epoch best-threshold valid F1: {f1_best_thresh_per_epoch_valid[-1]}\n"
    summary += f"Silhouette Score: {silhouette_avg}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin}\n"
    summary += f"Number of significant genes: {len(significant_stIds)}\n"

    save_file = os.path.join(
        results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt'
    )
    with open(save_file, 'w') as f:
        f.write(summary)

    # ============================================================
    # 24. FINAL INFORMATION
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL STAGE-1 RESULTS")
    print("=" * 70)
    print(f"Best epoch            : {best_epoch}")
    print(f"Best validation loss  : {best_valid_loss:.6f}")
    print(f"Best validation F1    : {best_f1_score:.6f} (thr={best_threshold:.2f})")
    print(f"Max training F1       : {max_f1_train:.6f}")
    print(f"Max validation F1     : {max_f1_valid:.6f}")
    print(f"Significant genes     : {len(significant_stIds)}")
    print(f"Model path:\n{model_path}")
    print("=" * 70)

    return model_path

def train_4(
    hyperparams=None,
    data_path='../data/omics/',
    plot=True,
    omics='cna',
    cancer='BLCA'
):
    """
    Stage-1 training:
        Predict generic biological significance from the graph.

    IMPORTANT:
        This stage does NOT use cancer-driver labels.
        'significance' is the Stage-1 pretraining target.

    Expected significance values:
        0 = not significant
        1 = significant

    If -1 is present:
        -1 = unlabeled / ignored during loss and F1 calculation
    """

    import os
    import copy
    import pickle

    import numpy as np
    import pandas as pd
    import torch
    import torch.optim as optim
    from tqdm import tqdm

    from sklearn import metrics
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score
    )

    # ============================================================
    # 1. Hyperparameters
    # ============================================================

    if hyperparams is None:
        raise ValueError("hyperparams must be provided.")

    num_epochs = hyperparams['num_epochs']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']

    print("\n" + "=" * 70)
    print("START STAGE-1 SIGNIFICANCE PRETRAINING")
    print("=" * 70)
    print(f"Omics       : {omics}")
    print(f"Cancer      : {cancer}")
    print(f"Device      : {device}")
    print(f"Epochs      : {num_epochs}")
    print(f"LR          : {learning_rate}")
    print(f"Batch size  : {batch_size}")
    print(f"Out feats   : {out_feats}")
    print(f"Layers      : {num_layers}")
    print(f"Loss type   : {hyperparams.get('loss_type', 'focal')}")
    print("=" * 70 + "\n")

    # ============================================================
    # 2. Model save path
    # ============================================================

    model_path = os.path.join(data_path, omics, cancer, 'emb', 'models')
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(
        model_path,
        f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth'
    )

    # ============================================================
    # 3. Load dataset and add node-degree structural features
    # ============================================================

    data_path_ = os.path.join(data_path, omics, cancer)
    ds = dataset.Dataset(data_path_)

    # Add node in-degree (normalized) as an extra structural feature to
    # both the training (0) and validation (1) graphs.
    for idx in [0, 1]:
        g, _ = ds[idx]

        node_degrees = g.in_degrees().float().unsqueeze(-1)
        node_degrees_norm = (
            (node_degrees - node_degrees.mean())
            / (node_degrees.std() + 1e-5)
        )

        if 'weight' in g.ndata:
            existing_feats = g.ndata['weight']
        elif 'feat' in g.ndata:
            existing_feats = g.ndata['feat']
        else:
            # No node features initialized yet — fall back to a constant.
            existing_feats = torch.ones((g.num_nodes(), 1), dtype=torch.float32)

        if existing_feats.dim() == 1:
            existing_feats = existing_feats.unsqueeze(-1)

        updated_feats = torch.cat([existing_feats, node_degrees_norm], dim=-1)

        # Keep both keys in sync so downstream code reading either won't break.
        g.ndata['feat'] = updated_feats
        g.ndata['weight'] = updated_feats

    graph, name = ds[0]

    # ============================================================
    # 4. SIGNIFICANCE LABEL DEBUG
    # ============================================================

    if 'significance' not in graph.ndata:
        raise KeyError(
            "graph.ndata['significance'] does not exist. "
            "Check your preprocessing and DGL graph conversion."
        )

    significance = graph.ndata['significance']

    print("\n" + "=" * 70)
    print("SIGNIFICANCE DEBUG")
    print("=" * 70)
    print("dtype :", significance.dtype)
    print("shape :", significance.shape)

    unique_labels, label_counts = torch.unique(significance, return_counts=True)
    print("unique labels:")
    for label, count in zip(unique_labels, label_counts):
        print(f"    label={label.item():>4} count={count.item():>6}")
    print("first 20:", significance[:20])
    print("=" * 70)

    valid_labels = significance[(significance == 0) | (significance == 1)]

    if valid_labels.numel() == 0:
        raise ValueError("No valid significance labels (0/1) were found.")

    num_positive = (valid_labels == 1).sum().item()
    num_negative = (valid_labels == 0).sum().item()
    num_unlabeled = (significance == -1).sum().item()

    print("\nLABEL DISTRIBUTION")
    print(f"Positive / significant : {num_positive}")
    print(f"Negative / non-signif. : {num_negative}")
    print(f"Unlabeled (-1)         : {num_unlabeled}")

    if num_positive == 0:
        raise ValueError("There are ZERO positive significance labels. F1 will necessarily be 0.")
    if num_negative == 0:
        raise ValueError("There are ZERO negative significance labels. Check your preprocessing.")

    positive_ratio = num_positive / valid_labels.numel()
    print(f"Positive ratio          : {positive_ratio:.4%}")

    # ============================================================
    # 5. Train / validation graphs
    # ============================================================

    ds_train = [ds[0]]
    ds_valid = [ds[1]]

    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # ============================================================
    # 6. Model / optimizer / best-model tracker
    # ============================================================

    net = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    # ============================================================
    # 7. Loss function
    # ============================================================

    # loss_type lets you A/B test FocalLoss against plain BCEWithLogitsLoss
    # without touching the rest of the training loop. Pass e.g.
    # hyperparams['loss_type'] = 'bce' to use the baseline.
    loss_type = hyperparams.get('loss_type', 'focal')

    total_valid = num_positive + num_negative

    if loss_type == 'bce':
        # Standard BCE class-balancing convention: pos_weight scales the
        # loss contribution of positive examples. Since positives are the
        # minority class here, pos_weight > 1 upweights them.
        #
        # NOTE: pos_weight = neg/pos makes the *weighted* cost of a false
        # negative and a false positive roughly equal at the population
        # level. That means a constant output of p* ~ 0.5 for every node
        # (ignoring the graph entirely) sits very close to loss-optimal —
        # a flat, easy-to-find attractor that gradient descent can settle
        # into instead of learning a genuinely discriminative solution.
        # Nudging pos_weight away from that exact tie point (via
        # pos_weight_multiplier) breaks the tie so the degenerate constant
        # solution is no longer near-optimal.
        pos_weight_multiplier = hyperparams.get('pos_weight_multiplier', 1.3)
        pos_weight_value = (num_negative / max(num_positive, 1)) * pos_weight_multiplier
        pos_weight = torch.tensor(pos_weight_value, device=device)

        print(
            f"Configuring BCEWithLogitsLoss (baseline) with "
            f"pos_weight={pos_weight_value:.4f} "
            f"(base={num_negative / max(num_positive, 1):.4f} x multiplier={pos_weight_multiplier:.2f})"
        )

        bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
        criterion = bce_criterion

    else:
        # alpha favors the rare positive class: weight ~ (negatives / total)
        dynamic_alpha = num_negative / total_valid

        print(f"Configuring Focal Loss with Dynamic Alpha: {dynamic_alpha:.4f} (favoring positive class)")

        criterion = FocalLoss(alpha=dynamic_alpha, gamma=2.0, reduction='mean')

        # NOTE: do NOT use a hardcoded weight tensor like [0.00001, 0.99999] —
        # that nearly fully suppresses class 0. And since FocalLoss(reduction='mean')
        # already returns a scalar, multiplying it by node-level weights afterwards
        # would be incorrect, so alpha is passed straight into FocalLoss instead.

    # ============================================================
    # 8. Training history
    # ============================================================

    loss_per_epoch_train = []
    loss_per_epoch_valid = []

    f1_per_epoch_train = []
    f1_per_epoch_valid = []
    f1_best_thresh_per_epoch_valid = []

    max_f1_scores_train = []
    max_f1_scores_valid = []

    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_f1_score = 0.0
    best_epoch = 0
    best_threshold = 0.5

    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    # ============================================================
    # 9. Results path
    # ============================================================

    results_path = os.path.abspath(
        os.path.join('results', 'multiomics_meth', 'node_embeddings', omics, cancer)
    )
    os.makedirs(results_path, exist_ok=True)

    # ============================================================
    # 10. INITIAL EMBEDDINGS
    # ============================================================

    print("\n" + "=" * 70)
    print("CALCULATING INITIAL EMBEDDINGS")
    print("=" * 70)

    all_embeddings_initial, cluster_labels_initial = calculate_cluster_labels(
        best_model, dl_train, device
    )
    all_embeddings_initial = all_embeddings_initial.reshape(all_embeddings_initial.shape[0], -1)

    save_path_heatmap_initial = os.path.join(
        results_path, f'embeddings_heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_matrix_initial = os.path.join(
        results_path, f'embeddings_matrix_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_pca_initial = os.path.join(
        results_path, f'embeddings_pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_t_SNE_initial = os.path.join(
        results_path, f'embeddings_t-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )

    # ============================================================
    # 11. INITIAL EMBEDDING INFORMATION
    # ============================================================

    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings_initial = (
            best_model.get_node_embeddings(graph).detach().cpu().numpy()
        )

        graph_path = os.path.join(data_path, omics, cancer, 'emb', 'raw', 'emb_train.pkl')
        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert len(cluster_labels_initial) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index_initial = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster_initial = {}
        first_node_embedding_in_cluster_initial = {}
        stid_dic_initial = {}

        for node in nx_graph.nodes:
            if 'stId' in nx_graph.nodes[node]:
                stId = nx_graph.nodes[node]['stId']
                stid_dic_initial[stId] = node_embeddings_initial[node_to_index_initial[node]]

        stid_df_initial = pd.DataFrame.from_dict(stid_dic_initial, orient='index')

        csv_save_path_initial = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.csv'
        )
        stid_df_initial.to_csv(csv_save_path_initial, index_label='stId')

        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if 'stId' in nx_graph.nodes[node]:
                if cluster not in first_node_stId_in_cluster_initial:
                    first_node_stId_in_cluster_initial[cluster] = nx_graph.nodes[node]['stId']
                    first_node_embedding_in_cluster_initial[cluster] = (
                        node_embeddings_initial[node_to_index_initial[node]]
                    )

        print("first_node_stId_in_cluster_initial:")
        print(first_node_stId_in_cluster_initial)

        sorted_clusters_initial = sorted(first_node_stId_in_cluster_initial.keys())
        stid_list_initial = [first_node_stId_in_cluster_initial[c] for c in sorted_clusters_initial]
        embedding_list_initial = [
            np.asarray(first_node_embedding_in_cluster_initial[c]).reshape(-1)
            for c in sorted_clusters_initial
        ]

        heatmap_data_initial = pd.DataFrame(embedding_list_initial, index=stid_list_initial)

        print(f"Number of clusters: {len(sorted_clusters_initial)}")
        print(f"Embedding matrix shape: {heatmap_data_initial.shape}")

        create_heatmap_with_stid(embedding_list_initial, stid_list_initial, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(
            embedding_list_initial, stid_list_initial, save_path_matrix_initial
        )

        break

    # ============================================================
    # 12. INITIAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(
        all_embeddings_initial, cluster_labels_initial, stid_list_initial, save_path_t_SNE_initial
    )
    visualize_embeddings_pca(
        all_embeddings_initial, cluster_labels_initial,
        first_node_stId_in_cluster_initial, save_path_pca_initial
    )

    if len(np.unique(cluster_labels_initial)) > 1:
        silhouette_avg_ = silhouette_score(all_embeddings_initial, cluster_labels_initial)
        davies_bouldin_ = davies_bouldin_score(all_embeddings_initial, cluster_labels_initial)
    else:
        silhouette_avg_ = np.nan
        davies_bouldin_ = np.nan

    summary_ = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_ = os.path.join(
        results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt'
    )
    with open(save_file_, 'w') as f:
        f.write(summary_)

    # ============================================================
    # 13. TRAINING
    # ============================================================

    print("\n" + "=" * 70)
    print("START TRAINING")
    print("=" * 70)

    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=True) as pbar:

        for epoch in range(num_epochs):

            # ---------------- TRAIN ----------------
            net.train()
            loss_per_graph = []
            f1_per_graph = []

            for data in dl_train:
                graph, name = data
                graph = graph.to(device)

                logits = net(graph).view(-1)
                labels = graph.ndata['significance'].float().view(-1)

                valid_mask = (labels == 0) | (labels == 1)
                if valid_mask.sum().item() == 0:
                    raise ValueError("Training graph contains no valid significance labels.")

                valid_logits = logits[valid_mask]
                valid_labels = labels[valid_mask]

                loss = criterion(valid_logits, valid_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_per_graph.append(loss.item())

                with torch.no_grad():
                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()
                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0
                    )
                    f1_per_graph.append(f1)

            running_loss_train = float(np.mean(loss_per_graph))
            running_f1_train = float(np.mean(f1_per_graph))
            loss_per_epoch_train.append(running_loss_train)
            f1_per_epoch_train.append(running_f1_train)

            # ---------------- VALIDATION ----------------
            net.eval()
            loss_per_graph = []
            f1_per_graph = []

            # Keep the last batch's probs/labels around for the threshold
            # diagnostic and best-model selection below.
            val_probs, val_labels_int = None, None

            with torch.no_grad():
                for data in dl_valid:
                    graph, name = data
                    graph = graph.to(device)

                    logits = net(graph).view(-1)
                    labels = graph.ndata['significance'].float().view(-1)

                    valid_mask = (labels == 0) | (labels == 1)
                    if valid_mask.sum().item() == 0:
                        continue

                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]

                    loss = criterion(valid_logits, valid_labels)
                    loss_per_graph.append(loss.item())

                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()

                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0
                    )
                    f1_per_graph.append(f1)

                    val_probs, val_labels_int = probs, labels_int

            if len(loss_per_graph) == 0:
                raise ValueError("Validation graph contains no valid significance labels.")

            running_loss_valid = float(np.mean(loss_per_graph))
            running_f1_val = float(np.mean(f1_per_graph))
            loss_per_epoch_valid.append(running_loss_valid)
            f1_per_epoch_valid.append(running_f1_val)

            max_f1_train = max(f1_per_epoch_train)
            max_f1_valid = max(f1_per_epoch_valid)
            max_f1_scores_train.append(max_f1_train)
            max_f1_scores_valid.append(max_f1_valid)

            # ---------------- BEST-THRESHOLD F1 (every epoch) ----------------
            # A fixed thr=0.2 F1 is a bad lens once probabilities drift off
            # center (e.g. the whole distribution sliding below 0.5): the
            # ranking underneath can still be fine or improving while
            # thr=0.2 F1 reads a flat zero. Sweeping thresholds every epoch
            # (not just when loss improves) avoids that blind spot and is
            # what actually gates the checkpoint below.
            best_f1_this_epoch = 0.0
            best_threshold_this_epoch = 0.5
            if val_probs is not None:
                for threshold in THRESHOLDS:
                    preds_tmp = (val_probs > threshold).int()
                    f1_tmp = metrics.f1_score(
                        val_labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0
                    )
                    if f1_tmp > best_f1_this_epoch:
                        best_f1_this_epoch = f1_tmp
                        best_threshold_this_epoch = threshold

            f1_best_thresh_per_epoch_valid.append(best_f1_this_epoch)

            # ---------------- BEST MODEL SELECTION ----------------
            # Primary criterion: best-across-threshold validation F1 must
            # improve. This is what actually reflects the model's ability
            # to separate classes, unlike raw loss (which a degenerate
            # constant-output solution can minimize) or thr=0.2 F1 (which
            # can read 0 even when the underlying ranking is fine). Loss is
            # used only as a tie-breaker when F1 is unchanged, so we still
            # prefer the more confident/better-calibrated model among ties.
            f1_improved = best_f1_this_epoch > best_f1_score
            f1_tied_but_loss_improved = (
                best_f1_this_epoch == best_f1_score and running_loss_valid < best_valid_loss
            )

            if f1_improved or f1_tied_but_loss_improved:

                best_train_loss = running_loss_train
                best_valid_loss = running_loss_valid
                best_f1_score = best_f1_this_epoch
                best_epoch = epoch + 1
                best_threshold = best_threshold_this_epoch

                best_model.load_state_dict(copy.deepcopy(net.state_dict()))

                print("\n*** BEST MODEL UPDATED ***")
                print(f"Epoch                : {best_epoch}")
                print(f"Train Loss           : {best_train_loss:.6f}")
                print(f"Valid Loss           : {best_valid_loss:.6f}")
                print(f"Valid F1 (thr=0.2)   : {running_f1_val:.6f}")
                print(f"Best F1 across thresh: {best_f1_score:.6f} (thr={best_threshold:.2f})")

            # ---------------- DETAILED DIAGNOSTIC ----------------
            # Every epoch for the first 15 (early training is where sudden
            # collapses/drifts like the epoch-7 case tend to happen and are
            # otherwise invisible between the epoch-1 and epoch-10 prints),
            # then every 10th epoch after that.
            if epoch < 15 or (epoch + 1) % 10 == 0:

                for data in dl_valid:
                    graph, _ = data
                    graph = graph.to(device)

                    logits = net(graph).view(-1)
                    labels = graph.ndata['significance'].float().view(-1)

                    valid_mask = (labels == 0) | (labels == 1)
                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]

                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > 0.5).int()
                    labels_int = valid_labels.int()

                    tp = ((labels_int == 1) & (preds == 1)).sum().item()
                    fp = ((labels_int == 0) & (preds == 1)).sum().item()
                    fn = ((labels_int == 1) & (preds == 0)).sum().item()
                    tn = ((labels_int == 0) & (preds == 0)).sum().item()

                    print("\n" + "-" * 70)
                    print(f"EPOCH {epoch + 1} DIAGNOSTIC")
                    print("-" * 70)
                    print(f"Train Loss : {running_loss_train:.6f}")
                    print(f"Valid Loss : {running_loss_valid:.6f}")
                    print(f"Train F1   : {running_f1_train:.6f}")
                    print(f"Valid F1   : {running_f1_val:.6f}")
                    print(f"TP={tp} | FP={fp} | FN={fn} | TN={tn}")
                    print(f"Probability min  : {probs.min().item():.6f}")
                    print(f"Probability max  : {probs.max().item():.6f}")
                    print(f"Probability mean : {probs.mean().item():.6f}")
                    print("-" * 70)

                    for threshold in THRESHOLDS:
                        preds_tmp = (probs > threshold).int()
                        f1_tmp = metrics.f1_score(
                            labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0
                        )
                        print(
                            f"threshold={threshold:.2f} | "
                            f"true_pos={(labels_int == 1).sum().item()} | "
                            f"true_neg={(labels_int == 0).sum().item()} | "
                            f"pred_pos={(preds_tmp == 1).sum().item()} | "
                            f"pred_neg={(preds_tmp == 0).sum().item()} | "
                            f"F1={f1_tmp:.4f}"
                        )

                    break

            pbar.update(1)
            pbar.set_postfix({
                'train_loss': f'{running_loss_train:.4f}',
                'valid_loss': f'{running_loss_valid:.4f}',
                'train_f1': f'{running_f1_train:.4f}',
                'valid_f1': f'{running_f1_val:.4f}'
            })

    # ============================================================
    # 14. FINAL BEST MODEL
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Best epoch       : {best_epoch}")
    print(f"Best train loss  : {best_train_loss:.6f}")
    print(f"Best valid loss  : {best_valid_loss:.6f}")
    print(f"Best valid F1    : {best_f1_score:.6f} (thr={best_threshold:.2f})")

    # ============================================================
    # 15. FINAL EMBEDDINGS
    # ============================================================

    all_embeddings, cluster_labels = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)

    # ============================================================
    # 16. COSINE SIMILARITY
    # ============================================================

    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized_embeddings = all_embeddings / norms
    cos_sim = np.dot(normalized_embeddings, normalized_embeddings.T)

    # ============================================================
    # 17. PLOTS
    # ============================================================

    if plot:
        loss_path = os.path.join(
            results_path, f'embeddings_loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        f1_path = os.path.join(
            results_path, f'embeddings_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        max_f1_path = os.path.join(
            results_path, f'embeddings_max_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        f1_best_thresh_path = os.path.join(
            results_path,
            f'embeddings_f1_best_thresh_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_max_f1_plot(max_f1_scores_train, max_f1_scores_valid, max_f1_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

        # This plot shows what the checkpoint selection actually sees, unlike
        # the fixed-thr=0.2 plot above: it can look completely flat at zero
        # (e.g. once probabilities drift below 0.5) even while this line
        # shows the model still improving underneath that blind spot.
        #
        # NOTE: we use matplotlib's Figure API directly (not pyplot) so this
        # doesn't touch the global backend/state — calling matplotlib.use()
        # or pyplot inside a library function can clobber an interactive
        # notebook backend for the rest of the session.
        from matplotlib.figure import Figure

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot(f1_per_epoch_valid, label='validation (thr=0.2)')
        ax.plot(f1_best_thresh_per_epoch_valid, label='validation (best threshold)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1-score')
        ax.set_title('Validation F1: fixed threshold vs. best threshold')
        ax.legend()
        fig.savefig(f1_best_thresh_path)

    # ============================================================
    # 18. SAVE BEST MODEL
    # ============================================================

    torch.save(best_model.state_dict(), model_path)
    print(f"\nBest model saved to:\n{model_path}")

    # ============================================================
    # 19. FINAL VISUALIZATION PATHS
    # ============================================================

    save_path_pca = os.path.join(
        results_path, f'embeddings_pca_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_t_SNE = os.path.join(
        results_path, f'embeddings_t-SNE_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_heatmap_ = os.path.join(
        results_path, f'embeddings_heatmap_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_matrix = os.path.join(
        results_path, f'embeddings_matrix_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )

    # ============================================================
    # 20. FINAL NODE / CLUSTER INFORMATION
    # ============================================================

    cluster_stId_dict = {}
    significant_stIds = []
    clusters_with_significant_stId = {}
    clusters_node_info = {}

    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()

        graph_path = os.path.join(data_path, omics, cancer, 'emb', 'raw', 'emb_train.pkl')
        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert len(cluster_labels) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster = {}
        first_node_embedding_in_cluster = {}
        stid_dic = {}

        for node in nx_graph.nodes:
            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]
            stid_dic[stid] = node_embeddings[idx]

            significance_value = graph.ndata['significance'][idx].item()
            if significance_value == 1:
                significant_stIds.append(stid)

        stid_df_final = pd.DataFrame.from_dict(stid_dic, orient='index')
        csv_save_path_final = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.csv'
        )
        stid_df_final.to_csv(csv_save_path_final, index_label='stId')

        for node, cluster in zip(nx_graph.nodes, cluster_labels):
            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]
            significance_value = graph.ndata['significance'][idx].item()

            if cluster not in first_node_stId_in_cluster:
                first_node_stId_in_cluster[cluster] = stid
                first_node_embedding_in_cluster[cluster] = node_embeddings[idx]

            cluster_stId_dict.setdefault(cluster, []).append(stid)

            clusters_with_significant_stId.setdefault(cluster, [])
            if significance_value == 1:
                clusters_with_significant_stId[cluster].append(stid)

            clusters_node_info.setdefault(cluster, []).append({
                'stId': stid,
                'significance': significance_value,
                'other_info': nx_graph.nodes[node]
            })

        print("\nFirst node in each cluster:")
        print(first_node_stId_in_cluster)

        sorted_clusters = sorted(first_node_stId_in_cluster.keys())
        stid_list = [first_node_stId_in_cluster[c] for c in sorted_clusters]
        embedding_list = [
            np.asarray(first_node_embedding_in_cluster[c]).reshape(-1)
            for c in sorted_clusters
        ]

        heatmap_data = pd.DataFrame(embedding_list, index=stid_list)

        print(f"Number of clusters: {len(sorted_clusters)}")
        print(f"Embedding matrix shape: {heatmap_data.shape}")

        create_heatmap_with_stid(embedding_list, stid_list, save_path_heatmap_)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list, stid_list, save_path_matrix)

        break

    # ============================================================
    # 21. FINAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(all_embeddings, cluster_labels, stid_list, save_path_t_SNE)
    visualize_embeddings_pca(all_embeddings, cluster_labels, first_node_stId_in_cluster, save_path_pca)

    # ============================================================
    # 22. FINAL CLUSTERING METRICS
    # ============================================================

    if len(np.unique(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(all_embeddings, cluster_labels)
    else:
        silhouette_avg = np.nan
        davies_bouldin = np.nan

    print(f"\nSilhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    # ============================================================
    # 23. FINAL SUMMARY
    # ============================================================

    summary = (
        f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    )
    summary += f"Best Epoch: {best_epoch}\n"
    summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {best_f1_score} (thr={best_threshold})\n"
    summary += f"Final-epoch best-threshold valid F1: {f1_best_thresh_per_epoch_valid[-1]}\n"
    summary += f"Silhouette Score: {silhouette_avg}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin}\n"
    summary += f"Number of significant genes: {len(significant_stIds)}\n"

    save_file = os.path.join(
        results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt'
    )
    with open(save_file, 'w') as f:
        f.write(summary)

    # ============================================================
    # 24. FINAL INFORMATION
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL STAGE-1 RESULTS")
    print("=" * 70)
    print(f"Best epoch            : {best_epoch}")
    print(f"Best validation loss  : {best_valid_loss:.6f}")
    print(f"Best validation F1    : {best_f1_score:.6f} (thr={best_threshold:.2f})")
    print(f"Max training F1       : {max_f1_train:.6f}")
    print(f"Max validation F1     : {max_f1_valid:.6f}")
    print(f"Significant genes     : {len(significant_stIds)}")
    print(f"Model path:\n{model_path}")
    print("=" * 70)

    return model_path


def train_4(
    hyperparams=None,
    data_path='../data/omics/',
    plot=True,
    omics='cna',
    cancer='BLCA'
):
    """
    Stage-1 training:
        Predict generic biological significance from the graph.

    IMPORTANT:
        This stage does NOT use cancer-driver labels.
        'significance' is the Stage-1 pretraining target.

    Expected significance values:
        0 = not significant
        1 = significant

    If -1 is present:
        -1 = unlabeled / ignored during loss and F1 calculation
    """

    import os
    import copy
    import pickle

    import numpy as np
    import pandas as pd
    import torch
    import torch.optim as optim
    from tqdm import tqdm

    from sklearn import metrics
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score
    )

    # ============================================================
    # 1. Hyperparameters
    # ============================================================

    if hyperparams is None:
        raise ValueError("hyperparams must be provided.")

    num_epochs = hyperparams['num_epochs']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']

    # The probability cutoff used to turn sigmoid outputs into 0/1
    # predictions everywhere in this function (training F1, diagnostics,
    # etc). Override via hyperparams['decision_threshold'] = 0.2, for
    # example, once you've picked an operating point from the
    # best-threshold plots rather than assuming 0.5 is correct.
    decision_threshold = hyperparams.get('decision_threshold', 0.2)

    print("\n" + "=" * 70)
    print("START STAGE-1 SIGNIFICANCE PRETRAINING")
    print("=" * 70)
    print(f"Omics       : {omics}")
    print(f"Cancer      : {cancer}")
    print(f"Device      : {device}")
    print(f"Epochs      : {num_epochs}")
    print(f"LR          : {learning_rate}")
    print(f"Batch size  : {batch_size}")
    print(f"Out feats   : {out_feats}")
    print(f"Layers      : {num_layers}")
    print(f"Loss type   : {hyperparams.get('loss_type', 'focal')}")
    print(f"Decision thr: {decision_threshold}")
    print("=" * 70 + "\n")

    # ============================================================
    # 2. Model save path
    # ============================================================

    model_path = os.path.join(data_path, omics, cancer, 'emb', 'models')
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(
        model_path,
        f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth'
    )

    # ============================================================
    # 3. Load dataset and add node-degree structural features
    # ============================================================

    data_path_ = os.path.join(data_path, omics, cancer)
    ds = dataset.Dataset(data_path_)

    # Add node in-degree (normalized) as an extra structural feature to
    # both the training (0) and validation (1) graphs.
    for idx in [0, 1]:
        g, _ = ds[idx]

        node_degrees = g.in_degrees().float().unsqueeze(-1)
        node_degrees_norm = (
            (node_degrees - node_degrees.mean())
            / (node_degrees.std() + 1e-5)
        )

        if 'weight' in g.ndata:
            existing_feats = g.ndata['weight']
        elif 'feat' in g.ndata:
            existing_feats = g.ndata['feat']
        else:
            # No node features initialized yet — fall back to a constant.
            existing_feats = torch.ones((g.num_nodes(), 1), dtype=torch.float32)

        if existing_feats.dim() == 1:
            existing_feats = existing_feats.unsqueeze(-1)

        updated_feats = torch.cat([existing_feats, node_degrees_norm], dim=-1)

        # Keep both keys in sync so downstream code reading either won't break.
        g.ndata['feat'] = updated_feats
        g.ndata['weight'] = updated_feats

    graph, name = ds[0]

    # ============================================================
    # 4. SIGNIFICANCE LABEL DEBUG
    # ============================================================

    if 'significance' not in graph.ndata:
        raise KeyError(
            "graph.ndata['significance'] does not exist. "
            "Check your preprocessing and DGL graph conversion."
        )

    significance = graph.ndata['significance']

    print("\n" + "=" * 70)
    print("SIGNIFICANCE DEBUG")
    print("=" * 70)
    print("dtype :", significance.dtype)
    print("shape :", significance.shape)

    unique_labels, label_counts = torch.unique(significance, return_counts=True)
    print("unique labels:")
    for label, count in zip(unique_labels, label_counts):
        print(f"    label={label.item():>4} count={count.item():>6}")
    print("first 20:", significance[:20])
    print("=" * 70)

    valid_labels = significance[(significance == 0) | (significance == 1)]

    if valid_labels.numel() == 0:
        raise ValueError("No valid significance labels (0/1) were found.")

    num_positive = (valid_labels == 1).sum().item()
    num_negative = (valid_labels == 0).sum().item()
    num_unlabeled = (significance == -1).sum().item()

    print("\nLABEL DISTRIBUTION")
    print(f"Positive / significant : {num_positive}")
    print(f"Negative / non-signif. : {num_negative}")
    print(f"Unlabeled (-1)         : {num_unlabeled}")

    if num_positive == 0:
        raise ValueError("There are ZERO positive significance labels. F1 will necessarily be 0.")
    if num_negative == 0:
        raise ValueError("There are ZERO negative significance labels. Check your preprocessing.")

    positive_ratio = num_positive / valid_labels.numel()
    print(f"Positive ratio          : {positive_ratio:.4%}")

    # ============================================================
    # 5. Train / validation graphs
    # ============================================================

    ds_train = [ds[0]]
    ds_valid = [ds[1]]

    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # ============================================================
    # 6. Model / optimizer / best-model tracker
    # ============================================================

    net = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    # ============================================================
    # 7. Loss function
    # ============================================================

    # loss_type lets you A/B test FocalLoss against plain BCEWithLogitsLoss
    # without touching the rest of the training loop. Pass e.g.
    # hyperparams['loss_type'] = 'bce' to use the baseline.
    loss_type = hyperparams.get('loss_type', 'focal')

    total_valid = num_positive + num_negative

    if loss_type == 'bce':
        # Standard BCE class-balancing convention: pos_weight scales the
        # loss contribution of positive examples. Since positives are the
        # minority class here, pos_weight > 1 upweights them.
        #
        # NOTE: pos_weight = neg/pos makes the *weighted* cost of a false
        # negative and a false positive roughly equal at the population
        # level. That means a constant output of p* ~ 0.5 for every node
        # (ignoring the graph entirely) sits very close to loss-optimal —
        # a flat, easy-to-find attractor that gradient descent can settle
        # into instead of learning a genuinely discriminative solution.
        # Nudging pos_weight away from that exact tie point (via
        # pos_weight_multiplier) breaks the tie so the degenerate constant
        # solution is no longer near-optimal.
        pos_weight_multiplier = hyperparams.get('pos_weight_multiplier', 1.3)
        pos_weight_value = (num_negative / max(num_positive, 1)) * pos_weight_multiplier
        pos_weight = torch.tensor(pos_weight_value, device=device)

        print(
            f"Configuring BCEWithLogitsLoss (baseline) with "
            f"pos_weight={pos_weight_value:.4f} "
            f"(base={num_negative / max(num_positive, 1):.4f} x multiplier={pos_weight_multiplier:.2f})"
        )

        bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
        criterion = bce_criterion

    else:
        # alpha favors the rare positive class: weight ~ (negatives / total)
        dynamic_alpha = num_negative / total_valid

        print(f"Configuring Focal Loss with Dynamic Alpha: {dynamic_alpha:.4f} (favoring positive class)")

        criterion = FocalLoss(alpha=dynamic_alpha, gamma=2.0, reduction='mean')

        # NOTE: do NOT use a hardcoded weight tensor like [0.00001, 0.99999] —
        # that nearly fully suppresses class 0. And since FocalLoss(reduction='mean')
        # already returns a scalar, multiplying it by node-level weights afterwards
        # would be incorrect, so alpha is passed straight into FocalLoss instead.

    # ============================================================
    # 8. Training history
    # ============================================================

    loss_per_epoch_train = []
    loss_per_epoch_valid = []

    f1_per_epoch_train = []
    f1_per_epoch_valid = []
    f1_best_thresh_per_epoch_valid = []

    max_f1_scores_train = []
    max_f1_scores_valid = []

    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_f1_score = 0.0
    best_epoch = 0
    best_threshold = 0.5

    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    # ============================================================
    # 9. Results path
    # ============================================================

    results_path = os.path.abspath(
        os.path.join('results', 'multiomics_meth', 'node_embeddings', omics, cancer)
    )
    os.makedirs(results_path, exist_ok=True)

    # ============================================================
    # 10. INITIAL EMBEDDINGS
    # ============================================================

    print("\n" + "=" * 70)
    print("CALCULATING INITIAL EMBEDDINGS")
    print("=" * 70)

    all_embeddings_initial, cluster_labels_initial = calculate_cluster_labels(
        best_model, dl_train, device
    )
    all_embeddings_initial = all_embeddings_initial.reshape(all_embeddings_initial.shape[0], -1)

    save_path_heatmap_initial = os.path.join(
        results_path, f'embeddings_heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_matrix_initial = os.path.join(
        results_path, f'embeddings_matrix_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_pca_initial = os.path.join(
        results_path, f'embeddings_pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_t_SNE_initial = os.path.join(
        results_path, f'embeddings_t-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )

    # ============================================================
    # 11. INITIAL EMBEDDING INFORMATION
    # ============================================================

    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings_initial = (
            best_model.get_node_embeddings(graph).detach().cpu().numpy()
        )

        graph_path = os.path.join(data_path, omics, cancer, 'emb', 'raw', 'emb_train.pkl')
        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert len(cluster_labels_initial) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index_initial = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster_initial = {}
        first_node_embedding_in_cluster_initial = {}
        stid_dic_initial = {}

        for node in nx_graph.nodes:
            if 'stId' in nx_graph.nodes[node]:
                stId = nx_graph.nodes[node]['stId']
                stid_dic_initial[stId] = node_embeddings_initial[node_to_index_initial[node]]

        stid_df_initial = pd.DataFrame.from_dict(stid_dic_initial, orient='index')

        csv_save_path_initial = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.csv'
        )
        stid_df_initial.to_csv(csv_save_path_initial, index_label='stId')

        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if 'stId' in nx_graph.nodes[node]:
                if cluster not in first_node_stId_in_cluster_initial:
                    first_node_stId_in_cluster_initial[cluster] = nx_graph.nodes[node]['stId']
                    first_node_embedding_in_cluster_initial[cluster] = (
                        node_embeddings_initial[node_to_index_initial[node]]
                    )

        print("first_node_stId_in_cluster_initial:")
        print(first_node_stId_in_cluster_initial)

        sorted_clusters_initial = sorted(first_node_stId_in_cluster_initial.keys())
        stid_list_initial = [first_node_stId_in_cluster_initial[c] for c in sorted_clusters_initial]
        embedding_list_initial = [
            np.asarray(first_node_embedding_in_cluster_initial[c]).reshape(-1)
            for c in sorted_clusters_initial
        ]

        heatmap_data_initial = pd.DataFrame(embedding_list_initial, index=stid_list_initial)

        print(f"Number of clusters: {len(sorted_clusters_initial)}")
        print(f"Embedding matrix shape: {heatmap_data_initial.shape}")

        create_heatmap_with_stid(embedding_list_initial, stid_list_initial, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(
            embedding_list_initial, stid_list_initial, save_path_matrix_initial
        )

        break

    # ============================================================
    # 12. INITIAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(
        all_embeddings_initial, cluster_labels_initial, stid_list_initial, save_path_t_SNE_initial
    )
    visualize_embeddings_pca(
        all_embeddings_initial, cluster_labels_initial,
        first_node_stId_in_cluster_initial, save_path_pca_initial
    )

    if len(np.unique(cluster_labels_initial)) > 1:
        silhouette_avg_ = silhouette_score(all_embeddings_initial, cluster_labels_initial)
        davies_bouldin_ = davies_bouldin_score(all_embeddings_initial, cluster_labels_initial)
    else:
        silhouette_avg_ = np.nan
        davies_bouldin_ = np.nan

    summary_ = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_ = os.path.join(
        results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt'
    )
    with open(save_file_, 'w') as f:
        f.write(summary_)

    # ============================================================
    # 13. TRAINING
    # ============================================================

    print("\n" + "=" * 70)
    print("START TRAINING")
    print("=" * 70)

    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=True) as pbar:

        for epoch in range(num_epochs):

            # ---------------- TRAIN ----------------
            net.train()
            loss_per_graph = []
            f1_per_graph = []

            for data in dl_train:
                graph, name = data
                graph = graph.to(device)

                logits = net(graph).view(-1)
                labels = graph.ndata['significance'].float().view(-1)

                valid_mask = (labels == 0) | (labels == 1)
                if valid_mask.sum().item() == 0:
                    raise ValueError("Training graph contains no valid significance labels.")

                valid_logits = logits[valid_mask]
                valid_labels = labels[valid_mask]

                loss = criterion(valid_logits, valid_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_per_graph.append(loss.item())

                with torch.no_grad():
                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > decision_threshold).int()
                    labels_int = valid_labels.int()
                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0
                    )
                    f1_per_graph.append(f1)

            running_loss_train = float(np.mean(loss_per_graph))
            running_f1_train = float(np.mean(f1_per_graph))
            loss_per_epoch_train.append(running_loss_train)
            f1_per_epoch_train.append(running_f1_train)

            # ---------------- VALIDATION ----------------
            net.eval()
            loss_per_graph = []
            f1_per_graph = []

            # Keep the last batch's probs/labels around for the threshold
            # diagnostic and best-model selection below.
            val_probs, val_labels_int = None, None

            with torch.no_grad():
                for data in dl_valid:
                    graph, name = data
                    graph = graph.to(device)

                    logits = net(graph).view(-1)
                    labels = graph.ndata['significance'].float().view(-1)

                    valid_mask = (labels == 0) | (labels == 1)
                    if valid_mask.sum().item() == 0:
                        continue

                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]

                    loss = criterion(valid_logits, valid_labels)
                    loss_per_graph.append(loss.item())

                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > decision_threshold).int()
                    labels_int = valid_labels.int()

                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0
                    )
                    f1_per_graph.append(f1)

                    val_probs, val_labels_int = probs, labels_int

            if len(loss_per_graph) == 0:
                raise ValueError("Validation graph contains no valid significance labels.")

            running_loss_valid = float(np.mean(loss_per_graph))
            running_f1_val = float(np.mean(f1_per_graph))
            loss_per_epoch_valid.append(running_loss_valid)
            f1_per_epoch_valid.append(running_f1_val)

            max_f1_train = max(f1_per_epoch_train)
            max_f1_valid = max(f1_per_epoch_valid)
            max_f1_scores_train.append(max_f1_train)
            max_f1_scores_valid.append(max_f1_valid)

            # ---------------- BEST-THRESHOLD F1 (every epoch) ----------------
            # decision_threshold as the fixed operating point can be a bad
            # lens once probabilities drift off center (e.g. the whole
            # distribution sliding below it): the ranking underneath can
            # still be fine or improving while fixed-threshold F1 reads a
            # flat zero. Sweeping thresholds every epoch (not just when loss
            # improves) avoids that blind spot and is what actually gates
            # the checkpoint below.
            best_f1_this_epoch = 0.0
            best_threshold_this_epoch = 0.5
            if val_probs is not None:
                for threshold in THRESHOLDS:
                    preds_tmp = (val_probs > threshold).int()
                    f1_tmp = metrics.f1_score(
                        val_labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0
                    )
                    if f1_tmp > best_f1_this_epoch:
                        best_f1_this_epoch = f1_tmp
                        best_threshold_this_epoch = threshold

            f1_best_thresh_per_epoch_valid.append(best_f1_this_epoch)

            # ---------------- BEST MODEL SELECTION ----------------
            # Primary criterion: best-across-threshold validation F1 must
            # improve. This is what actually reflects the model's ability
            # to separate classes, unlike raw loss (which a degenerate
            # constant-output solution can minimize) or fixed-threshold F1 (which
            # can read 0 even when the underlying ranking is fine). Loss is
            # used only as a tie-breaker when F1 is unchanged, so we still
            # prefer the more confident/better-calibrated model among ties.
            f1_improved = best_f1_this_epoch > best_f1_score
            f1_tied_but_loss_improved = (
                best_f1_this_epoch == best_f1_score and running_loss_valid < best_valid_loss
            )

            if f1_improved or f1_tied_but_loss_improved:

                best_train_loss = running_loss_train
                best_valid_loss = running_loss_valid
                best_f1_score = best_f1_this_epoch
                best_epoch = epoch + 1
                best_threshold = best_threshold_this_epoch

                best_model.load_state_dict(copy.deepcopy(net.state_dict()))

                print("\n*** BEST MODEL UPDATED ***")
                print(f"Epoch                : {best_epoch}")
                print(f"Train Loss           : {best_train_loss:.6f}")
                print(f"Valid Loss           : {best_valid_loss:.6f}")
                print(f"Valid F1 (thr={decision_threshold})   : {running_f1_val:.6f}")
                print(f"Best F1 across thresh: {best_f1_score:.6f} (thr={best_threshold:.2f})")

            # ---------------- DETAILED DIAGNOSTIC ----------------
            # Every epoch for the first 15 (early training is where sudden
            # collapses/drifts like the epoch-7 case tend to happen and are
            # otherwise invisible between the epoch-1 and epoch-10 prints),
            # then every 10th epoch after that.
            if epoch < 15 or (epoch + 1) % 10 == 0:

                for data in dl_valid:
                    graph, _ = data
                    graph = graph.to(device)

                    logits = net(graph).view(-1)
                    labels = graph.ndata['significance'].float().view(-1)

                    valid_mask = (labels == 0) | (labels == 1)
                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]

                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > decision_threshold).int()
                    labels_int = valid_labels.int()

                    tp = ((labels_int == 1) & (preds == 1)).sum().item()
                    fp = ((labels_int == 0) & (preds == 1)).sum().item()
                    fn = ((labels_int == 1) & (preds == 0)).sum().item()
                    tn = ((labels_int == 0) & (preds == 0)).sum().item()

                    print("\n" + "-" * 70)
                    print(f"EPOCH {epoch + 1} DIAGNOSTIC")
                    print("-" * 70)
                    print(f"Train Loss : {running_loss_train:.6f}")
                    print(f"Valid Loss : {running_loss_valid:.6f}")
                    print(f"Train F1   : {running_f1_train:.6f}")
                    print(f"Valid F1   : {running_f1_val:.6f}")
                    print(f"TP={tp} | FP={fp} | FN={fn} | TN={tn}")
                    print(f"Probability min  : {probs.min().item():.6f}")
                    print(f"Probability max  : {probs.max().item():.6f}")
                    print(f"Probability mean : {probs.mean().item():.6f}")
                    print("-" * 70)

                    for threshold in THRESHOLDS:
                        preds_tmp = (probs > threshold).int()
                        f1_tmp = metrics.f1_score(
                            labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0
                        )
                        print(
                            f"threshold={threshold:.2f} | "
                            f"true_pos={(labels_int == 1).sum().item()} | "
                            f"true_neg={(labels_int == 0).sum().item()} | "
                            f"pred_pos={(preds_tmp == 1).sum().item()} | "
                            f"pred_neg={(preds_tmp == 0).sum().item()} | "
                            f"F1={f1_tmp:.4f}"
                        )

                    break

            pbar.update(1)
            pbar.set_postfix({
                'train_loss': f'{running_loss_train:.4f}',
                'valid_loss': f'{running_loss_valid:.4f}',
                'train_f1': f'{running_f1_train:.4f}',
                'valid_f1': f'{running_f1_val:.4f}'
            })

    # ============================================================
    # 14. FINAL BEST MODEL
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Best epoch       : {best_epoch}")
    print(f"Best train loss  : {best_train_loss:.6f}")
    print(f"Best valid loss  : {best_valid_loss:.6f}")
    print(f"Best valid F1    : {best_f1_score:.6f} (thr={best_threshold:.2f})")

    # ============================================================
    # 15. FINAL EMBEDDINGS
    # ============================================================

    all_embeddings, cluster_labels = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)

    # ============================================================
    # 16. COSINE SIMILARITY
    # ============================================================

    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized_embeddings = all_embeddings / norms
    cos_sim = np.dot(normalized_embeddings, normalized_embeddings.T)

    # ============================================================
    # 17. PLOTS
    # ============================================================

    if plot:
        loss_path = os.path.join(
            results_path, f'embeddings_loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        f1_path = os.path.join(
            results_path, f'embeddings_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        max_f1_path = os.path.join(
            results_path, f'embeddings_max_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        f1_best_thresh_path = os.path.join(
            results_path,
            f'embeddings_f1_best_thresh_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_max_f1_plot(max_f1_scores_train, max_f1_scores_valid, max_f1_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

        # This plot shows what the checkpoint selection actually sees, unlike
        # the fixed-decision_threshold plot above: it can look completely
        # flat at zero (e.g. once probabilities drift past decision_threshold)
        # even while this line shows the model still improving underneath
        # that blind spot.
        #
        # NOTE: we use matplotlib's Figure API directly (not pyplot) so this
        # doesn't touch the global backend/state — calling matplotlib.use()
        # or pyplot inside a library function can clobber an interactive
        # notebook backend for the rest of the session.
        from matplotlib.figure import Figure

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot(f1_per_epoch_valid, label=f'validation (thr={decision_threshold})')
        ax.plot(f1_best_thresh_per_epoch_valid, label='validation (best threshold)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1-score')
        ax.set_title('Validation F1: fixed decision threshold vs. best threshold')
        ax.legend()
        fig.savefig(f1_best_thresh_path)

    # ============================================================
    # 18. SAVE BEST MODEL
    # ============================================================

    torch.save(best_model.state_dict(), model_path)
    print(f"\nBest model saved to:\n{model_path}")

    # ============================================================
    # 19. FINAL VISUALIZATION PATHS
    # ============================================================

    save_path_pca = os.path.join(
        results_path, f'embeddings_pca_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_t_SNE = os.path.join(
        results_path, f'embeddings_t-SNE_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_heatmap_ = os.path.join(
        results_path, f'embeddings_heatmap_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_matrix = os.path.join(
        results_path, f'embeddings_matrix_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )

    # ============================================================
    # 20. FINAL NODE / CLUSTER INFORMATION
    # ============================================================

    cluster_stId_dict = {}
    significant_stIds = []
    clusters_with_significant_stId = {}
    clusters_node_info = {}

    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()

        graph_path = os.path.join(data_path, omics, cancer, 'emb', 'raw', 'emb_train.pkl')
        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert len(cluster_labels) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster = {}
        first_node_embedding_in_cluster = {}
        stid_dic = {}

        for node in nx_graph.nodes:
            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]
            stid_dic[stid] = node_embeddings[idx]

            significance_value = graph.ndata['significance'][idx].item()
            if significance_value == 1:
                significant_stIds.append(stid)

        stid_df_final = pd.DataFrame.from_dict(stid_dic, orient='index')
        csv_save_path_final = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.csv'
        )
        stid_df_final.to_csv(csv_save_path_final, index_label='stId')

        for node, cluster in zip(nx_graph.nodes, cluster_labels):
            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]
            significance_value = graph.ndata['significance'][idx].item()

            if cluster not in first_node_stId_in_cluster:
                first_node_stId_in_cluster[cluster] = stid
                first_node_embedding_in_cluster[cluster] = node_embeddings[idx]

            cluster_stId_dict.setdefault(cluster, []).append(stid)

            clusters_with_significant_stId.setdefault(cluster, [])
            if significance_value == 1:
                clusters_with_significant_stId[cluster].append(stid)

            clusters_node_info.setdefault(cluster, []).append({
                'stId': stid,
                'significance': significance_value,
                'other_info': nx_graph.nodes[node]
            })

        print("\nFirst node in each cluster:")
        print(first_node_stId_in_cluster)

        sorted_clusters = sorted(first_node_stId_in_cluster.keys())
        stid_list = [first_node_stId_in_cluster[c] for c in sorted_clusters]
        embedding_list = [
            np.asarray(first_node_embedding_in_cluster[c]).reshape(-1)
            for c in sorted_clusters
        ]

        heatmap_data = pd.DataFrame(embedding_list, index=stid_list)

        print(f"Number of clusters: {len(sorted_clusters)}")
        print(f"Embedding matrix shape: {heatmap_data.shape}")

        create_heatmap_with_stid(embedding_list, stid_list, save_path_heatmap_)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list, stid_list, save_path_matrix)

        break

    # ============================================================
    # 21. FINAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(all_embeddings, cluster_labels, stid_list, save_path_t_SNE)
    visualize_embeddings_pca(all_embeddings, cluster_labels, first_node_stId_in_cluster, save_path_pca)

    # ============================================================
    # 22. FINAL CLUSTERING METRICS
    # ============================================================

    if len(np.unique(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(all_embeddings, cluster_labels)
    else:
        silhouette_avg = np.nan
        davies_bouldin = np.nan

    print(f"\nSilhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    # ============================================================
    # 23. FINAL SUMMARY
    # ============================================================

    summary = (
        f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    )
    summary += f"Best Epoch: {best_epoch}\n"
    summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {best_f1_score} (thr={best_threshold})\n"
    summary += f"Final-epoch best-threshold valid F1: {f1_best_thresh_per_epoch_valid[-1]}\n"
    summary += f"Silhouette Score: {silhouette_avg}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin}\n"
    summary += f"Number of significant genes: {len(significant_stIds)}\n"

    save_file = os.path.join(
        results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt'
    )
    with open(save_file, 'w') as f:
        f.write(summary)

    # ============================================================
    # 24. FINAL INFORMATION
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL STAGE-1 RESULTS")
    print("=" * 70)
    print(f"Best epoch            : {best_epoch}")
    print(f"Best validation loss  : {best_valid_loss:.6f}")
    print(f"Best validation F1    : {best_f1_score:.6f} (thr={best_threshold:.2f})")
    print(f"Max training F1       : {max_f1_train:.6f}")
    print(f"Max validation F1     : {max_f1_valid:.6f}")
    print(f"Significant genes     : {len(significant_stIds)}")
    print(f"Model path:\n{model_path}")
    print("=" * 70)

    return model_path

def train(
    hyperparams=None,
    data_path='../data/omics/',
    plot=True,
    omics='cna',
    cancer='BLCA'
):
    """
    Stage-1 training:
        Predict generic biological significance from the graph.

    IMPORTANT:
        This stage does NOT use cancer-driver labels.
        'significance' is the Stage-1 pretraining target.

    Expected significance values:
        0 = not significant
        1 = significant

    If -1 is present:
        -1 = unlabeled / ignored during loss and F1 calculation
    """

    import os
    import copy
    import pickle

    import numpy as np
    import pandas as pd
    import torch
    import torch.optim as optim
    from tqdm import tqdm

    from sklearn import metrics
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score
    )

    # ============================================================
    # 1. Hyperparameters
    # ============================================================

    if hyperparams is None:
        raise ValueError("hyperparams must be provided.")

    num_epochs = hyperparams['num_epochs']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']

    # The probability cutoff used to turn sigmoid outputs into 0/1
    # predictions everywhere in this function (training F1, diagnostics,
    # etc). Override via hyperparams['decision_threshold'] = 0.2, for
    # example, once you've picked an operating point from the
    # best-threshold plots rather than assuming 0.5 is correct.
    decision_threshold = hyperparams.get('decision_threshold', 0.2)

    print("\n" + "=" * 70)
    print("START STAGE-1 SIGNIFICANCE PRETRAINING")
    print("=" * 70)
    print(f"Omics       : {omics}")
    print(f"Cancer      : {cancer}")
    print(f"Device      : {device}")
    print(f"Epochs      : {num_epochs}")
    print(f"LR          : {learning_rate}")
    print(f"Batch size  : {batch_size}")
    print(f"Out feats   : {out_feats}")
    print(f"Layers      : {num_layers}")
    print(f"Loss type   : {hyperparams.get('loss_type', 'focal')}")
    print(f"Decision thr: {decision_threshold}")
    print("=" * 70 + "\n")

    # ============================================================
    # 2. Model save path
    # ============================================================

    model_path = os.path.join(data_path, omics, cancer, 'emb', 'models')
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(
        model_path,
        f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth'
    )

    # ============================================================
    # 3. Load dataset and add node-degree structural features
    # ============================================================

    data_path_ = os.path.join(data_path, omics, cancer)
    ds = dataset.Dataset(data_path_)

    # Add node in-degree (normalized) as an extra structural feature to
    # both the training (0) and validation (1) graphs.
    for idx in [0, 1]:
        g, _ = ds[idx]

        node_degrees = g.in_degrees().float().unsqueeze(-1)
        node_degrees_norm = (
            (node_degrees - node_degrees.mean())
            / (node_degrees.std() + 1e-5)
        )

        if 'weight' in g.ndata:
            existing_feats = g.ndata['weight']
        elif 'feat' in g.ndata:
            existing_feats = g.ndata['feat']
        else:
            # No node features initialized yet — fall back to a constant.
            existing_feats = torch.ones((g.num_nodes(), 1), dtype=torch.float32)

        if existing_feats.dim() == 1:
            existing_feats = existing_feats.unsqueeze(-1)

        updated_feats = torch.cat([existing_feats, node_degrees_norm], dim=-1)

        # Keep both keys in sync so downstream code reading either won't break.
        g.ndata['feat'] = updated_feats
        g.ndata['weight'] = updated_feats

    graph, name = ds[0]

    # ============================================================
    # 4. SIGNIFICANCE LABEL DEBUG
    # ============================================================

    if 'significance' not in graph.ndata:
        raise KeyError(
            "graph.ndata['significance'] does not exist. "
            "Check your preprocessing and DGL graph conversion."
        )

    significance = graph.ndata['significance']

    print("\n" + "=" * 70)
    print("SIGNIFICANCE DEBUG")
    print("=" * 70)
    print("dtype :", significance.dtype)
    print("shape :", significance.shape)

    unique_labels, label_counts = torch.unique(significance, return_counts=True)
    print("unique labels:")
    for label, count in zip(unique_labels, label_counts):
        print(f"    label={label.item():>4} count={count.item():>6}")
    print("first 20:", significance[:20])
    print("=" * 70)

    valid_labels = significance[(significance == 0) | (significance == 1)]

    if valid_labels.numel() == 0:
        raise ValueError("No valid significance labels (0/1) were found.")

    num_positive = (valid_labels == 1).sum().item()
    num_negative = (valid_labels == 0).sum().item()
    num_unlabeled = (significance == -1).sum().item()

    print("\nLABEL DISTRIBUTION")
    print(f"Positive / significant : {num_positive}")
    print(f"Negative / non-signif. : {num_negative}")
    print(f"Unlabeled (-1)         : {num_unlabeled}")

    if num_positive == 0:
        raise ValueError("There are ZERO positive significance labels. F1 will necessarily be 0.")
    if num_negative == 0:
        raise ValueError("There are ZERO negative significance labels. Check your preprocessing.")

    positive_ratio = num_positive / valid_labels.numel()
    print(f"Positive ratio          : {positive_ratio:.4%}")

    # ============================================================
    # 5. Train / validation graphs
    # ============================================================

    ds_train = [ds[0]]
    ds_valid = [ds[1]]

    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # ============================================================
    # 6. Model / optimizer / best-model tracker
    # ============================================================

    net = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = model.TAGCNModel(
        dim_latent=out_feats,
        num_layers=num_layers,
        do_train=True
    ).to(device)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    # # ============================================================
    # # 6. Model / optimizer / best-model tracker
    # # ============================================================

    # in_feats = 1

    # net = model.GCNModel(
    #     dim_latent=out_feats,
    #     num_layers=num_layers,
    #     in_feats=in_feats,
    #     do_train=True
    # ).to(device)

    # optimizer = optim.Adam(
    #     net.parameters(),
    #     lr=learning_rate
    # )

    # best_model = model.GCNModel(
    #     dim_latent=out_feats,
    #     num_layers=num_layers,
    #     in_feats=in_feats,
    #     do_train=True
    # ).to(device)

    # best_model.load_state_dict(
    #     copy.deepcopy(net.state_dict())
    # )

    # ============================================================
    # 7. Loss function
    # ============================================================

    # loss_type lets you A/B test FocalLoss against plain BCEWithLogitsLoss
    # without touching the rest of the training loop. Pass e.g.
    # hyperparams['loss_type'] = 'bce' to use the baseline.
    loss_type = hyperparams.get('loss_type', 'focal')

    total_valid = num_positive + num_negative

    if loss_type == 'bce':
        # Standard BCE class-balancing convention: pos_weight scales the
        # loss contribution of positive examples. Since positives are the
        # minority class here, pos_weight > 1 upweights them.
        #
        # NOTE: pos_weight = neg/pos makes the *weighted* cost of a false
        # negative and a false positive roughly equal at the population
        # level. That means a constant output of p* ~ 0.5 for every node
        # (ignoring the graph entirely) sits very close to loss-optimal —
        # a flat, easy-to-find attractor that gradient descent can settle
        # into instead of learning a genuinely discriminative solution.
        # Nudging pos_weight away from that exact tie point (via
        # pos_weight_multiplier) breaks the tie so the degenerate constant
        # solution is no longer near-optimal.
        pos_weight_multiplier = hyperparams.get('pos_weight_multiplier', 1.3)
        pos_weight_value = (num_negative / max(num_positive, 1)) * pos_weight_multiplier
        pos_weight = torch.tensor(pos_weight_value, device=device)

        print(
            f"Configuring BCEWithLogitsLoss (baseline) with "
            f"pos_weight={pos_weight_value:.4f} "
            f"(base={num_negative / max(num_positive, 1):.4f} x multiplier={pos_weight_multiplier:.2f})"
        )

        bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
        criterion = bce_criterion

    else:
        # alpha favors the rare positive class: weight ~ (negatives / total)
        dynamic_alpha = num_negative / total_valid

        print(f"Configuring Focal Loss with Dynamic Alpha: {dynamic_alpha:.4f} (favoring positive class)")

        criterion = FocalLoss(alpha=dynamic_alpha, gamma=2.0, reduction='mean')

        # NOTE: do NOT use a hardcoded weight tensor like [0.00001, 0.99999] —
        # that nearly fully suppresses class 0. And since FocalLoss(reduction='mean')
        # already returns a scalar, multiplying it by node-level weights afterwards
        # would be incorrect, so alpha is passed straight into FocalLoss instead.

    # ============================================================
    # 8. Training history
    # ============================================================

    loss_per_epoch_train = []
    loss_per_epoch_valid = []

    f1_per_epoch_train = []
    f1_per_epoch_valid = []
    f1_best_thresh_per_epoch_valid = []

    max_f1_scores_train = []
    max_f1_scores_valid = []

    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_f1_score = 0.0
    best_epoch = 0
    best_threshold = 0.5

    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    # ============================================================
    # 9. Results path
    # ============================================================

    results_path = os.path.abspath(
        os.path.join('results', 'multiomics_meth', 'node_embeddings', omics, cancer)
    )
    os.makedirs(results_path, exist_ok=True)

    # ============================================================
    # 10. INITIAL EMBEDDINGS
    # ============================================================

    print("\n" + "=" * 70)
    print("CALCULATING INITIAL EMBEDDINGS")
    print("=" * 70)

    all_embeddings_initial, cluster_labels_initial = calculate_cluster_labels(
        best_model, dl_train, device
    )
    all_embeddings_initial = all_embeddings_initial.reshape(all_embeddings_initial.shape[0], -1)

    save_path_heatmap_initial = os.path.join(
        results_path, f'embeddings_heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_matrix_initial = os.path.join(
        results_path, f'embeddings_matrix_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_pca_initial = os.path.join(
        results_path, f'embeddings_pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )
    save_path_t_SNE_initial = os.path.join(
        results_path, f'embeddings_t-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png'
    )

    # ============================================================
    # 11. INITIAL EMBEDDING INFORMATION
    # ============================================================

    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings_initial = (
            best_model.get_node_embeddings(graph).detach().cpu().numpy()
        )

        graph_path = os.path.join(data_path, omics, cancer, 'emb', 'raw', 'emb_train.pkl')
        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert len(cluster_labels_initial) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index_initial = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster_initial = {}
        first_node_embedding_in_cluster_initial = {}
        stid_dic_initial = {}

        for node in nx_graph.nodes:
            if 'stId' in nx_graph.nodes[node]:
                stId = nx_graph.nodes[node]['stId']
                stid_dic_initial[stId] = node_embeddings_initial[node_to_index_initial[node]]

        stid_df_initial = pd.DataFrame.from_dict(stid_dic_initial, orient='index')

        csv_save_path_initial = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.csv'
        )
        stid_df_initial.to_csv(csv_save_path_initial, index_label='stId')

        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if 'stId' in nx_graph.nodes[node]:
                if cluster not in first_node_stId_in_cluster_initial:
                    first_node_stId_in_cluster_initial[cluster] = nx_graph.nodes[node]['stId']
                    first_node_embedding_in_cluster_initial[cluster] = (
                        node_embeddings_initial[node_to_index_initial[node]]
                    )

        print("first_node_stId_in_cluster_initial:")
        print(first_node_stId_in_cluster_initial)

        sorted_clusters_initial = sorted(first_node_stId_in_cluster_initial.keys())
        stid_list_initial = [first_node_stId_in_cluster_initial[c] for c in sorted_clusters_initial]
        embedding_list_initial = [
            np.asarray(first_node_embedding_in_cluster_initial[c]).reshape(-1)
            for c in sorted_clusters_initial
        ]

        heatmap_data_initial = pd.DataFrame(embedding_list_initial, index=stid_list_initial)

        print(f"Number of clusters: {len(sorted_clusters_initial)}")
        print(f"Embedding matrix shape: {heatmap_data_initial.shape}")

        create_heatmap_with_stid(embedding_list_initial, stid_list_initial, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(
            embedding_list_initial, stid_list_initial, save_path_matrix_initial
        )

        break

    # ============================================================
    # 12. INITIAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(
        all_embeddings_initial, cluster_labels_initial, stid_list_initial, save_path_t_SNE_initial
    )
    visualize_embeddings_pca(
        all_embeddings_initial, cluster_labels_initial,
        first_node_stId_in_cluster_initial, save_path_pca_initial
    )

    if len(np.unique(cluster_labels_initial)) > 1:
        silhouette_avg_ = silhouette_score(all_embeddings_initial, cluster_labels_initial)
        davies_bouldin_ = davies_bouldin_score(all_embeddings_initial, cluster_labels_initial)
    else:
        silhouette_avg_ = np.nan
        davies_bouldin_ = np.nan

    summary_ = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_ = os.path.join(
        results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt'
    )
    with open(save_file_, 'w') as f:
        f.write(summary_)

    # ============================================================
    # 13. TRAINING
    # ============================================================

    print("\n" + "=" * 70)
    print("START TRAINING")
    print("=" * 70)

    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=True) as pbar:

        for epoch in range(num_epochs):

            # ---------------- TRAIN ----------------
            net.train()
            loss_per_graph = []
            f1_per_graph = []

            for data in dl_train:
                graph, name = data
                graph = graph.to(device)

                logits = net(graph).view(-1)
                labels = graph.ndata['significance'].float().view(-1)

                valid_mask = (labels == 0) | (labels == 1)
                if valid_mask.sum().item() == 0:
                    raise ValueError("Training graph contains no valid significance labels.")

                valid_logits = logits[valid_mask]
                valid_labels = labels[valid_mask]

                loss = criterion(valid_logits, valid_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_per_graph.append(loss.item())

                with torch.no_grad():
                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > decision_threshold).int()
                    labels_int = valid_labels.int()
                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0
                    )
                    f1_per_graph.append(f1)

            running_loss_train = float(np.mean(loss_per_graph))
            running_f1_train = float(np.mean(f1_per_graph))
            loss_per_epoch_train.append(running_loss_train)
            f1_per_epoch_train.append(running_f1_train)

            # ---------------- VALIDATION ----------------
            net.eval()
            loss_per_graph = []
            f1_per_graph = []

            # Keep the last batch's probs/labels around for the threshold
            # diagnostic and best-model selection below (and now also for
            # the epoch diagnostic block, instead of it recomputing this
            # from scratch -- see fix note there).
            val_probs, val_labels_int = None, None

            with torch.no_grad():
                for data in dl_valid:
                    graph, name = data
                    graph = graph.to(device)

                    logits = net(graph).view(-1)
                    labels = graph.ndata['significance'].float().view(-1)

                    valid_mask = (labels == 0) | (labels == 1)
                    if valid_mask.sum().item() == 0:
                        continue

                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask]

                    loss = criterion(valid_logits, valid_labels)
                    loss_per_graph.append(loss.item())

                    probs = torch.sigmoid(valid_logits)
                    preds = (probs > decision_threshold).int()
                    labels_int = valid_labels.int()

                    f1 = metrics.f1_score(
                        labels_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0
                    )
                    f1_per_graph.append(f1)

                    val_probs, val_labels_int = probs, labels_int

            if len(loss_per_graph) == 0:
                raise ValueError("Validation graph contains no valid significance labels.")

            running_loss_valid = float(np.mean(loss_per_graph))
            running_f1_val = float(np.mean(f1_per_graph))
            loss_per_epoch_valid.append(running_loss_valid)
            f1_per_epoch_valid.append(running_f1_val)

            max_f1_train = max(f1_per_epoch_train)
            max_f1_valid = max(f1_per_epoch_valid)
            max_f1_scores_train.append(max_f1_train)
            max_f1_scores_valid.append(max_f1_valid)

            # ---------------- BEST-THRESHOLD F1 (every epoch) ----------------
            # decision_threshold as the fixed operating point can be a bad
            # lens once probabilities drift off center (e.g. the whole
            # distribution sliding below it): the ranking underneath can
            # still be fine or improving while fixed-threshold F1 reads a
            # flat zero. Sweeping thresholds every epoch (not just when loss
            # improves) avoids that blind spot and is what actually gates
            # the checkpoint below.
            best_f1_this_epoch = 0.0
            best_threshold_this_epoch = 0.2
            if val_probs is not None:
                for threshold in THRESHOLDS:
                    preds_tmp = (val_probs > threshold).int()
                    f1_tmp = metrics.f1_score(
                        val_labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0
                    )
                    if f1_tmp > best_f1_this_epoch:
                        best_f1_this_epoch = f1_tmp
                        best_threshold_this_epoch = threshold

            f1_best_thresh_per_epoch_valid.append(best_f1_this_epoch)

            # ---------------- BEST MODEL SELECTION ----------------
            # Primary criterion: best-across-threshold validation F1 must
            # improve. This is what actually reflects the model's ability
            # to separate classes, unlike raw loss (which a degenerate
            # constant-output solution can minimize) or fixed-threshold F1 (which
            # can read 0 even when the underlying ranking is fine). Loss is
            # used only as a tie-breaker when F1 is unchanged, so we still
            # prefer the more confident/better-calibrated model among ties.
            f1_improved = best_f1_this_epoch > best_f1_score
            f1_tied_but_loss_improved = (
                best_f1_this_epoch == best_f1_score and running_loss_valid < best_valid_loss
            )

            if f1_improved or f1_tied_but_loss_improved:

                best_train_loss = running_loss_train
                best_valid_loss = running_loss_valid
                best_f1_score = best_f1_this_epoch
                best_epoch = epoch + 1
                best_threshold = best_threshold_this_epoch

                best_model.load_state_dict(copy.deepcopy(net.state_dict()))

                print("\n*** BEST MODEL UPDATED ***")
                print(f"Epoch                : {best_epoch}")
                print(f"Train Loss           : {best_train_loss:.6f}")
                print(f"Valid Loss           : {best_valid_loss:.6f}")
                print(f"Valid F1 (thr={decision_threshold})   : {running_f1_val:.6f}")
                print(f"Best F1 across thresh: {best_f1_score:.6f} (thr={best_threshold:.2f})")

            # ---------------- DETAILED DIAGNOSTIC ----------------
            # Every epoch for the first 15 (early training is where sudden
            # collapses/drifts like the epoch-7 case tend to happen and are
            # otherwise invisible between the epoch-1 and epoch-10 prints),
            # then every 10th epoch after that.
            #
            # FIXED: this used to re-run a second forward pass over dl_valid
            # here, outside torch.no_grad(), to get probs/labels_int for the
            # printout below -- redundant (dl_valid has exactly one graph,
            # and the validation loop directly above already computed the
            # same probs/labels_int into val_probs/val_labels_int, inside
            # torch.no_grad()) and needlessly built an autograd graph on
            # logits/probs/preds every time this block ran. Reusing
            # val_probs/val_labels_int removes both problems and prints the
            # exact same numbers.
            if epoch < 15 or (epoch + 1) % 10 == 0:

                probs = val_probs
                labels_int = val_labels_int
                preds = (probs > decision_threshold).int()

                tp = ((labels_int == 1) & (preds == 1)).sum().item()
                fp = ((labels_int == 0) & (preds == 1)).sum().item()
                fn = ((labels_int == 1) & (preds == 0)).sum().item()
                tn = ((labels_int == 0) & (preds == 0)).sum().item()

                print("\n" + "-" * 70)
                print(f"EPOCH {epoch + 1} DIAGNOSTIC")
                print("-" * 70)
                print(f"Train Loss : {running_loss_train:.6f}")
                print(f"Valid Loss : {running_loss_valid:.6f}")
                print(f"Train F1   : {running_f1_train:.6f}")
                print(f"Valid F1   : {running_f1_val:.6f}")
                print(f"TP={tp} | FP={fp} | FN={fn} | TN={tn}")
                print(f"Probability min  : {probs.min().item():.6f}")
                print(f"Probability max  : {probs.max().item():.6f}")
                print(f"Probability mean : {probs.mean().item():.6f}")
                print("-" * 70)

                for threshold in THRESHOLDS:
                    preds_tmp = (probs > threshold).int()
                    f1_tmp = metrics.f1_score(
                        labels_int.cpu().numpy(), preds_tmp.cpu().numpy(), zero_division=0
                    )
                    print(
                        f"threshold={threshold:.2f} | "
                        f"true_pos={(labels_int == 1).sum().item()} | "
                        f"true_neg={(labels_int == 0).sum().item()} | "
                        f"pred_pos={(preds_tmp == 1).sum().item()} | "
                        f"pred_neg={(preds_tmp == 0).sum().item()} | "
                        f"F1={f1_tmp:.4f}"
                    )

            pbar.update(1)
            pbar.set_postfix({
                'train_loss': f'{running_loss_train:.4f}',
                'valid_loss': f'{running_loss_valid:.4f}',
                'train_f1': f'{running_f1_train:.4f}',
                'valid_f1': f'{running_f1_val:.4f}'
            })

    # ============================================================
    # 14. FINAL BEST MODEL
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Best epoch       : {best_epoch}")
    print(f"Best train loss  : {best_train_loss:.6f}")
    print(f"Best valid loss  : {best_valid_loss:.6f}")
    print(f"Best valid F1    : {best_f1_score:.6f} (thr={best_threshold:.2f})")

    # ============================================================
    # 15. FINAL EMBEDDINGS
    # ============================================================

    all_embeddings, cluster_labels = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)

    # ============================================================
    # 16. COSINE SIMILARITY
    # ============================================================

    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized_embeddings = all_embeddings / norms
    cos_sim = np.dot(normalized_embeddings, normalized_embeddings.T)

    # ============================================================
    # 17. PLOTS
    # ============================================================

    if plot:
        loss_path = os.path.join(
            results_path, f'embeddings_loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        f1_path = os.path.join(
            results_path, f'embeddings_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        max_f1_path = os.path.join(
            results_path, f'embeddings_max_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )
        f1_best_thresh_path = os.path.join(
            results_path,
            f'embeddings_f1_best_thresh_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png'
        )

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_max_f1_plot(max_f1_scores_train, max_f1_scores_valid, max_f1_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

        # This plot shows what the checkpoint selection actually sees, unlike
        # the fixed-decision_threshold plot above: it can look completely
        # flat at zero (e.g. once probabilities drift past decision_threshold)
        # even while this line shows the model still improving underneath
        # that blind spot.
        #
        # NOTE: we use matplotlib's Figure API directly (not pyplot) so this
        # doesn't touch the global backend/state — calling matplotlib.use()
        # or pyplot inside a library function can clobber an interactive
        # notebook backend for the rest of the session.
        from matplotlib.figure import Figure

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot(f1_per_epoch_valid, label=f'validation (thr={decision_threshold})')
        ax.plot(f1_best_thresh_per_epoch_valid, label='validation (best threshold)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1-score')
        ax.set_title('Validation F1: fixed decision threshold vs. best threshold')
        ax.legend()
        fig.savefig(f1_best_thresh_path)

    # ============================================================
    # 18. SAVE BEST MODEL
    # ============================================================

    torch.save(best_model.state_dict(), model_path)
    print(f"\nBest model saved to:\n{model_path}")

    # ============================================================
    # 19. FINAL VISUALIZATION PATHS
    # ============================================================

    save_path_pca = os.path.join(
        results_path, f'embeddings_pca_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_t_SNE = os.path.join(
        results_path, f'embeddings_t-SNE_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_heatmap_ = os.path.join(
        results_path, f'embeddings_heatmap_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )
    save_path_matrix = os.path.join(
        results_path, f'embeddings_matrix_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )

    # ============================================================
    # 20. FINAL NODE / CLUSTER INFORMATION
    # ============================================================

    cluster_stId_dict = {}
    significant_stIds = []
    clusters_with_significant_stId = {}
    clusters_node_info = {}

    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()

        graph_path = os.path.join(data_path, omics, cancer, 'emb', 'raw', 'emb_train.pkl')
        with open(graph_path, 'rb') as f:
            nx_graph = pickle.load(f)

        assert len(cluster_labels) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster = {}
        first_node_embedding_in_cluster = {}
        stid_dic = {}

        for node in nx_graph.nodes:
            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]
            stid_dic[stid] = node_embeddings[idx]

            significance_value = graph.ndata['significance'][idx].item()
            if significance_value == 1:
                significant_stIds.append(stid)

        stid_df_final = pd.DataFrame.from_dict(stid_dic, orient='index')
        csv_save_path_final = os.path.join(
            results_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.csv'
        )
        stid_df_final.to_csv(csv_save_path_final, index_label='stId')

        for node, cluster in zip(nx_graph.nodes, cluster_labels):
            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]
            significance_value = graph.ndata['significance'][idx].item()

            if cluster not in first_node_stId_in_cluster:
                first_node_stId_in_cluster[cluster] = stid
                first_node_embedding_in_cluster[cluster] = node_embeddings[idx]

            cluster_stId_dict.setdefault(cluster, []).append(stid)

            clusters_with_significant_stId.setdefault(cluster, [])
            if significance_value == 1:
                clusters_with_significant_stId[cluster].append(stid)

            clusters_node_info.setdefault(cluster, []).append({
                'stId': stid,
                'significance': significance_value,
                'other_info': nx_graph.nodes[node]
            })

        print("\nFirst node in each cluster:")
        print(first_node_stId_in_cluster)

        sorted_clusters = sorted(first_node_stId_in_cluster.keys())
        stid_list = [first_node_stId_in_cluster[c] for c in sorted_clusters]
        embedding_list = [
            np.asarray(first_node_embedding_in_cluster[c]).reshape(-1)
            for c in sorted_clusters
        ]

        heatmap_data = pd.DataFrame(embedding_list, index=stid_list)

        print(f"Number of clusters: {len(sorted_clusters)}")
        print(f"Embedding matrix shape: {heatmap_data.shape}")

        create_heatmap_with_stid(embedding_list, stid_list, save_path_heatmap_)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list, stid_list, save_path_matrix)

        break

    # ============================================================
    # 21. FINAL EMBEDDING VISUALIZATION
    # ============================================================

    visualize_embeddings_tsne(all_embeddings, cluster_labels, stid_list, save_path_t_SNE)
    visualize_embeddings_pca(all_embeddings, cluster_labels, first_node_stId_in_cluster, save_path_pca)

    # ============================================================
    # 22. FINAL CLUSTERING METRICS
    # ============================================================

    if len(np.unique(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(all_embeddings, cluster_labels)
    else:
        silhouette_avg = np.nan
        davies_bouldin = np.nan

    print(f"\nSilhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    # ============================================================
    # 23. FINAL SUMMARY
    # ============================================================

    summary = (
        f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    )
    summary += f"Best Epoch: {best_epoch}\n"
    summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {best_f1_score} (thr={best_threshold})\n"
    summary += f"Final-epoch best-threshold valid F1: {f1_best_thresh_per_epoch_valid[-1]}\n"
    summary += f"Silhouette Score: {silhouette_avg}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin}\n"
    summary += f"Number of significant genes: {len(significant_stIds)}\n"

    save_file = os.path.join(
        results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt'
    )
    with open(save_file, 'w') as f:
        f.write(summary)

    # ============================================================
    # 24. FINAL INFORMATION
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL STAGE-1 RESULTS")
    print("=" * 70)
    print(f"Best epoch            : {best_epoch}")
    print(f"Best validation loss  : {best_valid_loss:.6f}")
    print(f"Best validation F1    : {best_f1_score:.6f} (thr={best_threshold:.2f})")
    print(f"Max training F1       : {max_f1_train:.6f}")
    print(f"Max validation F1     : {max_f1_valid:.6f}")
    print(f"Significant genes     : {len(significant_stIds)}")
    print(f"Model path:\n{model_path}")
    print("=" * 70)

    return model_path


def plot_cosine_similarity_matrix_for_clusters_with_values(embeddings, stids, save_path):
    cos_sim = np.dot(embeddings, np.array(embeddings).T)
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    plt.figure(figsize=(10, 8))
    
    vmin = cos_sim.min()
    vmax = cos_sim.max()
    # Create the heatmap with a custom color bar
    ##sns.heatmap(data, cmap='cividis')
    ##sns.heatmap(data, cmap='Blues') 'Greens' sns.heatmap(data, cmap='Spectral') 'coolwarm') 'YlGnBu') viridis cubehelix inferno

    ax = sns.heatmap(cos_sim, cmap="Spectral", annot=True, fmt=".3f", annot_kws={"size": 6},
                     xticklabels=stids, yticklabels=stids,
                     cbar_kws={"shrink": 0.2, "aspect": 15, "ticks": [vmin, vmax]})

    # Highlight the diagonal squares with value 1 by setting their background color to black
    for i in range(len(stids)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='black', alpha=0.5, zorder=3))
        
    ax.xaxis.tick_top()  # Move x-axis labels to the top
    ax.xaxis.set_label_position('top')  # Set x-axis label position to top
    plt.xticks(rotation=-30, fontsize=8, ha='right')  # Rotate x-axis labels, set font size, and align to the right
    plt.yticks(fontsize=8)  # Set font size for y-axis labels

    # Set the title below the plot
    ax.text(x=0.5, y=-0.03, s="Gene-gene similarities", fontsize=12, ha='center', va='top', transform=ax.transAxes)

    plt.savefig(save_path)
    ##plt.show()
    plt.close()
    
def create_gene_map(reactome_file, output_file):
    """
    Extracts gene IDs with the same gene STID and saves them to a new CSV file.

    Parameters:
    reactome_file (str): Path to the NCBI2Reactome.csv file.
    output_file (str): Path to save the output CSV file.
    """
    gene_map = {}  # Dictionary to store gene IDs for each gene STID

    # Read the NCBI2Reactome.csv file and populate the gene_map
    with open(reactome_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            gene_id = row[0]
            gene_stid = row[1]
            gene_map.setdefault(gene_stid, []).append(gene_id)

    # Write the gene_map to the output CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Protein STID", "Gene IDs"])  # Write header
        for gene_stid, gene_ids in gene_map.items():
            writer.writerow([gene_stid, ",".join(gene_ids)])
    
    return gene_map
        
def save_to_neo4j(graph, stid_dic, stid_mapping, gene_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, uri, user, password):
    from neo4j import GraphDatabase

    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))
    session = driver.session()

    # Clean the database
    session.run("MATCH (n) DETACH DELETE n")

    try:
        # Create nodes with embeddings and additional attributes
        for node_id in stid_dic:
            embedding = stid_dic[node_id].tolist()  
            stId = stid_mapping[node_id]  # Access stId based on node_id
            name = graph.graph_nx.nodes[node_id]['name']
            weight = graph.graph_nx.nodes[node_id]['weight']
            significance = graph.graph_nx.nodes[node_id]['significance']
            session.run(
                "CREATE (n:Protein {embedding: $embedding, stId: $stId, name: $name, weight: $weight, significance: $significance})",
                embedding=embedding, stId=stId, name=name, weight=weight, significance=significance
            )

            # Create gene nodes and relationships
            ##genes = get_genes_by_gene_stid(node_id, reactome_file, gene_names_file)
            genes = gene_map.get(node_id, [])


            ##print('stid_to_gene_info=========================-----------------------------\n', genes)
    
            # Create gene nodes and relationships
            for gene_id in genes:
                gene_name = gene_id_to_name_mapping.get(gene_id, None)
                gene_symbol = gene_id_to_symbol_mapping.get(gene_id, None)
                if gene_name:  # Only create node if gene name exists
                    session.run(
                        "MERGE (g:Gene {id: $gene_id, name: $gene_name, symbol: $gene_symbol})",
                        gene_id=gene_id, gene_name=gene_name, gene_symbol = gene_symbol
                    )
                    session.run(
                        "MATCH (p:Protein {stId: $stId}), (g:Gene {id: $gene_id}) "
                        "MERGE (p)-[:INVOLVES]->(g)",
                        stId=stId, gene_id=gene_id
                    )
                
                session.run(
                    "MATCH (p:Protein {stId: $stId}), (g:Gene {id: $gene_id}) "
                    "MERGE (p)-[:INVOLVES]->(g)",
                    stId=stId, gene_id=gene_id
                )
                
        # Create relationships using the stId mapping
        for source, target in graph.graph_nx.edges():
            source_stId = stid_mapping[source]
            target_stId = stid_mapping[target]
            session.run(
                "MATCH (a {stId: $source_stId}), (b {stId: $target_stId}) "
                "CREATE (a)-[:CONNECTED]->(b)",
                source_stId=source_stId, target_stId=target_stId
            )

    finally:
        session.close()
        driver.close()

def read_gene_names(file_path):
    """
    Reads the gene names from a CSV file and returns a dictionary mapping gene IDs to gene names.

    Parameters:
    file_path (str): Path to the gene names CSV file.

    Returns:
    dict: A dictionary mapping gene IDs to gene names.
    """
    gene_id_to_name_mapping = {}
    gene_id_to_symbol_mapping = {}

    # Read the gene names CSV file and populate the dictionary
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            gene_id = row['NCBI_Gene_ID']
            gene_name = row['Name']
            gene_symbol = row['Approved symbol']
            gene_id_to_name_mapping[gene_id] = gene_name
            gene_id_to_symbol_mapping[gene_id] = gene_symbol

    return gene_id_to_name_mapping, gene_id_to_symbol_mapping

def create_heatmap_with_stid_ori(embedding_list, stid_list, save_path):
    # Convert the embedding list to a DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list)
    
    # Create a clustermap
    ax = sns.clustermap(heatmap_data, cmap='tab20', standard_scale=1, figsize=(10, 10))
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels
    
    # Save the clustermap to a file
    plt.savefig(save_path)

    plt.close()

def create_heatmap_with_stid_dark_green(embedding_list, stid_list, save_path):
    # Convert the embedding list to a DataFrame and transpose it to switch axes
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list).transpose()
    
    # Create a clustermap
    ax = sns.clustermap(heatmap_data, cmap='viridis', standard_scale=1, figsize=(10, 10))
    
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels
    
    # Save the clustermap to a file
    plt.savefig(save_path)

    # Close the plot to free memory
    plt.close()

def create_heatmap_with_stid_grey(embedding_list, stid_list, save_path):
    """
    Creates a heatmap with hierarchical clustering using a grey-dark white colormap.

    Parameters:
    - embedding_list: List of embeddings.
    - stid_list: List of sample or feature IDs corresponding to embeddings.
    - save_path: Path to save the heatmap.
    """
    # Convert the embedding list to a DataFrame and transpose it
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list).transpose()
    
    # Create a clustermap with a grey-dark white colormap
    ax = sns.clustermap(
        heatmap_data, 
        cmap=cm.get_cmap('Greys'),  # Use Greys colormap
        standard_scale=1, 
        figsize=(10, 10)
    )
    
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels
    
    # Save the clustermap to a file
    plt.savefig(save_path)

    # Close the plot to free memory
    plt.close()

def create_heatmap_with_10_discrete_colors(embedding_list, stid_list, save_path):
    """
    Creates a heatmap with hierarchical clustering using 10 discrete colors.

    Parameters:
    - embedding_list: List of embeddings.
    - stid_list: List of sample or feature IDs corresponding to embeddings.
    - save_path: Path to save the heatmap.
    """
    # Convert the embedding list to a DataFrame and transpose it
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list).transpose()

    # Define 10 discrete colors
    discrete_colors = [
        '#f7fcf0', '#e0f3db', '#ccebc5', '#a8ddb5', '#7bccc4',
        '#4eb3d3', '#2b8cbe', '#0868ac', '#084081', '#081d58'
    ]  # Gradient from light green to dark blue

    # Define value ranges (bins) for the colors
    color_bounds = np.linspace(heatmap_data.min().min(), heatmap_data.max().max(), len(discrete_colors) + 1)

    # Create a discrete colormap and norm
    cmap = ListedColormap(discrete_colors)
    norm = BoundaryNorm(color_bounds, cmap.N)

    # Create a clustermap
    ax = sns.clustermap(
        heatmap_data,
        cmap=cmap, 
        norm=norm, 
        figsize=(10, 10),
        linewidths=0.5  # Add gridlines for clarity
    )
    
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels

    # Save the clustermap to a file
    plt.savefig(save_path)

    # Close the plot to free memory
    plt.close()

def create_heatmap_with_stid_light_blue(embedding_list, stid_list, save_path):
    """
    Creates a heatmap with hierarchical clustering using 10 discrete colors.

    Parameters:
    - embedding_list: List of embeddings.
    - stid_list: List of sample or feature IDs corresponding to embeddings.
    - save_path: Path to save the heatmap.
    """
    # Convert the embedding list to a DataFrame and transpose it
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list).transpose()
    ##heatmap_data = pd.DataFrame(embedding_list, index=stid_list)

    # Define 10 discrete colors
    discrete_colors = [
        '#f7fcf0', '#e0f3db', '#ccebc5', '#a8ddb5', '#7bccc4',
        '#4eb3d3', '#2b8cbe', '#0868ac', '#084081', '#081d58'
    ]  # Gradient from light green to dark blue

    # Define value ranges (bins) for the colors
    color_bounds = np.linspace(heatmap_data.min().min(), heatmap_data.max().max(), len(discrete_colors) + 1)

    # Create a discrete colormap and norm
    cmap = ListedColormap(discrete_colors)
    norm = BoundaryNorm(color_bounds, cmap.N)

    # Create a clustermap
    ax = sns.clustermap(
        heatmap_data,
        cmap=cmap, 
        norm=norm, 
        figsize=(10, 10),
        linewidths=0.5  # Add gridlines for clarity
    )
    
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels

    # Save the clustermap to a file
    plt.savefig(save_path)

    # Close the plot to free memory
    plt.close()

def create_heatmap_with_stid(embedding_list, stid_list, save_path):
    """
    Creates a heatmap with hierarchical clustering using a better colormap.
    Ensures no spaces between cells and limits x-axis labels to 30.

    Parameters:
    - embedding_list: List of embeddings.
    - stid_list: List of sample or feature IDs corresponding to embeddings.
    - save_path: Path to save the heatmap.
    """
    # Convert embeddings into a DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list)

    # Create the clustermap with improved settings
    ax = sns.clustermap(
        heatmap_data, 
        cmap="coolwarm",  # Use 'coolwarm' or 'viridis' for clarity
        standard_scale=1, 
        figsize=(8, 7.5),  # Larger size for better readability
        linewidths=0,  # Remove spaces between cells
        dendrogram_ratio=(0.1, 0.1),  # Reduce dendrogram size
        ##xticklabels=30,  # Show only 30 x-axis ticks
        cbar_pos=(1.00, 0.4, 0.01, 0.2),  # Adjusted for better spacing
    )

    # Adjust color bar font size
    cbar = ax.ax_cbar  # Get the color bar axis
    ##cbar.set_ylabel('Intensity', fontsize=6)  # Set color bar label font size
    cbar.tick_params(labelsize=6)  # Set tick label font size
    # Adjust x-axis labels
    ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
    ax.ax_heatmap.set_yticklabels(ax.ax_heatmap.get_yticklabels(), fontsize=8)

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def calculate_cluster_labels(net, dataloader, device, num_clusters=20):
    all_embeddings = []
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, _ = data
            embeddings = net.get_node_embeddings(graph.to(device))
            all_embeddings.append(embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Use KMeans clustering to assign cluster labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    return all_embeddings, cluster_labels

def visualize_embeddings_pca_(embeddings, cluster_labels, stid_list, save_path):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))

    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{stid_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and stid labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=stid_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def visualize_embeddings_pca_ori(embeddings, cluster_labels, stid_list, save_path):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))

    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{stid_list[cluster]}', s=20, color=palette[i])

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and stid labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=stid_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def visualize_embeddings_tsne_ori(embeddings, cluster_labels, stid_list, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))
    
    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{stid_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    plt.title('T-SNE of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and stid labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=stid_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_embeddings_tsne_(embeddings, cluster_labels, stid_list, save_path):
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Initialize the figure
    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style for the plot
    sns.set(style="whitegrid")  # White background with grid lines

    # Identify unique clusters and sort them for consistent ordering
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)

    # Define a color palette for the clusters
    palette = sns.color_palette("viridis", len(sorted_clusters))
    
    # Create a scatter plot for each cluster
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f'{stid_list[cluster]}',  # Label based on stid_list
            s=20,  # Size of the scatter points
            color=palette[i],  # Color based on the cluster
            edgecolor='k'  # Black edge around each point
        )

    # Add axis labels and a title
    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    plt.title('T-SNE of Embeddings')

    # Customize the plot grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')  # Light gray background for the plot area
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)

    # Maintain square aspect ratio for better visualization
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend to label clusters
    handles = [
        plt.Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=palette[i], markersize=8,
            label=stid_list[cluster]
        ) for i, cluster in enumerate(sorted_clusters)
    ]
    plt.legend(
        handles=handles,
        title='Label',
        bbox_to_anchor=(1.02, 0.5),
        loc='center left',
        borderaxespad=0.,
        fontsize='small',
        handlelength=0.5,
        handletextpad=0.5
    )

    # Save the plot to the specified path
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def visualize_embeddings_tsne(
    embeddings,
    cluster_labels,
    stid_list,
    save_path
):
    """
    Visualize embeddings using t-SNE.

    `stid_list` may contain representative stIds, but cluster IDs
    are not assumed to be contiguous list indices.
    """

    # Convert embeddings to 2D using t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, max(5, len(embeddings) // 3))
    )

    embeddings_2d = tsne.fit_transform(embeddings)

    cluster_labels = np.asarray(cluster_labels)

    sorted_clusters = sorted(np.unique(cluster_labels))

    # ---------------------------------------------------------
    # Build a safe cluster -> stId mapping
    # ---------------------------------------------------------
    stid_map = {}

    for cluster, stid in zip(sorted_clusters, stid_list):
        stid_map[cluster] = stid

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))

    palette = [
        CLUSTER_COLORS.get(
            int(cluster),
            '#808080'
        )
        for cluster in sorted_clusters
    ]

    for i, cluster in enumerate(sorted_clusters):

        cluster_points = embeddings_2d[
            cluster_labels == cluster
        ]

        # Use representative stId if available
        label = stid_map.get(
            cluster,
            f'Cluster {cluster}'
        )

        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=str(label),
            s=20,
            color=palette[i],
            edgecolor='k'
        )

    plt.xlabel(
        'dim_1',
        fontsize=14
    )

    plt.ylabel(
        'dim_2',
        fontsize=14
    )

    plt.title(
        't-SNE Visualization of Embeddings',
        fontsize=16,
        fontweight='bold'
    )

    plt.legend(
        fontsize=8,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

def visualize_embeddings_pca(
    embeddings,
    cluster_labels,
    cluster_stid_map,
    save_path
):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))

    sns.set(style="whitegrid")

    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)

    palette = sns.color_palette("viridis", len(sorted_clusters))

    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]

        stid = cluster_stid_map.get(
            cluster,
            f'Cluster {cluster}'
        )

        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=stid,
            s=20,
            color=palette[i],
            edgecolor='k'
        )

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Embeddings')

    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(
        True,
        which='both',
        color='white',
        linestyle='-',
        linewidth=1.0,
        alpha=0.9
    )

    ax.set_aspect('equal', adjustable='box')

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor=palette[i],
            markersize=8,
            label=cluster_stid_map.get(
                cluster,
                f'Cluster {cluster}'
            )
        )
        for i, cluster in enumerate(sorted_clusters)
    ]

    plt.legend(
        handles=handles,
        title='Label',
        bbox_to_anchor=(1.02, 0.5),
        loc='center left',
        borderaxespad=0.,
        fontsize='small',
        handlelength=0.5,
        handletextpad=0.5
    )

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def export_to_cytoscape(node_embeddings, cluster_labels, stid_list, output_path):
    # Create a DataFrame for Cytoscape export
    data = {
        'Node': stid_list,
        'Cluster': cluster_labels,
        'Embedding': list(node_embeddings)
    }
    df = pd.DataFrame(data)
    
    # Expand the embedding column into separate columns
    embeddings_df = pd.DataFrame(node_embeddings, columns=[f'Embed_{i}' for i in range(node_embeddings.shape[1])])
    df = df.drop('Embedding', axis=1).join(embeddings_df)

    # Save to CSV for Cytoscape import
    df.to_csv(output_path, index=False)
    print(f"Data exported to {output_path} for Cytoscape visualization.")

def draw_loss_plot(train_loss, valid_loss, save_path):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility
    
    plt.savefig(f'{save_path}')
    plt.close()

def draw_max_f1_plot(max_train_f1, max_valid_f1, save_path):
    plt.figure()
    plt.plot(max_train_f1, label='train')
    plt.plot(max_valid_f1, label='validation')
    plt.title('Max F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.savefig(f'{save_path}')
    plt.close()

def draw_f1_plot(train_f1, valid_f1, save_path):
    plt.figure()
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='validation')
    plt.title('F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    plt.savefig(f'{save_path}')
    plt.close()

if __name__ == '__main__':
    hyperparams = {
        'num_epochs': 100,
        'out_feats': 128,
        'num_layers': 2,
        'lr': 0.001,
        'batch_size': 1,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    train(hyperparams=hyperparams)
