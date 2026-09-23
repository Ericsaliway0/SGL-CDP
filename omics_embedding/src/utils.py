import os
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import dataset
import model
import train
from network import Network  # Importing the updated Network class

# Read stId mapping from the graph
def get_stid_mapping(graph):
    stid_mapping = {node_id: data['stId'] for node_id, data in graph.graph_nx.nodes(data=True)}
    return stid_mapping

# Save graph to disk
def save_to_disk(graph, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{graph.kge}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(graph.graph_nx, f)
    print(f"Graph saved to {save_path}")

# Save stId to CSV
def save_stid_to_csv(graph, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    stid_data = {'stId': [data['stId'] for _, data in graph.graph_nx.nodes(data=True)]}
    df = pd.DataFrame(stid_data)
    csv_path = os.path.join(save_dir, 'stId_nodes.csv')
    df.to_csv(csv_path, index=False)
    print(f"stId nodes saved to {csv_path}")

# Create network using protein interaction data
def create_network_from_genes_(data, kge, weight_path):
    ##graph = Network('data/split_files/protein_interaction_p_value_results_with_fdr_ptmod.csv') 
    ##graph = Network('data/protein_interaction_p_value_results_with_fdr_SHS27k.csv')
    graph = Network(weight_path)
    ########graph = Network('data/__protein_interaction_p_value_CPDB_ppi_0.99.csv')
    ##graph = Network('data/inhibition_protein_interaction_p_value_results_with_fdr_SHS27k.csv')
    # Initialize the protein network
    graph.interaction_data = data  # Assign the filtered data directly
    graph.build_graph()  # Build the graph with the interaction data
    graph.kge = kge  # Set the knowledge graph embedding identifier
    return graph

def create_network_from_genes_ori(data, kge, weight_path):
    graph = Network(weight_path)

    # Remove graph constructed from the complete CSV
    graph.graph_nx.clear()

    graph.interaction_data = data.copy()
    graph.build_graph()
    graph.kge = kge

    print(
        f"{kge}: "
        f"rows={len(data)}, "
        f"nodes={graph.graph_nx.number_of_nodes()}, "
        f"edges={graph.graph_nx.number_of_edges()}"
    )

    return graph
    

def create_network_from_genes(data, kge, weight_path):
    graph = Network(weight_path)

    # Remove graph constructed from the complete CSV
    graph.graph_nx.clear()

    graph.interaction_data = data.copy()
    graph.build_graph()
    graph.kge = kge

    print("\n========== INPUT TO create_network_from_genes ==========")
    print("Rows:", len(data))
    print("Columns:", data.columns.tolist())
    print("\nSignificance:")
    print(data['significance'].value_counts(dropna=False))
    print("\nUnique genes:")
    print("GeneA:", data['GeneA'].nunique())
    print("GeneB:", data['GeneB'].nunique())
    print("Union:",
        len(set(data['GeneA']) | set(data['GeneB'])))
        

    # ---------------------------------------------------------
    # Explicitly initialize significance for every graph node.
    #
    #  1 = significant
    #  0 = non-significant
    # -1 = unlabeled / no significance information available
    # ---------------------------------------------------------
    for node in graph.graph_nx.nodes():
        if 'significance' not in graph.graph_nx.nodes[node]:
            graph.graph_nx.nodes[node]['significance'] = -1

    print(
        f"{kge}: "
        f"rows={len(data)}, "
        f"nodes={graph.graph_nx.number_of_nodes()}, "
        f"edges={graph.graph_nx.number_of_edges()}"
    )

    # Debug
    from collections import Counter

    significance_counts = Counter(
        graph.graph_nx.nodes[node].get('significance', -1)
        for node in graph.graph_nx.nodes()
    )

    print("Significance distribution:", significance_counts)

    return graph

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def create_embedding_with_genes(
    p_value=0.05,
    save=True,
    data_dir='results/multiomics_meth/node_embeddings',
    omics_types=None,
    cancer_types=None
):
    """
    Create train/test embedding graphs for each omics-cancer pair.

    IMPORTANT:
    For Stage 1 pretraining, all genes are retained rather than filtering
    by p-value. The 'significance' column is preserved as the pretraining
    target.

    Stage 1:
        Generic biological relevance/significance pretraining.

    Stage 2:
        Fine-tuning for cancer driver gene prediction.

    Parameters
    ----------
    p_value : float
        Retained for backward compatibility. It is NOT used to filter genes
        in this version because Stage 1 should retain both significant and
        non-significant genes.
    save : bool
        Whether to save generated graphs to disk.
    data_dir : str
        Root directory for saved graph files.
    omics_types : list or None
        Omics types to process.
    cancer_types : list or None
        Cancer types to process.

    Returns
    -------
    all_graphs : dict
        Dictionary:
            {(omics, cancer): (graph_train, graph_test)}
    """

    # ---------------------------------------------------------
    # Default omics types
    # ---------------------------------------------------------
    if omics_types is None:
        omics_types = sorted([
            'cna',
            'ge',
            'meth',
            'mf'
        ])

    # ---------------------------------------------------------
    # Default cancer types
    # ---------------------------------------------------------
    if cancer_types is None:
        cancer_types = sorted([
            'KIRC',
            'BRCA',
            'READ',
            'PRAD',
            'STAD',
            'HNSC',
            'LUAD',
            'THCA',
            'BLCA',
            'ESCA',
            'LIHC',
            'UCEC',
            'COAD',
            'LUSC',
            'CESC',
            'KIRP'
        ])

    # ---------------------------------------------------------
    # Store all generated graphs
    # ---------------------------------------------------------
    all_graphs = {}

    # =========================================================
    # Process every omics-cancer combination
    # =========================================================
    for omics in omics_types:

        for cancer in cancer_types:

            print("\n" + "=" * 80)
            print(f"Processing: {omics}-{cancer}")
            print("=" * 80)

            # -------------------------------------------------
            # Input file
            # -------------------------------------------------
            p_value_path = os.path.join(
                'data',
                'omics',
                omics,
                f'{cancer}.csv'
            )

            # -------------------------------------------------
            # Check whether file exists
            # -------------------------------------------------
            if not os.path.exists(p_value_path):

                print(
                    f"⚠️ Skipping {omics}-{cancer}: "
                    f"File not found ({p_value_path})"
                )

                continue

            # # -------------------------------------------------
            # # Read CSV
            # # -------------------------------------------------
            # try:

            #     p_value_df = pd.read_csv(p_value_path)

            # except Exception as e:

            #     print(
            #         f"❌ Failed to read {omics}-{cancer}: {e}"
            #     )

            #     continue

            # # -------------------------------------------------
            # # Check required columns
            # # -------------------------------------------------
            # if 'p_value' not in p_value_df.columns:

            #     print(
            #         f"⚠️ Skipping {omics}-{cancer}: "
            #         f"Missing 'p_value' column"
            #     )

            #     continue

            # if 'significance' not in p_value_df.columns:

            #     print(
            #         f"⚠️ Skipping {omics}-{cancer}: "
            #         f"Missing 'significance' column"
            #     )

            #     continue

            # -------------------------------------------------
            # Read CSV
            # -------------------------------------------------
            try:
                p_value_df = pd.read_csv(p_value_path)

            except Exception as e:
                print(
                    f"❌ Failed to read {omics}-{cancer}: {e}"
                )
                continue


            # =================================================
            # Check required columns
            # =================================================

            if 'p_value' not in p_value_df.columns:

                print(
                    f"⚠️ Skipping {omics}-{cancer}: "
                    f"Missing 'p_value' column"
                )
                continue


            # =================================================
            # Create significance from mutation score
            # =================================================

            filtered_df = p_value_df.copy()

            # p_value is actually the mutation score
            filtered_df['p_value'] = pd.to_numeric(
                filtered_df['p_value'],
                errors='coerce'
            )

            # Remove invalid mutation scores
            filtered_df = filtered_df.dropna(
                subset=['p_value']
            ).copy()

            # -------------------------------------------------
            # Mutation-score-based significance
            #
            # mutation score > 0.05  -> significance = 1
            # mutation score <= 0.05 -> significance = 0
            # -------------------------------------------------

            filtered_df['significance'] = (
                filtered_df['p_value'] > 0.05
            ).astype(int)

            # =================================================
            # IMPORTANT:
            # Do NOT filter using p_value <= 0.05
            # =================================================
            #
            # Previous:
            #
            # filtered_df = p_value_df[
            #     p_value_df['p_value'] <= p_value
            # ].copy()
            #
            # This could result in:
            #
            # cna-BLCA:
            #     significance = [1]
            #
            # which eliminates the negative class.
            #
            # For Stage 1 pretraining, retain all genes.
            # =================================================

            filtered_df = p_value_df.copy()

            # -------------------------------------------------
            # Convert significance to numeric
            # -------------------------------------------------
            filtered_df['significance'] = pd.to_numeric(
                filtered_df['significance'],
                errors='coerce'
            )

            # -------------------------------------------------
            # Remove invalid/missing significance labels
            # -------------------------------------------------
            before_drop = len(filtered_df)

            filtered_df = filtered_df.dropna(
                subset=['significance']
            ).copy()

            after_drop = len(filtered_df)

            if before_drop != after_drop:

                print(
                    f"⚠️ Removed "
                    f"{before_drop - after_drop} rows with "
                    f"invalid/missing significance labels"
                )

            # -------------------------------------------------
            # Check whether anything remains
            # -------------------------------------------------
            if len(filtered_df) == 0:

                print(
                    f"⚠️ Skipping {omics}-{cancer}: "
                    f"No valid rows remaining"
                )

                continue

            # =================================================
            # Diagnostic information
            # =================================================

            unique_vals = sorted(
                filtered_df['significance'].unique().tolist()
            )

            class_counts = (
                filtered_df['significance']
                .value_counts()
                .sort_index()
            )

            print(
                f"📊 [{omics}-{cancer}] "
                f"Total genes: {len(filtered_df)}"
            )

            print(
                f"📊 [{omics}-{cancer}] "
                f"Unique significance labels: {unique_vals}"
            )

            print(
                f"📊 [{omics}-{cancer}] "
                f"Significance distribution:"
            )

            for label, count in class_counts.items():

                percentage = (
                    count / len(filtered_df) * 100
                )

                print(
                    f"    significance={label}: "
                    f"{count} "
                    f"({percentage:.2f}%)"
                )

            # =================================================
            # Train/test split
            # =================================================

            # -------------------------------------------------
            # Determine whether stratification is possible
            # -------------------------------------------------

            n_classes = filtered_df['significance'].nunique()

            min_class_count = (
                filtered_df['significance']
                .value_counts()
                .min()
            )

            can_stratify = (
                n_classes >= 2
                and min_class_count >= 2
            )

            if can_stratify:

                print(
                    f"🔀 Using stratified 80/20 split "
                    f"for {omics}-{cancer}"
                )

                genes_train, genes_test = train_test_split(
                    filtered_df,
                    test_size=0.2,
                    random_state=42,
                    stratify=filtered_df['significance']
                )

                print("\n========== GENES TRAIN SIGNIFICANCE ==========")
                print("shape:", genes_train.shape)
                print("columns:", genes_train.columns.tolist())
                print("dtype:", genes_train['significance'].dtype)
                print("unique:")
                print(genes_train['significance'].value_counts(dropna=False))
                print("values:")
                print(genes_train['significance'].unique())
                print(genes_train[['GeneA', 'GeneB', 'significance']].head(20))

            else:

                print(
                    f"⚠️ Cannot stratify {omics}-{cancer}. "
                    f"Using standard 80/20 split."
                )

                genes_train, genes_test = train_test_split(
                    filtered_df,
                    test_size=0.2,
                    random_state=42
                )

            # =================================================
            # Print split diagnostics
            # =================================================

            print(
                f"📦 Train: {len(genes_train)} rows"
            )

            print(
                f"📦 Test : {len(genes_test)} rows"
            )

            # -------------------------------------------------
            # Training label distribution
            # -------------------------------------------------

            train_dist = (
                genes_train['significance']
                .value_counts()
                .sort_index()
            )

            test_dist = (
                genes_test['significance']
                .value_counts()
                .sort_index()
            )

            print(
                "📈 Train significance distribution:"
            )

            for label, count in train_dist.items():

                percentage = (
                    count / len(genes_train) * 100
                )

                print(
                    f"    {label}: "
                    f"{count} "
                    f"({percentage:.2f}%)"
                )

            print(
                "📉 Test significance distribution:"
            )

            for label, count in test_dist.items():

                percentage = (
                    count / len(genes_test) * 100
                )

                print(
                    f"    {label}: "
                    f"{count} "
                    f"({percentage:.2f}%)"
                )

            # =================================================
            # Create graphs
            # =================================================

            print(
                f"🔨 Creating training graph: "
                f"{omics}-{cancer}"
            )

            graph_train = create_network_from_genes(
                genes_train,
                'emb_train',
                p_value_path
            )

            print(
                f"🔨 Creating test graph: "
                f"{omics}-{cancer}"
            )

            graph_test = create_network_from_genes(
                genes_test,
                'emb_test',
                p_value_path
            )

            # =================================================
            # Save graphs
            # =================================================

            if save:

                save_dir = os.path.join(
                    data_dir,
                    omics,
                    cancer,
                    'emb',
                    'raw'
                )

                os.makedirs(
                    save_dir,
                    exist_ok=True
                )

                save_to_disk(
                    graph_train,
                    save_dir
                )

                save_to_disk(
                    graph_test,
                    save_dir
                )

                print(
                    f"💾 Graphs saved to: {save_dir}"
                )

            # =================================================
            # Store graphs
            # =================================================

            all_graphs[
                (omics, cancer)
            ] = (
                graph_train,
                graph_test
            )

            print(
                f"✅ Processed {omics}-{cancer}"
            )

    # =========================================================
    # Final summary
    # =========================================================

    print("\n" + "=" * 80)
    print("🎉 ALL PROCESSING COMPLETED")
    print("=" * 80)

    print(
        f"Generated graph pairs: {len(all_graphs)}"
    )

    for key in all_graphs:

        omics, cancer = key

        print(
            f"    ✓ {omics}-{cancer}"
        )

    return all_graphs

# Function to create embeddings using GAT model
def create_embeddings(load_model=True, save=True, data_dir='data', hyperparams=None, plot=True, omics='cna', cancer='KIRC'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir_ = os.path.join(data_dir, omics, cancer)##, 'emb/raw')
    # Load dataset and set up directories
    data = dataset.Dataset(data_dir_)  # Adjust dataset to handle protein interactions
    emb_dir = os.path.abspath(os.path.join(data_dir_))##, 'embeddings'))
    os.makedirs(emb_dir, exist_ok=True)

    # Model parameters
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams.get('num_heads', 2)  # Default to 2 heads if not specified


    dim_latent = hyperparams['out_feats']
    '''num_layers = hyperparams['num_layers']
    
    net = model.SAGEModel(dim_latent=dim_latent, num_layers=num_layers).to(device)'''
    ## net = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads).to(device)
    ##net = model.GCNModel(dim_latent=dim_latent, num_layers=num_layers).to(device)
    net = model.TAGCNModel(dim_latent=out_feats, num_layers=num_layers).to(device)
    # net = model.GCNModel(
    #         dim_latent=out_feats,
    #         num_layers=num_layers,
    #         in_feats=in_feats,
    #         do_train=True
    #     ).to(device)

    # Load or train the model
    if load_model:
        model_path = os.path.join(data_dir, 'models', 'model.pth')
        net.load_state_dict(torch.load(model_path))
    else:
        model_path = train.train(hyperparams=hyperparams, data_path=data_dir, plot=plot, omics=omics, cancer=cancer)
        net.load_state_dict(torch.load(model_path))

    # Generate and save embeddings
    embedding_dict = {}
    for idx in tqdm(range(len(data))):
        graph, name = data[idx]
        graph = graph.to(device)
        
        with torch.no_grad():
            embedding = net(graph)
        embedding_dict[name] = embedding.cpu()

        if save:
            emb_path = os.path.join(emb_dir, f'{name[:-4]}.pth')
            torch.save(embedding.cpu(), emb_path)
            print(f"Embedding for {name} saved to {emb_path}")

    return embedding_dict
