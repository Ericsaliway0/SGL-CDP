import os
import pickle
import subprocess

import dgl
from dgl.data import DGLDataset


class Dataset(DGLDataset):

    def __init__(self, root='data'):##, omics='cna', cancer='KIRC'):
        self.root = os.path.abspath(root)
        save_dir = os.path.join(self.root, 'emb/processed')##, omics, cancer)
        os.makedirs(save_dir, exist_ok=True)        
        if 'processed' not in os.listdir(self.root):
            subprocess.run(f"mkdir 'tmp'", shell=True, cwd=self.root)
        raw_dir = os.path.join(self.root, 'emb/raw')##, omics, cancer)
        super().__init__(name='gene_graph', raw_dir=raw_dir, save_dir=save_dir)
        

    '''def has_cache(self):
        return len(os.listdir(self.save_dir)) == len(os.listdir(self.raw_dir))'''

    # def __len__(self):
    #     return len(os.listdir(self.save_dir))

    # def __getitem__(self, idx):
    #     print('self.save_dir=================================', self.save_dir)
    #     names = sorted(os.listdir(self.save_dir))
    #     print('names=================================', names)
    #     name = names[idx]
    #     print('name=================================', name)
    #     (graph,), _ = dgl.load_graphs(os.path.join(self.save_dir, name))
    #     return graph, name

    def __len__(self):
        return 2


    def __getitem__(self, idx):
        names = {
            0: 'emb_train.dgl',
            1: 'emb_test.dgl',
        }

        name = names[idx]

        print('Loading:', os.path.join(self.save_dir, name))

        (graph,), _ = dgl.load_graphs(
            os.path.join(self.save_dir, name)
        )

        return graph, name

    def process_(self):
        for cnt, graph_file in enumerate(os.listdir(self.raw_dir)):
            print('self.save_dir-------------------------------------', self.save_dir)
            print('self.raw_dir--------------------------------------', self.raw_dir)
            print('graph_file----------------------------------------', graph_file)
            graph_path = os.path.join(self.raw_dir, graph_file)
            nx_graph = pickle.load(open(graph_path, 'rb'))
            
            # Ensure that all nodes have 'significance' and 'weight' attributes
            for node in nx_graph.nodes:
                # Set default value for 'significance' if it does not exist
                significance = nx_graph.nodes[node].get('significance', 0.0)
                nx_graph.nodes[node]['significance'] = 1.0 if significance == 'significant' else 0.0
                
                # Set default value for 'weight' if it does not exist
                nx_graph.nodes[node]['weight'] = nx_graph.nodes[node].get('weight', 1.0)  # Default to 1.0

            # Convert to a DGL graph, setting both 'weight' and 'significance' attributes
            dgl_graph = dgl.from_networkx(nx_graph, node_attrs=['weight', 'significance'])
            
            # Save the DGL graph
            save_path = os.path.join(self.save_dir, f'{graph_file[:-4]}.dgl')
            dgl.save_graphs(save_path, dgl_graph)

    def process(self):
        for cnt, graph_file in enumerate(os.listdir(self.raw_dir)):
            print('self.save_dir-------------------------------------', self.save_dir)
            print('self.raw_dir--------------------------------------', self.raw_dir)
            print('graph_file----------------------------------------', graph_file)
            graph_path = os.path.join(self.raw_dir, graph_file)
            nx_graph = pickle.load(open(graph_path, 'rb'))
            
            # --- FIXED NODE LOGIC ---
            # Set node significance to 1.0 if the node connects to passing edges
            for node in nx_graph.nodes:
                # 1. Fallback default weight if not explicitly set
                nx_graph.nodes[node]['weight'] = nx_graph.nodes[node].get('weight', 1.0)
                
                # 2. Check if node significance was somehow set directly
                raw_sig = nx_graph.nodes[node].get('significance', None)
                
                if raw_sig == 'significant' or raw_sig == 1.0 or raw_sig == 1:
                    nx_graph.nodes[node]['significance'] = 1.0
                else:
                    # 3. If missing from node, extract it from connected edges
                    connected_edges = nx_graph.edges(node, data=True)
                    is_significant_node = False
                    
                    for _, _, edge_data in connected_edges:
                        # Inspect the true edge significance metrics from your file rows
                        edge_sig = edge_data.get('significance', 0)
                        if edge_sig == 1 or edge_sig == 1.0 or edge_sig == 'significant':
                            is_significant_node = True
                            break
                    
                    nx_graph.nodes[node]['significance'] = 1.0 if is_significant_node else 0.0
            # ------------------------

            # Convert to a DGL graph, setting both 'weight' and 'significance' attributes
            dgl_graph = dgl.from_networkx(nx_graph, node_attrs=['weight', 'significance'])
            
            # Save the DGL graph
            save_path = os.path.join(self.save_dir, f'{graph_file[:-4]}.dgl')
            dgl.save_graphs(save_path, dgl_graph)


    '''def process(self):
        for cnt, graph_file in enumerate(os.listdir(self.raw_dir)):
            graph_path = os.path.join(self.raw_dir, graph_file)
            nx_graph = pickle.load(open(graph_path, 'rb'))
            for node in nx_graph.nodes:
                if nx_graph.nodes[node]['significance'] == 'significant':
                    nx_graph.nodes[node]['significance'] = 1.0
                else:
                    nx_graph.nodes[node]['significance'] = 0.0
            dgl_graph = dgl.from_networkx(nx_graph, node_attrs=['weight', 'significance'])
            save_path = os.path.join(self.save_dir, f'{graph_file[:-4]}.dgl')
            dgl.save_graphs(save_path, dgl_graph)'''
