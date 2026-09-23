import pandas as pd
import networkx as nx
from collections import defaultdict

class Network:

    def __init__(self, interaction_data_path):
        # Initialize an empty directed graph
        self.graph_nx = nx.DiGraph()
        
        # Load interaction data from CSV
        self.interaction_data = self.load_interaction_data(interaction_data_path)
        self.build_graph()

    def load_interaction_data(self, path):
        # Load gene interaction data from CSV
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()  # Strip any leading/trailing whitespace from column names
        return df

    def build_graph_ori(self):
        # Build the graph by adding edges and attributes from the interaction data
        for _, row in self.interaction_data.iterrows():
            genea = row['GeneA']
            geneb = row['GeneB']
            stId = row['GeneA']
            name = row['GeneA']
            ##gene_type = row['gene_type']
            ####shared_partners = row['shared_partners']
            ####shared_count = row['shared_partners_count']
            weight = row['p_value']
            ##weight = row['expression_matrix']
            significance = row['significance']
            
            # Add edge between genes
            self.graph_nx.add_edge(genea, geneb)
            
            # Store gene info in a dictionary format
            self.graph_nx.nodes[genea]['stId'] = stId
            self.graph_nx.nodes[genea]['name'] = name
            ##self.graph_nx.nodes[genea]['gene_type'] = gene_type
            self.graph_nx.nodes[genea]['significance'] = significance
            ####self.graph_nx.nodes[genea]['shared_partners'] = shared_partners
            ####self.graph_nx.nodes[genea]['shared_count'] = shared_count
            self.graph_nx.nodes[genea]['weight'] = weight

    def build_graph_(self):
        # Build the graph by adding edges and gene-level attributes
        for _, row in self.interaction_data.iterrows():

            genea = row['GeneA']
            geneb = row['GeneB']

            weight = row['p_value']
            significance = int(row['significance'])

            # --------------------------------------------------
            # Add edge
            # --------------------------------------------------
            self.graph_nx.add_edge(genea, geneb)

            # --------------------------------------------------
            # GeneA
            # Only assign gene-level attributes if not already set
            # --------------------------------------------------
            if 'significance' not in self.graph_nx.nodes[genea]:
                self.graph_nx.nodes[genea]['stId'] = genea
                self.graph_nx.nodes[genea]['name'] = genea
                self.graph_nx.nodes[genea]['significance'] = significance
                self.graph_nx.nodes[genea]['weight'] = weight

            # --------------------------------------------------
            # GeneB
            # Only assign gene-level attributes if not already set
            # --------------------------------------------------
            if 'significance' not in self.graph_nx.nodes[geneb]:
                self.graph_nx.nodes[geneb]['stId'] = geneb
                self.graph_nx.nodes[geneb]['name'] = geneb
                self.graph_nx.nodes[geneb]['significance'] = significance
                self.graph_nx.nodes[geneb]['weight'] = weight

    def build_graph(self):
        # --------------------------------------------------
        # 1. Build edges
        # --------------------------------------------------
        for _, row in self.interaction_data.iterrows():
            genea = row['GeneA']
            geneb = row['GeneB']

            self.graph_nx.add_edge(
                genea,
                geneb,
                weight=row['p_value']
            )

        # --------------------------------------------------
        # 2. Build gene-level attributes
        # --------------------------------------------------
        gene_info = {}

        for _, row in self.interaction_data.iterrows():

            genea = row['GeneA']
            geneb = row['GeneB']
            # significance = int(row['significance'])
            # significance = bool(row['p_value'] > 0.05)
            p_value = float(row['p_value'])
            
            if p_value > 0.05:
                significance = 1
            else:
                significance = 0


            # GeneA
            if genea not in gene_info:
                gene_info[genea] = {
                    'stId': genea,
                    'name': genea,
                    'significance': significance,
                }

            # GeneB
            if geneb not in gene_info:
                gene_info[geneb] = {
                    'stId': geneb,
                    'name': geneb,
                    'significance': significance,
                }

        # --------------------------------------------------
        # 3. Assign attributes
        # --------------------------------------------------
        for gene, attrs in gene_info.items():
            self.graph_nx.nodes[gene].update(attrs)
                        
    def get_gene_info(self, gene):
        # Retrieve the stored gene info
        if gene in self.graph_nx:
            return {
                'stId': self.graph_nx.nodes[gene]['stId'],
                'name': self.graph_nx.nodes[gene]['name'],
                ##'gene_type': self.graph_nx.nodes[gene]['gene_type'],
                'significance': self.graph_nx.nodes[gene]['significance'],
                'shared_partners': self.graph_nx.nodes[gene]['shared_partners'],
                'shared_count': self.graph_nx.nodes[gene]['shared_count'],
                'weight': self.graph_nx.nodes[gene]['weight']
            }
        else:
            return None

    def display_graph_(self):
        # Display graph with node attributes (for debugging purposes)
        for node in self.graph_nx.nodes:
            info = self.get_gene_info(node)
            print(f"Gene: {node}, StId: {info['stId']}, Name: {info['name']}, Category: {info['gene_type']}, P-value: {info['weight']}")

    def save_name_to_id(self):
        # Save a mapping of gene names to IDs (stId)
        name_to_id = {node: self.graph_nx.nodes[node]['stId'] for node in self.graph_nx.nodes}
        file_path = 'name_to_id.txt'
        with open(file_path, 'w') as f:
            for name, stid in name_to_id.items():
                f.write(f"{name}: {stid}\n")

    def save_sorted_stids(self):
        # Save a sorted list of gene IDs (stIds)
        file_path = 'sorted_stids.txt'
        stids = sorted([self.graph_nx.nodes[node]['stId'] for node in self.graph_nx.nodes])
        with open(file_path, 'w') as f:
            for stid in stids:
                f.write(f"{stid}\n")


