import torch
import torch.nn as nn
from torch import Tensor
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn
from dgl.base import DGLError
from typing import Callable, Optional, Tuple, Union
##from dgl.nn import GATConv
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.nn import SAGEConv
import torch
import torch.nn as nn
import dgl
from dgl.nn import TAGConv, GINConv, ChebConv, GATConv

class TAGCN(nn.Module):
    def __init__(self, dim_latent: int, num_layers: int, do_train=False):
        super().__init__()
        self.do_train = do_train
        self.linear = nn.Linear(1, dim_latent)  # Linear layer to transform input weights to the desired latent dimension
        self.conv_0 = TAGConv(in_feats=dim_latent, out_feats=dim_latent, k=2)  # Initial TAGConv layer
        self.relu = nn.LeakyReLU()
        self.layers = nn.ModuleList(
            [TAGConv(in_feats=dim_latent, out_feats=dim_latent, k=2) for _ in range(num_layers - 1)]
        )  # Additional TAGConv layers
        self.predict = nn.Linear(dim_latent, 1)  # Linear layer for predictions

    def forward(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)  # Extract and unsqueeze node weights
        features = self.linear(weights)  # Transform weights to latent features
        graph = dgl.add_self_loop(graph)  # Add self-loops to the graph
        embedding = self.conv_0(graph, features)  # Apply the initial TAGConv layer

        for conv in self.layers:
            embedding = self.relu(embedding)  # Apply activation
            embedding = conv(graph, embedding)  # Apply each subsequent TAGConv layer

        if not self.do_train:
            return embedding.detach()  # Return detached embeddings during inference
        
        logits = self.predict(embedding)  # Predict logits
        return logits

    def get_node_embeddings(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)  # Extract and unsqueeze node weights
        features = self.linear(weights)  # Transform weights to latent features
        graph = dgl.add_self_loop(graph)  # Add self-loops to the graph
        embedding = self.conv_0(graph, features)  # Apply the initial TAGConv layer

        for conv in self.layers:
            embedding = self.relu(embedding)  # Apply activation
            embedding = conv(graph, embedding)  # Apply each subsequent TAGConv layer

        return embedding  # Return node embeddings

class GraphSAGE(nn.Module):
    def __init__(self, dim_latent, num_layers=1, do_train=False):
        super().__init__()
        self.do_train = do_train
        self.linear = nn.Linear(1, dim_latent)
        self.conv_0 = SAGEConv(in_feats=dim_latent, out_feats=dim_latent, aggregator_type='mean')
        self.relu = nn.LeakyReLU()
        self.layers = nn.ModuleList([SAGEConv(in_feats=dim_latent, out_feats=dim_latent, aggregator_type='mean')
                                     for _ in range(num_layers - 1)])
        self.predict = nn.Linear(dim_latent, 1)

    def forward(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)
        embedding = self.conv_0(graph, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)
        
        if not self.do_train:
            return embedding.detach()
        
        logits = self.predict(embedding)
        return logits

    def get_node_embeddings(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)
        embedding = self.conv_0(graph, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)

        return embedding

class GCN(nn.Module):
    def __init__(self, dim_latent: int, num_layers: int, do_train=False):
        super().__init__()
        self.do_train = do_train
        self.linear = nn.Linear(1, dim_latent)
        self.conv_0 = GraphConv(dim_latent, dim_latent, allow_zero_in_degree=True)
        self.relu = nn.LeakyReLU()
        self.layers = nn.ModuleList([GraphConv(dim_latent, dim_latent, allow_zero_in_degree=True)
                                     for _ in range(num_layers - 1)])
        self.predict = nn.Linear(dim_latent, 1)

    def forward(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)
        embedding = self.conv_0(graph, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)
        
        if not self.do_train:
            return embedding.detach()
        
        logits = self.predict(embedding)
        return logits

    def get_node_embeddings(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)
        embedding = self.conv_0(graph, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)

        return embedding

class GAT(nn.Module):
    def __init__(self, dim_latent: int, num_layers: int, num_heads: int = 4, do_train=False):
        super().__init__()
        self.do_train = do_train
        self.linear = nn.Linear(1, dim_latent)

        # Ensure out_feats is at least 1
        out_feats = max(1, dim_latent // num_heads)

        # First GAT layer
        self.conv_0 = GATConv(
            in_feats=dim_latent, 
            out_feats=out_feats,  
            num_heads=num_heads, 
            allow_zero_in_degree=True
        )

        self.relu = nn.LeakyReLU()

        # Stacking multiple GAT layers
        self.layers = nn.ModuleList([
            GATConv(
                in_feats=num_heads * out_feats,  # Adjust input size for multi-head attention
                out_feats=out_feats,  
                num_heads=num_heads, 
                allow_zero_in_degree=True
            ) 
            for _ in range(num_layers - 1)
        ])

        # Final linear layer to map from (num_heads * out_feats) → 1
        self.predict = nn.Linear(num_heads * out_feats, 1)

    def forward(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)

        # First layer
        embedding = self.conv_0(graph, features)
        embedding = embedding.view(embedding.shape[0], -1)  # Flatten multi-head output

        # Process through layers
        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)
            embedding = embedding.view(embedding.shape[0], -1)  # Flatten multi-head output

        if not self.do_train:
            return embedding.detach()

        logits = self.predict(embedding)  # Ensure output is (batch_size, 1)
        return logits

    def get_node_embeddings(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)

        embedding = self.conv_0(graph, features)
        embedding = embedding.view(embedding.shape[0], -1)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)
            embedding = embedding.view(embedding.shape[0], -1)

        return embedding
import torch.nn as nn
import dgl
from dgl.nn import ChebConv

class Chebnet(nn.Module):
    def __init__(self, dim_latent: int, num_layers: int, k: int = 3, do_train=False):
        super().__init__()
        self.do_train = do_train
        self.linear = nn.Linear(1, dim_latent)

        # First Chebnet layer
        self.conv_0 = ChebConv(
            in_feats=dim_latent,
            out_feats=dim_latent,
            k=k  # Chebyshev polynomial order
        )

        self.relu = nn.LeakyReLU()

        # Stacking multiple ChebConv layers
        self.layers = nn.ModuleList([
            ChebConv(
                in_feats=dim_latent, 
                out_feats=dim_latent,
                k=k
            ) 
            for _ in range(num_layers - 1)
        ])

        # Final linear layer to map to 1 output
        self.predict = nn.Linear(dim_latent, 1)

    def forward(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)  # This is fine

        # First layer
        embedding = self.conv_0(graph, features)

        # Process through layers
        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)

        if not self.do_train:
            return embedding.detach()

        logits = self.predict(embedding)  # Ensure output is (batch_size, 1)
        return logits

    def get_node_embeddings(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)

        embedding = self.conv_0(graph, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)

        return embedding

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for GIN update function."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class GIN(nn.Module):
    def __init__(self, dim_latent: int, num_layers: int, do_train=False):
        super().__init__()
        self.do_train = do_train
        self.linear = nn.Linear(1, dim_latent)  # Initial feature transformation
        
        # Define the first GIN layer
        self.conv_0 = GINConv(MLP(dim_latent, dim_latent, dim_latent), learn_eps=True)
        
        # Additional GIN layers
        self.layers = nn.ModuleList([
            GINConv(MLP(dim_latent, dim_latent, dim_latent), learn_eps=True)
            for _ in range(num_layers - 1)
        ])
        
        self.relu = nn.ReLU()
        self.predict = nn.Linear(dim_latent, 1)  # Final prediction layer

    def forward(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)  # Add self-loops

        embedding = self.conv_0(graph, features)
        
        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)

        if not self.do_train:
            return embedding.detach()

        logits = self.predict(embedding)
        return logits

    def get_node_embeddings(self, graph):
        """Extract node embeddings for inference."""
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)

        embedding = self.conv_0(graph, features)
        
        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)

        return embedding
