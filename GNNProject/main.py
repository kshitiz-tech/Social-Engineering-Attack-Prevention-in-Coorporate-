# Import required libraries
import ast
import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import from_networkx, negative_sampling
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import networkx as nx
import os

# Configure CUDA settings for GPU acceleration
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better CUDA error debugging
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)  # Explicitly set CUDA device
torch.cuda.empty_cache()  # Clear GPU cache

# Display GPU and CUDA information
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
print(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current GPU device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

# Load data from CSV files
# nodes.csv contains node attributes with a dictionary of day-count pairs
# edges.csv contains the connections between nodes
nodes_df = pd.read_csv('nodes.csv', header=None, names=['node_attribute'], index_col=0)
edges_df = pd.read_csv('edges.csv')

# Process nodes data
# Convert string representation of dictionary to actual dictionary using ast.literal_eval
nodes_list = [(index, ast.literal_eval(row.node_attribute)) for index, row in nodes_df.iterrows()]
edges_list = [(row.From, row.To) for index, row in edges_df.iterrows()]

# Initialize vectors for each node
number_of_days = 1448  # Total number of days in the dataset
nodes_list_vec = []
for index, dict_node in nodes_list:
    # Create a zero vector for each node with length equal to number of days
    vec = np.zeros(number_of_days, dtype=np.float32)
    # Fill in the vector with counts for each day
    for day, count in dict_node.items():
        vec[day] = count
    nodes_list_vec.append((index, vec))

# Create directed graph using networkx
Graph = nx.DiGraph()
# Add nodes with their feature vectors
for node_id, node_attr_vec in nodes_list_vec:
    Graph.add_node(node_id, x=node_attr_vec)
# Add edges to the graph
Graph.add_edges_from(edges_list)

# Print graph statistics
print(f"Number of nodes: {nx.number_of_nodes(Graph)}")
print(f"Number of edges: {nx.number_of_edges(Graph)}")

# Convert networkx graph to PyTorch Geometric data format
data = from_networkx(Graph)
# Convert node attributes to tensor and move to GPU
X = torch.stack([torch.tensor(attr['x'], dtype=torch.float32) for _, attr in Graph.nodes(data=True)])
data.x = X.to(device)
data.edge_index = data.edge_index.to(device)

print("\nData structure:")
print(data)
print(f"\nTensor devices:")
print(f"x tensor device: {data.x.device}")
print(f"edge_index device: {data.edge_index.device}")

# Phase 1: Statistical Analysis
# Calculate anomaly scores based on mean and standard deviation
phase1_score = []  # Holds every node's phase 1 score, index number corresponds to node id
def phase1(node_list_vec, start_day=0, current_day=1448):
    """
    Calculate phase 1 scores for each node based on statistical measures
    Args:
        node_list_vec: List of tuples containing node ID and its activity vector
        start_day: Start day for the analysis window
        current_day: Current day being analyzed
    """
    for node_id, node_attr_vec in node_list_vec:
        mean = np.mean(node_attr_vec[start_day:current_day])
        std = np.std(node_attr_vec[start_day:current_day])
        today_score = (node_attr_vec[current_day] - mean) / std if std > 0 else 0
        phase1_score.append(today_score)

# Calculate phase 1 scores
phase1(nodes_list_vec, start_day=0, current_day=1447)

# Phase 2: Graph Neural Network Analysis
phase2_score = []  # Holds every node's phase 2 score, index number corresponds to node id

# Define GraphSAGE model
class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for learning node embeddings
    Args:
        in_channels: Number of input features
        hidden_channels: Number of hidden features
        out_channels: Number of output features
        num_layers: Number of GraphSAGE layers
        dropout: Dropout probability
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        # Add intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Add output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass of the model
        Args:
            x: Node features
            edge_index: Graph connectivity in COO format
        Returns:
            Node embeddings
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

# Initialize model parameters
in_channels = data.num_node_features
hidden_channels = 128
out_channels = 64

# Create model and move to GPU
model = GraphSAGE(in_channels, hidden_channels, out_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training function
def train():
    """
    Train the GraphSAGE model using negative sampling
    Returns:
        tuple: (loss value, node embeddings)
    """
    model.train()
    optimizer.zero_grad()
    
    # Get node embeddings
    z = model(data.x, data.edge_index)

    # Positive edges are the existing edges
    pos_edge_index = data.edge_index
    # Generate negative edges through negative sampling
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1),
    ).to(device)

    # Calculate loss for positive edges
    pos_similarity = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    pos_loss = F.logsigmoid(pos_similarity).mean()

    # Calculate loss for negative edges
    neg_similarity = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    neg_loss = F.logsigmoid(-neg_similarity).mean()

    # Combined loss
    loss = -pos_loss - neg_loss
    loss.backward()
    optimizer.step()

    return loss.item(), z

# Print initial device information
print(f"\nTraining on device: {next(model.parameters()).device}")

# Training loop
for epoch in range(1, 201):
    loss, embeddings = train()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Final evaluation
model.eval()
with torch.no_grad():
    z = model(data.x, data.edge_index)
    print(f"\nFinal embeddings device: {z.device}")
import pandas as pd 
import ast

def historical_pattern(current_day = 1447):
    csv_file = 'node_day_recipients.csv'

    df = pd.read_csv(csv_file)
    df["day_recipients_str"] = df["day_recipients_str"].apply(ast.literal_eval)

    
    
    

            
       

historical_pattern()