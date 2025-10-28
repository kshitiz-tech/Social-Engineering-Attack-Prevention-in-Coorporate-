# Social Engineering Attack Prevention in Corporate Networks
# Graph Neural Network Analysis for Email Communication Patterns

import ast
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.nn import SAGEConv
import networkx as nx
import community.community_louvain as community_louvain
from collections import defaultdict

# =============================================================================
# PHASE 1: Data Loading and Graph Construction
# =============================================================================

print("=== Loading Data and Constructing Graph ===")

# Load nodes and edges data
nodes_df = pd.read_csv('nodes.csv', header=None, names=['node_attribute'], index_col=0)
edges_df = pd.read_csv('edges.csv')

# Parse node attributes (day -> email count dictionaries)
nodes_list = [(index, ast.literal_eval(row.node_attribute)) for index, row in nodes_df.iterrows()]
edges_list = [(row.From, row.To) for index, row in edges_df.iterrows()]

# Convert node attributes to vectors (1448 days)
number_of_days = 1448
nodes_list_vec = []
for index, dict_node in nodes_list:
    vec = np.zeros(number_of_days, dtype=np.float32)
    for day, count in dict_node.items():
        vec[day] = count
    nodes_list_vec.append((index, vec))

# Create directed graph with node attributes
Graph = nx.DiGraph()
for node_id, node_attr_vec in nodes_list_vec:
    Graph.add_node(node_id, x=node_attr_vec)
Graph.add_edges_from(edges_list)

print(f"Number of nodes: {nx.number_of_nodes(Graph)}")
print(f"Number of edges: {nx.number_of_edges(Graph)}")

# =============================================================================
# PHASE 2: Convert to PyTorch Geometric Format
# =============================================================================

print("\n=== Converting to PyTorch Geometric Format ===")

# Convert NetworkX graph to PyTorch Geometric data
data = from_networkx(Graph)

# Convert node attributes to tensor and move to device
X = torch.stack([torch.tensor(attr['x'], dtype=torch.float32) for _, attr in Graph.nodes(data=True)])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X = X.to(device)
data.x = X
data = data.to(device)

print(data)
print(f"Data is on device: {data.x.device}")

# =============================================================================
# PHASE 3: Phase 1 Scoring - Statistical Anomaly Detection
# =============================================================================

print("\n=== Phase 1: Statistical Anomaly Detection ===")

phase1_score = []  # Holds every node's phase 1 score, index number corresponds to node id

def phase1(node_list_vec, start_day=0, current_day=1448):
    """
    Calculate Phase 1 scores based on statistical anomaly detection.
    Compares current day activity to historical mean and standard deviation.
    """
    for node_id, node_attr_vec in node_list_vec:
        mean = np.mean(node_attr_vec[start_day:current_day])
        std = np.std(node_attr_vec[start_day:current_day])
        today_score = (node_attr_vec[current_day] - mean) / std if std > 0 else 0
        phase1_score.append(today_score)

# Calculate Phase 1 scores for all nodes
phase1(nodes_list_vec, start_day=0, current_day=1447)

# =============================================================================
# PHASE 4: Graph Neural Network Model Definition and Training
# =============================================================================

print("\n=== Phase 2: Graph Neural Network Training ===")

# Initialize Phase 2 scores storage
phase2_score = []  # Holds every node's phase 2 score, index number corresponds to node id
for i in range(len(nodes_list_vec)):
    phase2_score.append({})

# Define GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

# Initialize model and optimizer
in_channels = data.num_node_features
hidden_channels = 128 
out_channels = 64 

model = GraphSAGE(in_channels, hidden_channels, out_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    """
    Training function for GraphSAGE model using link prediction task.
    Uses positive and negative sampling for contrastive learning.
    """
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)

    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1),
    ).to(device)  

    pos_similarity = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    pos_loss = F.logsigmoid(pos_similarity).mean()

    neg_similarity = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    neg_loss = F.logsigmoid(-neg_similarity).mean()

    loss = -pos_loss - neg_loss
    loss.backward()
    optimizer.step()

    return loss.item(), z

# Train the model
print(f"Training on device: {next(model.parameters()).device}")

for epoch in range(1, 500):
    loss, embeddings = train()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Generate final embeddings
model.eval()
with torch.no_grad():
    z = model(data.x, data.edge_index)

# Save embeddings
torch.save(z, 'final_embeddings.pt')
print("\nEmbeddings saved to 'final_embeddings.pt'")

# =============================================================================
# PHASE 5: Historical Pattern Analysis with Cosine Similarity
# =============================================================================

print("\n=== Phase 2: Historical Pattern Analysis ===")

def historical_pattern(current_day=220, weight_distribution=0.3):
    """
    Analyze historical communication patterns and combine with cosine similarity.
    
    Args:
        current_day: The day to analyze (default: 220)
        weight_distribution: Weight for historical pattern vs cosine similarity (default: 0.3)
    """
    csv_file = 'node_day_recipients.csv'
    df = pd.read_csv(csv_file)
    df["day_recipients_str"] = df["day_recipients_str"].apply(ast.literal_eval)

    for index, row in df.iterrows():
        node = row['node_id']
        day_recipients = row['day_recipients_str']
        
        if (current_day < len(day_recipients) and 
            len(day_recipients[current_day]) > 0):  
            
            recipients_today = day_recipients[current_day]
            total_score_not_recipient = [0 for _ in range(len(recipients_today))]
            total_score_recipient = [0 for _ in range(len(recipients_today))]

            # Analyze historical patterns
            for past_day in range(current_day + 1):
                if past_day < len(day_recipients) and len(day_recipients[past_day]) > 0:
                    for past_recipient in day_recipients[past_day]:
                        for j_index, current_recipient in enumerate(recipients_today):  
                            if past_recipient == current_recipient:
                                total_score_recipient[j_index] += 1
                            else:
                                total_score_not_recipient[j_index] += 1

            # Calculate final scores with cosine similarity
            day_scores = []
            for j_index, recipient in enumerate(recipients_today):
                # Get historical pattern score
                if total_score_not_recipient[j_index] == 0:
                    historical_score = 1.0
                else:
                    historical_score = total_score_recipient[j_index] / total_score_not_recipient[j_index]
                    
                # If historical score is 1, keep it as is
                if historical_score == 1.0:
                    final_score = 1.0
                else:
                    # Calculate cosine similarity between current node and recipient
                    try:
                        # Get embeddings for current node and recipient
                        node_embedding = z[node]
                        recipient_embedding = z[recipient]
                        
                        # Calculate cosine similarity using PyTorch
                        cos = F.cosine_similarity(
                            node_embedding.unsqueeze(0), 
                            recipient_embedding.unsqueeze(0)
                        ).item()

                        cos = (cos + 1) / 2  # Normalize [-1,1] to [0,1]
                        
                        # Combine historical pattern and cosine similarity using weight
                        final_score = (weight_distribution * historical_score + 
                                     (1 - weight_distribution) * (1 - cos))
                        
                        # Ensure score is between 0 and 1
                        final_score = max(0.0, min(1.0, final_score))
                        
                    except (IndexError, ValueError) as e:
                        # Fallback if there's an issue with embeddings
                        print(f"Warning: Error calculating cosine similarity for nodes {node} and {recipient}: {e}")
                        final_score = historical_score
                    
                day_scores.append((recipient, final_score))
                
            # Store in phase2_score at the node's index position
            phase2_score[node][current_day] = day_scores

# Call the function for a single day
historical_pattern(current_day=220, weight_distribution=0.3)

# Helper function to get nodes with Phase 2 data
def get_phase2_nodes():
    """
    Get list of nodes that have Phase 2 scores calculated.
    """
    phase2_nodes = []
    for node_id in range(len(phase2_score)):
        if phase2_score[node_id]:  # Only stores that have data
            phase2_nodes.append(node_id)            
    return phase2_nodes

# =============================================================================
# PHASE 6: Community Detection and Normalization
# =============================================================================

print("\n=== Phase 3: Community Detection and Normalization ===")

# Initialize global scores
phase3_score = [0] * len(nodes_list)

def phase3_community_normalization(Graph):
    """
    Perform community detection using Louvain algorithm and normalize scores.
    
    Args:
        Graph: NetworkX graph object
        
    Returns:
        phase3_score: List of community-based scores
        partition: Community partition dictionary
    """
    global phase3_score

    # Step 1: Convert to undirected graph
    undirected_graph = Graph.to_undirected()

    # Step 2: Apply Louvain community detection
    partition = community_louvain.best_partition(undirected_graph)

    # Step 3: Get nodes to evaluate (from Phase 2)
    nodes_to_eval = set(get_phase2_nodes())

    # Step 4: Count nodes per community (only those being evaluated)
    community_count = defaultdict(int)
    for node in nodes_to_eval:
        community_id = partition.get(node, -1)
        community_count[community_id] += 1

    # Step 5: Assign scores based on community size
    for node in nodes_to_eval:
        community_id = partition.get(node, -1)
        if community_id != -1:
            phase3_score[node] = community_count[community_id] - 1
        else:
            phase3_score[node] = 0  # if not found in partition

    return phase3_score, partition

# Execute Phase 3 analysis
phase3_community_normalization(Graph)

def apply_normalization(phase1_score, phase2_score, phase3_score):
    pass

