#!/usr/bin/env python
# coding: utf-8

# In[2]:


import ast
import pandas as pd
import numpy as np
import os

nodes_df = pd.read_csv('nodes.csv' , header = None , names=['node_attribute'], index_col=0)
edges_df = pd.read_csv('edges.csv')


nodes_list = [(index, ast.literal_eval(row.node_attribute))  for index, row in nodes_df.iterrows()]
edges_list = [(row.From, row.To) for index, row in edges_df.iterrows()]

number_of_days = 1448
nodes_list_vec = []
for index,dict_node in nodes_list:
    vec = np.zeros(number_of_days,dtype=np.float32)
    for day, count in dict_node.items():
        vec[day] = count

    nodes_list_vec.append((index, vec))

import networkx as nx


Graph = nx.DiGraph()
for node_id, node_attr_vec in nodes_list_vec:
    Graph.add_node(node_id, x = node_attr_vec)
Graph.add_edges_from(edges_list)

 



# In[3]:


import torch
from torch_geometric.utils import from_networkx

# Convert to PyG data
data = from_networkx(Graph)

# Convert node attributes to tensor and move to device
X = torch.stack([torch.tensor(attr['x'], dtype=torch.float32) for _, attr in Graph.nodes(data=True)])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X = X.to(device)
data.x = X
data = data.to(device)

 


# In[56]:


def sigmoid_norm(x, alpha=0.5):
    """Applies sigmoid normalization with adjustable slope."""
    return float(1 / (1 + np.exp(-alpha * x)))


phase1_score = []  # Holds every node's normalized phase 1 score [0,1]

def phase1(node_list_vec, start_day=0, current_day=220):
    for node_id, node_attr_vec in node_list_vec:
        mean = np.mean(node_attr_vec[start_day:current_day])
        std = np.std(node_attr_vec[start_day:current_day])
        today_score = (node_attr_vec[current_day] - mean) / std if std > 0 else 0
        
        # Apply sigmoid normalization directly to get [0,1]
        normalized_score = sigmoid_norm(today_score, alpha=0.5)
        phase1_score.append(normalized_score)



# In[5]:


from torch_geometric.utils import negative_sampling

phase2_score = [0] * len(nodes_list) #Holds every node's phase 2 score , index number corresponds to node id


import torch 
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prefer loading existing embeddings; train only if missing
z = None
embeddings_path = 'final_embeddings.pt'
if os.path.exists(embeddings_path):
    z = torch.load(embeddings_path, map_location=device)
    
else:
    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels , num_layers=3 , dropout = 0.3):
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
                x = F.dropout(x, p = self.dropout, training= self.training)
            x = self.convs[-1](x, edge_index)
            return x 

    in_channels = data.num_node_features
    hidden_channels = 128 
    out_channels = 64 

    model = GraphSAGE(in_channels, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr =0.01, weight_decay=5e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        z_local = model(data.x, data.edge_index)

        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1),
        ).to(device)  

        pos_similarity = (z_local[pos_edge_index[0]] * z_local[pos_edge_index[1]]).sum(dim=1)
        pos_loss = F.logsigmoid(pos_similarity).mean()

        neg_similarity = (z_local[neg_edge_index[0]] * z_local[neg_edge_index[1]]).sum(dim=1)
        neg_loss = F.logsigmoid(-neg_similarity).mean()

        loss = -pos_loss - neg_loss
        loss.backward()
        optimizer.step()

        return loss.item(), z_local

    

    for epoch in range(1, 500):
        loss, embeddings = train()
        

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    # saving file
    torch.save(z, embeddings_path)
    


# In[ ]:


import pandas as pd 
import ast
from collections import defaultdict

def historical_pattern(current_day =  220, weight_distribution=0.3):
    csv_file = 'node_day_recipients.csv'

    df = pd.read_csv(csv_file)
    df["day_recipients_str"] = df["day_recipients_str"].apply(ast.literal_eval)

    

    for index, row in df.iterrows():
        node = row['node_id']
        day_recipients = row['day_recipients_str']
        if (current_day < len(day_recipients) and 
            len(day_recipients[current_day]) > 0):  
            
            recipients_today = day_recipients[current_day]
            total_score_not_recipent = [0 for _ in range(len(recipients_today))]
            total_score_recipent = [0 for _ in range(len(recipients_today))]

            for past_day in range(current_day + 1):
                if past_day < len(day_recipients) and len(day_recipients[past_day]) > 0:
                    for past_recipient in day_recipients[past_day]:
                        for j_index, current_recipient in enumerate(recipients_today):  
                            if past_recipient == current_recipient:
                                total_score_recipent[j_index] += 1
                            else:
                                total_score_not_recipent[j_index] += 1

           
                # Calculate final scores with cosine similarity
            day_scores = []
            for j_index, recipient in enumerate(recipients_today):
                    # Get historical pattern score
                if total_score_not_recipent[j_index] == 0:
                    historical_score = 1.0
                else:
                    historical_score = total_score_recipent[j_index] / total_score_not_recipent[j_index]
                    
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

                        cos = (cos+1)/2  #Normalize [-1,1] to [0,1]
                        
                            
                            # Combine historical pattern and cosine similarity using weight
                        raw_score = (weight_distribution * historical_score + 
                                         (1 - weight_distribution) * (1-cos))
                            
                            # Ensure score is between 0 and 1
                        
                            
                    except (IndexError, ValueError) as e:
                            # Fallback if there's an issue with embeddings
                        print(f"Warning: Error calculating cosine similarity for nodes {node} and {recipient}: {e}")
                        raw_score = historical_score

                final_score = sigmoid_norm(raw_score, alpha=0.5)
                    
                day_scores.append((recipient, final_score))
                
                # Store in phase2_score at the node's index position
            phase2_score[node] = day_scores

# Call the function for a single day
historical_pattern(current_day=220, weight_distribution=0.3)
 


# In[ ]:


import numpy as np
import community.community_louvain as community_louvain
from collections import defaultdict

# Initialize global scores
phase3_score = [0] * len(nodes_list)

def phase3_community_normalization(Graph):

    # Step 1: Undirected graph
    undirected_graph = Graph.to_undirected()

    # Step 2: Louvain partition
    partition = community_louvain.best_partition(undirected_graph)

    results = {}
    
# Step 1: collect all fraction_same values first
    all_fraction_same = []
    senders = []

    for sender, rec_scores in enumerate(phase2_score):
        if not rec_scores:
            continue
        
        c_send = partition.get(sender, -1)
        recipients = [r for r, _ in rec_scores]
        if c_send == -1:
            continue
        
        same_comm_count = sum(1 for r in recipients if partition.get(r, -1) == c_send)
        total_recipients = len(recipients)
        fraction_same = same_comm_count / total_recipients if total_recipients > 0 else 0
        
        all_fraction_same.append(fraction_same)
        senders.append(sender)

    # Step 2: compute mean across all senders
    mean_fraction = np.mean(all_fraction_same)

    # Step 3: apply sigmoid normalization relative to population
    for sender, fraction_same in zip(senders, all_fraction_same):
        phase3_score[sender] = float(1 / (1 + np.exp(0.5 * (fraction_same - mean_fraction))))







# In[117]:


final_score = [[] for _ in range(len(nodes_list))]

def aggregate_score(w2=0.6, w1=0.2, w3=0.2):
        for node_id in range(len(nodes_list)):
            entry = phase2_score[node_id]
            if entry:
                p1_score = phase1_score[node_id] if node_id < len(phase1_score) else 0.0
                p3_score = phase3_score[node_id] if node_id < len(phase3_score) and phase3_score[node_id] > 0 else 1

                for recipient, score in entry:
                    final_score_node = w2 * score + w1 * p1_score + w3 * p3_score
                    final_score[node_id].append((recipient, final_score_node))


            if final_score[node_id]:
                 print(node_id, final_score[node_id])




# -----------------------------------------------------------------------------
# Single-class wrapper
# -----------------------------------------------------------------------------
class SEAMain:
    def __init__(self, w2=0.6, w1=0.2, w3=0.2, hist_weight=0.3):
        self.w2 = w2
        self.w1 = w1
        self.w3 = w3
        self.hist_weight = hist_weight

    def reset_globals(self):
        global phase1_score, phase2_score, phase3_score, final_score
        phase1_score = []
        phase2_score = [[] for _ in range(len(nodes_list))]
        phase3_score = [0] * len(nodes_list)
        final_score = [[] for _ in range(len(nodes_list))]

    def phase1(self, day):
        cd = max(0, min(int(day), number_of_days - 1))
        phase1(nodes_list_vec, start_day=0, current_day=cd)

    def phase2(self, day):
        cd = max(0, min(int(day), number_of_days - 1))
        historical_pattern(current_day=cd, weight_distribution=self.hist_weight)

    def phase3(self):
        phase3_community_normalization(Graph)

    def aggregate(self):
        aggregate_score(w2=self.w2, w1=self.w1, w3=self.w3)
        return final_score

    def run(self, day):
        self.reset_globals()
        self.phase1(day)
        self.phase2(day)
        self.phase3()
        return self.aggregate()



app = SEAMain()                 # optional weights: w2=0.6, w1=0.2, w3=0.2, hist_weight=0.3
final_scores = app.run(220)     # pass day


anomalous_nodes = []
for node_id, result in enumerate(final_scores):
    for i in result:
        if len(i) >= 2 and i[1] > 0.55:
            anomalous_nodes.append({node_id: i})



