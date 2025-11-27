# Imports and data loading
import ast
import os
import pandas as pd
import numpy as np
import networkx as nx

# Load nodes and edges
nodes_df = pd.read_csv('nodes.csv', header=None, names=['node_attribute'], index_col=0)
edges_df = pd.read_csv('edges.csv')

# Parse node attributes
nodes_list = [(index, ast.literal_eval(row.node_attribute)) for index, row in nodes_df.iterrows()]
edges_list = [(row.From, row.To) for index, row in edges_df.iterrows()]

# Vectorize node attributes
number_of_days = 1448
nodes_list_vec = []
for index, dict_node in nodes_list:
    vec = np.zeros(number_of_days, dtype=np.float32)
    for day, count in dict_node.items():
        vec[day] = count
    nodes_list_vec.append((index, vec))

# Build graph
Graph = nx.DiGraph()
for node_id, node_attr_vec in nodes_list_vec:
    Graph.add_node(node_id, x=node_attr_vec)
Graph.add_edges_from(edges_list)

# Device, PyG conversion, and embeddings
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.nn import SAGEConv

# Convert to PyG data
data = from_networkx(Graph)

# Node features to tensor
X = torch.stack([torch.tensor(attr['x'], dtype=torch.float32) for _, attr in Graph.nodes(data=True)])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X = X.to(device)
data.x = X
data = data.to(device)

# Load or train embeddings once
z = None
model_path = "graphsage_model.pt"

# ----- FIX #1: GraphSAGE must be defined before loading -----
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
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

in_channels = data.num_node_features
model = GraphSAGE(in_channels, 128, 64).to(device)


if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

else:
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train_one():
        model.train(); opt.zero_grad()
        z_local = model(data.x, data.edge_index)
        pos = data.edge_index
        neg = negative_sampling(
            edge_index=data.edge_index, 
            num_nodes=data.num_nodes, 
            num_neg_samples=pos.size(1)
        ).to(device)

        pos_loss = F.logsigmoid((z_local[pos[0]] * z_local[pos[1]]).sum(dim=1)).mean()
        neg_loss = F.logsigmoid(-(z_local[neg[0]] * z_local[neg[1]]).sum(dim=1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward(); opt.step()
        return z_local

    for _ in range(500):
        _ = train_one()

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    torch.save(model.state_dict(), model_path)

# Generate embeddings for current graph
with torch.no_grad():
    z = model(data.x, data.edge_index)

# Load nodes and edges
nodes_df_synthetic = pd.read_csv('nodes-synthetic.csv', header=None, names=['node_attribute'], index_col=0)
edges_df_synthetic = pd.read_csv('edges-synthetic.csv')

# Parse node attributes
nodes_list_synthetic = [(index, ast.literal_eval(row.node_attribute)) for index, row in nodes_df_synthetic.iterrows()]
edges_list_synthetic = [(row.From, row.To) for index, row in edges_df_synthetic.iterrows()]

# Vectorize node attributes
number_of_days = 1448
nodes_list_vec_synthetic = []
for index, dict_node in nodes_list_synthetic:
    vec = np.zeros(number_of_days, dtype=np.float32)
    for day, count in dict_node.items():
        vec[day] = count
    nodes_list_vec_synthetic.append((index, vec))

#Create a new graph with new synthetic nodes to model the connection. No need to reporcess everything. will only add new nodes
#This can be extracted programatically , but since this is for research purposes , we work with hardcoded ids because we know which are new nodes
new_node_ids = [6600, 6601, 6602, 6603]
# Add new nodes with features
for nid, vec in nodes_list_vec_synthetic:
    if nid not in Graph:       # Graph = original NetworkX graph
        Graph.add_node(nid, x=vec)

# Add edges (may include new edges)
for u, v in edges_list_synthetic:
    Graph.add_edge(u, v)

data = from_networkx(Graph)

X = torch.stack([
    torch.tensor(attr['x'], dtype=torch.float32)
    for _, attr in Graph.nodes(data=True)
])

data.x = X.to(device)
data = data.to(device)

model = GraphSAGE(in_channels, 128, 64).to(device)
model.load_state_dict(torch.load("graphsage_model.pt", map_location=device))
model.eval()

with torch.no_grad():
    z = model(data.x, data.edge_index)



# Helper functions and globals

def sigmoid_norm(x, alpha=0.5):
    return float(1 / (1 + np.exp(-alpha * x)))

phase1_score = []
phase2_score = [0] * len(nodes_list_synthetic)
phase3_score = [0] * len(nodes_list_synthetic)
final_score = [[] for _ in range(len(nodes_list_synthetic))]

# Phase 1
def phase1(node_list_vec_synthetic, start_day=0, current_day=211):
    for _, node_attr_vec in node_list_vec_synthetic:
        mean = np.mean(node_attr_vec[start_day:current_day])
        std = np.std(node_attr_vec[start_day:current_day])
        today_score = (node_attr_vec[current_day] - mean) / std if std > 0 else 0
        phase1_score.append(sigmoid_norm(today_score, alpha=0.5))

# Phase 2
import ast as _ast  # avoid shadowing

def historical_pattern(current_day, weight_distribution=0.3):
    csv_file = 'node_day_recipients_synthetic.csv'
    df = pd.read_csv(csv_file)
    df['day_recipients_str'] = df['day_recipients_str'].apply(_ast.literal_eval)
    for _, row in df.iterrows():
        node = row['node_id']
        day_recipients = row['day_recipients_str']
        if current_day < len(day_recipients) and len(day_recipients[current_day]) > 0:
            recipients_today = day_recipients[current_day]
            total_not = [0 for _ in range(len(recipients_today))]
            total_yes = [0 for _ in range(len(recipients_today))]
            for past_day in range(current_day + 1):
                if past_day < len(day_recipients) and len(day_recipients[past_day]) > 0:
                    for past_recipient in day_recipients[past_day]:
                        for j_idx, current_recipient in enumerate(recipients_today):
                            if past_recipient == current_recipient:
                                total_yes[j_idx] += 1
                            else:
                                total_not[j_idx] += 1
            day_scores = []
            for j_idx, recipient in enumerate(recipients_today):
                if total_not[j_idx] == 0:
                    historical_score = 1.0
                else:
                    historical_score = total_yes[j_idx] / total_not[j_idx]
                try:
                    node_embedding = z[node]
                    recipient_embedding = z[recipient]
                    cos = F.cosine_similarity(node_embedding.unsqueeze(0), recipient_embedding.unsqueeze(0)).item()
                    cos = (cos + 1) / 2
                    raw_score = (weight_distribution * historical_score + (1 - weight_distribution) * (1 - cos))
                except Exception as e:
                    print("EMBED ERROR:", node, recipient, e)
                    raw_score = historical_score
                final_s = sigmoid_norm(raw_score, alpha=0.5)
                day_scores.append((recipient, final_s))
            phase2_score[node] = day_scores

# Phase 3
import community.community_louvain as community_louvain
from collections import defaultdict

def phase3_community_normalization(Graph):
    undirected_graph = Graph.to_undirected()
    partition = community_louvain.best_partition(undirected_graph)
    all_fraction_same, senders = [], []
    for sender, rec_scores in enumerate(phase2_score):
        if not rec_scores:
            continue
        c_send = partition.get(sender, -1)
        recipients = [r for r, _ in rec_scores]
        if c_send == -1:
            continue
        same_comm = sum(1 for r in recipients if partition.get(r, -1) == c_send)
        total = len(recipients)
        frac = same_comm / total if total > 0 else 0
        all_fraction_same.append(frac)
        senders.append(sender)
    mean_fraction = np.mean(all_fraction_same) if all_fraction_same else 0
    for sender, frac in zip(senders, all_fraction_same):
        phase3_score[sender] = float(1 / (1 + np.exp(0.5 * (frac - mean_fraction))))

# Aggregation
def aggregate_score(w2=0.6, w1=0.2, w3=0.2):
    for node_id in range(len(nodes_list)):
        entry = phase2_score[node_id]
        if entry:
            p1_score = phase1_score[node_id] if node_id < len(phase1_score) else 0.0
            p3_score = phase3_score[node_id] if node_id < len(phase3_score) and phase3_score[node_id] > 0 else 1
            for recipient, score in entry:
                final_score_node = w2 * score + w1 * p1_score + w3 * p3_score
                final_score[node_id].append((recipient, final_score_node))
            


# Class wrapper
class SEAMain:
    def __init__(self, w2=0.6, w1=0.2, w3=0.2, hist_weight=0.3):
        self.w2 = w2
        self.w1 = w1
        self.w3 = w3
        self.hist_weight = hist_weight
    def reset(self):
        global phase1_score, phase2_score, phase3_score, final_score
        phase1_score = []
        phase2_score = [[] for _ in range(len(nodes_list_synthetic))]
        phase3_score = [0] * len(nodes_list_synthetic)
        final_score = [[] for _ in range(len(nodes_list_synthetic))]
    def phase1(self, day):
        phase1(nodes_list_vec_synthetic, start_day=0, current_day= day)
    def phase2(self, day):
        historical_pattern(current_day=day, weight_distribution=self.hist_weight)
    def phase3(self):
        phase3_community_normalization(Graph)
    def aggregate(self):
        aggregate_score(w2=self.w2, w1=self.w1, w3=self.w3)
        return final_score
        
    def run(self, day):
        self.reset()
        self.phase1(day)
        self.phase2(day)
        self.phase3()
        return self.aggregate()

base_date = pd.Timestamp('1999-01-01', tz='UTC')

def day_index_to_date(day_index: int, tz: str = 'UTC') -> pd.Timestamp:
    di = int(day_index)
    dt = base_date + pd.Timedelta(days=di)
    return dt.tz_convert(tz) if tz and dt.tzinfo is not None and tz != 'UTC' else dt

def format_human_date(ts) -> str:
    """Return dates like '4 May 2001' (abbreviated month, no leading zero)."""
    ts = pd.Timestamp(ts)
    for fmt in ('%-d %b %Y', '%#d %b %Y', '%d %b %Y'):
        try:
            s = ts.strftime(fmt)
            if fmt == '%d %b %Y':
                s = s.lstrip('0')
            return s
        except ValueError:
            continue
    return ts.strftime('%Y-%m-%d')

def run_sender_receiver(day_index):
    """Run SEA pipeline and print sender → receiver anomalies."""
    app = SEAMain()
    results = app.run(day_index)
    print(f"Total communications {len(results)}")

    anomalies = []
    for sender, recs in enumerate(results):
        for recv, score in recs:
            if score > 0.6:
                anomalies.append({sender: (recv, score)})

    print(f"\nDay {day_index}: Detected {len(anomalies)} anomalous communications\n")

    return anomalies

day = 211
anomalies = run_sender_receiver(day)
anomalies

def show_emails_for_anomalies(anomalies, day_index):
    """Display email messages related to detected anomalies."""
    day_ts = day_index_to_date(day_index-3)
    day_str = " " + format_human_date(day_ts)
    print(f"\nSearching emails for date: {day_str}")

    emails = pd.read_csv("emails_synthetic.csv", names=['file', 'message'], header=0)
    id_email = pd.read_csv("id-email-synthetic.csv")

    count_matches = 0
    for anomaly in anomalies:
        sender_id, (recv_id, score) = list(anomaly.items())[0]
        sender_email = id_email.iloc[sender_id-1][1]
        receiver_email = id_email.iloc[recv_id-1][1]

        print(f"\n[Anomaly] Node {sender_id} → Node {recv_id} | Score={score:.3f}")
        print(f"Sender: {sender_email}")
        print(f"Receiver: {receiver_email}")

        matches = emails[
            emails['message'].str.contains(day_str, na=False)
            & emails['message'].str.contains(sender_email, na=False)
            & emails['message'].str.contains(receiver_email, na=False)
        ]
        
        if not matches.empty:
            print(f"\n📧 Found {len(matches)} matching email(s):\n")
            for i, row in matches.iterrows(): 
                print(row['message'], '...\n')
                count_matches += 1

    print(f"Count matches: {count_matches}")

show_emails_for_anomalies(anomalies, day)
