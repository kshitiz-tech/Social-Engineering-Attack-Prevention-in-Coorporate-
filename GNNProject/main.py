import os
import ast
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.nn import SAGEConv

import community.community_louvain as community_louvain


# ---------------------------------------------------------------------------
# Data loading and graph construction
# ---------------------------------------------------------------------------

N_DAYS = 1448  # length of node time-series vectors


def load_graph(
    nodes_path: str = "nodes.csv",
    edges_path: str = "edges.csv",
) -> Tuple[nx.DiGraph, List[Tuple[int, np.ndarray]], List[Tuple[int, Dict[str, Any]]]]:
    """
    Load nodes/edges from CSV and build a NetworkX DiGraph with node features.

    Returns:
        Graph: directed graph with node attribute 'x' as a float32 vector of length N_DAYS
        nodes_list_vec: list of (node_id, vector) pairs
        nodes_list_raw: list of (node_id, original_dict) pairs
    """
    nodes_df = pd.read_csv(nodes_path, header=None, names=["node_attribute"], index_col=0)
    edges_df = pd.read_csv(edges_path)

    # Parse node attributes from string to dict
    nodes_list_raw: List[Tuple[int, Dict[str, Any]]] = [
        (int(index), ast.literal_eval(row.node_attribute))  # type: ignore[arg-type]
        for index, row in nodes_df.iterrows()
    ]

    # Vectorize node attributes into length-N_DAYS arrays
    nodes_list_vec: List[Tuple[int, np.ndarray]] = []
    for index, dict_node in nodes_list_raw:
        vec = np.zeros(N_DAYS, dtype=np.float32)
        for day, count in dict_node.items():
            vec[int(day)] = float(count)
        nodes_list_vec.append((index, vec))

    # Build graph
    G = nx.DiGraph()
    for node_id, node_attr_vec in nodes_list_vec:
        G.add_node(int(node_id), x=node_attr_vec)

    edges_list = [(int(row.From), int(row.To)) for _, row in edges_df.iterrows()]  # type: ignore[attr-defined]
    G.add_edges_from(edges_list)

    return G, nodes_list_vec, nodes_list_raw


# ---------------------------------------------------------------------------
# Day index / date helpers
# ---------------------------------------------------------------------------

BASE_DATE = pd.Timestamp("1999-01-01", tz="UTC")


def day_index_to_date(day_index: int, tz: str = "UTC") -> pd.Timestamp:
    di = int(day_index)
    dt = BASE_DATE + pd.Timedelta(days=di)
    return dt.tz_convert(tz) if tz and dt.tzinfo is not None and tz != "UTC" else dt


def format_human_date(ts) -> str:
    """
    Return dates like '4 May 2001' (no leading zero), Windows-safe.
    """
    ts = pd.Timestamp(ts)
    for fmt in ("%-d %B %Y", "%#d %B %Y", "%d %B %Y"):
        try:
            s = ts.strftime(fmt)
            if fmt == "%d %B %Y":
                s = s.lstrip("0")
            return s
        except ValueError:
            continue
    return ts.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# PyG conversion and GraphSAGE embeddings
# ---------------------------------------------------------------------------


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def build_pyg_data(G: nx.DiGraph, device: torch.device) -> Tuple[torch.Tensor, Any]:
    data = from_networkx(G)

    X = torch.stack([torch.tensor(attr["x"], dtype=torch.float32) for _, attr in G.nodes(data=True)])
    X = X.to(device)
    data.x = X
    data = data.to(device)
    return X, data


def get_or_train_embeddings(
    G: nx.DiGraph,
    embeddings_path: str = "final_embeddings.pt",
    hidden_channels: int = 128,
    out_channels: int = 64,
    num_layers: int = 3,
    dropout: float = 0.3,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 500,
) -> Tuple[torch.Tensor, torch.device]:
    """
    Load precomputed embeddings if available; otherwise train GraphSAGE and save.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(embeddings_path):
        z = torch.load(embeddings_path, map_location=device)
        return z, device

    X, data = build_pyg_data(G, device)
    in_channels = data.num_node_features

    model = GraphSAGE(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_one() -> torch.Tensor:
        model.train()
        opt.zero_grad()
        z_local = model(data.x, data.edge_index)

        pos = data.edge_index
        neg = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos.size(1),
        ).to(device)

        pos_loss = F.logsigmoid((z_local[pos[0]] * z_local[pos[1]]).sum(dim=1)).mean()
        neg_loss = F.logsigmoid(-(z_local[neg[0]] * z_local[neg[1]]).sum(dim=1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        opt.step()
        return z_local

    for _ in range(epochs):
        _ = train_one()

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    torch.save(z, embeddings_path)
    return z, device


# ---------------------------------------------------------------------------
# Phase 1, 2, 3 and aggregation
# ---------------------------------------------------------------------------


def sigmoid_norm(x: float, alpha: float = 0.5) -> float:
    return float(1.0 / (1.0 + np.exp(-alpha * x)))


def phase1(nodes_list_vec: List[Tuple[int, np.ndarray]], start_day: int, current_day: int) -> List[float]:
    phase1_score: List[float] = []
    for _, node_attr_vec in nodes_list_vec:
        mean = np.mean(node_attr_vec[start_day:current_day])
        std = np.std(node_attr_vec[start_day:current_day])
        today_score = (node_attr_vec[current_day] - mean) / std if std > 0 else 0.0
        phase1_score.append(sigmoid_norm(today_score, alpha=0.5))
    return phase1_score


def historical_pattern(
    z: torch.Tensor,
    current_day: int,
    weight_distribution: float,
    csv_file: str = "node_day_recipients.csv",
) -> List[List[Tuple[int, float]]]:
    """
    Phase 2: historical communication pattern + embedding similarity for each sender.

    Returns:
        phase2_score: list indexed by node_id, each entry is a list of (recipient, score)
    """
    df = pd.read_csv(csv_file)
    df["day_recipients_str"] = df["day_recipients_str"].apply(ast.literal_eval)

    max_node_id = int(df["node_id"].max())
    phase2_score: List[List[Tuple[int, float]]] = [[] for _ in range(max_node_id + 1)]

    for _, row in df.iterrows():
        node = int(row["node_id"])
        day_recipients = row["day_recipients_str"]

        if current_day >= len(day_recipients) or len(day_recipients[current_day]) == 0:
            continue

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

        day_scores: List[Tuple[int, float]] = []
        for j_idx, recipient in enumerate(recipients_today):
            if total_not[j_idx] == 0:
                historical_score = 1.0
            else:
                historical_score = total_yes[j_idx] / total_not[j_idx]

            try:
                node_embedding = z[node]
                recipient_embedding = z[recipient]
                cos = F.cosine_similarity(node_embedding.unsqueeze(0), recipient_embedding.unsqueeze(0)).item()
                cos = (cos + 1.0) / 2.0
                raw_score = weight_distribution * historical_score + (1.0 - weight_distribution) * (1.0 - cos)
            except Exception:
                raw_score = historical_score

            final_s = sigmoid_norm(raw_score, alpha=0.5)
            day_scores.append((int(recipient), final_s))

        phase2_score[node] = day_scores

    return phase2_score


def phase3_community_normalization(G: nx.DiGraph, phase2_score: List[List[Tuple[int, float]]]) -> List[float]:
    """
    Phase 3: community-based normalization of sender behavior.
    """
    undirected_graph = G.to_undirected()
    partition = community_louvain.best_partition(undirected_graph)

    max_node_id = max(G.nodes)
    phase3_score: List[float] = [0.0] * (max_node_id + 1)

    all_fraction_same: List[float] = []
    senders: List[int] = []

    for sender, rec_scores in enumerate(phase2_score):
        if not rec_scores:
            continue

        c_send = partition.get(sender, -1)
        if c_send == -1:
            continue

        recipients = [r for r, _ in rec_scores]
        same_comm = sum(1 for r in recipients if partition.get(r, -1) == c_send)
        total = len(recipients)
        frac = same_comm / total if total > 0 else 0.0
        all_fraction_same.append(frac)
        senders.append(sender)

    mean_fraction = np.mean(all_fraction_same) if all_fraction_same else 0.0

    for sender, frac in zip(senders, all_fraction_same):
        phase3_score[sender] = float(1.0 / (1.0 + np.exp(0.5 * (frac - mean_fraction))))

    return phase3_score


def aggregate_score(
    nodes_list_vec: List[Tuple[int, np.ndarray]],
    phase1_score: List[float],
    phase2_score: List[List[Tuple[int, float]]],
    phase3_score: List[float],
    w2: float = 0.6,
    w1: float = 0.2,
    w3: float = 0.2,
) -> List[List[Tuple[int, float]]]:
    """
    Aggregate Phase 1/2/3 into final scores per sender/recipient.
    """
    max_node_id = max(node_id for node_id, _ in nodes_list_vec)
    final_score: List[List[Tuple[int, float]]] = [[] for _ in range(max_node_id + 1)]

    for node_id in range(max_node_id + 1):
        entry = phase2_score[node_id] if node_id < len(phase2_score) else []
        if not entry:
            continue

        p1_score = phase1_score[node_id] if node_id < len(phase1_score) else 0.0
        p3_score = phase3_score[node_id] if node_id < len(phase3_score) and phase3_score[node_id] > 0 else 1.0

        for recipient, score in entry:
            final_score_node = w2 * score + w1 * p1_score + w3 * p3_score
            final_score[node_id].append((int(recipient), float(final_score_node)))

    return final_score


# ---------------------------------------------------------------------------
# High-level SEAMain wrapper
# ---------------------------------------------------------------------------


class SEAMain:
    def __init__(self, G: nx.DiGraph, nodes_list_vec: List[Tuple[int, np.ndarray]], z: torch.Tensor, hist_weight: float = 0.3,
                 w2: float = 0.6, w1: float = 0.2, w3: float = 0.2):
        self.G = G
        self.nodes_list_vec = nodes_list_vec
        self.z = z
        self.hist_weight = hist_weight
        self.w2 = w2
        self.w1 = w1
        self.w3 = w3

    def run(self, day: int) -> List[List[Tuple[int, float]]]:
        p1 = phase1(self.nodes_list_vec, start_day=0, current_day=day)
        p2 = historical_pattern(self.z, current_day=day, weight_distribution=self.hist_weight)
        p3 = phase3_community_normalization(self.G, p2)
        final_scores = aggregate_score(self.nodes_list_vec, p1, p2, p3, w2=self.w2, w1=self.w1, w3=self.w3)
        return final_scores


# ---------------------------------------------------------------------------
# Anomaly detection and email lookup
# ---------------------------------------------------------------------------


def find_anomalous_nodes(final_scores: List[List[Tuple[int, float]]], threshold: float = 0.60) -> List[Dict[int, Tuple[int, float]]]:
    anomalous_nodes: List[Dict[int, Tuple[int, float]]] = []
    for node_id, result in enumerate(final_scores):
        for i in result:
            if len(i) >= 2 and i[1] > threshold:
                anomalous_nodes.append({node_id: i})
    return anomalous_nodes


def build_anomalies_dataframe(
    anomalous_nodes: List[Dict[int, Tuple[int, float]]],
    day_index: int,
    id_email_path: str = "id-email.csv",
) -> pd.DataFrame:
    # Flatten to rows
    rows: List[Dict[str, Any]] = []
    for entry in anomalous_nodes:
        if isinstance(entry, dict):
            for sender_id, tup in entry.items():
                if isinstance(tup, tuple) and len(tup) >= 2:
                    receiver_id, score = tup[0], tup[1]
                    rows.append(
                        {
                            "day_index": day_index,
                            "sender_id": int(sender_id),
                            "receiver_id": int(receiver_id),
                            "score": float(score),
                        }
                    )

    anomalies_df = pd.DataFrame(rows)
    if anomalies_df.empty:
        return anomalies_df

    # Add date columns
    anomalies_df["target_date"] = anomalies_df["day_index"].apply(day_index_to_date)
    anomalies_df["date_str"] = anomalies_df["target_date"].apply(format_human_date)

    # Map IDs to emails
    id_email_df = pd.read_csv(id_email_path, header=None)
    id_email_df.columns = ["id", "email"]
    id_to_email = dict(zip(id_email_df["id"], id_email_df["email"]))

    anomalies_df["sender_email"] = anomalies_df["sender_id"].map(id_to_email)
    anomalies_df["receiver_email"] = anomalies_df["receiver_id"].map(id_to_email)

    return anomalies_df


def find_first_message(
    emails_df: pd.DataFrame,
    date_str: str,
    sender_email: str | None = None,
    receiver_email: str | None = None,
) -> str | None:
    """
    Find first message in the Enron emails that matches date (+ optional sender/receiver).
    Uses a leading space to avoid '4 May' vs '14 May' collisions.
    """
    needle = " " + date_str
    subset = emails_df[emails_df["message"].str.contains(needle, na=False)]
    if sender_email:
        subset = subset[subset["message"].str.contains(sender_email, na=False)]
    if receiver_email:
        subset = subset[subset["message"].str.contains(receiver_email, na=False)]
    if not subset.empty:
        return subset.iloc[0]["message"]
    return None


def attach_email_messages(
    anomalies_df: pd.DataFrame,
    emails_path: str | None = None,
) -> pd.DataFrame:
    if anomalies_df.empty:
        return anomalies_df

    if emails_path is None:
        if os.path.exists("modified-emails.csv"):
            emails_path = "modified-emails.csv"
        elif os.path.exists("emails.csv"):
            emails_path = "emails.csv"
        else:
            return anomalies_df

    emails_df = pd.read_csv(emails_path)
    if list(emails_df.columns) == ["file", "message"]:
        pass
    else:
        # try to coerce
        emails_df.columns = ["file", "message"]

    messages: List[str] = []
    for _, row in anomalies_df.iterrows():
        msg = find_first_message(
            emails_df,
            row["date_str"],
            row.get("sender_email"),
            row.get("receiver_email"),
        )
        messages.append(msg if isinstance(msg, str) else "")

    anomalies_df["first_message"] = messages
    return anomalies_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(day: int = 854, threshold: float = 0.60) -> None:
    print(f"Loading graph...")
    G, nodes_list_vec, _ = load_graph()

    print("Getting / training embeddings...")
    z, device = get_or_train_embeddings(G)
    print(f"Embeddings on device: {device}")

    print(f"Running SEAMain pipeline for day {day}...")
    app = SEAMain(G, nodes_list_vec, z)
    final_scores = app.run(day=day)

    anomalous_nodes = find_anomalous_nodes(final_scores, threshold=threshold)
    print(f"Day {day}: Detected {len(anomalous_nodes)} anomalous communications")

    anomalies_df = build_anomalies_dataframe(anomalous_nodes, day_index=day)
    anomalies_df = attach_email_messages(anomalies_df)

    if not anomalies_df.empty:
        out_path = f"anomalies_summary_day{day}.csv"
        anomalies_df.to_csv(out_path, index=False)
        print(f"Saved anomalies to {out_path}")
        print(anomalies_df[["day_index", "date_str", "sender_id", "receiver_id", "score"]].head(10))
    else:
        print("No anomalies found.")


if __name__ == "__main__":
    # Simple entry point; for more control, call main(day=..., threshold=...) from another script.
    main()


