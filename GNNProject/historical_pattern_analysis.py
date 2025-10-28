# Social Engineering Detection - Historical Pattern Analysis
# Corrected version of the historical_pattern function

import pandas as pd 
import ast
import torch
import torch.nn.functional as F
from collections import defaultdict

def historical_pattern(current_day=220, weight_distribution=0.3, embeddings_file='final_embeddings.pt'):
    """
    Analyze historical communication patterns and combine with cosine similarity.
    
    Args:
        current_day: The day to analyze (default: 220)
        weight_distribution: Weight for historical pattern vs cosine similarity (default: 0.3)
        embeddings_file: Path to the embeddings file
        
    Returns:
        Dictionary of phase 2 scores by node
    """
    # Load embeddings
    z = torch.load(embeddings_file, map_location='cpu')
    
    csv_file = 'node_day_recipients.csv'
    df = pd.read_csv(csv_file)
    df["day_recipients_str"] = df["day_recipients_str"].apply(ast.literal_eval)

    # Initialize phase2_score as a dictionary
    phase2_score = {}
    
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
                
            # Store in phase2_score dictionary
            print(f"Processing node {node}")
            phase2_score[node] = {current_day: day_scores}
    
    return phase2_score

def get_phase2_nodes(phase2_score):
    """
    Get list of nodes that have Phase 2 scores calculated.
    
    Args:
        phase2_score: Dictionary of phase 2 scores
        
    Returns:
        List of node IDs with phase 2 data
    """
    phase2_nodes = []
    for node_id in phase2_score.keys():
        if phase2_score[node_id]:  # Only nodes that have data
            phase2_nodes.append(node_id)            
    return phase2_nodes

# Example usage
if __name__ == "__main__":
    # Analyze day 930
    print("Analyzing day 930...")
    phase2_results = historical_pattern(current_day=930, weight_distribution=0.3)
    
    # Get nodes with phase 2 data
    phase2_nodes = get_phase2_nodes(phase2_results)
    
    print(f"\nFound {len(phase2_nodes)} nodes with Phase 2 scores")
    print("Sample results:")
    
    # Show first 5 nodes
    for i, node_id in enumerate(phase2_nodes[:5]):
        if node_id in phase2_results:
            print(f"Node {node_id} - Phase 2 Score: {phase2_results[node_id]}")
    
    print(f"\nTotal nodes processed: {len(phase2_results)}")
