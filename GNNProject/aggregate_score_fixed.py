# Fixed aggregate_score function
# Corrected version to handle the scoring aggregation properly

import numpy as np

def aggregate_score(phase1_score, phase2_score, phase3_score, nodes_list):
    """
    Aggregate scores from all three phases.
    
    Args:
        phase1_score: List of Phase 1 scores
        phase2_score: Dictionary of Phase 2 scores by node
        phase3_score: List of Phase 3 scores
        nodes_list: List of nodes
        
    Returns:
        Dictionary of aggregated scores by node
    """
    aggregated_scores = {}
    
    for i in range(len(nodes_list)):
        if i in phase2_score and phase2_score[i]:  # Check if node has phase2 data
            # Get the first phase2 score for this node
            if isinstance(phase2_score[i], dict):
                # If phase2_score[i] is a dictionary with day keys
                first_day = list(phase2_score[i].keys())[0]
                first_recipient_score = phase2_score[i][first_day][0]  # First recipient's score
                phase2_value = first_recipient_score[1]  # The actual score value
            else:
                # If phase2_score[i] is a list
                phase2_value = phase2_score[i][0][1]  # First recipient's score
            
            print(f"Node {i} - Phase2 value: {phase2_value}")
            
            # Get phase1 score for this node
            phase1_value = phase1_score[i] if i < len(phase1_score) else 0.0
            
            # Apply threshold to phase1 score
            if phase1_value < 0.5:
                phase1_value = 0.5
            
            # Normalize phase1 score
            phase1_value = phase1_value - 0.5
            
            # Get phase3 score for this node
            phase3_value = phase3_score[i] if i < len(phase3_score) else 0
            
            print(f"Node {i} - Phase1: {phase1_value}, Phase3: {phase3_value}")
            
            # Store aggregated information
            aggregated_scores[i] = {
                'phase1_score': phase1_value,
                'phase2_score': phase2_value,
                'phase3_score': phase3_value,
                'node_id': i
            }
    
    return aggregated_scores

# Alternative simpler version if you want to fix the original function
def aggregate_score_simple():
    """
    Simplified version that fixes the original function's issues.
    """
    global phase1_score, phase2_score, phase3_score, nodes_list
    
    for i in range(len(nodes_list)):
        if i in phase2_score and phase2_score[i]:
            # Get phase2 score (first recipient's score)
            if isinstance(phase2_score[i], dict):
                first_day = list(phase2_score[i].keys())[0]
                phase2_value = phase2_score[i][first_day][0][1]
            else:
                phase2_value = phase2_score[i][0][1]
            
            print(f"Node {i} - Phase2 value: {phase2_value}")
            
            # Get phase1 score (use node index, not the score value as index)
            phase1_value = phase1_score[i] if i < len(phase1_score) else 0.0
            
            # Apply your logic
            if phase1_value < 0.5:
                phase1_value = 0.5
            
            phase1_value = phase1_value - 0.5
            
            # Get phase3 score (use node index, not the score value as index)
            phase3_value = phase3_score[i] if i < len(phase3_score) else 0
            
            print(f"Node {i} - Phase1: {phase1_value}, Phase3: {phase3_value}")

# Example usage with proper data structure
def example_usage():
    """
    Example of how to use the corrected function.
    """
    # Assuming you have these variables defined:
    # phase1_score = [list of scores]
    # phase2_score = {node_id: {day: [(recipient, score), ...]}}
    # phase3_score = [list of scores]
    # nodes_list = [list of nodes]
    
    # Call the function
    results = aggregate_score(phase1_score, phase2_score, phase3_score, nodes_list)
    
    # Print results
    for node_id, scores in results.items():
        print(f"Node {node_id}: Phase1={scores['phase1_score']:.3f}, "
              f"Phase2={scores['phase2_score']:.3f}, Phase3={scores['phase3_score']}")

if __name__ == "__main__":
    print("Fixed aggregate_score function ready to use!")
    print("Key fixes:")
    print("1. Use node index 'i' instead of score value as index")
    print("2. Properly access phase2_score structure")
    print("3. Handle different phase2_score formats")
    print("4. Add bounds checking for all score lists")
