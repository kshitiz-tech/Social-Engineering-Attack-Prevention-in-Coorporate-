import pandas as pd

# Load CSV
df = pd.read_csv("ablation_detailed_log.csv")

# Columns to evaluate
phase_cols = [
    "Phase 1 Only", "Phase 2 Only", "Phase 3 Only",
    "Phase 1+2", "Phase 1+3", "Phase 2+3"
]

# Set threshold to 0.70 (to match combined result of 28.8%)
threshold = 0.55

# User specified total samples
total_samples = 7913

print(f"Filter Load (%)':<15")
print("-" * 55)

for col in phase_cols:
    # 1. Filter Load Calculation: (Flagged / 626) * 100
    flagged_count = (df[col] >= threshold).sum()
    filter_load = (flagged_count / total_samples) * 100
    

    print(f"{col:<20} |  {filter_load:.3f}%")