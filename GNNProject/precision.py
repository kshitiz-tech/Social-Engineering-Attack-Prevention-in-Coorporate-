import pandas as pd

# Load CSV
df = pd.read_csv("ablation_detailed_log.csv")

# Define actual positives: Sender >= 6600
actual_positive = df["Sender"] >= 6600

# Columns to evaluate
phase_cols = [
    "S1", "S2", "S3",
    "Phase 1 Only", "Phase 2 Only", "Phase 3 Only",
    "Phase 1+2", "Phase 1+3", "Phase 2+3"
]

threshold = 0.55

precisions = {}

for col in phase_cols:
    # Predicted positive
    predicted_positive = df[col] >= threshold

    TP = (predicted_positive & actual_positive).sum()
    FP = (predicted_positive & ~actual_positive).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    precisions[col] = precision

# Print results
for phase, prec in precisions.items():
    print(f"{phase}: Precision = {prec * 1000:.4f}")
