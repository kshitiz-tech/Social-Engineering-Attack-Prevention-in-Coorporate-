import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import torch

# 1️⃣ Load CSV
df = pd.read_csv("bert_baseline_labeled.csv")  # must have 'message' and 'label'

# 2️⃣ Split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
datasets = DatasetDict({"train": train_dataset, "test": test_dataset})

# 3️⃣ Load tokenizer and model
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4️⃣ Tokenize
def tokenize(batch):
    return tokenizer(batch['message'], truncation=True, padding='max_length', max_length=512)

datasets = datasets.map(tokenize, batched=True)
datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 5️⃣ Define metrics for Trainer (still required but we’ll compute full report separately) 
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {}

# 6️⃣ Training arguments
# 6️⃣ Training arguments - Optimized for 8GB VRAM
training_args = TrainingArguments(
    output_dir="./bert_phishing",
    eval_strategy="epoch",
    learning_rate=2e-5,
    
    # --- Memory Optimizations ---
    per_device_train_batch_size=2,       # Small physical batch to fit in VRAM
    gradient_accumulation_steps=4,      # 2 * 4 = Effective batch size of 8
    gradient_checkpointing=True,        # Massive VRAM savings
    fp16=True,                          # Use mixed precision (if using NVIDIA GPU)
    # ----------------------------

    per_device_eval_batch_size=4,       # Eval doesn't need gradients, so 4 usually fits
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="no",
    logging_steps=10,
    # ModernBERT optimization
    dataloader_num_workers=0,           # Set to 0 if on Windows to avoid multiprocess bugs
)

# 7️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets['train'],
    eval_dataset=datasets['test'],
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# 8️⃣ Train
trainer.train()

# 9️⃣ Evaluate
print("Generating predictions on test set...")
predictions_output = trainer.predict(datasets['test'])
preds = np.argmax(predictions_output.predictions, axis=1)
labels = predictions_output.label_ids

#  🔹 Classification report
print("--- Classification Report ---")
print(classification_report(labels, preds))

# 🔹 Confusion matrix
print("--- Confusion Matrix ---")
print(confusion_matrix(labels, preds))
