#code for finetuning extractive model of legalbert

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import json
import re

# Load preprocessed data
with open("./data/preprocessed_legbert_dataset.json", "r") as f:
    case_data = json.load(f)

# Build extractive dataset
extractive_data = []
for case_id, data in case_data.items():
    raw_sentences = [s.strip() for s in re.split(r'[.!?]', data["raw_text"]) if s.strip()]
    summary_sentences = set()
    for annotator in ["summaries_A", "summaries_B"]:
        for section in data[annotator].values():
            if section:
                summary_sentences.update([s.strip().lower() for s in re.split(r'[.!?]', section)])
    labels = [1 if sent.lower() in summary_sentences else 0 for sent in raw_sentences]
    extractive_data.extend([{"text": sent, "label": label} for sent, label in zip(raw_sentences, labels)])

# Split data
train_data, val_data = train_test_split(extractive_data, test_size=0.2, random_state=42)

# Dataset class
class ExtractiveDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

# Tokenize
tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
train_texts = [d["text"] for d in train_data]
train_labels = [d["label"] for d in train_data]
val_texts = [d["text"] for d in val_data]
val_labels = [d["label"] for d in val_data]

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

train_dataset = ExtractiveDataset(train_encodings, train_labels)
val_dataset = ExtractiveDataset(val_encodings, val_labels)

# Model setup
model = BertForSequenceClassification.from_pretrained(
    "nlpaueb/legal-bert-base-uncased",
    num_labels=2  # Binary classification
)

# Freeze layers for CPU efficiency
for param in model.bert.encoder.layer[:6].parameters():
    param.requires_grad = False

# Training args (optimized for CPU)
training_args = TrainingArguments(
    output_dir="./extractive_model",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-5,
    no_cuda=True  # Force CPU
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()
trainer.save_model("./extractive_model/final")