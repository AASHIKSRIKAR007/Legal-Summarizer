import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import json
import re

# Load preprocessed data
with open("./data/preprocessed_legbert_dataset.json", "r") as f:
    case_data = json.load(f)

# Build classification dataset
classification_data = []
section_to_idx = {"facts":0, "judgment":1, "analysis":2, "argument":3, "statute":4}

for case_id, data in case_data.items():
    for annotator in ["summaries_A", "summaries_B"]:
        for section, summary in data[annotator].items():
            if not summary:
                continue
            section_idx = section_to_idx[section]
            for sent in re.split(r'[.!?]', summary):
                if sent.strip():
                    classification_data.append({
                        "text": sent.strip(),
                        "labels": [1 if i == section_idx else 0 for i in range(5)]
                    })

# Split data
train_cls, val_cls = train_test_split(classification_data, test_size=0.2, random_state=42)

# Dataset class
class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    def __len__(self):
        return len(self.labels)

# Tokenize
tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
train_texts = [d["text"] for d in train_cls]
train_labels = [d["labels"] for d in train_cls]
val_texts = [d["text"] for d in val_cls]
val_labels = [d["labels"] for d in val_cls]

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

train_dataset = ClassificationDataset(train_encodings, train_labels)
val_dataset = ClassificationDataset(val_encodings, val_labels)

# Model setup
model = BertForSequenceClassification.from_pretrained(
    "nlpaueb/legal-bert-base-uncased",
    num_labels=5,
    problem_type="multi_label_classification"
)

# Freeze layers for CPU efficiency
for param in model.bert.encoder.layer[:8].parameters():
    param.requires_grad = False

# Training args (optimized for CPU)
training_args = TrainingArguments(
    output_dir="./classification_model",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
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
trainer.save_model("./classification_model/final")
print("Model saved")