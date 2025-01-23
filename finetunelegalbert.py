import json
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Function to split the dataset
def split_dataset(input_file, train_file, test_file, test_size=0.2, seed=42):
    random.seed(seed)
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    with open(train_file, "w", encoding="utf-8") as file:
        json.dump(train_data, file, indent=4, ensure_ascii=False)
    with open(test_file, "w", encoding="utf-8") as file:
        json.dump(test_data, file, indent=4, ensure_ascii=False)
    return train_data, test_data

# Function to compute metrics
def compute_metrics(pred):
    predictions, labels = pred
    predictions = predictions.argmax(axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1_score": f1}

# Load and preprocess the dataset
def load_classification_dataset(train_data, test_data, tokenizer, label_to_id):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # Convert labels to IDs
    train_dataset = train_dataset.map(lambda e: {"labels": label_to_id[e["label"]]})
    test_dataset = test_dataset.map(lambda e: {"labels": label_to_id[e["label"]]})

    # Remove unnecessary columns
    train_dataset = train_dataset.remove_columns(["text", "label"])
    test_dataset = test_dataset.remove_columns(["text", "label"])

    return DatasetDict({"train": train_dataset, "test": test_dataset})

# Paths
input_path = "final_dataset.json"
train_path = "train_dataset.json"
test_path = "test_dataset.json"

# Split dataset
train_data, test_data = split_dataset(input_path, train_path, test_path)

# Labels
unique_labels = sorted(set([entry["label"] for entry in train_data]))
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# Tokenizer and model
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(unique_labels))

# Prepare datasets
datasets = load_classification_dataset(train_data, test_data, tokenizer, label_to_id)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",  # Correct parameter
    save_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./legalbert-finetuned")
tokenizer.save_pretrained("./legalbert-finetuned")

print("Fine-tuning completed and model saved.")
