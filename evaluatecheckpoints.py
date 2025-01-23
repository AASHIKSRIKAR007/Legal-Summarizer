import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# Function to load and preprocess the dataset
def load_classification_dataset(test_data, tokenizer, label_to_id):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    test_dataset = Dataset.from_list(test_data)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # Convert labels to IDs
    test_dataset = test_dataset.map(lambda e: {"labels": label_to_id[e["label"]]})

    # Remove unnecessary columns
    test_dataset = test_dataset.remove_columns(["text", "label"])

    return test_dataset

# Function to compute metrics
def compute_metrics(pred):
    predictions, labels = pred
    predictions = predictions.argmax(axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1_score": f1}

# Paths
test_path = "test_dataset.json"
checkpoint_path = "results"  # Replace with the path to the checkpoint

# Load test dataset
with open(test_path, "r", encoding="utf-8") as file:
    test_data = json.load(file)

# Labels
unique_labels = sorted(set([entry["label"] for entry in test_data]))
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# Tokenizer and model
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the checkpointed model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=len(unique_labels))

# Prepare the test dataset
test_dataset = load_classification_dataset(test_data, tokenizer, label_to_id)

# Trainer setup for evaluation
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=16,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate the model on the test dataset
results = trainer.evaluate()

# Print the evaluation results
print("Evaluation results:", results)
