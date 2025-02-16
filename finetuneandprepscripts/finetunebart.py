#code for finetuning bart
import os
import torch
from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Paths
train_judgment_folder = r"train-data\judgement"
train_summary_folder = r"train-data\summary"
test_judgment_folder = r"test-data\judgement"
test_summary_folder = r"test-data\summary"

# Function to load judgment-summary pairs from a folder structure
def load_data(judgment_folder, summary_folder):
    judgments = []
    summaries = []
    
    # Get the list of all judgment files
    judgment_files = sorted(os.listdir(judgment_folder))
    
    for filename in judgment_files:
        # Skip files without corresponding summary
        summary_filename = filename.split('.')[0] + '.txt'  # Assuming they have .txt extension
        if summary_filename not in os.listdir(summary_folder):
            continue
        
        # Read the judgment file
        with open(os.path.join(judgment_folder, filename), 'r', encoding='utf-8') as f:
            judgment_text = f.read().strip()
        
        # Read the summary file
        with open(os.path.join(summary_folder, summary_filename), 'r', encoding='utf-8') as f:
            summary_text = f.read().strip()
        
        judgments.append(judgment_text)
        summaries.append(summary_text)
    
    return judgments, summaries

# Load training data
train_judgments, train_summaries = load_data(train_judgment_folder, train_summary_folder)

# Load testing data
test_judgments, test_summaries = load_data(test_judgment_folder, test_summary_folder)

# Split the train data into train and validation sets (80% train, 20% validation)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_judgments, train_summaries, test_size=0.2, random_state=42)

# Initialize the tokenizer and model
model_name = "facebook/bart-large-cnn"  # You can use a smaller model like bart-base if needed
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    model_inputs = tokenizer(examples['judgment'], max_length=1024, truncation=True, padding="max_length")
    # Tokenize the summaries
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['summary'], max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Create a dataset from the loaded text pairs
train_dataset = Dataset.from_dict({"judgment": train_texts, "summary": train_labels})
val_dataset = Dataset.from_dict({"judgment": val_texts, "summary": val_labels})
test_dataset = Dataset.from_dict({"judgment": test_judgments, "summary": test_summaries})

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",  # evaluate at the end of every epoch
    save_strategy="epoch"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=None,  # Use default collator for BART
)

# Fine-tuning the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./bart-finetuned")
tokenizer.save_pretrained("./bart-finetuned")

print("Fine-tuning completed and model saved.")
