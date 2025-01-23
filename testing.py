import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Specify the path to your saved model
checkpoint_path = "legalbert-finetuned"

# List of required files for the model to be loaded correctly
required_files = [
    "config.json", 
    "pytorch_model.bin", 
    "tokenizer_config.json",
    "vocab.txt"  # Or merges.txt if using a BPE-based tokenizer (like GPT)
]

# Check if the necessary files are present in the checkpoint directory
missing_files = [file for file in required_files if not os.path.isfile(os.path.join(checkpoint_path, file))]

if missing_files:
    print(f"Missing files: {', '.join(missing_files)}")
else:
    print("All required files are present. Attempting to load the model...")

    try:
        # Try loading the model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print("Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error while loading model: {e}")
