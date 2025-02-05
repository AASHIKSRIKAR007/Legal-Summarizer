# code for testing the fientuned bart model output is in outputs folder as test_summ_bart.txt

import torch
import warnings
from transformers import BartForConditionalGeneration, BartTokenizer

# Paths
model_path = r"bart-finetuned2"
input_file_path = "processed.txt"

# Load resources
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# Read input
with open(input_file_path, "r", encoding="utf-8") as file:
    input_text = file.read().strip()

# Add training prefix if needed
# input_text = "summarize: " + input_text

# Tokenize
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)

# Generate with suppressed warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    generated_ids = model.generate(
        inputs["input_ids"],
        max_length=256,
        min_length=50,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

# Decode
generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\nGenerated Summary pre:\n", generated_summary)