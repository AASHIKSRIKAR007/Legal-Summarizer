#code to evaluate finetuned BART against baseline model, 
#output of this code are baselinesummaries.txt, fine_tuned_summaries.txt,(in outputs folder) and 
# reslut is in evaluatebart resul.txt(in tests folder)

import os
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import Dataset
from evaluate import load
import warnings  # Added for warning suppression

# Paths
test_judgment_folder = r"test-data/judgement"
test_summary_folder = r"test-data/summary"

# Function to load test data
def load_data(judgment_folder, summary_folder):
    judgments = []
    summaries = []
    
    judgment_files = sorted(os.listdir(judgment_folder))
    
    for filename in judgment_files:
        summary_filename = filename.split('.')[0] + '.txt'  # Assuming .txt extension
        if summary_filename not in os.listdir(summary_folder):
            continue
        
        with open(os.path.join(judgment_folder, filename), 'r', encoding='utf-8') as f:
            judgment_text = f.read().strip()
        
        with open(os.path.join(summary_folder, summary_filename), 'r', encoding='utf-8') as f:
            summary_text = f.read().strip()
        
        judgments.append(judgment_text)
        summaries.append(summary_text)
    
    return judgments, summaries

# Load test data
test_judgments, test_summaries = load_data(test_judgment_folder, test_summary_folder)

# Initialize the ROUGE evaluator
rouge = load("rouge")

# MODIFIED: Updated generation function
def generate_summaries(judgments, model, tokenizer):
    summaries = []
    for judgment in judgments:
        # Add training prefix if needed
        # formatted_text = "summarize: " + judgment
        
        inputs = tokenizer(
            judgment,  # or formatted_text if using prefix
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        
        with warnings.catch_warnings():  # Suppress generation warnings
            warnings.simplefilter("ignore")
            outputs = model.generate(
                inputs["input_ids"],
                max_length=256,
                min_length=50,  # CRUCIAL: Prevent empty outputs
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            
        summaries.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return summaries

# 1. Baseline Model Evaluation
print("Generating summaries using the baseline model...")
baseline_model_name = "facebook/bart-large-cnn"
baseline_tokenizer = BartTokenizer.from_pretrained(baseline_model_name)
baseline_model = BartForConditionalGeneration.from_pretrained(baseline_model_name)

baseline_summaries = generate_summaries(test_judgments, baseline_model, baseline_tokenizer)

# Save baseline summaries
with open("baseline_summaries2.txt", "w", encoding="utf-8") as f:
    for i, summary in enumerate(baseline_summaries):
        f.write(f"Baseline Test Case {i + 1}:\n{summary}\n\n")

# Compute ROUGE for baseline model
baseline_rouge_results = rouge.compute(predictions=baseline_summaries, references=test_summaries)
print("Baseline Model ROUGE Results:", baseline_rouge_results)

# 2. Fine-Tuned Model Evaluation (MODIFIED)
print("Generating summaries using the fine-tuned model...")
fine_tuned_model_path = "./bart-finetuned2"
fine_tuned_tokenizer = BartTokenizer.from_pretrained(fine_tuned_model_path)
fine_tuned_model = BartForConditionalGeneration.from_pretrained(fine_tuned_model_path)
fine_tuned_summaries = generate_summaries(test_judgments, fine_tuned_model, fine_tuned_tokenizer)

# Verify BOS token config (optional)
print(f"Fine-tuned model bos_token: {fine_tuned_tokenizer.bos_token_id}")  # Should be 0

# Generate summaries with proper parameters
fine_tuned_summaries = generate_summaries(test_judgments, fine_tuned_model, fine_tuned_tokenizer)

# Save fine-tuned summaries
with open("fine_tuned_summaries2.txt", "w", encoding="utf-8") as f:
    for i, summary in enumerate(fine_tuned_summaries):
        f.write(f"Fine-Tuned Test Case {i + 1}:\n{summary}\n\n")

# Compute ROUGE for fine-tuned model
fine_tuned_rouge_results = rouge.compute(predictions=fine_tuned_summaries, references=test_summaries)
print("Fine-Tuned Model ROUGE Results:", fine_tuned_rouge_results)

# 3. Compare Results
print("\nComparison of ROUGE Results:")
print(f"Baseline Model ROUGE-L: {baseline_rouge_results['rougeL']}")
print(f"Fine-Tuned Model ROUGE-L: {fine_tuned_rouge_results['rougeL']}")
