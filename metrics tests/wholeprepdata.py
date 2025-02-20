import os
import json
import warnings
from transformers import BartForConditionalGeneration, BartTokenizer

# ----------------------- CONFIGURATION -----------------------
# Folder containing preprocessed case files for whole summary inference
CASE_FILES_DIR = "data/data_metric/whole_case"  # e.g., "./data/whole_cases/"
# Folder containing corresponding reference summaries (plain text)
REFERENCE_SUMMARY_DIR = "data/data_metric/whole_summ"  # e.g., "./data/whole_refs/"

# Output JSON file paths
CANDIDATE_OUTPUT_JSON = "data/data_metric/try7LENmod3whole_candidate.json"
REFERENCE_OUTPUT_JSON = "data/data_metric/try7LENmod3whole_reference.json"

# Fine-tuned BART model directory (for whole summary)
MODEL_PATH = "bart-finetuned"  # adjust as needed
PREFIX = "summarize:"  # set to "" if no prefix is required
# -------------------------------------------------------------

def load_model(model_path):
    """Load the fine-tuned BART model and tokenizer."""
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model

def read_text_file(file_path):
    """Read and clean the text content of a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        # Remove excessive newlines
        text = text.replace('\n', ' ').strip()  # Replace newline characters with space
        return text

def generate_summary(model, tokenizer, input_text, max_input_length=1024, 
                     max_output_length=1024, min_output_length=50):
    """Generate a summary from input text using the model."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_input_length)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generated_ids = model.generate(
            inputs["input_ids"],
            max_length=max_output_length,
            min_length=min_output_length,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return summary

def process_whole_summaries():
    tokenizer, model = load_model(MODEL_PATH)
    candidate_summaries = {}
    reference_summaries = {}

    i = 0   
    # Process candidate (generated) summaries
    for filename in sorted(os.listdir(CASE_FILES_DIR)):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(CASE_FILES_DIR, filename)
        input_text = read_text_file(file_path)
        # Optionally prepend a prefix to boost performance
        if PREFIX:
            input_text = f"{PREFIX} {input_text}"
        summary = generate_summary(model, tokenizer, input_text)
        candidate_summaries[filename] = summary
        i = i + 1
        if i%4==0 or i==25:
            print(i)

    # Save candidate summaries to JSON
    with open(CANDIDATE_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(candidate_summaries, f, indent=2, ensure_ascii=False)
    print(f"Candidate summaries saved to {CANDIDATE_OUTPUT_JSON}")

    # Process reference summaries from reference folder
    for filename in sorted(os.listdir(REFERENCE_SUMMARY_DIR)):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(REFERENCE_SUMMARY_DIR, filename)
        ref_text = read_text_file(file_path)
        reference_summaries[filename] = ref_text
        # print(f"Processed reference for {filename}")

    # Save reference summaries to JSON
    with open(REFERENCE_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(reference_summaries, f, indent=2, ensure_ascii=False)
    print(f"Reference summaries saved to {REFERENCE_OUTPUT_JSON}")

if __name__ == "__main__":
    process_whole_summaries()
