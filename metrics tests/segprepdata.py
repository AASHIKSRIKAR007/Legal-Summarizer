import os
import re
import json
import warnings
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
import chardet

# ----------------------- CONFIGURATION -----------------------
# Folder containing preprocessed case files for segmented summary inference
SEG_CASE_FILES_DIR = "data/data_metric/segmented_case"  # e.g., "./data/seg_cases/"
# Folder containing segmented reference summaries (plain text files)
# Each file should contain segments labeled like:
# FACTS
# <text>
# JUDGMENT
# <text>
# ANALYSIS
# <text>
# ARGUMENT
# <text>
# STATUTE
# <text>
REFERENCE_SEG_DIR = "data/data_metric/segmented_summ"  # e.g., "./data/seg_refs/"

# Output JSON file paths
CANDIDATE_SEG_OUTPUT_JSON = "data/data_metric/segmented_candidate.json"
REFERENCE_SEG_OUTPUT_JSON = "data/data_metric/segmented_reference.json"

# Models for segmented summary inference:
# LegalBERT extraction & classification models (assumed to be fine-tuned)
EXTRACTIVE_MODEL_PATH = "./extractive_model/final"
CLASSIFICATION_MODEL_PATH = "./classification_model/final"
LEGAL_BERT_TOKENIZER = "nlpaueb/legal-bert-base-uncased"

# Generic BART model (for abstractive summarization)
BART_MODEL_NAME = "facebook/bart-large-cnn"

# Define segments (use consistent labels as per your finetuning)
# Note: In training the classifier, segments were: facts, judgment, analysis, argument, statute.
SEGMENTS = ["facts", "judgment", "analysis", "argument", "statute"]
# -------------------------------------------------------------

def detect_encoding(file_path):
    """Detect file encoding using chardet."""
    with open(file_path, 'rb') as f:
        return chardet.detect(f.read())['encoding']

def clean_text(text):
    """Clean and normalize whitespace in text."""
    # Remove excessive newlines and tabs, replace with a single space
    text = re.sub(r'[\n\t]+', ' ', text)
    # Normalize multiple spaces to single space
    return re.sub(r'\s{2,}', ' ', text).strip()

def split_sentences(text):
    """Simple sentence splitter based on punctuation."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# ----------------------- Model Loading -----------------------
def load_segmentation_models():
    # Load LegalBERT extraction model (for sentence selection)
    extractive_model = pipeline(
        "text-classification",
        model=EXTRACTIVE_MODEL_PATH,
        tokenizer=LEGAL_BERT_TOKENIZER,
        device=-1  # CPU
    )
    # Load LegalBERT classification model (for segment labeling)
    classification_model = pipeline(
        "text-classification",
        model=CLASSIFICATION_MODEL_PATH,
        tokenizer=LEGAL_BERT_TOKENIZER,
        device=-1,
        function_to_apply="sigmoid"
    )
    # Load generic BART for abstractive summarization
    bart_tokenizer = BartTokenizer.from_pretrained(BART_MODEL_NAME)
    bart_model = BartForConditionalGeneration.from_pretrained(BART_MODEL_NAME).cpu()
    return extractive_model, classification_model, bart_tokenizer, bart_model

# ----------------------- Inference Pipeline -----------------------
def generate_segmented_summary(text, extractive_model, classification_model, bart_tokenizer, bart_model):
    """
    Given raw text, perform extraction & classification,
    then generate an abstractive summary for each segment.
    Returns a dict with keys as segments and values as generated summaries.
    """
    sentences = split_sentences(clean_text(text))  # Clean the text here before splitting sentences
    # Extraction: select key sentences (threshold can be tuned)
    scores = [extractive_model(s, top_k=1)[0]['score'] for s in sentences]
    key_sentences = [s for s, score in zip(sentences, scores) if score > 0.5]
    
    # Initialize a dict for storing sentences per segment
    segments_content = {seg: [] for seg in SEGMENTS}
    # Classification: assign each key sentence to segments
    for sent in key_sentences:
        outputs = classification_model(sent)
        for pred in outputs:
            # Extract label index from "LABEL_X" (e.g., "LABEL_1")
            label_idx = int(pred['label'].split("_")[-1])
            # Here, we assume the classifier was trained with labels corresponding to SEGMENTS order:
            # 0: facts, 1: judgment, 2: analysis, 3: argument, 4: statute.
            if label_idx < len(SEGMENTS) and pred['score'] > 0.3:
                segments_content[SEGMENTS[label_idx]].append(sent)
    
    # Abstractive summarization per segment
    generated_segments = {}
    for seg in SEGMENTS:
        seg_sents = segments_content[seg]
        if seg_sents:
            input_text = f"Summarize {seg}: " + " ".join(seg_sents)
            inputs = bart_tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            summary_ids = bart_model.generate(
                inputs["input_ids"],
                max_length=300,
                min_length=100,
                num_beams=4,
                length_penalty=1.5,
                no_repeat_ngram_size=4,
                early_stopping=True
            )
            summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            generated_segments[seg] = clean_text(summary)  # Apply clean_text after generation
        else:
            generated_segments[seg] = "No relevant content"
    return generated_segments

def process_segmented_candidates():
    extractive_model, classification_model, bart_tokenizer, bart_model = load_segmentation_models()
    candidate_seg = {}
    i = 0
    for filename in sorted(os.listdir(SEG_CASE_FILES_DIR)):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(SEG_CASE_FILES_DIR, filename)
        # Detect encoding and read the file
        encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            raw_text = f.read()
        # Generate segmented summary (a dict of segments)
        seg_summary = generate_segmented_summary(raw_text, extractive_model, classification_model, bart_tokenizer, bart_model)
        candidate_seg[filename] = seg_summary
        i = i + 1
        if i%5 ==0:
            print(i)
        # print(f"Processed candidate segmented summary for {filename}")
    
    # Save candidate segmented summaries to JSON
    with open(CANDIDATE_SEG_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(candidate_seg, f, indent=2, ensure_ascii=False)
    print(f"Candidate segmented summaries saved to {CANDIDATE_SEG_OUTPUT_JSON}")

# ----------------------- Reference Parsing for Segmented Summaries -----------------------
def parse_segmented_text(text):
    """
    Parse a segmented summary text file into a dictionary.
    Assumes segments are marked by headers (e.g., FACTS, JUDGMENT, ANALYSIS, ARGUMENT, STATUTE).
    The headers are assumed to be on a separate line.
    """
    seg_dict = {}
    current_seg = None
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # If the line matches one of the segment headers (case-insensitive)
        if line.upper() in [s.upper() for s in SEGMENTS]:
            current_seg = line.lower()
            seg_dict[current_seg] = ""
        elif current_seg:
            seg_dict[current_seg] += " " + line
    # Clean up extra spaces
    seg_dict = {k: v.strip() for k, v in seg_dict.items()}
    return seg_dict

def process_segmented_references():
    reference_seg = {}
    
    for filename in sorted(os.listdir(REFERENCE_SEG_DIR)):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(REFERENCE_SEG_DIR, filename)
        encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            text = f.read()
        seg_dict = parse_segmented_text(text)
        # Ensure all segments exist (fill missing ones with an empty string or a placeholder)
        for seg in SEGMENTS:
            if seg not in seg_dict:
                seg_dict[seg] = "No content available"
        reference_seg[filename] = seg_dict
        print(f"Processed reference segmented summary for {filename}")
    
    # Save reference segmented summaries to JSON
    with open(REFERENCE_SEG_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(reference_seg, f, indent=2, ensure_ascii=False)
    # print(f"Reference segmented summaries saved to {REFERENCE_SEG_OUTPUT_JSON}")

def process_segmented_summaries():
    process_segmented_candidates()
    process_segmented_references()

if __name__ == "__main__":
    process_segmented_summaries()
