#code for inference of combined extractive and abstractive legalbert model

from transformers import pipeline, BertTokenizer, BartForConditionalGeneration, BartTokenizer
import os
import re
import chardet

# =================================================================================
# CONFIGURATION (Edit these paths)
# =================================================================================
INPUT_FILE = "processed.txt"
OUTPUT_ABSTRACTIVE_FILE = "abstractive_summarylegbert3.txt"  # BART output
OUTPUT_EXTRACTIVE_FILE = "extractive_segmentslegbert3.txt"   # LegalBERT output

# =================================================================================
# UTILITY FUNCTIONS 
# =================================================================================
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        return chardet.detect(f.read())['encoding']

def clean_text(text):
    text = re.sub(r'[\n\t]+', ' ', text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# =================================================================================
# LOAD MODELS (CPU ONLY)
# =================================================================================
# 1. LegalBERT Models
extractive_model = pipeline(
    "text-classification",
    model="./extractive_model/final",
    tokenizer="nlpaueb/legal-bert-base-uncased",
    device=-1  # CPU
)

classification_model = pipeline(
    "text-classification",
    model="./classification_model/final",
    tokenizer="nlpaueb/legal-bert-base-uncased",
    device=-1,
    function_to_apply="sigmoid"
)

# 2. BART Model
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").cpu()


# =================================================================================
# PROCESSING PIPELINE
# =================================================================================
def generate_summaries():
    # Read input file
    try:
        encoding = detect_encoding(INPUT_FILE)
        with open(INPUT_FILE, 'r', encoding=encoding, errors='ignore') as f:
            raw_text = clean_text(f.read())
    except Exception as e:
        print(f"Error reading {INPUT_FILE}: {str(e)}")
        return

    # Load models (same as before)
    # [Keep the model loading code identical to previous version]

    # Extract and classify sentences
    sentences = split_sentences(raw_text)
    scores = [extractive_model(s, top_k=1)[0]['score'] for s in sentences]
    key_sentences = [s for s, score in zip(sentences, scores) if score > 0.5]

    section_map = {0:"facts", 1:"judgment", 2:"analysis", 3:"argument", 4:"statute"}
    classified = {v: [] for v in section_map.values()}
    
    for sent in key_sentences:
    # Get raw model outputs
        outputs = classification_model(sent)
    
        # For multi-label classification, outputs is a list of dicts:
        # [{'label': 'LABEL_0', 'score': 0.95}, ...]
        for pred in outputs:
            label_idx = int(pred['label'].split("_")[-1])  # Extract 0 from "LABEL_0"
            if pred['score'] > 0.3:
                classified[section_map[label_idx]].append(sent)

    # =================================================================
    # Save EXTRACTIVE segments
    # =================================================================
    extractive_output = []
    for section, sents in classified.items():
        extractive_output.append(f"=== {section.upper()} ===")
        extractive_output.extend(sents if sents else ["No sentences classified"])
        extractive_output.append("\n")
    
    extractive_dir = os.path.dirname(OUTPUT_EXTRACTIVE_FILE)
    if extractive_dir:  # Only create directories if the path is not empty
        os.makedirs(extractive_dir, exist_ok=True)

    with open(OUTPUT_EXTRACTIVE_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(extractive_output))
    print(f"Extractive segments saved to {OUTPUT_EXTRACTIVE_FILE}")

    # =================================================================
    # Generate & Save ABSTRACTIVE summaries 
    # =================================================================
    abstractive_output = []
    for section, sentences in classified.items():
        if not sentences:
            abstractive_output.append(f"{section.upper()}:\nNo relevant content\n")
            continue
            
        inputs = bart_tokenizer(
            f"Summarize {section}: {' '.join(sentences)}",
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        summary_ids = bart_model.generate(
            inputs["input_ids"].to("cpu"),
            max_length=300,
            min_length=100,
            num_beams=4,
            length_penalty=1.5,
            no_repeat_ngram_size=4,
            early_stopping=True  # Stops when the model feels it's complete
        )
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        abstractive_output.append(f"{section.upper()}:\n{clean_text(summary)}\n")
    abstractive_dir = os.path.dirname(OUTPUT_ABSTRACTIVE_FILE)
    if abstractive_dir:
        os.makedirs(abstractive_dir, exist_ok=True)
    with open(OUTPUT_ABSTRACTIVE_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(abstractive_output))
    print(f"Abstractive summary saved to {OUTPUT_ABSTRACTIVE_FILE}")

# =================================================================================
# EXECUTE
# =================================================================================
if __name__ == "__main__":
    generate_summaries()