from flask import Flask, request, jsonify, render_template, Response
import re
import spacy
import warnings
import torch
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer, AutoModelForSequenceClassification
import json
import os

app = Flask(__name__)

# Load spaCy model for sentence segmentation
nlp = spacy.load("en_core_web_sm")

# Directory where case files are stored
CASEFILES_DIR = 'data/casefiles'

######################################
# Utility Functions
######################################
def clean_text(text):
    """Cleans the text by removing citations, dates, and extra spaces."""
    text = re.sub(r'\[.*?\]', '', text)  # Remove citations like [1980] INSC 216
    text = re.sub(r'\b\d{1,2} [A-Za-z]+ \d{4}\b', '', text)  # Remove dates like 13 November 1980
    text = re.sub(r'\s+', ' ', text).strip()  # Remove excess spaces and newlines
    return text

def preprocess_text(text):    
    cleaned_text = clean_text(text)
    doc = nlp(cleaned_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return " ".join(sentences)  # Return as a single string for summarization

######################################
# Models for Summarization
######################################

# --- Updated Extractive & Classification Models (Load from Hugging Face repo) ---

# Load the extractive model from Hugging Face repository (subfolder "extractive_model")
extractive_model_obj = AutoModelForSequenceClassification.from_pretrained(
    "lksai19/Legal-summarizer-models", subfolder="extractive_model"
)
extractive_model = pipeline(
    "text-classification",
    model=extractive_model_obj,
    tokenizer="nlpaueb/legal-bert-base-uncased",
    device=-1  # CPU
)

# Load the classification model from Hugging Face repository (subfolder "classification_model")
classification_model_obj = AutoModelForSequenceClassification.from_pretrained(
    "lksai19/Legal-summarizer-models", subfolder="classification_model"
)
classification_model = pipeline(
    "text-classification",
    model=classification_model_obj,
    tokenizer="nlpaueb/legal-bert-base-uncased",
    device=-1,
    function_to_apply="sigmoid"
)

# BART model for segmented summarization (unchanged)
bart_tokenizer_segmented = BartTokenizer.from_pretrained("facebook/bart-large")
bart_model_segmented = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").cpu()

# Mapping for section labels
section_map = {0: "facts", 1: "judgment", 2: "analysis", 3: "argument", 4: "statute"}

def generate_segmented_summary(input_text):
    processed_text = preprocess_text(input_text)
    sentences = processed_text.split(". ")  # Simple segmentation if needed

    scores = [extractive_model(s, top_k=1)[0]['score'] for s in sentences]
    key_sentences = [s for s, score in zip(sentences, scores) if score > 0.5]

    classified = {v: [] for v in section_map.values()}
    for sent in key_sentences:
        outputs = classification_model(sent)
        for pred in outputs:
            label_idx = int(pred['label'].split("_")[-1])
            if pred['score'] > 0.3:
                classified[section_map[label_idx]].append(sent)

    abstractive_output = []
    for section, sents in classified.items():
        if not sents:
            abstractive_output.append(f"{section.upper()}:\nNo relevant content\n")
            continue
        inputs = bart_tokenizer_segmented(
            f"Summarize {section}: {' '.join(sents)}",
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        summary_ids = bart_model_segmented.generate(
            inputs["input_ids"].to("cpu"),
            max_length=300,
            min_length=100,
            num_beams=4,
            length_penalty=1.5,
            no_repeat_ngram_size=4,
            early_stopping=True
        )
        summary = bart_tokenizer_segmented.decode(summary_ids[0], skip_special_tokens=True)
        abstractive_output.append(f"{section.upper()}:\n{clean_text(summary)}\n")

    return "\n".join(abstractive_output)

# --- Updated Whole Summary Model (Load from Hugging Face repo with correct subfolder) ---

def load_model(model_id, subfolder):
    """
    Load the tokenizer and model from the specified Hugging Face repository and subfolder.
    """
    tokenizer = BartTokenizer.from_pretrained(model_id, subfolder=subfolder)
    model = BartForConditionalGeneration.from_pretrained(model_id, subfolder=subfolder)
    return tokenizer, model

# Use the Hugging Face repository "lksai19/Legal-summarizer-models" and the subfolder "bart-finetuned2"
tokenizer_whole, model_whole = load_model("lksai19/Legal-summarizer-models", "bart-finetuned")

def generate_whole_summary(input_text):
    processed_text = preprocess_text(input_text)  # Apply preprocessing
    return generate_summary_whole(model_whole, tokenizer_whole, processed_text)

def generate_summary_whole(model, tokenizer, input_text, max_input_length=1024, 
                           max_output_length=1024, min_output_length=50):
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

######################################
# Flask Routes
######################################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_casefiles', methods=['GET'])
def get_casefiles():
    try:
        if not os.path.exists(CASEFILES_DIR):
            return jsonify({"error": "Casefiles directory not found"}), 404
            
        casefiles = os.listdir(CASEFILES_DIR)
        casefiles = [f for f in casefiles if f.endswith(".txt")]
        
        if not casefiles:
            return jsonify({"error": "No casefiles found"}), 404
            
        return jsonify({"casefiles": casefiles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Load the selected case file
@app.route('/load_casefile', methods=['POST'])
def load_casefile():
    data = request.get_json()
    casefile_name = data.get('casefile_name')

    # Check if filename is provided
    if not casefile_name:
        return jsonify({"error": "No casefile name provided"}), 400
    
    casefile_path = os.path.join(CASEFILES_DIR, casefile_name)

    # Check if file exists
    if not os.path.exists(casefile_path):
        return jsonify({"error": f"File '{casefile_name}' not found"}), 404

    try:
        with open(casefile_path, 'r', encoding='utf-8', errors='replace') as file:
            casefile_text = file.read()
        return jsonify({"text": casefile_text})
    except Exception as e:
        return jsonify({"error": f"Error loading case file: {str(e)}"}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    input_text = data["text"]
    summary_type = data.get("summary_type", "whole-summary")

    try:
        if summary_type == "segmented-summary":
            summary = generate_segmented_summary(input_text)
        elif summary_type == "whole-summary":
            summary = generate_whole_summary(input_text)
        else:
            return jsonify({"error": "Invalid summary type"}), 400

        return Response(json.dumps({"summary": summary}, ensure_ascii=False), mimetype="application/json")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
