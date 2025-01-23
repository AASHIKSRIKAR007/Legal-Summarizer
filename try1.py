from transformers import AutoTokenizer, AutoModel, BartTokenizer, BartForConditionalGeneration
import torch

print("Summarizing just wait")
# Load models and tokenizers
legalbert_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
legalbert_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def preprocess_text(file_path):
    """Load and preprocess legal document"""
    with open(file_path, 'r') as f:
        text = f.read()
    # Add cleaning or chunking logic if needed
    print("File is opened\n\n")
    return text

def extractive_summary_legalbert(text):
    """Use LegalBERT to extract important sentences"""
    sentences = text.split(". ")  # Split text into sentences
    inputs = legalbert_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = legalbert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
    
    # Example: Rank sentences by some custom logic (e.g., cosine similarity)
    # For simplicity, return the top 3 sentences
    top_sentences = sentences  # Replace this with actual ranking logic
    return " ".join(top_sentences)

def abstractive_summary_bart(text):
    """Use BART to generate an abstractive summary"""
    inputs = bart_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(
        inputs["input_ids"], max_length=100000, min_length=50, length_penalty=2.0, num_beams=4
    )
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("Yes, Extraction done.")
    return summary

if __name__ == "__main__":
    # Input legal document
    file_path = "legal_document.txt"  # Replace with your file path
    text = preprocess_text(file_path)

    # Step 1: Extractive summary with LegalBERT
    extractive_summary = extractive_summary_legalbert(text)
    # print("Extractive Summary:\n", extractive_summary)

    # Step 2: Abstractive summary with BART
    abstractive_summary = abstractive_summary_bart(extractive_summary)
    print("\nAbstractive Summary:\n", abstractive_summary)
