import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction



# File paths (Update these with actual file locations)
CANDIDATE_JSON_PATH = "data/data_metric/try7LENmod3whole_candidate.json"
REFERENCE_JSON_PATH = "data/data_metric/try7LENmod3whole_reference.json"

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def tokenize_text(text):
    """Tokenize text into word-level tokens."""
    return nltk.word_tokenize(text.lower())

def calculate_bleu(candidate_summaries, reference_summaries):
    """Calculate BLEU precision, recall, and F1 scores for all n-grams."""
    smooth_fn = SmoothingFunction().method1  # Smoothing for short texts

    total_scores = {
        'BLEU-1': {'precision': [], 'recall': [], 'f1': []},
        'BLEU-2': {'precision': [], 'recall': [], 'f1': []},
        'BLEU-3': {'precision': [], 'recall': [], 'f1': []},
        'BLEU-4': {'precision': [], 'recall': [], 'f1': []}
    }

    for case_id, candidate_text in candidate_summaries.items():
        if case_id in reference_summaries:
            reference_text = reference_summaries[case_id]

            # Tokenize the text
            candidate_tokens = tokenize_text(candidate_text)
            reference_tokens = [tokenize_text(reference_text)]  # BLEU expects list of references

            # Compute BLEU scores for each n-gram
            for n in range(1, 5):
                weights = [1.0 / n] * n + [0] * (4 - n)  # Create weight for BLEU (1,0,0,0) for BLEU-1, etc.
                bleu_score = sentence_bleu(reference_tokens, candidate_tokens, weights=weights, smoothing_function=smooth_fn)

                # Calculate precision
                precision = bleu_score

                # Calculate recall
                recall = sum(1 for word in reference_tokens[0] if word in candidate_tokens) / len(reference_tokens[0])

                # Calculate F1 score
                if precision + recall != 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0

                # Store the scores
                total_scores[f'BLEU-{n}']['precision'].append(precision)
                total_scores[f'BLEU-{n}']['recall'].append(recall)
                total_scores[f'BLEU-{n}']['f1'].append(f1)

    # Return the computed scores
    return total_scores

def main():
    """Load JSON files and evaluate BLEU scores."""
    candidate_summaries = load_json(CANDIDATE_JSON_PATH)
    reference_summaries = load_json(REFERENCE_JSON_PATH)

    bleu_scores = calculate_bleu(candidate_summaries, reference_summaries)

    print("\nBLEU Scores (Submetrics):")
    for key in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']:
        print(f"\n{key}:")
        print(f"  Precision: {sum(bleu_scores[key]['precision']) / len(bleu_scores[key]['precision']):.4f}")
        print(f"  Recall: {sum(bleu_scores[key]['recall']) / len(bleu_scores[key]['recall']):.4f}")
        print(f"  F1 Score: {sum(bleu_scores[key]['f1']) / len(bleu_scores[key]['f1']):.4f}")

if __name__ == "__main__":
    main()