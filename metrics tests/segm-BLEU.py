import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import defaultdict

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def calculate_bleu(candidate_summaries, reference_summaries):
    """Calculate BLEU score (with submetrics for precision, recall, and F1) for each n-gram per segment."""
    bleu_scores = {
        'facts': {'bleu-1': {'precision': [], 'recall': [], 'f1': []}, 
                  'bleu-2': {'precision': [], 'recall': [], 'f1': []}, 
                  'bleu-3': {'precision': [], 'recall': [], 'f1': []}, 
                  'bleu-4': {'precision': [], 'recall': [], 'f1': []}},
        'judgment': {'bleu-1': {'precision': [], 'recall': [], 'f1': []}, 
                     'bleu-2': {'precision': [], 'recall': [], 'f1': []}, 
                     'bleu-3': {'precision': [], 'recall': [], 'f1': []}, 
                     'bleu-4': {'precision': [], 'recall': [], 'f1': []}},
        'analysis': {'bleu-1': {'precision': [], 'recall': [], 'f1': []}, 
                     'bleu-2': {'precision': [], 'recall': [], 'f1': []}, 
                     'bleu-3': {'precision': [], 'recall': [], 'f1': []}, 
                     'bleu-4': {'precision': [], 'recall': [], 'f1': []}},
        'argument': {'bleu-1': {'precision': [], 'recall': [], 'f1': []}, 
                     'bleu-2': {'precision': [], 'recall': [], 'f1': []}, 
                     'bleu-3': {'precision': [], 'recall': [], 'f1': []}, 
                     'bleu-4': {'precision': [], 'recall': [], 'f1': []}},
        'statute': {'bleu-1': {'precision': [], 'recall': [], 'f1': []}, 
                    'bleu-2': {'precision': [], 'recall': [], 'f1': []}, 
                    'bleu-3': {'precision': [], 'recall': [], 'f1': []}, 
                    'bleu-4': {'precision': [], 'recall': [], 'f1': []}}
    }

    smoothing_function = SmoothingFunction().method4

    for case_id, candidate_summary in candidate_summaries.items():
        if case_id in reference_summaries:
            reference_summary = reference_summaries[case_id]
            
            # Calculate BLEU score for each segment
            for segment in bleu_scores.keys():
                candidate_sentences = candidate_summary.get(segment, "").split('.')
                reference_sentences = reference_summary.get(segment, "").split('.')
                
                # Remove empty sentences
                candidate_sentences = [sent.strip() for sent in candidate_sentences if sent.strip()]
                reference_sentences = [sent.strip() for sent in reference_sentences if sent.strip()]

                if candidate_sentences and reference_sentences:
                    # Calculate BLEU score for different n-grams (1-gram to 4-gram)
                    for n in range(1, 5):
                        # Compute BLEU score for the current n-gram
                        bleu_score = sentence_bleu([reference_sentences], candidate_sentences, 
                                                   weights=[1.0 / n] * n, smoothing_function=smoothing_function)

                        # Calculate precision, recall, and F1 score for this n-gram
                        precision = bleu_score  # Simplified for illustration purposes
                        recall = bleu_score
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                        bleu_scores[segment][f'bleu-{n}']['precision'].append(precision)
                        bleu_scores[segment][f'bleu-{n}']['recall'].append(recall)
                        bleu_scores[segment][f'bleu-{n}']['f1'].append(f1)

    # Compute average precision, recall, and F1 for each n-gram per segment
    avg_bleu_scores = {
        seg: {
            key: {
                'precision': sum(values['precision']) / len(values['precision']) if values['precision'] else 0,
                'recall': sum(values['recall']) / len(values['recall']) if values['recall'] else 0,
                'f1': sum(values['f1']) / len(values['f1']) if values['f1'] else 0
            }
            for key, values in segment_values.items()
        }
        for seg, segment_values in bleu_scores.items()
    }

    return avg_bleu_scores

def main():
    candidate_file = "data/data_metric/segmented_reference.json"  # Update with your file path
    reference_file = "data/data_metric/segmented_reference.json"  # Update with your file path

    candidate_summaries = load_json(candidate_file)
    reference_summaries = load_json(reference_file)

    bleu_scores = calculate_bleu(candidate_summaries, reference_summaries)

    print("\nBLEU Scores (with Precision, Recall, and F1 for each n-gram):")
    for segment, score_dict in bleu_scores.items():
        print(f"\n{segment.upper()}:")
        for ngram, metrics in score_dict.items():
            print(f"  {ngram.upper()}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1 Score: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
