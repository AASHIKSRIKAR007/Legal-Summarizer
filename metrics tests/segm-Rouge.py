import json
from rouge_score import rouge_scorer
from collections import defaultdict

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
def calculate_rouge(candidate_summaries, reference_summaries):
    """Calculate ROUGE scores (with submetrics for precision, recall, and F1) for each segment."""
    rouge_scores = {
        'facts': {'rouge-1': {'precision': [], 'recall': [], 'f1': []}, 
                  'rouge-2': {'precision': [], 'recall': [], 'f1': []}, 
                  'rouge-L': {'precision': [], 'recall': [], 'f1': []}},
        'judgment': {'rouge-1': {'precision': [], 'recall': [], 'f1': []}, 
                     'rouge-2': {'precision': [], 'recall': [], 'f1': []}, 
                     'rouge-L': {'precision': [], 'recall': [], 'f1': []}},
        'analysis': {'rouge-1': {'precision': [], 'recall': [], 'f1': []}, 
                     'rouge-2': {'precision': [], 'recall': [], 'f1': []}, 
                     'rouge-L': {'precision': [], 'recall': [], 'f1': []}},
        'argument': {'rouge-1': {'precision': [], 'recall': [], 'f1': []}, 
                     'rouge-2': {'precision': [], 'recall': [], 'f1': []}, 
                     'rouge-L': {'precision': [], 'recall': [], 'f1': []}},
        'statute': {'rouge-1': {'precision': [], 'recall': [], 'f1': []}, 
                    'rouge-2': {'precision': [], 'recall': [], 'f1': []}, 
                    'rouge-L': {'precision': [], 'recall': [], 'f1': []}}
    }

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for case_id, candidate_summary in candidate_summaries.items():
        if case_id in reference_summaries:
            reference_summary = reference_summaries[case_id]
            
            # Calculate ROUGE score for each segment
            for segment in rouge_scores.keys():
                candidate_text = candidate_summary.get(segment, "")
                reference_text = reference_summary.get(segment, "")
                
                if not candidate_text:
                    print(f"Warning: Candidate text for segment '{segment}' is empty for case '{case_id}'")
                if not reference_text:
                    print(f"Warning: Reference text for segment '{segment}' is empty for case '{case_id}'")

                # Calculate ROUGE score for ROUGE-1, ROUGE-2, ROUGE-L
                if candidate_text and reference_text:
                    try:
                        scores = scorer.score(reference_text, candidate_text)

                        for key in ['rouge1', 'rouge2', 'rougeL']:
                            if key in scores:
                                rouge_scores[segment][key]['precision'].append(scores[key].precision)
                                rouge_scores[segment][key]['recall'].append(scores[key].recall)
                                rouge_scores[segment][key]['f1'].append(scores[key].fmeasure)
                            else:
                                print(f"Warning: {key} not found in the scores for {segment}")
                    except Exception as e:
                        print(f"Error calculating ROUGE for segment {segment}: {e}")

    # Compute average precision, recall, and F1 for each n-gram per segment
    avg_rouge_scores = {
        seg: {
            key: {
                'precision': sum(values['precision']) / len(values['precision']) if values['precision'] else 0,
                'recall': sum(values['recall']) / len(values['recall']) if values['recall'] else 0,
                'f1': sum(values['f1']) / len(values['f1']) if values['f1'] else 0
            }
            for key, values in rouge_scores[seg].items()
        }
        for seg in rouge_scores
    }

    return avg_rouge_scores

def main():
    candidate_file = "data/data_metric/segmented_candidate.json"  # Update with your file path
    reference_file = "data/data_metric/segmented_reference.json"  # Update with your file path

    candidate_summaries = load_json(candidate_file)
    reference_summaries = load_json(reference_file)

    rouge_scores = calculate_rouge(candidate_summaries, reference_summaries)

    print("\nROUGE Scores (with Precision, Recall, and F1 for each n-gram):")
    for segment, score_dict in rouge_scores.items():
        print(f"\n{segment.upper()}:")
        for ngram, metrics in score_dict.items():
            print(f"  {ngram.upper()}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1 Score: {metrics['f1']:.4f}")



if __name__ == "__main__":
    main()
