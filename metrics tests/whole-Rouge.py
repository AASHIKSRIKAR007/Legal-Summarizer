# import json
# from rouge_score import rouge_scorer

# def load_json(filename):
#     """Loads a JSON file containing summaries."""
#     with open(filename, "r", encoding="utf-8") as file:
#         return json.load(file)

# def calculate_rouge(candidate_summaries, reference_summaries):
#     """Computes ROUGE scores for all summaries."""
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     total_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

#     for doc_id, candidate_summary in candidate_summaries.items():
#         if doc_id in reference_summaries:  # Ensure corresponding reference exists
#             reference_summary = reference_summaries[doc_id]
#             scores = scorer.score(reference_summary, candidate_summary)
            
#             total_scores["rouge1"].append(scores["rouge1"].fmeasure)
#             total_scores["rouge2"].append(scores["rouge2"].fmeasure)
#             total_scores["rougeL"].append(scores["rougeL"].fmeasure)

#     # Compute average scores
#     avg_scores = {metric: sum(scores) / len(scores) for metric, scores in total_scores.items()}
#     return avg_scores

# def main():
#     candidate_file = "data/data_metric/try3LENwhole_candidate.json"  # Replace with your candidate JSON
#     reference_file = "data/data_metric/try3LENwhole_reference.json"  # Replace with your reference JSON

#     candidate_summaries = load_json(candidate_file)
#     reference_summaries = load_json(reference_file)

#     rouge_results = calculate_rouge(candidate_summaries, reference_summaries)

#     print("ROUGE Scores (Whole Summary):")
#     print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
#     print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
#     print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")

# if __name__ == "__main__":
#     main()


import json
from rouge_score import rouge_scorer

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def calculate_rouge(candidate_summaries, reference_summaries):
    """Calculate ROUGE precision, recall, and F1 scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    total_scores = {
        'rouge1': {'precision': [], 'recall': [], 'f1': []},
        'rouge2': {'precision': [], 'recall': [], 'f1': []},
        'rougeL': {'precision': [], 'recall': [], 'f1': []}
    }

    for case_id, candidate_text in candidate_summaries.items():
        if case_id in reference_summaries:
            reference_text = reference_summaries[case_id]
            scores = scorer.score(reference_text, candidate_text)

            for key in ['rouge1', 'rouge2', 'rougeL']:
                total_scores[key]['precision'].append(scores[key].precision)
                total_scores[key]['recall'].append(scores[key].recall)
                total_scores[key]['f1'].append(scores[key].fmeasure)

    # Compute average scores
    avg_scores = {
        key: {
            'precision': sum(values['precision']) / len(values['precision']),
            'recall': sum(values['recall']) / len(values['recall']),
            'f1': sum(values['f1']) / len(values['f1'])
        }
        for key, values in total_scores.items()
    }

    return avg_scores

def main():
    candidate_file = "data/data_metric/try3LENwhole_candidate.json"  # Update with your file path
    reference_file = "data/data_metric/try3LENwhole_reference.json"  # Update with your file path

    candidate_summaries = load_json(candidate_file)
    reference_summaries = load_json(reference_file)

    rouge_scores = calculate_rouge(candidate_summaries, reference_summaries)

    print("\nROUGE Scores (Whole Summary):")
    for key in ['rouge1', 'rouge2', 'rougeL']:
        print(f"\n{key.upper()}:")
        print(f"  Precision: {rouge_scores[key]['precision']:.4f}")
        print(f"  Recall: {rouge_scores[key]['recall']:.4f}")
        print(f"  F1 Score: {rouge_scores[key]['f1']:.4f}")

if __name__ == "__main__":
    main()

