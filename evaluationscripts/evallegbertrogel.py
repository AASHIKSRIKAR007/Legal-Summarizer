#code to evaluate finetuned legalbert model
#results are in testresults folder,RougueLscores.txt

from rouge_score import rouge_scorer

# =================================================================================
# FILE PATHS (Modify if needed)
# =================================================================================
EXTRACTIVE_SUMMARY_FILE = "tests\extrumm.txt"  # Model-generated summary
REFERENCE_SUMMARY_A = "tests\A.txt"  # Human-written summary (Expert A)
REFERENCE_SUMMARY_B = "tests\B.txt"  # Human-written summary (Expert B)

# =================================================================================
# UTILITY FUNCTION: Read text from file
# =================================================================================
def read_text_file(file_path):
    """Reads a text file and returns its content as a single string."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return ""

# =================================================================================
# LOAD TEXT DATA
# =================================================================================
generated_summary = read_text_file(EXTRACTIVE_SUMMARY_FILE)
reference_summary_a = read_text_file(REFERENCE_SUMMARY_A)
reference_summary_b = read_text_file(REFERENCE_SUMMARY_B)

# Ensure text files are not empty
if not generated_summary or not reference_summary_a or not reference_summary_b:
    print("Error: One or more input files are empty or missing.")
    exit()

# =================================================================================
# COMPUTE ROUGE-L SCORE
# =================================================================================
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Compute ROUGE-L scores against both references
scores_a = scorer.score(reference_summary_a, generated_summary)
scores_b = scorer.score(reference_summary_b, generated_summary)

# Average ROUGE-L scores across both references
rouge_l_precision = (scores_a["rougeL"].precision + scores_b["rougeL"].precision) / 2
rouge_l_recall = (scores_a["rougeL"].recall + scores_b["rougeL"].recall) / 2
rouge_l_f1 = (scores_a["rougeL"].fmeasure + scores_b["rougeL"].fmeasure) / 2

# =================================================================================
# PRINT RESULTS
# =================================================================================
print("ROUGE-L Scores:")
print(f"Precision: {rouge_l_precision:.4f}")
print(f"Recall:    {rouge_l_recall:.4f}")
print(f"F1 Score:  {rouge_l_f1:.4f}")
