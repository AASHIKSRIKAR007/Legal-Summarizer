import os
import re
import json
from glob import glob
import chardet

def clean_text(text):
    """Normalize whitespace and remove unwanted characters"""
    # Replace newlines, tabs, and multiple spaces with single space
    text = re.sub(r'[\n\t]+', ' ', text)
    # Remove non-breaking spaces and other special whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace and quotes
    return text.strip().strip('"\'')  

def detect_encoding(file_path):
    """Detect file encoding with fallback"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        return 'utf-8'
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                f.read()
            return 'latin-1'
        except:
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read())
            return result['encoding']

def extract_case_id(filename):
    """Extract case ID from filename using regex"""
    match = re.match(r'(\d{4}_[A-Z]_\d+)\.txt$', filename)
    return match.group(1) if match else None

def load_section_summaries(root_dir, case_id, sections):
    """Load and clean section summaries with encoding detection"""
    summaries = {}
    for section in sections:
        file_path = os.path.join(root_dir, section, f"{case_id}.txt")
        if os.path.exists(file_path):
            encoding = detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = clean_text(f.read())
                summaries[section] = content if content else None
        else:
            summaries[section] = None
    return summaries

def preprocess_dataset(case_files_dir, summaries_A_dir, summaries_B_dir, output_json_path):
    """Main preprocessing pipeline"""
    sections = ["facts", "judgment", "analysis", "argument", "statute"]
    case_data = {}

    case_files = glob(os.path.join(case_files_dir, "*.txt"))
    for case_file in case_files:
        filename = os.path.basename(case_file)
        case_id = extract_case_id(filename)
        if not case_id:
            print(f"Skipping invalid filename: {filename}")
            continue

        # Read and clean raw case text
        encoding = detect_encoding(case_file)
        with open(case_file, 'r', encoding=encoding, errors='ignore') as f:
            raw_text = clean_text(f.read())

        # Load and clean summaries
        summaries_A = load_section_summaries(summaries_A_dir, case_id, sections)
        summaries_B = load_section_summaries(summaries_B_dir, case_id, sections)

        # Validate case data
        if not raw_text or (all(v is None for v in summaries_A.values()) and all(v is None for v in summaries_B.values())):
            print(f"Skipping {case_id}: No valid text/summaries")
            continue

        # Store cleaned data
        case_data[case_id] = {
            "raw_text": raw_text,
            "summaries_A": summaries_A,
            "summaries_B": summaries_B
        }

    # Save cleaned dataset
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(case_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully processed {len(case_data)} cases")
    print(f"Cleaned dataset saved to: {output_json_path}")


# Example Usage
preprocess_dataset(
    case_files_dir="./data/data_legbert/case_files_bert",
    summaries_A_dir="./data/data_legbert/summ_A1",
    summaries_B_dir="./data/data_legbert/summ_A2",
    output_json_path="./data/preprocessed_legbert_dataset.json"
)
