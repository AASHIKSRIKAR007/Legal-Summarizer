import os
import json
import chardet

def load_segments_from_folder(folder_path, label):
    """
    Load the text segments from a given folder and label them, including case IDs.

    Args:
        folder_path (str): Path to the folder containing text files.
        label (str): The label to assign to each segment (e.g., 'Facts', 'Judgment').

    Returns:
        list: A list of dictionaries with text, label, and case_id.
    """
    segments = []
    
    # Ensure the folder exists before attempting to load files
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Skip if it's a non-text file or hidden file
            if not file_name.endswith(".txt"):
                continue

            try:
                # Extract case_id from file name
                case_id = os.path.splitext(file_name)[0]

                # Automatically detect encoding using chardet
                with open(file_path, "rb") as file:
                    raw_data = file.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding']
                
                # Open file using detected encoding
                with open(file_path, "r", encoding=encoding) as file:
                    text = file.read().strip()
                
                segments.append({"text": text, "label": label, "case_id": case_id})
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    else:
        print(f"Folder '{folder_path}' not found. Skipping...")

    return segments


def create_unified_dataset(annotator_a_folder, annotator_b_folder, case_file_folder, output_file):
    """
    Create a unified dataset for both segment classification and summarization.

    Args:
        annotator_a_folder (str): Path to Annotator A's folder.
        annotator_b_folder (str): Path to Annotator B's folder.
        case_file_folder (str): Path to the folder containing the original case files.
        output_file (str): Path to save the final unified dataset.
    """
    dataset = []

    # Segment Classification
    labels = ['analysis', 'judgment', 'facts', 'argument', 'statute']

    # Process data from annotator A
    for label in labels:
        a_label_folder = os.path.join(annotator_a_folder, label)
        data_a = load_segments_from_folder(a_label_folder, label)
        dataset.extend(data_a)

    # Process data from annotator B
    for label in labels:
        b_label_folder = os.path.join(annotator_b_folder, label)
        data_b = load_segments_from_folder(b_label_folder, label)
        dataset.extend(data_b)

    # Adding Case Files for Summarization (only 'Judgment' sections)
    for case_file in os.listdir(case_file_folder):
        case_file_path = os.path.join(case_file_folder, case_file)
        try:
            # Automatically detect encoding using chardet
            with open(case_file_path, "rb") as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            
            # Extract case_id from file name
            case_id = os.path.splitext(case_file)[0]

            # Open the case file using the detected encoding
            with open(case_file_path, "r", encoding=encoding) as file:
                case_text = file.read().strip()
            
            # Here, we add the original case file for summarization as a 'Judgment'
            dataset.append({"text": case_text, "label": "case_file", "case_id": case_id})
        except Exception as e:
            print(f"Error reading file {case_file_path}: {e}")

    # Save the unified dataset to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"Unified dataset saved to {output_file}")
    print(f"Total number of segments: {len(dataset)}")

# Paths
annotator_a_folder = r'C:\Users\lksai\Downloads\legal\dataset\IN-Ext\summary\segment-wise\A1'  # Replace with actual path
annotator_b_folder = r'C:\Users\lksai\Downloads\legal\dataset\IN-Ext\summary\segment-wise\A2'  # Replace with actual path
case_file_folder = r'C:\Users\lksai\Downloads\legal\dataset\IN-Ext\judgement'  # Replace with actual path
output_file = "final_dataset.json"  # Replace with desired output path

# Create the unified dataset
create_unified_dataset(annotator_a_folder, annotator_b_folder, case_file_folder, output_file)
