# import spacy
# import re 
# # Load spaCy model for sentence segmentation
# nlp = spacy.load("en_core_web_sm")

# def preprocess_case_file(input_file, output_file):
#     """
#     Preprocess the case file to split text into sentences.
    
#     Args:
#         input_file (str): Path to the input .txt file.
#         output_file (str): Path to the output .txt file for comparison.
#     """
#     # Read the input case file
#     with open(input_file, "r", encoding="utf-8") as file:
#         text = file.read()
    
#     cleaned_text = " ".join(text.splitlines())

#     # Use spaCy to segment text into sentences
#     doc = nlp(cleaned_text)
#     sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
#     # Write the processed sentences to the output file
#     with open(output_file, "w", encoding="utf-8") as file:
#         for sentence in sentences:
#             file.write(sentence + "\n")
    
#     print(f"Processed data saved to {output_file}")
#     print(f"Number of sentences: {len(sentences)}")
#     print(sentences[1:2])

# # Example Usage
# input_path = "legal_document.txt"  # Replace with your input file path
# output_path = "processed.txt"  # Replace with your desired output file path
# preprocess_case_file(input_path, output_path)
import spacy
import re

# Load spaCy model for sentence segmentation
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
    Cleans the case file text by removing citations, dates, and irrelevant sections.
    
    Args:
        text (str): The input raw text from the case file.
        
    Returns:
        str: The cleaned text.
    """
    # Remove citation references (e.g., [1980] INSC 216)
    text = re.sub(r'\[.*?\]', '', text)  # Removes anything between square brackets
    
    # Remove dates (e.g., 13 November 1980)
    text = re.sub(r'\b\d{1,2} [A-Za-z]+ \d{4}\b', '', text)
     
    # Remove excess newlines or spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one space
    text = text.strip()  # Remove leading and trailing whitespace
    
    return text

def preprocess_case_file(input_file, output_file):
    """
    Preprocess the case file to clean text and split it into sentences.
    
    Args:
        input_file (str): Path to the input .txt file.
        output_file (str): Path to the output .txt file for comparison.
    """
    # Read the input case file
    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()
    
    # Clean the text by removing irrelevant sections
    cleaned_text = clean_text(text)

    # Use spaCy to segment the cleaned text into sentences
    doc = nlp(cleaned_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    # Write the processed sentences to the output file
    with open(output_file, "w", encoding="utf-8") as file:
        for sentence in sentences:
            file.write(sentence + "\n")
    
    print(f"Processed data saved to {output_file}")
    print(f"Number of sentences: {len(sentences)}")
    print(sentences[:5])  # Display the first 5 sentences for review

input_path = "legal_document.txt"  
output_path = "processed.txt" 
preprocess_case_file(input_path, output_path)
