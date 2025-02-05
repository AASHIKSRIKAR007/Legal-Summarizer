#code for inference of fine tuned bart model for whole summary

import torch
import warnings
from transformers import BartForConditionalGeneration, BartTokenizer

def load_model(model_path):
    """
    Load the tokenizer and model from the specified path.
    """
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model

def read_input_file(input_file_path):
    """
    Read and return the content from the input text file.
    """
    with open(input_file_path, "r", encoding="utf-8") as file:
        input_text = file.read().strip()
    return input_text

def generate_summary(model, tokenizer, input_text, max_input_length=1024, 
                     max_output_length=256, min_output_length=50):
    """
    Tokenize the input text, generate the summary, and decode it.
    """
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_input_length)
    
    # Generate output while suppressing warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generated_ids = model.generate(
            inputs["input_ids"],
            max_length=max_output_length,
            min_length=min_output_length,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    
    # Decode the generated ids to text
    generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_summary

def main():
    # Explicitly defined parameters
    model_path = "bart-finetuned2"
    input_file_path = r"data\preprocessedcasefiles\processedcasefile4.txt"
    prefix = "summarize:"  # set to empty string ("") if no prefix is needed

    # Load the model and tokenizer
    tokenizer, model = load_model(model_path)
    
    # Read the input file
    input_text = read_input_file(input_file_path)
    print(input_text[::30])
    
    # Optionally add a prefix to the input text if required
    if prefix:
        input_text = f"{prefix} {input_text}"
    
    # Generate the summary
    summary = generate_summary(model, tokenizer, input_text)
    print("\nGenerated Summary:\n", summary)

if __name__ == "__main__":
    main()
