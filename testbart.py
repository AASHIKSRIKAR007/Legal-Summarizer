# import torch
# from transformers import BartForConditionalGeneration, BartTokenizer

# # Set the paths
# input_file_path = "processed.txt"      # Path to your test input text file
# model_path = "./bart-finetuned"           # Path to your fine-tuned BART model

# # Load the fine-tuned model and tokenizer
# tokenizer = BartTokenizer.from_pretrained(model_path)
# model = BartForConditionalGeneration.from_pretrained(model_path)
# model.eval()  # Set model to evaluation mode

# # (Optional) Ensure the model's configuration is correct for generation
# model.config.forced_bos_token_id = tokenizer.bos_token_id

# # Read the input text from the file
# with open(input_file_path, "r", encoding="utf-8") as file:
#     input_text = file.read().strip()

# if not input_text:
#     print("Input text is empty!")
# else:
#     # Print a preview of the input text (first 500 characters)
#     print("Input text (first 500 characters):")
#     print(input_text[:500])
#     print("-" * 50)

# # Tokenize the input text
# inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
# print("Tokenized input IDs:")
# print(inputs["input_ids"])

# # Generate the summary
# generated_ids = model.generate(
#     inputs["input_ids"],
#     max_length=256,
#     num_beams=4,
#     early_stopping=True
# )
# print("Generated IDs:")
# print(generated_ids)

# # Decode the generated summary
# generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# print("Generated Summary:")
# print(generated_summary)


# import torch
# from transformers import BartForConditionalGeneration, BartTokenizer

# # Set paths for the input text file and the fine-tuned model
# input_file_path = "processed.txt"      # Update with your input file path
# model_path = "./bart-finetuned"           # Update with your fine-tuned model path

# # Load the fine-tuned model and tokenizer
# tokenizer = BartTokenizer.from_pretrained(model_path)
# model = BartForConditionalGeneration.from_pretrained(model_path)

# # Update the model configuration to ensure proper generation behavior
# # This sets the forced beginning-of-sentence (BOS) token and decoder start token
# model.config.forced_bos_token_id = tokenizer.bos_token_id
# model.config.decoder_start_token_id = tokenizer.bos_token_id

# # Read the input text from the file
# with open(input_file_path, "r", encoding="utf-8") as file:
#     input_text = file.read().strip()

# if not input_text:
#     print("Input text is empty!")
# else:
#     # Print a preview of the input text
#     print("Input text (first 500 characters):")
#     print(input_text[:500])
#     print("-" * 50)

# # Tokenize the input text with truncation
# inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
# print("Tokenized input IDs:", inputs["input_ids"])

# # Generate the summary with additional parameters to encourage better generation
# generated_ids = model.generate(
#     inputs["input_ids"],
#     max_length=256,
#     num_beams=4,
#     early_stopping=True,
#     no_repeat_ngram_size=2,
#     length_penalty=2.0
# )
# print("Generated IDs:", generated_ids)

# # Decode the generated summary
# generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# print("Generated Summary:")
# print(generated_summary)


import torch
import warnings
from transformers import BartForConditionalGeneration, BartTokenizer

# Paths
# model_path = r"bart-finetuned1"
model_path = "facebook/bart-large-cnn"
input_file_path = "processed.txt"

# Load resources
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# Read input
with open(input_file_path, "r", encoding="utf-8") as file:
    input_text = file.read().strip()

# Add training prefix if needed
# input_text = "summarize: " + input_text

# Tokenize
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)

# Generate with suppressed warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    generated_ids = model.generate(
        inputs["input_ids"],
        max_length=256,
        min_length=50,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

# Decode
generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\nGenerated Summary pre:\n", generated_summary)