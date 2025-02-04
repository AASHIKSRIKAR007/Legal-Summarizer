import os
import torch
from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Paths
train_judgment_folder = r"train-data\judgement"
train_summary_folder = r"train-data\summary"
test_judgment_folder = r"test-data\judgement"
test_summary_folder = r"test-data\summary"

# Function to load judgment-summary pairs from a folder structure
def load_data(judgment_folder, summary_folder):
    judgments = []
    summaries = []
    
    # Get the list of all judgment files
    judgment_files = sorted(os.listdir(judgment_folder))
    
    for filename in judgment_files:
        # Skip files without corresponding summary
        summary_filename = filename.split('.')[0] + '.txt'  # Assuming they have .txt extension
        if summary_filename not in os.listdir(summary_folder):
            continue
        
        # Read the judgment file
        with open(os.path.join(judgment_folder, filename), 'r', encoding='utf-8') as f:
            judgment_text = f.read().strip()
        
        # Read the summary file
        with open(os.path.join(summary_folder, summary_filename), 'r', encoding='utf-8') as f:
            summary_text = f.read().strip()
        
        judgments.append(judgment_text)
        summaries.append(summary_text)
    
    return judgments, summaries

# Load training data
train_judgments, train_summaries = load_data(train_judgment_folder, train_summary_folder)

# Load testing data
test_judgments, test_summaries = load_data(test_judgment_folder, test_summary_folder)

# Split the train data into train and validation sets (80% train, 20% validation)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_judgments, train_summaries, test_size=0.2, random_state=42)

# Initialize the tokenizer and model
model_name = "facebook/bart-large-cnn"  # You can use a smaller model like bart-base if needed
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    model_inputs = tokenizer(examples['judgment'], max_length=1024, truncation=True, padding="max_length")
    # Tokenize the summaries
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['summary'], max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Create a dataset from the loaded text pairs
train_dataset = Dataset.from_dict({"judgment": train_texts, "summary": train_labels})
val_dataset = Dataset.from_dict({"judgment": val_texts, "summary": val_labels})
test_dataset = Dataset.from_dict({"judgment": test_judgments, "summary": test_summaries})

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",  # evaluate at the end of every epoch
    save_strategy="epoch"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=None,  # Use default collator for BART
)

# Fine-tuning the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./bart-finetuned")
tokenizer.save_pretrained("./bart-finetuned")

print("Fine-tuning completed and model saved.")

# Evaluate on test data
test_results = trainer.evaluate(test_dataset)
print("Test Results:", test_results)

# try 2

# import os
# import torch
# import numpy as np
# from datasets import Dataset
# from transformers import (
#     BartTokenizer,
#     BartForConditionalGeneration,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForSeq2Seq,
#     BartConfig
# )
# from sklearn.model_selection import train_test_split
# from evaluate import load
# import nltk

# # Download required NLTK data
# nltk.download('punkt')

# # Initialize metrics
# rouge = load("rouge")
# bertscore = load("bertscore")

# # Paths
# train_judgment_folder = r"train-data\judgement"
# train_summary_folder = r"train-data\summary"
# test_judgment_folder = r"test-data\judgement"
# test_summary_folder = r"test-data\summary"

# # Function to load judgment-summary pairs
# def load_data(judgment_folder, summary_folder):
#     judgments = []
#     summaries = []
    
#     judgment_files = sorted(os.listdir(judgment_folder))
    
#     for filename in judgment_files:
#         summary_filename = filename.split('.')[0] + '.txt'
#         if summary_filename not in os.listdir(summary_folder):
#             continue
        
#         with open(os.path.join(judgment_folder, filename), 'r', encoding='utf-8') as f:
#             judgment_text = f.read().strip()
        
#         with open(os.path.join(summary_folder, summary_filename), 'r', encoding='utf-8') as f:
#             summary_text = f.read().strip()
        
#         judgments.append(judgment_text)
#         summaries.append(summary_text)
    
#     return judgments, summaries

# # Load and split data
# train_judgments, train_summaries = load_data(train_judgment_folder, train_summary_folder)
# test_judgments, test_summaries = load_data(test_judgment_folder, test_summary_folder)
# train_texts, val_texts, train_labels, val_labels = train_test_split(
#     train_judgments, train_summaries, test_size=0.2, random_state=42
# )

# # Initialize model and tokenizer with custom config
# model_name = "facebook/bart-large-cnn"
# tokenizer = BartTokenizer.from_pretrained(model_name)
# config = BartConfig.from_pretrained(model_name)
# config.forced_bos_token_id = 0  # Fix config warning
# config.attention_probs_dropout_prob = 0.1  # Add regularization
# model = BartForConditionalGeneration.from_pretrained(model_name, config=config)

# # Curriculum learning setup (gradual unfreezing)
# for param in model.parameters():
#     param.requires_grad = False
    
# for layer in [model.model.encoder.layers[-4:], model.model.decoder.layers[-4:]]:
#     for param in layer.parameters():
#         param.requires_grad = True

# # Enhanced tokenization with legal-specific formatting
# def tokenize_function(examples):
#     # Add domain-specific prefix
#     inputs = ["Summarize legal case: " + text for text in examples['judgment']]
#     targets = ["[LEGAL_SUM] " + text for text in examples['summary']]
    
#     model_inputs = tokenizer(
#         inputs,
#         max_length=1024,
#         truncation=True,
#         padding=False  # Dynamic padding handled by collator
#     )
    
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(
#             targets,
#             max_length=256,
#             truncation=True,
#             padding=False
#         )
    
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

# # Create datasets
# train_dataset = Dataset.from_dict({"judgment": train_texts, "summary": train_labels})
# val_dataset = Dataset.from_dict({"judgment": val_texts, "summary": val_labels})
# test_dataset = Dataset.from_dict({"judgment": test_judgments, "summary": test_summaries})

# # Tokenize datasets
# train_dataset = train_dataset.map(tokenize_function, batched=True)
# val_dataset = val_dataset.map(tokenize_function, batched=True)
# test_dataset = test_dataset.map(tokenize_function, batched=True)

# # Data collator for dynamic padding
# data_collator = DataCollatorForSeq2Seq(
#     tokenizer,
#     model=model,
#     padding="longest",
#     max_length=1024,
#     pad_to_multiple_of=8
# )

# # Metrics computation
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
#     # Calculate ROUGE
#     rouge_results = rouge.compute(
#         predictions=decoded_preds,
#         references=decoded_labels,
#         use_stemmer=True
#     )
    
#     # Calculate BERTScore
#     bert_results = bertscore.compute(
#         predictions=decoded_preds,
#         references=decoded_labels,
#         lang="en",
#         model_type="bert-base-multilingual-cased"
#     )
    
#     return {
#         "rouge1": round(rouge_results["rouge1"], 4),
#         "rougeL": round(rouge_results["rougeL"], 4),
#         "bertscore": round(np.mean(bert_results["f1"]), 4)
#     }

# # Optimized training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=5,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=2,
#     learning_rate=3e-5,
#     warmup_ratio=0.1,
#     weight_decay=0.01,
#     gradient_accumulation_steps=4,
#     fp16=False,
#     no_cuda=True,
#     logging_dir="./logs",
#     logging_steps=50,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",  
#     load_best_model_at_end=True,
#     metric_for_best_model="rougeL",
#     greater_is_better=True,
#     group_by_length=True,
#     report_to="none"
# )

# # Enhanced trainer setup
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics
# )

# # Training
# trainer.train()

# # Save model with proper config
# model.config.save_pretrained("./bart-finetuned2")
# model.save_pretrained("./bart-finetuned2")
# tokenizer.save_pretrained("./bart-finetuned2")

# # Final evaluation
# test_results = trainer.evaluate(test_dataset)
# print("\nFinal Test Results:")
# print(f"ROUGE-L: {test_results['eval_rougeL']}")
# print(f"BERTScore F1: {test_results['eval_bertscore']}")

# # Post-training optimization
# quantized_model = torch.quantization.quantize_dynamic(
#     model,
#     {torch.nn.Linear},
#     dtype=torch.qint8
# )
# quantized_model.save_pretrained("./bart-finetuned2-quantized")

# print("\nOptimized model saved with quantization.")

# # import os
# # import torch
# # import numpy as np
# # import gc  # Garbage collector
# # from datasets import Dataset
# # from transformers import (
# #     BartTokenizer,
# #     BartForConditionalGeneration,
# #     Trainer,
# #     TrainingArguments,
# #     DataCollatorForSeq2Seq,
# #     BartConfig
# # )
# # from sklearn.model_selection import train_test_split
# # from evaluate import load
# # import nltk

# # # Download required NLTK data
# # nltk.download('punkt')

# # # Initialize metrics
# # rouge = load("rouge")
# # bertscore = load("bertscore")

# # # Paths
# # train_judgment_folder = r"train-data\judgement"
# # train_summary_folder = r"train-data\summary"
# # test_judgment_folder = r"test-data\judgement"
# # test_summary_folder = r"test-data\summary"

# # # Function to load judgment-summary pairs
# # def load_data(judgment_folder, summary_folder):
# #     judgments = []
# #     summaries = []
    
# #     judgment_files = sorted(os.listdir(judgment_folder))
    
# #     for filename in judgment_files:
# #         summary_filename = filename.split('.')[0] + '.txt'
# #         if summary_filename not in os.listdir(summary_folder):
# #             continue
        
# #         with open(os.path.join(judgment_folder, filename), 'r', encoding='utf-8') as f:
# #             judgment_text = f.read().strip()
        
# #         with open(os.path.join(summary_folder, summary_filename), 'r', encoding='utf-8') as f:
# #             summary_text = f.read().strip()
        
# #         judgments.append(judgment_text)
# #         summaries.append(summary_text)
    
# #     return judgments, summaries

# # # Load and split data
# # train_judgments, train_summaries = load_data(train_judgment_folder, train_summary_folder)
# # test_judgments, test_summaries = load_data(test_judgment_folder, test_summary_folder)
# # train_texts, val_texts, train_labels, val_labels = train_test_split(
# #     train_judgments, train_summaries, test_size=0.2, random_state=42
# # )

# # # Initialize model and tokenizer with custom config
# # model_name = "facebook/bart-large-cnn"
# # tokenizer = BartTokenizer.from_pretrained(model_name)
# # config = BartConfig.from_pretrained(model_name)
# # config.forced_bos_token_id = 0  # Fix config warning
# # config.attention_probs_dropout_prob = 0.1  # Add regularization
# # model = BartForConditionalGeneration.from_pretrained(model_name, config=config)

# # # Curriculum learning setup (gradual unfreezing)
# # for param in model.parameters():
# #     param.requires_grad = False
    
# # for layer in [model.model.encoder.layers[-4:], model.model.decoder.layers[-4:]]:
# #     for param in layer.parameters():
# #         param.requires_grad = True

# # # Enhanced tokenization with legal-specific formatting
# # def tokenize_function(examples):
# #     inputs = ["Summarize legal case: " + text for text in examples['judgment']]
# #     targets = ["[LEGAL_SUM] " + text for text in examples['summary']]
    
# #     model_inputs = tokenizer(
# #         inputs,
# #         max_length=512,  # Reduced from 1024 to save memory
# #         truncation=True,
# #         padding=False  # Dynamic padding handled by collator
# #     )
    
# #     with tokenizer.as_target_tokenizer():
# #         labels = tokenizer(
# #             targets,
# #             max_length=256,
# #             truncation=True,
# #             padding=False
# #         )
    
# #     model_inputs["labels"] = labels["input_ids"]
# #     return model_inputs

# # # Create datasets
# # train_dataset = Dataset.from_dict({"judgment": train_texts, "summary": train_labels})
# # val_dataset = Dataset.from_dict({"judgment": val_texts, "summary": val_labels})
# # test_dataset = Dataset.from_dict({"judgment": test_judgments, "summary": test_summaries})

# # # Tokenize datasets
# # train_dataset = train_dataset.map(tokenize_function, batched=True)
# # val_dataset = val_dataset.map(tokenize_function, batched=True)
# # test_dataset = test_dataset.map(tokenize_function, batched=True)

# # # Data collator for dynamic padding
# # data_collator = DataCollatorForSeq2Seq(
# #     tokenizer,
# #     model=model,
# #     padding="longest",
# #     max_length=512,  # Reduced from 1024
# #     pad_to_multiple_of=8
# # )

# # # Metrics computation
# # def compute_metrics(eval_pred):
# #     predictions, labels = eval_pred
    
# #     # Decode predictions one at a time to save memory
# #     decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True, max_length=256) for pred in predictions]
# #     decoded_labels = [tokenizer.decode(label, skip_special_tokens=True, max_length=256) for label in labels]
    
# #     # Calculate ROUGE
# #     rouge_results = rouge.compute(
# #         predictions=decoded_preds,
# #         references=decoded_labels,
# #         use_stemmer=True
# #     )
    
# #     # Calculate BERTScore
# #     bert_results = bertscore.compute(
# #         predictions=decoded_preds,
# #         references=decoded_labels,
# #         lang="en",
# #         model_type="bert-base-multilingual-cased"
# #     )
    
# #     return {
# #         "rouge1": round(rouge_results["rouge1"], 4),
# #         "rougeL": round(rouge_results["rougeL"], 4),
# #         "bertscore": round(np.mean(bert_results["f1"]), 4)
# #     }

# # # Optimized training arguments
# # training_args = TrainingArguments(
# #     output_dir="./results",
# #     num_train_epochs=5,
# #     per_device_train_batch_size=1,
# #     per_device_eval_batch_size=1,  # Reduced from 2 to save memory
# #     learning_rate=3e-5,
# #     warmup_ratio=0.1,
# #     weight_decay=0.01,
# #     gradient_accumulation_steps=4,
# #     fp16=False,
# #     no_cuda=True,
# #     logging_dir="./logs",
# #     logging_steps=50,
# #     evaluation_strategy="epoch",
# #     save_strategy="epoch",  
# #     load_best_model_at_end=True,
# #     metric_for_best_model="rougeL",
# #     greater_is_better=True,
# #     group_by_length=True,
# #     report_to="none"
# # )

# # # Enhanced trainer setup
# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=train_dataset,
# #     eval_dataset=val_dataset,
# #     tokenizer=tokenizer,
# #     data_collator=data_collator,
# #     compute_metrics=compute_metrics
# # )

# # # Training
# # trainer.train()

# # # Save model with proper config
# # model.config.save_pretrained("./bart-finetuned2")
# # model.save_pretrained("./bart-finetuned2")
# # tokenizer.save_pretrained("./bart-finetuned2")

# # # Free memory before evaluation
# # gc.collect()

# # # Final evaluation
# # test_results = trainer.evaluate(test_dataset)
# # print("\nFinal Test Results:")
# # print(f"ROUGE-L: {test_results['eval_rougeL']}")
# # print(f"BERTScore F1: {test_results['eval_bertscore']}")

# # # Post-training optimization with quantization
# # quantized_model = torch.quantization.quantize_dynamic(
# #     model,
# #     {torch.nn.Linear},
# #     dtype=torch.qint8
# # )

# # # Save the quantized model
# # quantized_model.save_pretrained("./bart-finetuned2-quantized")

# # # Use quantized model for evaluation
# # trainer.model = quantized_model

# # # Re-evaluate with quantized model
# # gc.collect()
# # test_results = trainer.evaluate(test_dataset)
# # print("\nOptimized Model Test Results:")
# # print(f"ROUGE-L: {test_results['eval_rougeL']}")
# # print(f"BERTScore F1: {test_results['eval_bertscore']}")

# # print("\nOptimized model saved with quantization.")

