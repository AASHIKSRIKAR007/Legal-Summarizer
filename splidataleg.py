import json
import random

def split_dataset(input_file, train_file, test_file, test_size=0.2, seed=42):
    """
    Splits a dataset into training and testing datasets.
    
    Args:
        input_file (str): Path to the input JSON file.
        train_file (str): Path to save the training dataset JSON file.
        test_file (str): Path to save the testing dataset JSON file.
        test_size (float): Proportion of the dataset to include in the test split.
        seed (int): Random seed for reproducibility.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load the unified dataset
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # Shuffle the data
    random.shuffle(data)
    
    # Split the data
    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Save the train and test datasets
    with open(train_file, "w", encoding="utf-8") as file:
        json.dump(train_data, file, indent=4, ensure_ascii=False)
    with open(test_file, "w", encoding="utf-8") as file:
        json.dump(test_data, file, indent=4, ensure_ascii=False)
    
    print(f"Dataset split completed.")
    print(f"Training data: {len(train_data)} samples")
    print(f"Testing data: {len(test_data)} samples")

# Specify file paths
input_path = "final_dataset.json"  # Path to your unified dataset
train_path = "train_dataset.json"    # Path to save the training dataset
test_path = "test_dataset.json"      # Path to save the testing dataset

# Split the dataset
split_dataset(input_path, train_path, test_path)
