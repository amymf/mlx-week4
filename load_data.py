import torch
from datasets import load_dataset

ds = load_dataset("nlphuji/flickr30k")

torch.manual_seed(42)  # For reproducibility

split_dataset = ds["test"].train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
