#!/usr/bin/env python3
"""Hugging Face Transformers Fine-tuning with Trainer API
This script demonstrates how to fine-tune a pretrained model using the Trainer API.
"""

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_data():
    """Load and prepare the dataset for fine-tuning."""
    print("\n=== Loading Dataset ===")
    
    # Load the IMDB dataset (a binary sentiment classification dataset)
    # Using a small subset for demonstration purposes
    dataset = load_dataset("imdb", split="train[:1000]")
    
    # Split the dataset into training and validation sets
    dataset = dataset.train_test_split(test_size=0.2)
    
    print(f"Training set size: {len(dataset['train'])}")
    print(f"Validation set size: {len(dataset['test'])}")
    
    return dataset

def preprocess_data(dataset, tokenizer):
    """Tokenize and prepare the dataset for training."""
    print("\n=== Preprocessing Data ===")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    
    # Tokenize the datasets
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Set the format for PyTorch
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    print("Data preprocessing complete")
    
    return tokenized_datasets

def compute_metrics(eval_pred):
    """Compute evaluation metrics for the model."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(tokenized_datasets, tokenizer, model):
    """Fine-tune the model using the Trainer API."""
    print("\n=== Training Model ===")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",  # Evaluate at specific steps
        eval_steps=500,              # Evaluate every 500 steps
        save_strategy="steps",       # Save at specific steps
        save_steps=500,              # Save checkpoint every 500 steps
        load_best_model_at_end=True, # Load the best model at the end of training
        push_to_hub=False,
    )
    
    # Create a data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate the model
    print("\n=== Evaluating Model ===")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    return trainer

def test_finetuned_model(trainer):
    """Test the fine-tuned model on sample inputs."""
    print("\n=== Testing Fine-tuned Model ===")
    
    # Sample texts for testing
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "The acting was terrible and the plot made no sense at all.",
        "It was an okay film, had some good moments but also some flaws."
    ]
    
    # Get the tokenizer and model from the trainer
    tokenizer = trainer.tokenizer
    model = trainer.model
    
    # Move model to the appropriate device
    model = model.to(device)
    
    # Process each test text
    for text in test_texts:
        # Tokenize the input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process the outputs
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        positive_score = predictions[0][1].item()
        negative_score = predictions[0][0].item()
        
        print(f"\nText: {text}")
        print(f"Positive score: {positive_score:.4f}")
        print(f"Negative score: {negative_score:.4f}")
        print(f"Predicted sentiment: {'POSITIVE' if positive_score > negative_score else 'NEGATIVE'}")

if __name__ == "__main__":
    # Load the pretrained model and tokenizer
    model_name = "distilbert-base-uncased"
    print(f"Loading base model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Load and preprocess the dataset
    dataset = load_data()
    tokenized_datasets = preprocess_data(dataset, tokenizer)
    
    # Train the model
    trainer = train_model(tokenized_datasets, tokenizer, model)
    
    # Test the fine-tuned model
    test_finetuned_model(trainer)
    
    print("\n=== Fine-tuning Complete ===")
    print("The model has been fine-tuned on the IMDB dataset for sentiment analysis.")
