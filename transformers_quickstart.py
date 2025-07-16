#!/usr/bin/env python3
"""
Hugging Face Transformers Quickstart
This script demonstrates how to:
1. Load a pretrained model and tokenizer
2. Run inference using the Pipeline API
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Check if CUDA is available and set the device accordingly
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")

def load_model_and_tokenizer():
    """Load a pretrained model and tokenizer for sentiment analysis."""
    print("\n=== Loading Model and Tokenizer ===")
    
    # Load a pretrained model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    return model, tokenizer

def run_pipeline_inference():
    """Demonstrate using the Pipeline API for inference."""
    print("\n=== Running Inference with Pipeline API ===")
    
    # Create a sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", device=device)
    
    # Example texts for sentiment analysis
    texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is terrible. I'm very disappointed with the quality.",
        "The movie was okay, not great but not bad either."
    ]
    
    # Run inference
    print("Running sentiment analysis on example texts:")
    for text in texts:
        result = sentiment_pipeline(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result[0]['label']}")
        print(f"Confidence: {result[0]['score']:.4f}")

def manual_inference(model, tokenizer):
    """Demonstrate manual inference using the model and tokenizer directly."""
    print("\n=== Manual Inference with Model and Tokenizer ===")
    
    # Move model to appropriate device
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device_name)
    
    # Example text
    text = "I really enjoyed using this product, it exceeded my expectations!"
    
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt").to(device_name)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process the outputs
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = predictions[0][1].item()
    negative_score = predictions[0][0].item()
    
    print(f"Text: {text}")
    print(f"Positive score: {positive_score:.4f}")
    print(f"Negative score: {negative_score:.4f}")
    print(f"Predicted sentiment: {'POSITIVE' if positive_score > negative_score else 'NEGATIVE'}")

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Run inference using the Pipeline API
    run_pipeline_inference()
    
    # Run manual inference
    manual_inference(model, tokenizer)
