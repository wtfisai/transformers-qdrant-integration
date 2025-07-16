# Transformers Quickstart Documentation

This document summarizes the steps taken to set up and use Hugging Face Transformers with PyTorch and CUDA support.

## Environment Setup

1. Created a Python virtual environment for Transformers work:
   ```bash
   python -m venv transformers_env
   source transformers_env/bin/activate
   ```

2. Installed required packages:
   ```bash
   pip install transformers torch datasets scikit-learn
   ```

3. Authenticated with Hugging Face CLI:
   ```bash
   huggingface-cli login
   # Used token: <YOUR_HUGGINGFACE_TOKEN>
   ```

4. Verified CUDA availability:
   ```python
   # From check_cuda.py
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
   ```

## Transformers Quickstart

### 1. Loading a Pretrained Model and Tokenizer

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

### 2. Running Inference with Pipeline API

```python
from transformers import pipeline

# Create a sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)

# Analyze text
results = sentiment_analyzer([
    "I love this movie!",
    "This was a terrible experience."
])
```

### 3. Manual Inference with Model and Tokenizer

```python
import torch

# Tokenize input text
inputs = tokenizer("I love this movie!", return_tensors="pt").to("cuda")

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)

# Process outputs
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
positive_score = predictions[0][1].item()
negative_score = predictions[0][0].item()
```

## Fine-tuning a Model

We fine-tuned a DistilBERT model on the IMDB dataset for sentiment classification:

1. Loaded and preprocessed the IMDB dataset
2. Set up training arguments with the Trainer API
3. Trained the model for 3 epochs
4. Evaluated the model on a test set

```python
# Key components of fine-tuning
from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
```

## Results and Observations

- Successfully loaded and used pretrained models for sentiment analysis
- Pipeline API provides a simple interface for common NLP tasks
- Fine-tuning was completed successfully, but the model showed bias toward negative predictions
- For better fine-tuning results, consider:
  - Using more training data
  - Adjusting hyperparameters
  - Trying different model architectures
  - Implementing class balancing techniques

## Next Steps

- Push fine-tuned models to Hugging Face Hub
- Explore other NLP tasks (text generation, question answering, etc.)
- Integrate Transformers models with Qdrant for vector search
- Deploy models for production use

## Files Created

1. `check_cuda.py` - Script to verify CUDA availability
2. `transformers_quickstart.py` - Basic usage of Transformers models
3. `transformers_finetuning.py` - Original fine-tuning script (had compatibility issues)
4. `simple_finetuning.py` - Simplified fine-tuning script compatible with transformers 4.53.2
5. `transformers_documentation.md` - This documentation file
