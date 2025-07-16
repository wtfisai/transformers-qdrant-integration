# Transformers & Qdrant Integration

This repository demonstrates integration between Hugging Face Transformers, Qdrant vector database, and various LLM APIs including OpenAI, Anthropic, and Mistral.

## Components

1. **Qdrant Client Setup**
   - Connection to Qdrant vector database
   - Creating and managing vector collections

2. **Transformers Integration**
   - Loading pretrained models
   - Running inference with the Pipeline API
   - Fine-tuning models on custom datasets

3. **LLM API Integration**
   - OpenAI
   - Anthropic
   - Mistral

4. **LangSmith Monitoring**
   - Tracking and debugging LLM applications

## Scripts Overview

- `qdrant_setup.py`: Setup and test Qdrant client connection
- `transformers_quickstart.py`: Basic usage of Hugging Face Transformers
- `transformers_finetuning.py`: Fine-tuning a model on a dataset
- `simple_finetuning.py`: Simplified fine-tuning script
- `llm_qdrant_integration.py`: Integration of Qdrant with various LLM APIs
- `rag_demonstration.py`: Complete RAG workflow demonstration

## RAG Demonstration

The `rag_demonstration.py` script demonstrates a complete Retrieval Augmented Generation (RAG) workflow:

1. **Document Loading**: Loads sample documents about Qdrant, OpenAI, RAG systems, Transformers, and LangSmith.
2. **Document Processing**: Splits documents into chunks and creates embeddings.
3. **Vector Storage**: Stores document embeddings in a Qdrant collection.
4. **Semantic Search**: Performs similarity search to retrieve relevant documents.
5. **Response Generation**: Generates responses to queries using the retrieved context.

### Running the RAG Demo

Before running the demo, ensure you have set up the required API keys in `api_config.env`:

```
# API Keys Configuration
OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
QDRANT_URL="<YOUR_QDRANT_URL>"
QDRANT_API_KEY="<YOUR_QDRANT_API_KEY>"
```

Install the required packages:

```bash
pip install langchain langchain-community langchain-openai qdrant-client python-dotenv
```

Then run the demonstration:

```bash
python rag_demonstration.py
```

### Sample Queries

The demo includes the following sample queries:

1. "How does Qdrant help with vector search?"
2. "What is RAG and how does it work?"
3. "Can you explain how Hugging Face Transformers and LangSmith can work together?"

## Environment Setup

This project uses Python virtual environments:

```bash
# Create and activate virtual environment
python -m venv transformers_env
source transformers_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Documentation

For more detailed documentation, see:

- [Transformers Documentation](transformers_documentation.md)
- [LLM Integration Documentation](llm_integration_documentation.md)

## License

This project is open-source and available under the MIT License.
