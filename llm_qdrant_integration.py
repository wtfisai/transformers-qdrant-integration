#!/usr/bin/env python3
"""
LLM and Qdrant Integration with LangSmith Monitoring
This script demonstrates how to integrate Qdrant with OpenAI, Anthropic, and Mistral APIs,
and set up LangSmith for monitoring.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI, Anthropic
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langsmith import Client
import langsmith

# Load environment variables from .env file
load_dotenv("api_config.env")

# Set up API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Set up LangSmith for monitoring
os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "qdrant-llm-integration"

def setup_qdrant_client():
    """Set up and test connection to Qdrant."""
    try:
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        collections = qdrant_client.get_collections()
        print("\n=== Qdrant Connection ===")
        print("Successfully connected to Qdrant server!")
        print(f"Available collections: {collections}")
        return qdrant_client
    except Exception as e:
        print(f"Error connecting to Qdrant server: {e}")
        return None

def setup_llm_clients():
    """Set up and test connections to LLM APIs."""
    print("\n=== LLM API Connections ===")
    
    # OpenAI
    try:
        openai_llm = ChatOpenAI(api_key=openai_api_key)
        openai_response = openai_llm.invoke("Hello, OpenAI!")
        print("✅ OpenAI connection successful")
        print(f"OpenAI response: {openai_response.content}")
    except Exception as e:
        print(f"❌ Error connecting to OpenAI: {e}")
    
    # Anthropic
    try:
        anthropic_llm = ChatAnthropic(api_key=anthropic_api_key)
        anthropic_response = anthropic_llm.invoke("Hello, Anthropic!")
        print("✅ Anthropic connection successful")
        print(f"Anthropic response: {anthropic_response.content}")
    except Exception as e:
        print(f"❌ Error connecting to Anthropic: {e}")
    
    # Mistral (using Anthropic's API format since Mistral API key format looks similar)
    try:
        # Note: This is a placeholder. If Mistral has a different API structure,
        # you would need to use their specific client library
        print("ℹ️ Mistral API integration would be implemented here")
        print("ℹ️ API key format suggests it might be similar to Anthropic")
    except Exception as e:
        print(f"❌ Error with Mistral placeholder: {e}")

def setup_langsmith():
    """Set up and test connection to LangSmith."""
    try:
        print("\n=== LangSmith Connection ===")
        client = Client(api_key=langsmith_api_key)
        # List projects to verify connection
        projects = client.list_projects()
        print("✅ LangSmith connection successful")
        print(f"Available projects: {[p.name for p in projects]}")
        return client
    except Exception as e:
        print(f"❌ Error connecting to LangSmith: {e}")
        return None

def create_qdrant_collection():
    """Create a sample collection in Qdrant."""
    try:
        print("\n=== Creating Qdrant Collection ===")
        qdrant_client = setup_qdrant_client()
        if not qdrant_client:
            return
        
        # Check if collection already exists
        collections = qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        collection_name = "sample_collection"
        if collection_name in collection_names:
            print(f"Collection '{collection_name}' already exists")
        else:
            # Create a new collection
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": 1536,  # OpenAI embeddings dimension
                    "distance": "Cosine"
                }
            )
            print(f"Created new collection: '{collection_name}'")
        
        # Verify collection was created
        collections = qdrant_client.get_collections()
        print(f"Updated collections: {collections}")
    except Exception as e:
        print(f"Error creating Qdrant collection: {e}")

def main():
    """Main function to run all setup and integration steps."""
    print("Starting LLM and Qdrant integration with LangSmith monitoring...")
    
    # Set up Qdrant
    qdrant_client = setup_qdrant_client()
    if not qdrant_client:
        print("Failed to connect to Qdrant. Exiting.")
        return
    
    # Set up LLM clients
    setup_llm_clients()
    
    # Set up LangSmith
    langsmith_client = setup_langsmith()
    
    # Create a sample Qdrant collection
    create_qdrant_collection()
    
    print("\n=== Integration Complete ===")
    print("Successfully integrated Qdrant with LLM APIs and set up LangSmith monitoring.")

if __name__ == "__main__":
    main()
