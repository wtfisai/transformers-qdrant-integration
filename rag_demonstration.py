#!/usr/bin/env python
"""
RAG (Retrieval Augmented Generation) Demonstration using Qdrant and LLMs

This script demonstrates a complete RAG workflow:
1. Load sample documents
2. Create embeddings
3. Store in Qdrant vector database
4. Perform semantic search
5. Generate responses using LLMs with context from retrieved documents
"""

import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from api_config.env
load_dotenv("api_config.env")

# Check if required packages are installed
try:
    from langchain_community.vectorstores import Qdrant
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import ChatOpenAI
    from langchain.schema import Document
    from qdrant_client import QdrantClient
except ImportError:
    print("Required packages not found. Please install them with:")
    print("pip install langchain langchain-community langchain-openai qdrant-client")
    sys.exit(1)

# Configuration
COLLECTION_NAME = "demo_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with connections to Qdrant and OpenAI."""
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not all([self.qdrant_url, self.qdrant_api_key, self.openai_api_key]):
            raise ValueError("Missing required environment variables. Please check api_config.env file.")
        
        # Initialize clients
        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        self.embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            api_key=self.openai_api_key
        )
        
        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        print(f"✅ Successfully initialized RAG system")

    def create_collection(self):
        """Create a new Qdrant collection for document storage."""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if COLLECTION_NAME in collection_names:
            print(f"Collection '{COLLECTION_NAME}' already exists.")
            return
        
        # Get vector size from embeddings
        vector_size = len(self.embeddings.embed_query("Test"))
        
        # Create collection
        self.qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": vector_size, "distance": "Cosine"}
        )
        
        print(f"✅ Created collection '{COLLECTION_NAME}' with vector size {vector_size}")

    def load_sample_documents(self) -> List[Document]:
        """Load sample documents for demonstration."""
        documents = [
            Document(
                page_content="Qdrant is a vector database designed for storing, managing, and retrieving high-dimensional vectors. "
                "It's particularly useful for similarity search, which is a core component in many machine learning applications, "
                "including semantic search, recommendation systems, and more. Qdrant provides real-time filtering during search "
                "operations, making it possible to combine traditional boolean conditions with vector similarity search.",
                metadata={"source": "Qdrant Documentation", "topic": "vector_databases"}
            ),
            Document(
                page_content="OpenAI's GPT (Generative Pre-trained Transformer) models are large language models known for their "
                "ability to generate coherent and contextually relevant text. These models have been trained on vast amounts of "
                "text data and can perform a wide range of language tasks including translation, summarization, question answering, "
                "and creative writing. The latest versions have significantly improved capabilities compared to earlier versions.",
                metadata={"source": "AI Documentation", "topic": "language_models"}
            ),
            Document(
                page_content="Retrieval-Augmented Generation (RAG) is an AI framework that combines the strengths of retrieval-based "
                "and generation-based approaches. In RAG systems, a retrieval component first fetches relevant documents or information "
                "from an external knowledge source, and then a generator creates responses based on both the retrieved information and "
                "the original query. This approach helps ground the model's responses in factual information and reduces hallucinations.",
                metadata={"source": "AI Research Papers", "topic": "rag_systems"}
            ),
            Document(
                page_content="Hugging Face Transformers is a popular library that provides pre-trained models for natural language "
                "processing tasks. It supports a wide variety of transformer models like BERT, GPT, T5, and others. The library makes "
                "it easy to fine-tune these models on custom datasets and deploy them for specific tasks such as classification, "
                "named entity recognition, question answering, and more.",
                metadata={"source": "Hugging Face Documentation", "topic": "transformers"}
            ),
            Document(
                page_content="LangSmith is a platform for monitoring and debugging Language Model (LLM) applications. It provides "
                "tools for tracking prompts and completions, evaluating model responses, and identifying areas for improvement. "
                "LangSmith integrates with LangChain, making it easier to develop, debug, and improve LLM applications.",
                metadata={"source": "LangSmith Documentation", "topic": "langsmith"}
            )
        ]
        
        print(f"✅ Loaded {len(documents)} sample documents")
        return documents

    def process_documents(self, documents: List[Document]):
        """Process documents and store them in Qdrant."""
        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(documents)
        print(f"✅ Split {len(documents)} documents into {len(split_docs)} chunks")
        
        # Create Qdrant vector store
        vector_store = Qdrant.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection_name=COLLECTION_NAME,
            force_recreate=True
        )
        
        print(f"✅ Documents stored in Qdrant collection '{COLLECTION_NAME}'")
        return vector_store

    def retrieve_similar_documents(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve documents similar to the query."""
        # Create a vector store client
        vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
            embeddings=self.embeddings
        )
        
        # Retrieve similar documents
        documents = vector_store.similarity_search(query, k=k)
        
        print(f"✅ Retrieved {len(documents)} documents similar to query: '{query}'")
        return documents

    def generate_response(self, query: str, documents: List[Document]) -> str:
        """Generate a response based on the query and retrieved documents."""
        # Prepare context from documents
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Prepare prompt
        prompt = f"""
        Answer the question based on the context provided. If the context doesn't contain 
        relevant information, state that you don't have enough information.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        
        # Get response from LLM
        response = self.llm.invoke(prompt).content
        return response


def main():
    print("Starting RAG Demonstration...")
    
    # Initialize RAG system
    rag_system = RAGSystem()
    
    # Create collection
    rag_system.create_collection()
    
    # Load and process documents
    documents = rag_system.load_sample_documents()
    rag_system.process_documents(documents)
    
    # Example queries
    queries = [
        "How does Qdrant help with vector search?",
        "What is RAG and how does it work?",
        "Can you explain how Hugging Face Transformers and LangSmith can work together?"
    ]
    
    # Process each query
    print("\n" + "="*50)
    print("Demonstrating RAG Question-Answering")
    print("="*50)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        # Retrieve relevant documents
        retrieved_docs = rag_system.retrieve_similar_documents(query)
        
        print("Retrieved Documents:")
        for j, doc in enumerate(retrieved_docs, 1):
            print(f"Document {j} (Source: {doc.metadata['source']}, Topic: {doc.metadata['topic']}):")
            print(f"  {doc.page_content[:100]}...")
        
        # Generate response
        response = rag_system.generate_response(query, retrieved_docs)
        
        print("\nGenerated Response:")
        print(response)
        print("="*50)


if __name__ == "__main__":
    main()
