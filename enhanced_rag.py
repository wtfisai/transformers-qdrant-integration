#!/usr/bin/env python
"""
Enhanced RAG (Retrieval Augmented Generation) Demonstration

This script extends the basic RAG workflow with:
1. Multiple embedding models support
2. Document source loaders
3. More advanced query formulation
4. Response evaluation and feedback
5. Caching for improved performance
"""

import os
import sys
import time
import json
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from api_config.env
load_dotenv("api_config.env")

# Check if required packages are installed
try:
    from langchain_community.vectorstores import Qdrant
    from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    from langchain.schema.output_parser import StrOutputParser
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.cache import InMemoryCache
    from langchain.globals import set_llm_cache
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError:
    print("Required packages not found. Please install them with:")
    print("pip install langchain langchain-community langchain-openai qdrant-client")
    sys.exit(1)

# Set up caching
set_llm_cache(InMemoryCache())

# Configuration
COLLECTION_NAME = "enhanced_demo_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models"""
    name: str
    dim: int
    model_config: Dict[str, Any] = field(default_factory=dict)


class DocumentSource:
    """Base class for document sources"""
    def __init__(self, name: str):
        self.name = name
    
    def load_documents(self) -> List[Document]:
        """Load documents from the source"""
        raise NotImplementedError("Subclasses must implement load_documents")


class SampleDocumentSource(DocumentSource):
    """Sample document source with predefined documents"""
    def __init__(self):
        super().__init__("sample_documents")
    
    def load_documents(self) -> List[Document]:
        """Load sample documents"""
        documents = [
            Document(
                page_content="Qdrant is a vector database designed for storing, managing, and retrieving high-dimensional vectors. "
                "It's particularly useful for similarity search, which is a core component in many machine learning applications, "
                "including semantic search, recommendation systems, and more. Qdrant provides real-time filtering during search "
                "operations, making it possible to combine traditional boolean conditions with vector similarity search.",
                metadata={"source": "Qdrant Documentation", "topic": "vector_databases", "importance": "high"}
            ),
            Document(
                page_content="OpenAI's GPT (Generative Pre-trained Transformer) models are large language models known for their "
                "ability to generate coherent and contextually relevant text. These models have been trained on vast amounts of "
                "text data and can perform a wide range of language tasks including translation, summarization, question answering, "
                "and creative writing. The latest versions have significantly improved capabilities compared to earlier versions.",
                metadata={"source": "AI Documentation", "topic": "language_models", "importance": "high"}
            ),
            Document(
                page_content="Retrieval-Augmented Generation (RAG) is an AI framework that combines the strengths of retrieval-based "
                "and generation-based approaches. In RAG systems, a retrieval component first fetches relevant documents or information "
                "from an external knowledge source, and then a generator creates responses based on both the retrieved information and "
                "the original query. This approach helps ground the model's responses in factual information and reduces hallucinations.",
                metadata={"source": "AI Research Papers", "topic": "rag_systems", "importance": "high"}
            ),
            Document(
                page_content="Hugging Face Transformers is a popular library that provides pre-trained models for natural language "
                "processing tasks. It supports a wide variety of transformer models like BERT, GPT, T5, and others. The library makes "
                "it easy to fine-tune these models on custom datasets and deploy them for specific tasks such as classification, "
                "named entity recognition, question answering, and more.",
                metadata={"source": "Hugging Face Documentation", "topic": "transformers", "importance": "medium"}
            ),
            Document(
                page_content="LangSmith is a platform for monitoring and debugging Language Model (LLM) applications. It provides "
                "tools for tracking prompts and completions, evaluating model responses, and identifying areas for improvement. "
                "LangSmith integrates with LangChain, making it easier to develop, debug, and improve LLM applications.",
                metadata={"source": "LangSmith Documentation", "topic": "langsmith", "importance": "medium"}
            ),
            Document(
                page_content="Hybrid search combines vector similarity search with traditional keyword-based search to get the "
                "best of both worlds. It's especially useful when dealing with edge cases where pure vector search might miss "
                "relevant results. In a hybrid search approach, both vector similarity and keyword relevance scores are computed, "
                "then combined using various methods such as weighted sums or re-ranking.",
                metadata={"source": "Search Systems Documentation", "topic": "hybrid_search", "importance": "medium"}
            ),
            Document(
                page_content="When implementing RAG systems, chunking strategies are crucial for effective retrieval. Text chunks "
                "should be sized appropriately - not too small to lose context, not too large to introduce noise. Overlapping chunks "
                "can help maintain contextual continuity. Advanced techniques include hierarchical chunking, where documents are "
                "represented at different levels of granularity, and semantic chunking, where natural semantic boundaries guide the splits.",
                metadata={"source": "RAG Implementation Guide", "topic": "chunking_strategies", "importance": "high"}
            ),
        ]
        logger.info(f"Loaded {len(documents)} sample documents")
        return documents


class TextFileDocumentSource(DocumentSource):
    """Document source that loads documents from text files in a directory"""
    def __init__(self, directory: str):
        super().__init__(f"text_files_{os.path.basename(directory)}")
        self.directory = directory
    
    def load_documents(self) -> List[Document]:
        """Load documents from text files"""
        documents = []
        if not os.path.exists(self.directory):
            logger.warning(f"Directory {self.directory} does not exist")
            return documents
        
        for filename in os.listdir(self.directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.directory, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        doc = Document(
                            page_content=content,
                            metadata={"source": file_path, "filename": filename}
                        )
                        documents.append(doc)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {self.directory}")
        return documents


class EnhancedRAGSystem:
    def __init__(self):
        """Initialize the enhanced RAG system with connections to Qdrant and LLM APIs."""
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not all([self.qdrant_url, self.qdrant_api_key, self.openai_api_key]):
            raise ValueError("Missing required environment variables. Please check api_config.env file.")
        
        # Initialize clients
        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        
        # Define available embedding models
        self.embedding_models = {
            "openai": EmbeddingModelConfig(
                name="openai",
                dim=1536,
                model_config={"model": "text-embedding-ada-002"}
            ),
            "huggingface-mpnet": EmbeddingModelConfig(
                name="huggingface-mpnet",
                dim=768,
                model_config={"model_name": "sentence-transformers/all-mpnet-base-v2"}
            )
        }
        
        # Default embedding model
        self.embeddings = self._get_embedding_model("openai")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            api_key=self.openai_api_key,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Initialize response cache
        self.response_cache = {}
        
        logger.info("Successfully initialized Enhanced RAG system")

    def _get_embedding_model(self, model_name: str):
        """Get embedding model by name"""
        if model_name not in self.embedding_models:
            raise ValueError(f"Unknown embedding model: {model_name}")
        
        config = self.embedding_models[model_name]
        
        if model_name == "openai":
            return OpenAIEmbeddings(
                api_key=self.openai_api_key,
                **config.model_config
            )
        elif model_name.startswith("huggingface"):
            return HuggingFaceEmbeddings(**config.model_config)
        else:
            raise ValueError(f"Unsupported embedding model type: {model_name}")

    def change_embedding_model(self, model_name: str):
        """Change the current embedding model"""
        self.embeddings = self._get_embedding_model(model_name)
        logger.info(f"Changed embedding model to {model_name}")
        return self.embeddings

    def create_collection(self, model_name: str = None):
        """Create a new Qdrant collection for document storage."""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if COLLECTION_NAME in collection_names:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists.")
            return
        
        # Get embedding model
        if model_name:
            embeddings = self._get_embedding_model(model_name)
        else:
            embeddings = self.embeddings
        
        # Get vector size from config
        model_config = next((c for c in self.embedding_models.values() if c.name == embeddings.__class__.__name__), None)
        vector_size = model_config.dim if model_config else len(embeddings.embed_query("Test"))
        
        # Create collection
        self.qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": vector_size, "distance": "Cosine"}
        )
        
        logger.info(f"Created collection '{COLLECTION_NAME}' with vector size {vector_size}")

    def load_documents(self, document_sources: List[DocumentSource]) -> List[Document]:
        """Load documents from multiple sources"""
        all_documents = []
        for source in document_sources:
            documents = source.load_documents()
            all_documents.extend(documents)
        
        logger.info(f"Loaded {len(all_documents)} documents from {len(document_sources)} sources")
        return all_documents

    def process_documents(self, documents: List[Document]) -> Qdrant:
        """Process documents and store them in Qdrant."""
        start_time = time.time()
        
        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        
        # Create Qdrant vector store
        vector_store = Qdrant.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection_name=COLLECTION_NAME,
            force_recreate=True
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Documents processed and stored in {elapsed_time:.2f} seconds")
        return vector_store

    def search_documents(self, query: str, filter_criteria: Optional[Dict[str, Any]] = None, 
                        k: int = 4, method: str = "similarity") -> List[Document]:
        """
        Search for documents using different methods.
        
        Args:
            query: The search query
            filter_criteria: Optional filter for metadata
            k: Number of documents to retrieve
            method: Search method ('similarity', 'mmr', or 'hybrid')
        
        Returns:
            List of retrieved documents
        """
        # Create vector store client
        vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
            embeddings=self.embeddings
        )
        
        # Prepare filter if provided
        qdrant_filter = None
        if filter_criteria:
            filter_conditions = []
            for key, value in filter_criteria.items():
                if isinstance(value, list):
                    filter_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchAny(any=value)
                        )
                    )
                else:
                    filter_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=value)
                        )
                    )
            
            qdrant_filter = models.Filter(
                must=filter_conditions
            )
        
        # Perform search based on method
        start_time = time.time()
        
        if method == "mmr":
            # Maximum Marginal Relevance search for diversity
            documents = vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=k*2, filter=qdrant_filter
            )
        elif method == "hybrid":
            # Hybrid search (requires langchain-community >= 0.0.10)
            documents = vector_store.hybrid_search(
                query, k=k, filter=qdrant_filter, 
                alpha=0.5  # Balance between vector and keyword search
            )
        else:
            # Default similarity search
            documents = vector_store.similarity_search(
                query, k=k, filter=qdrant_filter
            )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Retrieved {len(documents)} documents in {elapsed_time:.2f} seconds using {method} search")
        return documents

    def generate_response(self, query: str, documents: List[Document], 
                         response_type: str = "detailed") -> str:
        """
        Generate a response based on the query and retrieved documents.
        
        Args:
            query: The user query
            documents: Retrieved documents
            response_type: Type of response ('concise', 'detailed', or 'expert')
        
        Returns:
            Generated response
        """
        # Check cache first
        cache_key = f"{query}_{response_type}_{','.join([doc.page_content[:50] for doc in documents])}"
        if cache_key in self.response_cache:
            logger.info("Using cached response")
            return self.response_cache[cache_key]
        
        # Prepare context from documents
        context = "\n\n".join([
            f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        
        # Select prompt based on response type
        if response_type == "concise":
            prompt_template = """
            Answer the question concisely based on the provided context. Focus only on the most important points.
            
            Context:
            {context}
            
            Question: {query}
            
            Concise Answer:
            """
        elif response_type == "expert":
            prompt_template = """
            You are an expert in the subject matter. Answer the question based on the provided context,
            using technical language and detailed explanations. Cite the specific documents you're drawing from.
            
            Context:
            {context}
            
            Question: {query}
            
            Expert Answer:
            """
        else:  # detailed
            prompt_template = """
            Answer the question based on the context provided. If the context doesn't contain
            relevant information, state that you don't have enough information.
            
            Context:
            {context}
            
            Question: {query}
            
            Detailed Answer:
            """
        
        # Create prompt
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Get response from LLM
        start_time = time.time()
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "query": query})
        elapsed_time = time.time() - start_time
        
        logger.info(f"Generated {response_type} response in {elapsed_time:.2f} seconds")
        
        # Cache the response
        self.response_cache[cache_key] = response
        
        return response

    def evaluate_response(self, query: str, response: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Evaluate the quality of the response based on relevance, factuality, and completeness.
        
        Args:
            query: The original query
            response: The generated response
            documents: The retrieved documents used for generation
            
        Returns:
            Dictionary with evaluation scores and feedback
        """
        # Create evaluation prompt
        evaluation_prompt = f"""
        Evaluate the quality of the following response to a query based on the provided context documents.
        Rate each criteria on a scale of 1-5 (5 being best).
        
        Query: {query}
        
        Response: {response}
        
        Context Documents:
        {', '.join([doc.metadata.get('source', 'Unknown') for doc in documents])}
        
        Provide a JSON object with the following:
        1. relevance_score (1-5): How relevant is the response to the query?
        2. factuality_score (1-5): How factually accurate is the response based on the context?
        3. completeness_score (1-5): How completely does the response answer the query?
        4. feedback: Brief feedback explaining the ratings
        """
        
        # Get evaluation from LLM
        evaluation_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            api_key=self.openai_api_key
        )
        
        # Parse result as JSON
        try:
            result = evaluation_llm.invoke(evaluation_prompt).content
            # Extract JSON from response (handle cases where LLM might add text around the JSON)
            json_str = result.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].strip()
            
            evaluation = json.loads(json_str)
            logger.info(f"Response evaluation completed: {evaluation}")
            return evaluation
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {
                "relevance_score": 0,
                "factuality_score": 0,
                "completeness_score": 0,
                "feedback": f"Error evaluating response: {str(e)}"
            }


def main():
    """Main function to demonstrate the Enhanced RAG system"""
    print("\n" + "="*70)
    print("ENHANCED RAG SYSTEM DEMONSTRATION")
    print("="*70 + "\n")
    
    try:
        # Initialize RAG system
        rag_system = EnhancedRAGSystem()
        
        # Create collection
        rag_system.create_collection()
        
        # Load documents from multiple sources
        document_sources = [
            SampleDocumentSource(),
            # Uncomment to load from a directory of text files
            # TextFileDocumentSource("./sample_documents")
        ]
        
        documents = rag_system.load_documents(document_sources)
        
        # Process and store documents
        rag_system.process_documents(documents)
        
        # Example queries with different response types
        queries = [
            ("How does Qdrant help with vector search?", "concise"),
            ("What is RAG and how does it work?", "detailed"),
            ("Explain chunking strategies for RAG implementations.", "expert")
        ]
        
        # Example metadata filters
        filters = [
            None,
            {"topic": "rag_systems"},
            {"importance": "high"}
        ]
        
        # Example search methods
        search_methods = ["similarity", "mmr", "hybrid"]
        
        # Process each query with different parameters
        print("\n" + "="*70)
        print("DEMONSTRATING ENHANCED RAG CAPABILITIES")
        print("="*70)
        
        for i, (query, response_type) in enumerate(queries, 1):
            print(f"\n\nQUERY {i}: {query}")
            print("-" * 70)
            print(f"Response Type: {response_type}")
            
            # Use different filter and search method for each query
            filter_criteria = filters[i % len(filters)]
            search_method = search_methods[i % len(search_methods)]
            
            if filter_criteria:
                filter_str = ", ".join(f"{k}={v}" for k, v in filter_criteria.items())
                print(f"Filter: {filter_str}")
            print(f"Search Method: {search_method}")
            print("-" * 70)
            
            # Retrieve relevant documents
            retrieved_docs = rag_system.search_documents(
                query, 
                filter_criteria=filter_criteria,
                method=search_method
            )
            
            print("\nRetrieved Documents:")
            for j, doc in enumerate(retrieved_docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                topic = doc.metadata.get('topic', 'Unknown')
                print(f"Document {j} (Source: {source}, Topic: {topic}):")
                print(f"  {doc.page_content[:100]}...")
            
            # Generate response
            print("\nGenerating Response:")
            response = rag_system.generate_response(query, retrieved_docs, response_type)
            
            # Evaluate response
            print("\nEvaluating Response Quality...")
            evaluation = rag_system.evaluate_response(query, response, retrieved_docs)
            
            print("\nResponse Evaluation:")
            print(f"Relevance: {evaluation.get('relevance_score', 0)}/5")
            print(f"Factuality: {evaluation.get('factuality_score', 0)}/5")
            print(f"Completeness: {evaluation.get('completeness_score', 0)}/5")
            print(f"Feedback: {evaluation.get('feedback', 'No feedback available')}")
            
            print("\n" + "="*70)
        
        print("\nEnhanced RAG Demonstration Complete!")
        
    except Exception as e:
        logger.error(f"Error in RAG demonstration: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
