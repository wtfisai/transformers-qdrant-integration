#!/usr/bin/env python
"""
Unit Tests for RAG Demonstration System
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAG system
from rag_demonstration import RAGSystem, Document


class MockQdrantClient:
    """Mock QdrantClient for testing"""
    def __init__(self, *args, **kwargs):
        self.collections = []
        self.vectors = {}
        self.records = {}
    
    def get_collections(self):
        class Collections:
            def __init__(self, collections):
                self.collections = [type('obj', (object,), {'name': c}) for c in collections]
        return Collections(self.collections)
    
    def create_collection(self, collection_name, vectors_config):
        self.collections.append(collection_name)
        self.vectors[collection_name] = vectors_config
        self.records[collection_name] = []
        return True
    
    def upsert(self, collection_name, points):
        self.records[collection_name].extend(points)
        return True
    
    def search(self, collection_name, query_vector, limit=10, **kwargs):
        # Just return dummy results
        return [
            type('obj', (object,), {
                'id': f"test-id-{i}",
                'score': 0.9 - (i * 0.1),
                'payload': {'page_content': f"Test content {i}", 'metadata': {'source': f"test-{i}"}}
            }) for i in range(min(3, limit))
        ]


class MockEmbeddings:
    """Mock Embeddings for testing"""
    def __init__(self, *args, **kwargs):
        pass
    
    def embed_query(self, text):
        # Return a fixed-size dummy vector
        return [0.1] * 768
    
    def embed_documents(self, documents):
        # Return fixed-size dummy vectors
        return [[0.1] * 768 for _ in documents]


class TestRAGSystem(unittest.TestCase):
    """Test cases for RAG System"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            "QDRANT_URL": "http://fake-qdrant-url",
            "QDRANT_API_KEY": "fake-api-key",
            "OPENAI_API_KEY": "fake-openai-key"
        })
        self.env_patcher.start()
        
        # Create mock for QdrantClient
        self.qdrant_patcher = patch('rag_demonstration.QdrantClient', MockQdrantClient)
        self.mock_qdrant = self.qdrant_patcher.start()
        
        # Create mock for OpenAIEmbeddings
        self.embeddings_patcher = patch('rag_demonstration.OpenAIEmbeddings', MockEmbeddings)
        self.mock_embeddings = self.embeddings_patcher.start()
        
        # Create mock for ChatOpenAI
        self.openai_patcher = patch('rag_demonstration.ChatOpenAI')
        self.mock_openai = self.openai_patcher.start()
        self.mock_openai.return_value.invoke.return_value.content = "Test response"
    
    def tearDown(self):
        """Clean up after tests"""
        self.env_patcher.stop()
        self.qdrant_patcher.stop()
        self.embeddings_patcher.stop()
        self.openai_patcher.stop()
        
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization of RAG system"""
        rag = RAGSystem()
        self.assertIsNotNone(rag)
        self.assertIsNotNone(rag.qdrant_client)
        self.assertIsNotNone(rag.embeddings)
        self.assertIsNotNone(rag.llm)
    
    def test_create_collection(self):
        """Test creating a collection"""
        rag = RAGSystem()
        rag.create_collection()
        self.assertIn("demo_documents", rag.qdrant_client.collections)
    
    def test_load_sample_documents(self):
        """Test loading sample documents"""
        rag = RAGSystem()
        docs = rag.load_sample_documents()
        self.assertIsInstance(docs, list)
        self.assertTrue(len(docs) > 0)
        self.assertIsInstance(docs[0], Document)
    
    def test_process_documents(self):
        """Test processing documents"""
        rag = RAGSystem()
        docs = [
            Document(page_content="Test document 1", metadata={"source": "test1"}),
            Document(page_content="Test document 2", metadata={"source": "test2"})
        ]
        vector_store = rag.process_documents(docs)
        self.assertIsNotNone(vector_store)
    
    def test_retrieve_similar_documents(self):
        """Test retrieving similar documents"""
        rag = RAGSystem()
        # We need to set up the collection first
        rag.create_collection()
        # And add some documents
        docs = [
            Document(page_content="Test document 1", metadata={"source": "test1"}),
            Document(page_content="Test document 2", metadata={"source": "test2"})
        ]
        rag.process_documents(docs)
        
        # Now try to retrieve documents
        retrieved_docs = rag.retrieve_similar_documents("test query")
        self.assertIsInstance(retrieved_docs, list)
    
    def test_generate_response(self):
        """Test generating a response"""
        rag = RAGSystem()
        docs = [
            Document(page_content="Test document 1", metadata={"source": "test1"}),
            Document(page_content="Test document 2", metadata={"source": "test2"})
        ]
        response = rag.generate_response("test query", docs)
        self.assertEqual(response, "Test response")


if __name__ == "__main__":
    unittest.main()
