#!/usr/bin/env python3

from qdrant_client import QdrantClient

def test_qdrant_connection():
    """Test connection to the Qdrant server and print available collections."""
    try:
        # Create client with the provided URL and API key
        qdrant_client = QdrantClient(
            url="https://cb8db371-6407-4b4b-8ee6-b7ec792fe4ec.us-east4-0.gcp.cloud.qdrant.io:6333", 
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.bJ4rgZ8hL6tD2yw8MnHbMLOCURk-2G2BKDW10sCDGBI",
        )
        
        # Test connection by getting collections
        collections = qdrant_client.get_collections()
        print("\nSuccessfully connected to Qdrant server!")
        print("\nAvailable collections:")
        print(collections)
        
        return True, qdrant_client
    
    except Exception as e:
        print(f"\nError connecting to Qdrant server: {e}")
        return False, None

if __name__ == "__main__":
    print("Testing connection to Qdrant server...")
    success, client = test_qdrant_connection()
    
    if success:
        print("\nQdrant client setup complete and connection verified.")
    else:
        print("\nFailed to connect to Qdrant server. Please check your connection details.")
