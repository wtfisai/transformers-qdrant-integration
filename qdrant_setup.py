#!/usr/bin/env python3

import subprocess
import sys

def install_package(package):
    """Install a Python package if it's not already installed."""
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} has been installed.")

# Install qdrant-client if not already installed
install_package("qdrant_client")

# Now import the client
from qdrant_client import QdrantClient

def setup_qdrant_client():
    """Set up and test connection to the Qdrant server."""
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
        
        return qdrant_client
    
    except Exception as e:
        print(f"\nError connecting to Qdrant server: {e}")
        return None

if __name__ == "__main__":
    print("Setting up Qdrant client...")
    client = setup_qdrant_client()
    
    if client:
        print("\nQdrant client setup complete and connection verified.")
    else:
        print("\nFailed to set up Qdrant client. Please check your connection details.")
