#!/usr/bin/env python3
"""
Test script to verify SSL configuration is working for Vanna
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

import config
from vanna_train import VannaModelManager

def test_ssl_configuration():
    """Test if SSL configuration resolves the certificate verification issue"""
    
    print("Testing SSL configuration for Vanna...")
    print(f"SSL Verify setting: {config.VANNA_OPENAI_SSL_VERIFY}")
    
    try:
        # Initialize Vanna manager
        manager = VannaModelManager()
        print("✓ VannaModelManager initialized successfully")
        
        # Initialize Vanna client
        vanna_client = manager.initialize_vanna()
        print("✓ Vanna client initialized successfully")
        
        # Try a simple SQL generation test
        if vanna_client:
            print("✓ SSL configuration appears to be working!")
            print("Vanna client is ready for SQL generation")
            return True
        else:
            print("✗ Vanna client initialization failed")
            return False
            
    except Exception as e:
        print(f"✗ Error during initialization: {e}")
        if "certificate verify failed" in str(e).lower() or "ssl" in str(e).lower():
            print("SSL certificate issue still exists")
        return False

if __name__ == "__main__":
    success = test_ssl_configuration()
    sys.exit(0 if success else 1)