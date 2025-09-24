"""
Simple test to verify the library imports and basic functionality.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Test imports
    from wikipedia_async import WikipediaClient, ClientConfig
    from wikipedia_async.models import SearchResult, WikiPage
    from wikipedia_async.exceptions import WikipediaException
    
    print("✅ All imports successful!")
    
    # Test config creation
    config = ClientConfig()
    print(f"✅ Config created: language={config.language}, rate_limit={config.rate_limit_calls}")
    
    # Test client creation
    client = WikipediaClient(config=config)
    print("✅ Client created successfully!")
    
    # Test validation
    try:
        bad_config = ClientConfig(rate_limit_calls=0)
    except ValueError as e:
        print(f"✅ Validation working: {e}")
    
    print("\n🎉 Library setup is complete and working!")
    print("\nTo test with real API calls, run: python wiki.py")
    print("To run benchmarks: python benchmark.py")
    print("To see examples: python examples.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check the library setup")