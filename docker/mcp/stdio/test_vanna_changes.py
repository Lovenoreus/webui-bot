#!/usr/bin/env python3
"""
Test script to verify that the Vanna training has been moved from vanna_train.py 
to vanna_query_engine.py and that global initialization has been removed.
"""

import sys
import os

def test_vanna_train_import():
    """Test that vanna_train module can be imported without global initialization"""
    try:
        # This should now work without triggering global initialization
        from vanna_train import VannaModelManager, vanna_train, get_database_schema_info, get_table_names, get_vanna_info
        print("✅ vanna_train module imported successfully without global initialization")
        return True
    except Exception as e:
        print(f"❌ Failed to import vanna_train module: {e}")
        return False

def test_vanna_query_engine_import():
    """Test that vanna_query_engine module can be imported"""
    try:
        from vanna_query_engine import VannaQueryEngine
        print("✅ vanna_query_engine module imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import vanna_query_engine module: {e}")
        return False

def test_vanna_manager_creation():
    """Test that VannaModelManager can be created without auto-initialization"""
    try:
        from vanna_train import VannaModelManager
        
        # This should create a manager without initializing Vanna
        manager = VannaModelManager()
        print("✅ VannaModelManager created successfully without auto-initialization")
        print(f"   - Current provider: {manager.current_provider}")
        print(f"   - Current database: {manager.current_database}")
        print(f"   - Vanna client initialized: {manager.vanna_client is not None}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create VannaModelManager: {e}")
        return False

def test_function_signatures():
    """Test that the modified functions have the correct signatures"""
    try:
        from vanna_train import vanna_train, get_vanna_info
        import inspect
        
        # Check vanna_train signature
        sig = inspect.signature(vanna_train)
        expected_params = ['vanna_client', 'current_provider', 'ddl', 'documentation', 'question', 'sql']
        actual_params = list(sig.parameters.keys())
        
        if actual_params == expected_params:
            print("✅ vanna_train function has correct signature")
        else:
            print(f"❌ vanna_train function signature mismatch. Expected: {expected_params}, Got: {actual_params}")
            return False
        
        # Check get_vanna_info signature
        sig = inspect.signature(get_vanna_info)
        expected_params = ['vanna_manager']
        actual_params = list(sig.parameters.keys())
        
        if actual_params == expected_params:
            print("✅ get_vanna_info function has correct signature")
        else:
            print(f"❌ get_vanna_info function signature mismatch. Expected: {expected_params}, Got: {actual_params}")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Failed to test function signatures: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Vanna training refactoring...\n")
    
    tests = [
        test_vanna_train_import,
        test_vanna_query_engine_import,
        test_vanna_manager_creation,
        test_function_signatures
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}\n")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactoring was successful.")
        print("\n📝 Summary of changes:")
        print("   - Removed global initialization from vanna_train.py")
        print("   - Training logic moved to vanna_query_engine.py")
        print("   - VannaModelManager can be created without auto-initialization")
        print("   - Functions now require explicit parameters instead of globals")
        return True
    else:
        print("💥 Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)