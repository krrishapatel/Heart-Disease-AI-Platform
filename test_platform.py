#!/usr/bin/env python3
"""
Test Script for Heart Disease AI Platform
Verifies that the platform works correctly
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        import shap
        print("✅ SHAP imported successfully")
    except ImportError as e:
        print(f"❌ SHAP import failed: {e}")
        return False
    
    try:
        from lime import lime_tabular
        print("✅ LIME imported successfully")
    except ImportError as e:
        print(f"❌ LIME import failed: {e}")
        return False
    
    return True

def test_data_files():
    """Test if required data files exist"""
    print("\n📁 Testing data files...")
    
    required_files = [
        'cleveland.data',
        'hungarian.data',
        'switzerland.data',
        'long-beach-va.data',
        'heart-disease.names'
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        print("   Some features may not work without these files")
        return False
    
    return True

def test_platform_import():
    """Test if the main platform can be imported"""
    print("\n🔧 Testing platform import...")
    
    try:
        from heart_disease_ai_platform import HeartDiseasePlatform
        print("✅ HeartDiseasePlatform imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Platform import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic platform functionality"""
    print("\n🚀 Testing basic functionality...")
    
    try:
        from heart_disease_ai_platform import HeartDiseasePlatform
        
        # Create platform instance
        platform = HeartDiseasePlatform()
        print("✅ Platform instance created")
        
        # Test data processor
        from heart_disease_ai_platform import HeartDiseaseDataProcessor
        data_processor = HeartDiseaseDataProcessor()
        print("✅ Data processor created")
        
        # Test predictor
        from heart_disease_ai_platform import HeartDiseasePredictor
        predictor = HeartDiseasePredictor()
        print("✅ Predictor created")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def run_quick_test():
    """Run a quick test with minimal data"""
    print("\n⚡ Running quick test...")
    
    try:
        from heart_disease_ai_platform import HeartDiseaseDataProcessor
        
        # Test data loading
        data_processor = HeartDiseaseDataProcessor()
        
        # Check if we can load any data
        try:
            raw_data = data_processor.load_multi_institutional_data()
            if not raw_data.empty:
                print(f"✅ Data loaded successfully: {len(raw_data)} patients")
                
                # Test feature extraction
                try:
                    feature_data = data_processor.extract_key_features(raw_data)
                    print(f"✅ Features extracted: {len(feature_data.columns)} features")
                    return True
                except Exception as e:
                    print(f"❌ Feature extraction failed: {e}")
                    return False
            else:
                print("❌ No data loaded")
                return False
                
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🏥 Heart Disease AI Platform - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Files Test", test_data_files),
        ("Platform Import Test", test_platform_import),
        ("Basic Functionality Test", test_basic_functionality),
        ("Quick Test", run_quick_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Platform is ready to use.")
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. Platform should work with some limitations.")
    else:
        print("❌ Many tests failed. Please check dependencies and setup.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
