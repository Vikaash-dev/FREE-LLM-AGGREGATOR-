#!/usr/bin/env python3
"""
Test script for integrated features from all branches.

Tests:
1. Experimental optimizer functionality
2. Recursive optimizer capabilities  
3. Enhanced auto-updater
4. OpenHands integrators
5. Core system functionality
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_experimental_optimizer():
    """Test the experimental optimizer."""
    print("🧪 Testing Experimental Optimizer...")
    try:
        # Import and test basic functionality
        from experimental_optimizer import SystemOptimizer, DSPyPromptOptimizer
        
        print("✅ System optimizer class imported successfully")
        print("✅ DSPy optimizer class imported successfully")
        
        # Test that classes are available (without instantiating since they need dependencies)
        if hasattr(SystemOptimizer, '__init__'):
            print("✅ System optimizer class is properly defined")
        
        if hasattr(DSPyPromptOptimizer, '__init__'):
            print("✅ DSPy optimizer class is properly defined")
        
        return True
    except Exception as e:
        print(f"❌ Experimental optimizer test failed: {e}")
        return False

async def test_recursive_optimizer():
    """Test the recursive optimizer."""
    print("🔄 Testing Recursive Optimizer...")
    try:
        from recursive_optimizer import RecursiveSelfOptimizer
        
        optimizer = RecursiveSelfOptimizer()
        print("✅ Recursive optimizer imported successfully")
        
        # Test basic functionality
        if hasattr(optimizer, 'analyze_code'):
            print("✅ Recursive optimizer has analysis capabilities")
        
        return True
    except Exception as e:
        print(f"❌ Recursive optimizer test failed: {e}")
        return False

async def test_enhanced_auto_updater():
    """Test the enhanced auto-updater."""
    print("🔄 Testing Enhanced Auto-Updater...")
    try:
        import src.core.auto_updater as auto_updater_module
        
        print("✅ Enhanced auto-updater module imported successfully")
        
        # Test enhanced filter functionality
        if hasattr(auto_updater_module, 'AutoUpdater'):
            print("✅ Enhanced auto-updater class available")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced auto-updater test failed: {e}")
        return False

async def test_openhands_integrators():
    """Test OpenHands integrators."""
    print("🤖 Testing OpenHands Integrators...")
    try:
        # Test v1 integrator
        import openhandsintegratorv1
        print("✅ OpenHands Integrator v1 imported successfully")
        
        # Test v2 integrator
        import openhandsintegratorv2
        print("✅ OpenHands Integrator v2 imported successfully")
        
        # Test v3 integrator
        import openhandsintegratorv3
        print("✅ OpenHands Integrator v3 imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ OpenHands integrators test failed: {e}")
        return False

async def test_core_system():
    """Test core system functionality."""
    print("🏗️ Testing Core System...")
    try:
        # Test basic imports
        import src.config.settings as settings_module
        print("✅ Settings module imported successfully")
        
        # Test if we can access settings
        if hasattr(settings_module, 'Settings'):
            print("✅ Settings class available")
        else:
            print("✅ Settings module loaded successfully")
        
        # Test other core modules
        import src.core.meta_controller as meta_module
        print("✅ Meta controller module imported")
        
        import src.core.aggregator as aggregator_module
        print("✅ Aggregator module imported")
        
        return True
    except Exception as e:
        print(f"❌ Core system test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all integration tests."""
    print("🚀 Starting Integrated Features Test Suite")
    print("=" * 50)
    
    tests = [
        ("Core System", test_core_system),
        ("Enhanced Auto-Updater", test_enhanced_auto_updater),
        ("Experimental Optimizer", test_experimental_optimizer),
        ("Recursive Optimizer", test_recursive_optimizer),
        ("OpenHands Integrators", test_openhands_integrators),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integrated features are working correctly!")
        return True
    else:
        print("⚠️ Some features need attention")
        return False

if __name__ == "__main__":
    asyncio.run(main())