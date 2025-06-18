#!/usr/bin/env python3
"""
Quick test script to verify FinGPT dependencies are installed correctly
"""

import sys


def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")

    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT (Parameter Efficient Fine-Tuning)"),
        ("huggingface_hub", "Hugging Face Hub"),
    ]

    failed_imports = []

    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {description} - OK")
        except ImportError as e:
            print(f"❌ {description} - FAILED: {e}")
            failed_imports.append(package)

    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("💡 Please run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages imported successfully!")
        return True


def test_torch():
    """Test PyTorch configuration"""
    print("\n🔧 Testing PyTorch configuration...")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"MPS available: {torch.backends.mps.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"Recommended device: {device}")

        # Test tensor creation
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"✅ Tensor creation test: {x}")

        return True

    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False


def test_huggingface_auth():
    """Test Hugging Face authentication"""
    print("\n🔐 Testing Hugging Face authentication...")

    try:
        from huggingface_hub import login, whoami

        # Test with the provided token
        token = "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"

        try:
            login(token=token)
            user_info = whoami()
            print(f"✅ Authenticated as: {user_info.get('name', 'Unknown')}")
            return True
        except Exception as auth_e:
            print(f"❌ Authentication failed: {auth_e}")
            return False

    except Exception as e:
        print(f"❌ Hugging Face test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 FinGPT Dependency Test")
    print("=" * 40)

    tests_passed = 0
    total_tests = 3

    # Test imports
    if test_imports():
        tests_passed += 1

    # Test PyTorch
    if test_torch():
        tests_passed += 1

    # Test Hugging Face auth
    if test_huggingface_auth():
        tests_passed += 1

    print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("✅ All tests passed! You're ready to run FinGPT forecaster.")
        print("🚀 Run: python3 fingpt_forecaster_setup.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        if tests_passed < 2:
            print("💡 Try running: pip install -r requirements.txt")

    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
