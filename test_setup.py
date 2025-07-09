#!/usr/bin/env python3
"""
🧪 Virtual Environment Setup Test Script
==========================================
This script verifies that your virtual environment is properly configured
with all necessary packages for the GenerativeAI_Demystified project.
"""

import sys
import importlib
import platform
from pathlib import Path

def print_header():
    """Print a nice header for the test results."""
    print("🧪 Testing Virtual Environment Setup")
    print("=" * 50)
    print(f"🐍 Python version: {sys.version}")
    print(f"📍 Python location: {sys.executable}")
    print(f"💻 Platform: {platform.system()} {platform.release()}")
    print(f"🏗️  Architecture: {platform.machine()}")
    print(f"📁 Working directory: {Path.cwd()}")
    print()

def test_import(module_name, description="", show_version=False):
    """Test if a module can be imported and optionally show version."""
    try:
        module = importlib.import_module(module_name)

        version_info = ""
        if show_version:
            # Try to get version information
            version_attrs = ['__version__', 'version', 'VERSION']
            for attr in version_attrs:
                if hasattr(module, attr):
                    version = getattr(module, attr)
                    if callable(version):
                        version = version()
                    version_info = f" (v{version})"
                    break

        print(f"✅ {module_name}{version_info} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name} - {description}: Imported but error getting version: {e}")
        return True

def test_environment_variables():
    """Test if important environment variables are set."""
    import os

    print("\n🔐 Environment Variables Check:")
    print("-" * 30)

    env_vars = [
        ("OPENAI_API_KEY", "OpenAI API access"),
        ("GOOGLE_API_KEY", "Google Gemini API access"),
        ("GROQ_API_KEY", "Groq API access"),
        ("LANGCHAIN_API_KEY", "LangSmith tracing"),
        ("LANGCHAIN_TRACING_V2", "LangSmith tracing enabled"),
    ]

    for var_name, description in env_vars:
        value = os.getenv(var_name)
        if value:
            # Don't print the actual key, just indicate it's set
            masked_value = value[:4] + "***" if len(value) > 4 else "***"
            print(f"✅ {var_name}: {masked_value} - {description}")
        else:
            print(f"⚠️  {var_name}: Not set - {description}")

def test_file_system():
    """Test if important files and directories exist."""
    print("\n📁 File System Check:")
    print("-" * 20)

    important_paths = [
        ("requirements.txt", "Project dependencies"),
        ("setup_venv.sh", "Virtual environment setup script"),
        ("activate_venv.sh", "Quick activation script"),
        ("deactivate_venv.sh", "Quick deactivation script"),
        ("venv/", "Virtual environment directory"),
        (".env", "Environment variables file (optional)"),
    ]

    for path_name, description in important_paths:
        path = Path(path_name)
        if path.exists():
            print(f"✅ {path_name} - {description}")
        else:
            print(f"⚠️  {path_name} - {description} (missing)")

def main():
    """Run all tests."""
    print_header()

    # Test core Python packages
    print("📦 Core Package Tests:")
    print("-" * 20)

    core_tests = [
        ("sys", "Python system interface"),
        ("os", "Operating system interface"),
        ("pathlib", "Object-oriented filesystem paths"),
        ("json", "JSON encoder/decoder"),
        ("urllib", "URL handling modules"),
        ("ssl", "SSL/TLS support"),
    ]

    core_passed = sum(test_import(module, desc) for module, desc in core_tests)

    # Test LangChain ecosystem
    print("\n🦜️ LangChain Ecosystem Tests:")
    print("-" * 30)

    langchain_tests = [
        ("langchain", "LangChain framework", True),
        ("langchain_core", "LangChain core", True),
        ("langchain_community", "LangChain community", True),
        ("langchain_openai", "OpenAI integration", True),
        ("langchain_google_genai", "Google Gemini integration", True),
        ("langchain_groq", "Groq integration", True),
        ("langchain_huggingface", "Hugging Face integration", True),
    ]

    langchain_passed = sum(test_import(module, desc, ver) for module, desc, ver in langchain_tests)

    # Test LLM API clients
    print("\n🤖 LLM API Client Tests:")
    print("-" * 25)

    llm_tests = [
        ("openai", "OpenAI API client", True),
        ("google.generativeai", "Google Generative AI", True),
        ("groq", "Groq API client", True),
    ]

    llm_passed = sum(test_import(module, desc, ver) for module, desc, ver in llm_tests)

    # Test ML/AI libraries
    print("\n🧠 Machine Learning Libraries:")
    print("-" * 30)

    ml_tests = [
        ("torch", "PyTorch", True),
        ("torchvision", "PyTorch vision", True),
        ("transformers", "Hugging Face Transformers", True),
        ("accelerate", "Hugging Face Accelerate", True),
        ("numpy", "NumPy", True),
        ("pandas", "Pandas", True),
    ]

    ml_passed = sum(test_import(module, desc, ver) for module, desc, ver in ml_tests)

    # Test document processing
    print("\n📄 Document Processing Libraries:")
    print("-" * 35)

    doc_tests = [
        ("pypdf", "PyPDF", True),
        ("fitz", "PyMuPDF", True),
        ("pdfminer", "PDFMiner", True),
        ("unstructured", "Unstructured", True),
        ("bs4", "BeautifulSoup4", True),
        ("lxml", "LXML", True),
    ]

    doc_passed = sum(test_import(module, desc, ver) for module, desc, ver in doc_tests)

    # Test utilities
    print("\n🛠️ Utility Libraries:")
    print("-" * 20)

    util_tests = [
        ("dotenv", "Python-dotenv", True),
        ("tiktoken", "TikToken", True),
        ("nltk", "NLTK", True),
        ("requests", "Requests", True),
        ("jupyter", "Jupyter", True),
    ]

    util_passed = sum(test_import(module, desc, ver) for module, desc, ver in util_tests)

    # Test environment and file system
    test_environment_variables()
    test_file_system()

    # Calculate totals
    total_passed = core_passed + langchain_passed + llm_passed + ml_passed + doc_passed + util_passed
    total_tests = len(core_tests) + len(langchain_tests) + len(llm_tests) + len(ml_tests) + len(doc_tests) + len(util_tests)

    # Print summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    print(f"Core Python:      {core_passed}/{len(core_tests)} tests passed")
    print(f"LangChain:        {langchain_passed}/{len(langchain_tests)} tests passed")
    print(f"LLM APIs:         {llm_passed}/{len(llm_tests)} tests passed")
    print(f"ML Libraries:     {ml_passed}/{len(ml_tests)} tests passed")
    print(f"Document Proc:    {doc_passed}/{len(doc_tests)} tests passed")
    print(f"Utilities:        {util_passed}/{len(util_tests)} tests passed")
    print("-" * 50)
    print(f"TOTAL:            {total_passed}/{total_tests} tests passed")
    print()

    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Your environment is ready for GenerativeAI development!")
        print("🚀 You can now start building amazing AI applications!")
    elif total_passed >= total_tests * 0.8:
        print("✅ MOSTLY WORKING! Your environment is mostly ready.")
        print("⚠️  Some optional packages failed. Check the output above.")
        print("💡 You can still proceed with most functionalities.")
    else:
        print("❌ SETUP ISSUES DETECTED! Several important packages are missing.")
        print("🔧 Please check the failed imports and install missing packages.")
        print("📖 Refer to VIRTUAL_ENV_SETUP.md for troubleshooting help.")

    print("\n🔧 Quick fixes:")
    print("- Recreate environment: bash setup_venv.sh")
    print("- Install missing packages: pip install -r requirements.txt")
    print("- Check documentation: open VIRTUAL_ENV_SETUP.md")

if __name__ == "__main__":
    main()