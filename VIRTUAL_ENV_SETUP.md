# 🚀 Virtual Environment Setup Guide

## 📋 Table of Contents
- [Quick Start](#quick-start)
- [Manual Setup](#manual-setup)
- [Troubleshooting](#troubleshooting)
- [Package Management](#package-management)
- [Development Workflow](#development-workflow)
- [Common Issues](#common-issues)

---

## 🎯 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Run the automated setup script
bash setup_venv.sh
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 Manual Setup Guide

### 1. Prerequisites
- **Python 3.11+** (recommended)
- **pip** (latest version)
- **Git** (for version control)

### 2. Check Your Python Version
```bash
python3 --version
# Should show Python 3.11.x or higher
```

### 3. Create Virtual Environment
```bash
# Navigate to your project directory
cd /path/to/GenerativeAI_Demystified

# Create virtual environment
python3 -m venv venv
```

### 4. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```cmd
venv\Scripts\activate
```

### 5. Upgrade Package Managers
```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install build tools
pip install wheel setuptools
```

### 6. Install Project Dependencies
```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### 7. Verify Installation
```bash
# Test LangChain installation
python -c "import langchain; print('✅ LangChain works!')"

# Test OpenAI integration
python -c "import openai; print('✅ OpenAI works!')"

# Test PyTorch installation
python -c "import torch; print('✅ PyTorch works!')"
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. SSL Certificate Errors (macOS)
```bash
# Install certificates
/Applications/Python\ 3.x/Install\ Certificates.command

# Or use pip with trusted hosts
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```

#### 2. Permission Errors
```bash
# Use --user flag for global installations
pip install --user package_name

# Or fix permissions (macOS/Linux)
sudo chown -R $(whoami) /usr/local/lib/python3.x/site-packages
```

#### 3. Package Conflicts
```bash
# Clear pip cache
pip cache purge

# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 4. M1 Mac Specific Issues
```bash
# Install Rosetta 2 if needed
softwareupdate --install-rosetta

# Use conda for better M1 support
conda create -n genai python=3.11
conda activate genai
pip install -r requirements.txt
```

---

## 📦 Package Management

### Installing New Packages
```bash
# Install a new package
pip install package_name

# Install with specific version
pip install package_name==1.0.0

# Install from GitHub
pip install git+https://github.com/user/repo.git

# Install in development mode
pip install -e .
```

### Updating Packages
```bash
# Update a specific package
pip install --upgrade package_name

# Update all packages (be careful!)
pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U
```

### Managing Dependencies
```bash
# Generate requirements.txt
pip freeze > requirements.txt

# Install from requirements.txt
pip install -r requirements.txt

# Check for security vulnerabilities
pip-audit

# Show package dependencies
pip show package_name
```

---

## 🔄 Development Workflow

### Daily Usage
```bash
# Start your day
source venv/bin/activate  # or bash activate_venv.sh

# Work on your project
jupyter notebook
# or
python your_script.py

# End your day
deactivate  # or bash deactivate_venv.sh
```

### Quick Commands
```bash
# Activate environment
bash activate_venv.sh

# Deactivate environment
bash deactivate_venv.sh

# Recreate environment
bash setup_venv.sh
```

### Environment Variables
Create a `.env` file in your project root:
```bash
# API Keys
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
GROQ_API_KEY=your_groq_key_here

# Project Settings
PROJECT_NAME=GenerativeAI_Demystified
DEBUG=True
```

---

## 🧪 Testing Your Setup

### Test Script
Create a test file `test_setup.py`:
```python
#!/usr/bin/env python3
"""Test script to verify virtual environment setup."""

import sys
import importlib

def test_import(module_name, description=""):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description}: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Virtual Environment Setup")
    print("=" * 50)

    # Test Python version
    print(f"🐍 Python version: {sys.version}")
    print(f"📍 Python location: {sys.executable}")
    print()

    # Test core packages
    tests = [
        ("langchain", "LangChain framework"),
        ("langchain_core", "LangChain core"),
        ("langchain_community", "LangChain community"),
        ("langchain_openai", "OpenAI integration"),
        ("openai", "OpenAI API client"),
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("jupyter", "Jupyter Notebook"),
    ]

    passed = 0
    total = len(tests)

    for module, description in tests:
        if test_import(module, description):
            passed += 1

    print()
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your environment is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above.")

if __name__ == "__main__":
    main()
```

Run the test:
```bash
python test_setup.py
```

---

## 📝 Best Practices

### 1. Environment Management
- Always use virtual environments
- Keep requirements.txt updated
- Use specific package versions for production
- Document your setup process

### 2. Security
- Never commit API keys to version control
- Use environment variables for sensitive data
- Regularly update packages for security patches
- Use `pip-audit` to check for vulnerabilities

### 3. Development
- Use `.gitignore` to exclude virtual environment
- Keep your virtual environment outside your project if preferred
- Use descriptive names for environments
- Document any special setup requirements

### 4. Performance
- Use `pip cache` to speed up installations
- Consider using `pip-tools` for better dependency management
- Use `conda` for scientific computing packages on M1 Macs

---

## 🆘 Getting Help

### Resources
- **LangChain Documentation**: https://langchain.readthedocs.io/
- **OpenAI API Documentation**: https://platform.openai.com/docs
- **Python Virtual Environments**: https://docs.python.org/3/tutorial/venv.html
- **pip Documentation**: https://pip.pypa.io/en/stable/

### Community
- **LangChain Discord**: https://discord.gg/langchain
- **Stack Overflow**: Tag your questions with `langchain`, `openai`, `python`
- **GitHub Issues**: Check the respective project repositories

---

## 📄 License

This setup guide is part of the GenerativeAI_Demystified project. See the main project license for details.

---

**Happy coding! 🚀**