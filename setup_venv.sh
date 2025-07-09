#!/bin/bash

# =============================================================================
# 🚀 VIRTUAL ENVIRONMENT SETUP SCRIPT
# =============================================================================
# Project: GenerativeAI_Demystified
# Description: Automated script to create and configure Python virtual environment
# Usage: bash setup_venv.sh
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emojis for better UX
ROCKET="🚀"
CHECKMARK="✅"
CROSS="❌"
WRENCH="🔧"
PACKAGE="📦"
PYTHON="🐍"
SPARKLES="✨"

echo -e "${BLUE}${ROCKET} Starting Virtual Environment Setup${NC}"
echo -e "${BLUE}=======================================${NC}"

# Check if Python 3 is installed
echo -e "${YELLOW}${PYTHON} Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}${CHECKMARK} Python found: ${PYTHON_VERSION}${NC}"
else
    echo -e "${RED}${CROSS} Python 3 not found. Please install Python 3.11 or higher.${NC}"
    exit 1
fi

# Check Python version (should be 3.11+)
PYTHON_VERSION_NUM=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if (( $(echo "$PYTHON_VERSION_NUM >= 3.11" | bc -l) )); then
    echo -e "${GREEN}${CHECKMARK} Python version is compatible (${PYTHON_VERSION_NUM})${NC}"
else
    echo -e "${YELLOW}⚠️  Warning: Python ${PYTHON_VERSION_NUM} detected. Recommended: 3.11+${NC}"
fi

# Check if venv already exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment 'venv' already exists.${NC}"
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${WRENCH} Removing existing virtual environment...${NC}"
        rm -rf venv
    else
        echo -e "${BLUE}${CHECKMARK} Using existing virtual environment.${NC}"
        source venv/bin/activate
        echo -e "${GREEN}${CHECKMARK} Virtual environment activated!${NC}"
        echo -e "${CYAN}To deactivate, run: deactivate${NC}"
        exit 0
    fi
fi

# Create virtual environment
echo -e "${WRENCH} Creating virtual environment...${NC}"
python3 -m venv venv

if [ $? -eq 0 ]; then
    echo -e "${GREEN}${CHECKMARK} Virtual environment created successfully!${NC}"
else
    echo -e "${RED}${CROSS} Failed to create virtual environment.${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${WRENCH} Activating virtual environment...${NC}"
source venv/bin/activate

if [ $? -eq 0 ]; then
    echo -e "${GREEN}${CHECKMARK} Virtual environment activated!${NC}"
else
    echo -e "${RED}${CROSS} Failed to activate virtual environment.${NC}"
    exit 1
fi

# Upgrade pip
echo -e "${PACKAGE} Upgrading pip...${NC}"
pip install --upgrade pip

# Install wheel and setuptools
echo -e "${PACKAGE} Installing build tools...${NC}"
pip install wheel setuptools

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo -e "${PACKAGE} Installing dependencies from requirements.txt...${NC}"
    pip install -r requirements.txt

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}${CHECKMARK} Dependencies installed successfully!${NC}"
    else
        echo -e "${RED}${CROSS} Some dependencies failed to install. Check the output above.${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  requirements.txt not found. Skipping dependency installation.${NC}"
fi

# Create activation script
echo -e "${WRENCH} Creating activation script...${NC}"
cat > activate_venv.sh << 'EOF'
#!/bin/bash
# Quick activation script for GenerativeAI_Demystified virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "🐍 Virtual environment activated!"
    echo "📦 Python: $(python --version)"
    echo "📍 Location: $(which python)"
    echo "🔧 To deactivate: deactivate"
else
    echo "❌ Virtual environment not found. Run: bash setup_venv.sh"
fi
EOF

chmod +x activate_venv.sh

# Create deactivation script
cat > deactivate_venv.sh << 'EOF'
#!/bin/bash
# Quick deactivation script
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
    echo "🔴 Virtual environment deactivated!"
else
    echo "ℹ️  No virtual environment is currently active."
fi
EOF

chmod +x deactivate_venv.sh

# Display summary
echo -e "${SPARKLES}${GREEN}=======================================${NC}"
echo -e "${SPARKLES}${GREEN} VIRTUAL ENVIRONMENT SETUP COMPLETE! ${NC}"
echo -e "${SPARKLES}${GREEN}=======================================${NC}"
echo ""
echo -e "${CYAN}📁 Virtual environment location: ${PWD}/venv${NC}"
echo -e "${CYAN}🐍 Python version: $(python --version)${NC}"
echo -e "${CYAN}📦 Pip version: $(pip --version)${NC}"
echo ""
echo -e "${BLUE}🚀 QUICK COMMANDS:${NC}"
echo -e "  ${GREEN}Activate:${NC}   source venv/bin/activate  OR  bash activate_venv.sh"
echo -e "  ${GREEN}Deactivate:${NC} deactivate                OR  bash deactivate_venv.sh"
echo -e "  ${GREEN}Install:${NC}    pip install <package>"
echo -e "  ${GREEN}Update:${NC}     pip install --upgrade <package>"
echo -e "  ${GREEN}List:${NC}       pip list"
echo ""
echo -e "${BLUE}📚 JUPYTER NOTEBOOK:${NC}"
echo -e "  ${GREEN}Start:${NC}      jupyter notebook"
echo -e "  ${GREEN}Lab:${NC}        jupyter lab"
echo ""
echo -e "${BLUE}🔧 TROUBLESHOOTING:${NC}"
echo -e "  ${GREEN}Recreate:${NC}   bash setup_venv.sh"
echo -e "  ${GREEN}Test:${NC}       python -c \"import langchain; print('LangChain works!')\""
echo ""
echo -e "${PURPLE}✨ Your virtual environment is ready for GenerativeAI development!${NC}"
echo -e "${PURPLE}✨ Current environment is activated and ready to use.${NC}"