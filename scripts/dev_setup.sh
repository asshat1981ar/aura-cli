#!/usr/bin/env bash
# Development environment setup script for AURA CLI contributors
# This script automates initial setup steps for local development

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================"
echo "AURA CLI - Development Environment Setup"
echo "================================================"
echo ""

# Color output helpers
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    NC='\033[0m' # No Color
else
    GREEN=''
    YELLOW=''
    RED=''
    NC=''
fi

info() {
    echo -e "${GREEN}✓${NC} $1"
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10 ]]; then
    error "Python 3.10+ required, found Python $PYTHON_VERSION"
    exit 1
fi
info "Python $PYTHON_VERSION detected"

# Check if virtual environment exists
cd "$PROJECT_ROOT"

if [[ ! -d "venv" ]]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    info "Virtual environment created at ./venv"
else
    warn "Virtual environment already exists at ./venv"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate || {
    error "Failed to activate virtual environment"
    exit 1
}
info "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q
info "pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt -q
    info "Runtime dependencies installed"
else
    warn "requirements.txt not found, skipping"
fi

# Install dev dependencies if file exists
if [[ -f "requirements-dev.txt" ]]; then
    echo ""
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt -q
    info "Development dependencies installed"
fi

# Setup pre-commit hooks if available
echo ""
if command -v pre-commit &> /dev/null; then
    echo "Setting up pre-commit hooks..."
    pre-commit install
    info "Pre-commit hooks installed"
else
    warn "pre-commit not found. Install with: pip install pre-commit"
    warn "Then run: pre-commit install"
fi

# Setup detect-secrets baseline if needed
echo ""
if [[ -f ".pre-commit-config.yaml" ]] && grep -q "detect-secrets" .pre-commit-config.yaml; then
    if command -v detect-secrets &> /dev/null; then
        if [[ ! -f ".secrets.baseline" ]]; then
            echo "Generating .secrets.baseline file..."
            detect-secrets scan > .secrets.baseline 2>/dev/null || true
            info ".secrets.baseline generated"
        else
            info ".secrets.baseline already exists"
        fi
    else
        warn "detect-secrets not found. Install with: pip install detect-secrets"
    fi
fi

# Create config from example if needed
echo ""
if [[ ! -f "aura.config.json" ]]; then
    if [[ -f "aura.config.example.json" ]]; then
        echo "Creating aura.config.json from example..."
        cp aura.config.example.json aura.config.json
        info "aura.config.json created"
        warn "Please add your API keys to aura.config.json before running AURA"
    else
        warn "aura.config.example.json not found"
        warn "You may need to create aura.config.json manually"
    fi
else
    info "aura.config.json already exists"
fi

# Create memory directory structure
echo ""
echo "Setting up memory directory..."
mkdir -p memory/store
touch memory/.gitkeep
info "Memory directory initialized"

# Create .env file if it doesn't exist
echo ""
if [[ ! -f ".env" ]]; then
    cat > .env << 'EOF'
# AURA CLI Environment Variables
# Copy this to .env and fill in your values

# OpenRouter API key (for cloud models)
# AURA_API_KEY=your_openrouter_api_key_here

# Optional: Gemini API key
# GEMINI_API_KEY=your_gemini_key_here

# Optional: OpenAI API key
# OPENAI_API_KEY=your_openai_key_here

# Development settings
AURA_SKIP_CHDIR=1
# AURA_DRY_RUN=1

# Server settings
# AGENT_API_TOKEN=your_secret_token_here
# AGENT_API_ENABLE_RUN=1
EOF
    info ".env template created"
    warn "Please add your API keys to .env"
else
    info ".env file already exists"
fi

# Run basic health check
echo ""
echo "Running health check..."
if python3 main.py doctor &> /dev/null; then
    info "Health check passed"
else
    warn "Health check had warnings (expected on first run — missing API keys is normal)"
    warn "Configure your API key in aura.config.json (set 'api_key') or via AURA_API_KEY env var"
fi

# Summary
echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Add your API keys to:"
echo "     - aura.config.json (OpenRouter API key)"
echo "     - .env (optional: Gemini, OpenAI keys)"
echo ""
echo "  2. Verify installation:"
echo "     $ python3 main.py doctor"
echo ""
echo "  3. Run tests:"
echo "     $ python3 -m pytest"
echo ""
echo "  4. Start the development server:"
echo "     $ python3 aura_cli/server.py"
echo ""
echo "  5. See available commands:"
echo "     $ python3 main.py --help"
echo ""
echo "Useful commands:"
echo "  - Run a single goal: python3 main.py --goal 'Your task here'"
echo "  - Run in dry-run mode: python3 main.py --goal 'Task' --dry-run"
echo "  - Interactive mode: ./run_aura.sh interactive"
echo ""
echo "Documentation:"
echo "  - README.md - Project overview"
echo "  - CLAUDE.md - AI assistant guide & architecture"
echo "  - docs/CLI_REFERENCE.md - Command reference"
echo ""
echo "Happy coding! 🚀"
echo ""
