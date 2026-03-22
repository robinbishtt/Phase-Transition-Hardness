#!/bin/bash
# setup_dev.sh
# Development environment setup script
# Usage: ./scripts/setup_dev.sh

set -e

echo "========================================"
echo "Setting up development environment"
echo "========================================"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install || echo "pre-commit not configured, skipping"

# Run initial tests
echo ""
echo "Running initial tests..."
pytest tests/unit/ -v --tb=short -x || true

echo ""
echo "========================================"
echo "Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Run tests: pytest tests/"
echo "  3. Run quick test: ./scripts/quick_test.sh"
echo "========================================"
