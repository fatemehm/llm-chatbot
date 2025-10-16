#!/bin/bash
set -e

echo "Setting up code quality tools..."

# 1. Install tools
echo "Installing quality tools..."
pip install -q black isort flake8 mypy pre-commit pytest pytest-cov

# 2. Format code
echo ""
echo "Formatting code with Black..."
black app/ tests/ train.py

echo ""
echo "Sorting imports with isort..."
isort app/ tests/ train.py --profile black

# 3. Check quality
echo ""
echo "Running Flake8..."
flake8 app/ tests/ train.py --max-line-length=88 --extend-ignore=E203,W503 || true

# 4. Setup pre-commit
echo ""
echo "Setting up pre-commit hooks..."
if [ -f .pre-commit-config.yaml ]; then
    pre-commit install
    echo "✓ Pre-commit hooks installed"
else
    echo ".pre-commit-config.yaml not found. Creating it..."
    cat > .pre-commit-config.yaml << 'EOL'
repos:
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]
EOL
    pre-commit install
    echo "✓ Created .pre-commit-config.yaml and installed hooks"
fi

echo ""
echo "Quality tools setup complete!"
echo ""
echo "Next steps:"
echo "  1. Review any Flake8 warnings above"
echo "  2. Run: make test (to run tests)"
echo "  3. Run: git commit (pre-commit will auto-check)"
