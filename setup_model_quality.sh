#!/bin/bash
set -e

echo "Setting up Model Quality & Monitoring..."
echo ""

# Create directories
echo "Creating directories..."
mkdir -p monitoring
mkdir -p explainability/plots
mkdir -p ab_testing

# Install additional dependencies
echo "Installing dependencies..."
pip install -q shap lime rouge scipy alibi-detect

# Create __init__.py files
touch monitoring/__init__.py
touch explainability/__init__.py
touch ab_testing/__init__.py

echo ""
echo "Setup complete!"
echo ""
echo "Available commands:"
echo "  make evaluate    - Evaluate model performance"
echo "  make monitor     - Run drift detection"
echo "  make explain     - Generate SHAP explanations"
echo "  make ab-test     - Run A/B test simulation"
echo ""
echo "Next steps:"
echo "  1. Run: make evaluate"
echo "  2. Check: mlflow ui"
echo "  3. Review: MODEL_QUALITY.md"
