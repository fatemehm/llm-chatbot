#!/bin/bash
set -e

echo "🔍 Setting up Data Quality & Validation..."
echo ""

# Create directories
echo "Creating directories..."
mkdir -p data_validation/reports
mkdir -p data_validation/feature_store

# Create __init__.py
touch data_validation/__init__.py

echo ""
echo "Setup complete!"
echo ""
echo "Available commands:"
echo "  make validate-data  - Validate data quality"
echo "  make feature-store  - Manage feature store"
echo "  make data-quality   - Run all data checks"
echo ""
echo "Next steps:"
echo "  1. Run: make validate-data"
echo "  2. Run: make feature-store"
echo "  3. Review: data_validation/reports/"
