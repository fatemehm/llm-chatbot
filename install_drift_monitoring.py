#!/usr/bin/env python3
"""Simple installer for drift monitoring"""
import os
import subprocess
import sys
from pathlib import Path

print("=" * 60)
print(" Drift Monitoring Installer")
print("=" * 60)

# Check we're in the right place
if not Path("train.py").exists():
    print(" Error: Not in project root!")
    print("Please run from llm-chatbot directory")
    sys.exit(1)

print("✓ Running from project root")

# Create directories
print("\n Creating directories...")
dirs = [".github/workflows", "monitoring/reports", "data/reference", "data/current"]
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f" {d}")

# Install dependencies
print("\n Installing dependencies...")
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "evidently==0.4.25", "plotly", "scipy"],
        check=True,
    )
    print(" evidently, plotly, scipy installed")
except:
    print(" Installation failed, trying individually...")
    for pkg in ["evidently==0.4.25", "plotly", "scipy"]:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
            print(f"  ✓ {pkg}")
        except:
            print(f"  ✗ {pkg} failed")

# Create baseline
print("\n Creating baseline data...")
try:
    import json

    import numpy as np
    import pandas as pd

    if Path("data/tech_support_qa.json").exists():
        with open("data/tech_support_qa.json") as f:
            data = json.load(f)

        baseline = []
        for item in data[:1000]:
            baseline.append(
                {
                    "input_length": len(item.get("question", "")),
                    "output_length": len(item.get("answer", "")),
                    "response_time": np.random.exponential(2),
                    "confidence": np.random.beta(8, 2),
                    "sentiment_score": np.random.uniform(-1, 1),
                    "prediction": item.get("category", "general"),
                }
            )

        df = pd.DataFrame(baseline)
        df.to_csv("data/reference/baseline_data.csv", index=False)
        print(f" Created baseline with {len(df)} records")
    else:
        print(" Training data not found, skipping baseline")
except Exception as e:
    print(f"    Could not create baseline: {e}")

# Create test script
print("\n Creating test script...")
with open("test_drift.sh", "w") as f:
    f.write(
        """#!/bin/bash
echo " Testing drift detection..."
source venv/bin/activate 2>/dev/null || true
python monitoring/drift_detection_enhanced.py --mode evidently --days 7
echo ""
echo " View report: monitoring/reports/drift_report.html"
"""
    )
os.chmod("test_drift.sh", 0o755)
print("  ✓ test_drift.sh created")

print("\n" + "=" * 60)
print(" Installation Complete!")
print("=" * 60)
