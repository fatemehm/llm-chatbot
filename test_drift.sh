#!/bin/bash
echo " Testing drift detection..."
source venv/bin/activate 2>/dev/null || true
python monitoring/drift_detection_enhanced.py --mode evidently --days 7
echo ""
echo " View report: monitoring/reports/drift_report.html"
