#!/usr/bin/env python3
"""Convert tech support QA to production logs format"""
import json
import random
from datetime import datetime, timedelta

# Load your tech support data
with open("data/tech_support_qa.json", "r") as f:
    qa_data = json.load(f)

# Convert to production logs format
production_logs = []
base_time = datetime.now() - timedelta(days=7)

for i, item in enumerate(qa_data):
    log_entry = {
        "timestamp": (base_time + timedelta(hours=i)).isoformat(),
        "input": item["question"],
        "output": item["answer"],
        "input_length": len(item["question"]),
        "output_length": len(item["answer"]),
        "response_time": random.uniform(0.5, 2.5),  # Simulated
        "confidence": random.uniform(0.7, 0.95),  # Simulated
        "sentiment_score": random.uniform(-0.2, 0.8),  # Simulated
        "prediction": "technical",  # All are technical support
    }
    production_logs.append(log_entry)

# Save production logs
with open("monitoring/production_logs.json", "w") as f:
    json.dump(production_logs, f, indent=2)

print(f" Converted {len(production_logs)} records to production logs format")
print(f" Saved to: monitoring/production_logs.json")
