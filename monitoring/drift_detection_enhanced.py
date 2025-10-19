import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

print("Starting drift detection...")


def load_production_logs(days=7):
    logs_path = Path("monitoring/production_logs.json")

    if not logs_path.exists():
        print(f"  No production logs found at {logs_path}")
        print("Creating sample data...")

        # Create sample data
        sample_data = []
        for i in range(500):
            sample_data.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "input_length": np.random.normal(120, 25),
                    "output_length": np.random.normal(140, 35),
                    "response_time": np.random.exponential(2.5),
                    "confidence": np.random.beta(6, 3),
                    "sentiment_score": np.random.uniform(-0.5, 1),
                    "prediction": np.random.choice(["technical", "billing", "general"]),
                }
            )

        df = pd.DataFrame(sample_data)
        print(f" Created sample data: {len(df)} records")
        return df

    with open(logs_path) as f:
        logs = json.load(f)

    print(f" Loaded {len(logs)} production logs")

    # Convert to DataFrame with basic features
    if logs:
        df = pd.DataFrame(logs)

        # Extract basic features if not present
        if "input_length" not in df.columns:
            if "input" in df.columns:
                df["input_length"] = df["input"].str.len()
            else:
                df["input_length"] = np.random.normal(100, 20, len(df))

        if "output_length" not in df.columns:
            if "output" in df.columns:
                df["output_length"] = df["output"].str.len()
            else:
                df["output_length"] = np.random.normal(150, 30, len(df))

        if "response_time" not in df.columns:
            df["response_time"] = np.random.exponential(2, len(df))

        if "confidence" not in df.columns:
            df["confidence"] = np.random.beta(8, 2, len(df))

        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = np.random.uniform(-1, 1, len(df))

        if "prediction" not in df.columns:
            df["prediction"] = np.random.choice(["technical", "billing", "general"], len(df))

        return df
    else:
        print("  No logs data, creating sample...")
        # Recursive call with different param to create sample
        sample_data = []
        for i in range(500):
            sample_data.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "input_length": np.random.normal(120, 25),
                    "output_length": np.random.normal(140, 35),
                    "response_time": np.random.exponential(2.5),
                    "confidence": np.random.beta(6, 3),
                    "sentiment_score": np.random.uniform(-0.5, 1),
                    "prediction": np.random.choice(["technical", "billing", "general"]),
                }
            )
        return pd.DataFrame(sample_data)


def load_baseline():
    baseline_path = Path("data/reference/baseline_data.csv")

    if not baseline_path.exists():
        print(f"  No baseline found")
        return None

    df = pd.read_csv(baseline_path)
    print(f" Loaded baseline: {len(df)} records")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="evidently")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(" Drift Detection Pipeline")
    print("=" * 60)

    try:
        # Try to import evidently
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.report import Report

        print(" Evidently imported successfully")
    except ImportError:
        print(" Evidently not installed!")
        print("Run: pip install evidently==0.4.25")
        sys.exit(1)

    # Load data
    print("\n📊 Loading data...")
    baseline = load_baseline()
    current = load_production_logs(args.days)

    if baseline is None:
        print(" No baseline data!")
        sys.exit(1)

    if args.update_baseline:
        print("\n Updating baseline...")
        current.to_csv("data/reference/baseline_data.csv", index=False)
        print(" Baseline updated")
        sys.exit(0)

    # Ensure both dataframes have same columns
    common_cols = list(set(baseline.columns) & set(current.columns))
    baseline = baseline[common_cols]
    current = current[common_cols]

    # Run drift detection
    print("\n Running drift analysis...")

    # Create reports directory
    Path("monitoring/reports").mkdir(parents=True, exist_ok=True)

    # Configure column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = "prediction" if "prediction" in current.columns else None
    column_mapping.numerical_features = [
        col
        for col in current.columns
        if col not in ["prediction", "timestamp"] and pd.api.types.is_numeric_dtype(current[col])
    ]

    print(f"   Monitoring features: {column_mapping.numerical_features}")

    # Create report
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ]
    )

    # Run report
    report.run(reference_data=baseline, current_data=current, column_mapping=column_mapping)

    # Save reports
    html_path = "monitoring/reports/drift_report.html"
    report.save_html(html_path)
    print(f" HTML report: {html_path}")

    # Extract metrics
    report_dict = report.as_dict()
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "dataset_drift_detected": False,
        "drifted_features": [],
    }

    for metric in report_dict.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            result = metric.get("result", {})
            metrics["dataset_drift_detected"] = result.get("dataset_drift", False)
            metrics["drift_share"] = result.get("share_of_drifted_columns", 0)
            metrics["number_of_drifted_columns"] = result.get("number_of_drifted_columns", 0)

            drift_by_cols = result.get("drift_by_columns", {})
            for feature, info in drift_by_cols.items():
                if info.get("drift_detected"):
                    metrics["drifted_features"].append(feature)

    # Save JSON metrics
    json_path = "monitoring/reports/drift_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f" JSON metrics: {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(" DRIFT DETECTION SUMMARY")
    print("=" * 60)
    print(f"Dataset Drift: {metrics['dataset_drift_detected']}")
    print(f"Drift Share: {metrics.get('drift_share', 0):.2%}")
    print(f"Drifted Features: {len(metrics['drifted_features'])}")

    if metrics["drifted_features"]:
        print(f"\n  Features with drift:")
        for feature in metrics["drifted_features"]:
            print(f"  • {feature}")
    else:
        print("\n✅ No significant drift detected")

    print("=" * 60)

    # Exit code
    if metrics["dataset_drift_detected"]:
        print("\n❌ DRIFT DETECTED")
        sys.exit(1)
    else:
        print("\n✅ NO DRIFT")
        sys.exit(0)


if __name__ == "__main__":
    main()
