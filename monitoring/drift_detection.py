import json
import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
from scipy.stats import ks_2samp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data and concept drift in production"""

    def __init__(self, reference_data_path: str = "data/tech_support_qa.json"):
        self.reference_data = self._load_reference_data(reference_data_path)
        self.production_data: List[Dict] = []

    def _load_reference_data(self, path: str) -> List[Dict]:
        """Load reference dataset from training"""
        with open(path, "r") as f:
            return json.load(f)

    def log_prediction(
        self, question: str, prediction: str, confidence: float = None, response_time: float = None
    ):
        """Log production prediction for drift analysis"""
        self.production_data.append(
            {
                "question": question,
                "prediction": prediction,
                "confidence": confidence,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def detect_input_drift(self, threshold: float = 0.05) -> Dict:
        """Detect drift in input distribution using KS test"""
        if len(self.production_data) < 30:
            logger.warning("Need at least 30 samples for drift detection")
            return {"drift_detected": False, "reason": "insufficient_data"}

        # Extract features (e.g., question length)
        ref_lengths = [len(d["question"].split()) for d in self.reference_data]
        prod_lengths = [len(d["question"].split()) for d in self.production_data]

        # Kolmogorov-Smirnov test
        statistic, p_value = ks_2samp(ref_lengths, prod_lengths)

        drift_detected = p_value < threshold

        result = {
            "drift_detected": drift_detected,
            "statistic": statistic,
            "p_value": p_value,
            "threshold": threshold,
            "ref_mean_length": np.mean(ref_lengths),
            "prod_mean_length": np.mean(prod_lengths),
        }

        if drift_detected:
            logger.warning(f"⚠️  INPUT DRIFT DETECTED! p-value: {p_value:.4f}")
        else:
            logger.info(f"✅ No input drift detected. p-value: {p_value:.4f}")

        return result

    def detect_performance_degradation(
        self, baseline_accuracy: float = 0.8, threshold: float = 0.1
    ) -> Dict:
        """Detect if model performance has degraded"""
        if len(self.production_data) < 10:
            return {"degradation_detected": False, "reason": "insufficient_data"}

        # Calculate recent accuracy (if ground truth available)
        # This is simplified - in production you'd need feedback loop
        recent_predictions = self.production_data[-50:]

        # Check response times
        response_times = [d["response_time"] for d in recent_predictions if d.get("response_time")]

        if response_times:
            avg_response_time = np.mean(response_times)
            degradation = avg_response_time > 5.0  # 5 second threshold

            result = {
                "degradation_detected": degradation,
                "avg_response_time": avg_response_time,
                "samples_analyzed": len(response_times),
            }

            if degradation:
                logger.warning(f"⚠️  PERFORMANCE DEGRADATION! Avg time: {avg_response_time:.2f}s")

            return result

        return {"degradation_detected": False, "reason": "no_timing_data"}

    def generate_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": len(self.production_data),
            "reference_size": len(self.reference_data),
        }

        if len(self.production_data) >= 30:
            report["drift_analysis"] = self.detect_input_drift()
            report["performance_analysis"] = self.detect_performance_degradation()
        else:
            report["status"] = "collecting_data"
            report["samples_needed"] = 30 - len(self.production_data)

        return report

    def save_production_data(self, path: str = "monitoring/production_logs.json"):
        """Save production data for analysis"""
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.production_data, f, indent=2)

        logger.info(f"Saved {len(self.production_data)} predictions to {path}")


# Example usage in main.py
def integrate_monitoring():
    """Example of how to integrate into main.py"""
    import time

    from fastapi import Request

    from app.main import app

    drift_detector = DriftDetector()

    @app.middleware("http")
    async def log_predictions(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log to drift detector
        if request.url.path == "/chat":
            # Extract question and prediction from request/response
            # This is simplified - actual implementation would parse body
            drift_detector.log_prediction(
                question="example", prediction="example response", response_time=process_time
            )

        return response


if __name__ == "__main__":
    # Test drift detection
    print("🔍 Drift Detection Demo\n")

    detector = DriftDetector()

    print("Simulating 50 production requests...")
    # Simulate production data with slight drift
    for i in range(50):
        # Add some drift: longer questions over time
        extra_words = " " * (i // 10) if i > 25 else ""
        detector.log_prediction(
            question=f"Test question {i} with some words{extra_words}",
            prediction="Test response",
            response_time=np.random.uniform(0.5, 2.0),
        )

    print("\n" + "=" * 60)
    print("Drift Detection Report")
    print("=" * 60)

    # Generate report
    report = detector.generate_report()

    print(f"\nTotal predictions logged: {report['total_predictions']}")
    print(f"Reference dataset size: {report['reference_size']}")

    if "drift_analysis" in report:
        drift = report["drift_analysis"]
        print(f"\n📊 Input Drift Analysis:")
        print(f"  Drift detected: {'⚠️  YES' if drift['drift_detected'] else '✅ NO'}")
        print(f"  P-value: {drift['p_value']:.4f}")
        print(f"  Reference avg length: {drift['ref_mean_length']:.1f} words")
        print(f"  Production avg length: {drift['prod_mean_length']:.1f} words")

        if drift["drift_detected"]:
            print(f"\n  ⚠️  Action required: Consider retraining model!")

    if "performance_analysis" in report:
        perf = report["performance_analysis"]
        print(f"\n⚡ Performance Analysis:")
        print(f"  Degradation detected: {'⚠️  YES' if perf['degradation_detected'] else '✅ NO'}")
        if "avg_response_time" in perf:
            print(f"  Avg response time: {perf['avg_response_time']:.2f}s")

    # Save data
    detector.save_production_data()

    print("\n" + "=" * 60)
    print("✅ Monitoring complete!")
    print("=" * 60)
