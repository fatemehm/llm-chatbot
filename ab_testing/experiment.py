import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import chi2_contingency


@dataclass
class Variant:
    """A/B test variant configuration"""

    name: str
    model_name: str
    traffic_percentage: float
    metrics: Dict = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "avg_response_time": 0.0,
                "response_times": [],
            }


@dataclass
class ABTestResult:
    """Result of A/B test analysis"""

    variant_a: str
    variant_b: str
    metric: str
    a_value: float
    b_value: float
    p_value: float
    significant: bool
    winner: Optional[str]


class ABTestManager:
    """Manage A/B testing experiments"""

    def __init__(self, experiment_name: str = "default"):
        self.experiment_name = experiment_name
        self.variants: Dict[str, Variant] = {}
        self.results_log: List[Dict] = []

    def create_experiment(self, variants: List[Variant]):
        """Create new A/B test experiment"""
        # Validate traffic percentages sum to 100
        total_traffic = sum(v.traffic_percentage for v in variants)
        if not (99.9 <= total_traffic <= 100.1):
            raise ValueError(f"Traffic percentages must sum to 100, got {total_traffic}")

        self.variants = {v.name: v for v in variants}
        print(f"✅ Experiment '{self.experiment_name}' created with {len(variants)} variants")
        for v in variants:
            print(f"  - {v.name}: {v.model_name} ({v.traffic_percentage}%)")

    def assign_variant(self, user_id: str) -> Variant:
        """Assign user to a variant (deterministic based on user_id)"""
        # Hash user_id for consistent assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        percentage = (hash_value % 100) / 100

        cumulative = 0.0
        for variant in self.variants.values():
            cumulative += variant.traffic_percentage / 100
            if percentage < cumulative:
                return variant

        # Fallback to first variant
        return list(self.variants.values())[0]

    def log_result(
        self,
        variant_name: str,
        success: bool,
        response_time: float,
        user_feedback: Optional[int] = None,
    ):
        """Log experiment result"""
        if variant_name not in self.variants:
            raise ValueError(f"Unknown variant: {variant_name}")

        variant = self.variants[variant_name]
        variant.metrics["requests"] += 1

        if success:
            variant.metrics["successes"] += 1
        else:
            variant.metrics["failures"] += 1

        # Update response time
        variant.metrics["response_times"].append(response_time)
        variant.metrics["avg_response_time"] = np.mean(variant.metrics["response_times"])

        # Log individual result
        self.results_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "variant": variant_name,
                "success": success,
                "response_time": response_time,
                "user_feedback": user_feedback,
            }
        )

    def analyze_results(self, metric: str = "success_rate") -> ABTestResult:
        """Analyze A/B test results for statistical significance"""
        if len(self.variants) != 2:
            raise ValueError("Analysis currently supports 2 variants only")

        variant_a, variant_b = list(self.variants.values())

        if metric == "success_rate":
            # Chi-square test for success rates
            contingency_table = [
                [variant_a.metrics["successes"], variant_a.metrics["failures"]],
                [variant_b.metrics["successes"], variant_b.metrics["failures"]],
            ]

            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            a_rate = variant_a.metrics["successes"] / max(variant_a.metrics["requests"], 1)
            b_rate = variant_b.metrics["successes"] / max(variant_b.metrics["requests"], 1)

            significant = p_value < 0.05
            winner = variant_a.name if a_rate > b_rate else variant_b.name if significant else None

            result = ABTestResult(
                variant_a=variant_a.name,
                variant_b=variant_b.name,
                metric="success_rate",
                a_value=a_rate,
                b_value=b_rate,
                p_value=p_value,
                significant=significant,
                winner=winner,
            )

        elif metric == "response_time":
            # T-test for response times
            from scipy.stats import ttest_ind

            a_times = variant_a.metrics["response_times"]
            b_times = variant_b.metrics["response_times"]

            if len(a_times) < 2 or len(b_times) < 2:
                return None

            t_stat, p_value = ttest_ind(a_times, b_times)

            a_avg = np.mean(a_times)
            b_avg = np.mean(b_times)

            significant = p_value < 0.05
            winner = variant_a.name if a_avg < b_avg else variant_b.name if significant else None

            result = ABTestResult(
                variant_a=variant_a.name,
                variant_b=variant_b.name,
                metric="response_time",
                a_value=a_avg,
                b_value=b_avg,
                p_value=p_value,
                significant=significant,
                winner=winner,
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return result

    def print_report(self):
        """Print detailed experiment report"""
        print(f"\n{'='*60}")
        print(f"A/B Test Report: {self.experiment_name}")
        print(f"{'='*60}\n")

        for name, variant in self.variants.items():
            metrics = variant.metrics
            success_rate = metrics["successes"] / max(metrics["requests"], 1) * 100

            print(f"Variant: {name} ({variant.model_name})")
            print(f"  Requests: {metrics['requests']}")
            print(f"  Success Rate: {success_rate:.2f}%")
            print(f"  Avg Response Time: {metrics['avg_response_time']:.3f}s")
            print()

        # Statistical analysis
        if len(self.variants) == 2:
            print("Statistical Analysis:")
            print("-" * 60)

            # Success rate
            result = self.analyze_results("success_rate")
            if result:
                print(f"\n✓ Success Rate Comparison:")
                print(f"  {result.variant_a}: {result.a_value:.2%}")
                print(f"  {result.variant_b}: {result.b_value:.2%}")
                print(f"  p-value: {result.p_value:.4f}")
                if result.significant:
                    print(f"  🎉 Winner: {result.winner} (statistically significant)")
                else:
                    print(f"  ⚠️  No significant difference (continue testing)")

            # Response time
            result = self.analyze_results("response_time")
            if result:
                print(f"\n✓ Response Time Comparison:")
                print(f"  {result.variant_a}: {result.a_value:.3f}s")
                print(f"  {result.variant_b}: {result.b_value:.3f}s")
                print(f"  p-value: {result.p_value:.4f}")
                if result.significant:
                    print(f"  ⚡ Winner: {result.winner} (statistically significant)")

        print(f"\n{'='*60}\n")

    def save_results(self, path: str = "ab_testing/results.json"):
        """Save experiment results"""
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "experiment_name": self.experiment_name,
            "variants": {name: asdict(variant) for name, variant in self.variants.items()},
            "results_log": self.results_log,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"💾 Results saved to {path}")


# Example: Integrate with FastAPI
def integrate_ab_testing():
    """Example integration with main.py"""
    from fastapi import Request, Response

    from app.main import app

    # Create experiment
    ab_manager = ABTestManager("model_comparison_v1")
    ab_manager.create_experiment(
        [
            Variant("control", "google/flan-t5-small", 50.0),
            Variant("treatment", "distilgpt2", 50.0),
        ]
    )

    @app.middleware("http")
    async def ab_test_middleware(request: Request, call_next):
        # Assign variant based on user/session
        user_id = request.headers.get("X-User-ID", "anonymous")
        variant = ab_manager.assign_variant(user_id)

        # Store variant in request state
        request.state.ab_variant = variant

        # Process request
        import time

        start = time.time()
        response = await call_next(request)
        response_time = time.time() - start

        # Log result
        success = response.status_code == 200
        ab_manager.log_result(variant.name, success, response_time)

        # Add variant to response header
        response.headers["X-AB-Variant"] = variant.name

        return response


if __name__ == "__main__":
    # Demo A/B test
    print("🧪 A/B Testing Demo\n")

    # Create experiment
    manager = ABTestManager("flan_vs_gpt2")
    manager.create_experiment(
        [
            Variant("flan_t5", "google/flan-t5-small", 50.0),
            Variant("gpt2", "distilgpt2", 50.0),
        ]
    )

    # Simulate 100 users
    print("\nSimulating 100 user requests...\n")
    for i in range(100):
        user_id = f"user_{i}"
        variant = manager.assign_variant(user_id)

        # Simulate results
        success = random.random() > 0.1  # 90% success rate
        response_time = np.random.normal(1.5, 0.3)  # ~1.5s avg

        manager.log_result(variant.name, success, response_time)

    # Print report
    manager.print_report()

    # Save results
    manager.save_results()

    print("✅ A/B test simulation complete!")
