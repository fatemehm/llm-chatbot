import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class DataValidator:
    """Validate data quality for training and production"""

    def __init__(self, data_path: str = "data/tech_support_qa.json"):
        self.data_path = data_path
        self.validation_results = {}

    def load_data(self) -> List[Dict]:
        """Load data from JSON"""
        with open(self.data_path, "r") as f:
            return json.load(f)

    def validate_schema(self, data: List[Dict]) -> Dict[str, Any]:
        """Validate data schema"""
        print("\n📋 Schema Validation")
        print("-" * 60)

        required_fields = ["question", "answer"]
        issues = []

        for i, item in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in item:
                    issues.append(f"Row {i}: Missing field '{field}'")

            # Check field types
            if "question" in item and not isinstance(item["question"], str):
                issues.append(f"Row {i}: 'question' must be string")

            if "answer" in item and not isinstance(item["answer"], str):
                issues.append(f"Row {i}: 'answer' must be string")

        if issues:
            print("❌ Schema validation FAILED:")
            for issue in issues[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
        else:
            print("✅ Schema validation PASSED")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "total_rows": len(data),
        }

    def validate_content(self, data: List[Dict]) -> Dict[str, Any]:
        """Validate content quality"""
        print("\n📝 Content Validation")
        print("-" * 60)

        issues = []
        stats = {
            "empty_questions": 0,
            "empty_answers": 0,
            "too_short_questions": 0,
            "too_short_answers": 0,
            "too_long_questions": 0,
            "duplicate_questions": 0,
        }

        questions_seen = set()

        for i, item in enumerate(data):
            question = item.get("question", "")
            answer = item.get("answer", "")

            # Empty checks
            if not question.strip():
                stats["empty_questions"] += 1
                issues.append(f"Row {i}: Empty question")

            if not answer.strip():
                stats["empty_answers"] += 1
                issues.append(f"Row {i}: Empty answer")

            # Length checks
            if len(question.split()) < 3:
                stats["too_short_questions"] += 1
                issues.append(f"Row {i}: Question too short (< 3 words)")

            if len(answer.split()) < 3:
                stats["too_short_answers"] += 1
                issues.append(f"Row {i}: Answer too short (< 3 words)")

            if len(question.split()) > 100:
                stats["too_long_questions"] += 1
                issues.append(f"Row {i}: Question too long (> 100 words)")

            # Duplicate checks
            if question.lower() in questions_seen:
                stats["duplicate_questions"] += 1
                issues.append(f"Row {i}: Duplicate question")
            questions_seen.add(question.lower())

        # Print summary
        print(f"Total rows: {len(data)}")
        print(f"Empty questions: {stats['empty_questions']}")
        print(f"Empty answers: {stats['empty_answers']}")
        print(f"Too short questions: {stats['too_short_questions']}")
        print(f"Too short answers: {stats['too_short_answers']}")
        print(f"Too long questions: {stats['too_long_questions']}")
        print(f"Duplicate questions: {stats['duplicate_questions']}")

        total_issues = sum(stats.values())
        if total_issues > 0:
            print(f"\n⚠️  Found {total_issues} content issues")
        else:
            print("\n✅ Content validation PASSED")

        return {
            "passed": total_issues == 0,
            "stats": stats,
            "issues": issues[:20],  # First 20 issues
        }

    def validate_distribution(self, data: List[Dict]) -> Dict[str, Any]:
        """Validate data distribution"""
        print("\n📊 Distribution Validation")
        print("-" * 60)

        questions = [item["question"] for item in data]
        answers = [item["answer"] for item in data]

        # Calculate statistics
        q_lengths = [len(q.split()) for q in questions]
        a_lengths = [len(a.split()) for a in answers]

        stats = {
            "total_samples": len(data),
            "avg_question_length": sum(q_lengths) / len(q_lengths),
            "min_question_length": min(q_lengths),
            "max_question_length": max(q_lengths),
            "avg_answer_length": sum(a_lengths) / len(a_lengths),
            "min_answer_length": min(a_lengths),
            "max_answer_length": max(a_lengths),
        }

        print(f"Total samples: {stats['total_samples']}")
        print(f"\nQuestion lengths:")
        print(f"  Avg: {stats['avg_question_length']:.1f} words")
        print(f"  Min: {stats['min_question_length']} words")
        print(f"  Max: {stats['max_question_length']} words")

        print(f"\nAnswer lengths:")
        print(f"  Avg: {stats['avg_answer_length']:.1f} words")
        print(f"  Min: {stats['min_answer_length']} words")
        print(f"  Max: {stats['max_answer_length']} words")

        # Check for imbalanced data
        if stats["avg_question_length"] > 50:
            print("\n⚠️  Questions are very long on average")

        if stats["avg_answer_length"] > 100:
            print("\n⚠️  Answers are very long on average")

        print("\n✅ Distribution analysis complete")

        return stats

    def validate_labels(self, data: List[Dict]) -> Dict[str, Any]:
        """Validate label distribution for classification"""
        print("\n🏷️  Label Validation")
        print("-" * 60)

        # Add labels based on keywords (same as training)
        keywords = [
            "error",
            "slow",
            "crash",
            "bug",
            "issue",
            "problem",
            "not working",
            "fail",
        ]

        technical_count = 0
        for item in data:
            question = item["question"].lower()
            if any(kw in question for kw in keywords):
                technical_count += 1

        non_technical_count = len(data) - technical_count

        print(f"Technical questions: {technical_count} ({technical_count/len(data)*100:.1f}%)")
        print(
            f"Non-technical questions: {non_technical_count} ({non_technical_count/len(data)*100:.1f}%)"
        )

        # Check for imbalance
        imbalance_ratio = max(technical_count, non_technical_count) / max(
            min(technical_count, non_technical_count), 1
        )

        if imbalance_ratio > 3:
            print(f"\n⚠️  Significant class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
            print("   Consider data augmentation or class weighting")
        else:
            print("\n✅ Label distribution is acceptable")

        return {
            "technical": technical_count,
            "non_technical": non_technical_count,
            "imbalance_ratio": imbalance_ratio,
        }

    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation checks"""
        print("=" * 60)
        print("🔍 Data Quality Validation Report")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Data path: {self.data_path}")

        # Load data
        data = self.load_data()

        # Run validations
        results = {
            "timestamp": datetime.now().isoformat(),
            "data_path": self.data_path,
            "schema_validation": self.validate_schema(data),
            "content_validation": self.validate_content(data),
            "distribution_validation": self.validate_distribution(data),
            "label_validation": self.validate_labels(data),
        }

        # Overall status
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        all_passed = (
            results["schema_validation"]["passed"] and results["content_validation"]["passed"]
        )

        if all_passed:
            print("✅ All validations PASSED")
        else:
            print("❌ Some validations FAILED")
            print("   Review issues above and fix data quality problems")

        # Save report
        os.makedirs("data_validation/reports", exist_ok=True)
        report_path = (
            f"data_validation/reports/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Validation report saved to: {report_path}")

        return results


class DataLineageTracker:
    """Track data lineage and transformations"""

    def __init__(self):
        self.lineage_path = "data_validation/lineage.json"
        self.lineage = self._load_lineage()

    def _load_lineage(self) -> List[Dict]:
        """Load existing lineage"""
        if os.path.exists(self.lineage_path):
            with open(self.lineage_path, "r") as f:
                return json.load(f)
        return []

    def track_data_load(self, source: str, num_records: int):
        """Track data loading event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "data_load",
            "source": source,
            "num_records": num_records,
        }
        self.lineage.append(event)
        self._save_lineage()

    def track_transformation(
        self,
        operation: str,
        input_records: int,
        output_records: int,
        details: Dict = None,
    ):
        """Track data transformation"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "transformation",
            "operation": operation,
            "input_records": input_records,
            "output_records": output_records,
            "details": details or {},
        }
        self.lineage.append(event)
        self._save_lineage()

    def track_model_training(self, model_name: str, train_size: int, val_size: int, test_size: int):
        """Track model training data split"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "model_training",
            "model_name": model_name,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
        }
        self.lineage.append(event)
        self._save_lineage()

    def _save_lineage(self):
        """Save lineage to file"""
        os.makedirs(os.path.dirname(self.lineage_path), exist_ok=True)
        with open(self.lineage_path, "w") as f:
            json.dump(self.lineage, f, indent=2)

    def print_lineage(self):
        """Print data lineage"""
        print("\n" + "=" * 60)
        print("📊 Data Lineage")
        print("=" * 60)

        if not self.lineage:
            print("No lineage events tracked yet")
            return

        for event in self.lineage[-10:]:  # Last 10 events
            print(f"\n[{event['timestamp']}]")
            print(f"  Type: {event['event_type']}")

            if event["event_type"] == "data_load":
                print(f"  Source: {event['source']}")
                print(f"  Records: {event['num_records']}")

            elif event["event_type"] == "transformation":
                print(f"  Operation: {event['operation']}")
                print(f"  Input: {event['input_records']} → Output: {event['output_records']}")

            elif event["event_type"] == "model_training":
                print(f"  Model: {event['model_name']}")
                print(
                    f"  Train: {event['train_size']}, Val: {event['val_size']}, Test: {event['test_size']}"
                )


if __name__ == "__main__":
    # Run data validation
    validator = DataValidator()
    results = validator.run_all_validations()

    # Demo lineage tracking
    print("\n\n")
    tracker = DataLineageTracker()
    tracker.track_data_load("data/tech_support_qa.json", 114)
    tracker.track_transformation(
        "train_test_split", 114, 114, {"train_ratio": 0.9, "test_ratio": 0.1}
    )
    tracker.track_model_training("flan-t5-small", 102, 12, 23)
    tracker.print_lineage()

    print("\n✅ Data quality validation complete!")
