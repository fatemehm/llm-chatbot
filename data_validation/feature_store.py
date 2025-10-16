import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List


class SimpleFeatureStore:
    """Lightweight feature store for ML features"""

    def __init__(self, store_path: str = "data_validation/feature_store"):
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        self.metadata_path = os.path.join(store_path, "metadata.json")
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load feature store metadata"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {"features": {}, "versions": []}

    def _save_metadata(self):
        """Save metadata"""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def register_feature(self, name: str, description: str, feature_type: str, source: str):
        """Register a new feature"""
        feature_id = hashlib.md5(name.encode()).hexdigest()[:8]

        self.metadata["features"][name] = {
            "id": feature_id,
            "name": name,
            "description": description,
            "type": feature_type,
            "source": source,
            "created_at": datetime.now().isoformat(),
        }

        self._save_metadata()
        print(f"✅ Registered feature: {name}")

    def save_features(self, feature_name: str, features: List[Dict], version: str = None):
        """Save computed features"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        feature_path = os.path.join(self.store_path, f"{feature_name}_v{version}.json")

        with open(feature_path, "w") as f:
            json.dump(features, f, indent=2)

        # Update metadata
        self.metadata["versions"].append(
            {
                "feature_name": feature_name,
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "num_samples": len(features),
                "path": feature_path,
            }
        )

        self._save_metadata()
        print(f"💾 Saved {len(features)} features: {feature_name} v{version}")

    def load_features(self, feature_name: str, version: str = None) -> List[Dict]:
        """Load features by name and version"""
        if version is None:
            # Get latest version
            versions = [v for v in self.metadata["versions"] if v["feature_name"] == feature_name]
            if not versions:
                raise ValueError(f"No features found: {feature_name}")
            version = versions[-1]["version"]

        feature_path = os.path.join(self.store_path, f"{feature_name}_v{version}.json")

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Features not found: {feature_path}")

        with open(feature_path, "r") as f:
            return json.load(f)

    def list_features(self):
        """List all registered features"""
        print("\n📦 Feature Store Inventory")
        print("=" * 60)

        if not self.metadata["features"]:
            print("No features registered")
            return

        for name, info in self.metadata["features"].items():
            print(f"\nFeature: {name}")
            print(f"  ID: {info['id']}")
            print(f"  Type: {info['type']}")
            print(f"  Source: {info['source']}")
            print(f"  Created: {info['created_at']}")

        print(f"\nTotal features: {len(self.metadata['features'])}")
        print(f"Total versions: {len(self.metadata['versions'])}")


def extract_features(data: List[Dict]) -> List[Dict]:
    """Extract features from raw data"""
    features = []

    for item in data:
        question = item["question"]
        answer = item["answer"]

        # Extract basic features
        feature = {
            "question": question,
            "answer": answer,
            "question_length": len(question.split()),
            "answer_length": len(answer.split()),
            "question_char_count": len(question),
            "has_question_mark": "?" in question,
            "starts_with_how": question.lower().startswith("how"),
            "starts_with_why": question.lower().startswith("why"),
            "starts_with_what": question.lower().startswith("what"),
            "contains_error": "error" in question.lower(),
            "contains_fix": "fix" in question.lower(),
            "technical_keywords_count": sum(
                1 for kw in ["error", "crash", "slow", "bug", "issue"] if kw in question.lower()
            ),
        }

        features.append(feature)

    return features


if __name__ == "__main__":
    # Demo feature store
    print("🏪 Feature Store Demo\n")

    # Initialize store
    store = SimpleFeatureStore()

    # Register features
    store.register_feature(
        name="question_length",
        description="Number of words in question",
        feature_type="numeric",
        source="question",
    )

    store.register_feature(
        name="technical_keywords",
        description="Count of technical keywords",
        feature_type="numeric",
        source="question",
    )

    # Load data and extract features
    with open("data/tech_support_qa.json", "r") as f:
        data = json.load(f)

    features = extract_features(data)

    # Save features
    store.save_features("qa_features", features)

    # List features
    store.list_features()

    # Load features
    loaded = store.load_features("qa_features")
    print(f"\n✅ Loaded {len(loaded)} feature vectors")

    # Show sample
    print("\nSample feature vector:")
    print(json.dumps(loaded[0], indent=2))
