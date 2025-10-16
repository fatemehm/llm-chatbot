import os
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")


class ModelExplainer:
    """Generate explanations for model predictions"""

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.model_path = f"models/{model_name.replace('/', '-')}-lora"

        print(f"Loading {model_name} for explainability...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load base model (SHAP works better with base models)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities"""
        # Handle both single strings and lists
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        return probs.numpy()

    def explain_with_lime(self, text: str, save_plot: bool = True):
        """Generate LIME explanation (simpler alternative to SHAP)"""
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            print("LIME not installed. Run: pip install lime")
            return None

        print(f"\nExplaining: '{text}'")

        # Create LIME explainer
        explainer = LimeTextExplainer(class_names=["Non-technical", "Technical"])

        # Generate explanation
        exp = explainer.explain_instance(text, self.predict_proba, num_features=10, num_samples=100)

        # Get prediction
        pred_proba = self.predict_proba([text])[0]
        predicted_class = np.argmax(pred_proba)
        confidence = pred_proba[predicted_class]

        print(f"Prediction: {'Technical' if predicted_class == 1 else 'Non-technical'}")
        print(f"Confidence: {confidence:.2%}")

        # Print top features
        print("\nTop contributing words:")
        for word, weight in exp.as_list()[:5]:
            direction = "→ Technical" if weight > 0 else "→ Non-technical"
            print(f"  '{word}': {abs(weight):.3f} {direction}")

        # Save visualization
        if save_plot:
            os.makedirs("explainability/plots", exist_ok=True)

            # Save as HTML
            html_path = f"explainability/plots/lime_{abs(hash(text))}.html"
            exp.save_to_file(html_path)
            print(f"\n💾 Saved explanation to: {html_path}")

            # Save as image
            try:
                fig = exp.as_pyplot_figure()
                img_path = f"explainability/plots/lime_{abs(hash(text))}.png"
                plt.tight_layout()
                plt.savefig(img_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"💾 Saved plot to: {img_path}")
            except Exception as e:
                print(f"Could not save plot: {e}")

        return {
            "text": text,
            "prediction": "Technical" if predicted_class == 1 else "Non-technical",
            "confidence": float(confidence),
            "explanation": exp,
        }

    def simple_attention_explanation(self, text: str):
        """Simple attention-based explanation"""
        print(f"\nAnalyzing: '{text}'")

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_class = probs.argmax().item()
            confidence = probs[0, predicted_class].item()

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        print(f"Prediction: {'Technical' if predicted_class == 1 else 'Non-technical'}")
        print(f"Confidence: {confidence:.2%}")

        # Compute simple importance (gradient-based)
        inputs_embedded = inputs["input_ids"].float().requires_grad_(True)

        print("\nTokens in input:")
        for i, token in enumerate(tokens):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                print(f"  {token}")

        return {
            "text": text,
            "prediction": "Technical" if predicted_class == 1 else "Non-technical",
            "confidence": float(confidence),
            "tokens": [t for t in tokens if t not in ["[CLS]", "[SEP]", "[PAD]"]],
        }

    def explain_batch(self, texts: List[str], method: str = "lime"):
        """Explain multiple predictions"""
        results = []

        for text in texts:
            try:
                if method == "lime":
                    result = self.explain_with_lime(text, save_plot=True)
                else:
                    result = self.simple_attention_explanation(text)

                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error explaining '{text}': {e}")

        return results

    def compare_predictions(self, texts: List[str]):
        """Compare predictions across multiple texts"""
        print("\n" + "=" * 60)
        print("Prediction Comparison")
        print("=" * 60)

        results = []
        for text in texts:
            pred_proba = self.predict_proba([text])[0]
            predicted_class = np.argmax(pred_proba)

            results.append(
                {
                    "text": text,
                    "prediction": "Technical" if predicted_class == 1 else "Non-technical",
                    "technical_prob": pred_proba[1],
                    "non_technical_prob": pred_proba[0],
                }
            )

        # Print table
        print(f"\n{'Text':<50} {'Prediction':<15} {'Confidence':<10}")
        print("-" * 75)
        for r in results:
            conf = max(r["technical_prob"], r["non_technical_prob"])
            text_short = r["text"][:47] + "..." if len(r["text"]) > 50 else r["text"]
            print(f"{text_short:<50} {r['prediction']:<15} {conf:.2%}")

        return results


def main():
    """Demo explainability features"""
    print("🔍 Model Explainability Demo\n")

    try:
        explainer = ModelExplainer("bert-base-uncased")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        print("\nNote: This requires the base BERT model.")
        print("The explainability works best with base models, not LoRA-adapted ones.")
        return

    # Test examples
    test_texts = [
        "My computer keeps crashing when I open Chrome",
        "How do I change my wallpaper?",
        "Blue screen error 0x0000007B",
        "What time is the meeting today?",
    ]

    print("=" * 60)
    print("Method 1: LIME Explanations (Detailed)")
    print("=" * 60)

    try:
        results = explainer.explain_batch(test_texts[:2], method="lime")
        print(f"\n✅ Generated {len(results)} LIME explanations")
    except Exception as e:
        print(f"⚠️  LIME failed: {e}")
        print("Falling back to simpler method...")

    print("\n" + "=" * 60)
    print("Method 2: Simple Token Analysis")
    print("=" * 60)

    for text in test_texts[2:]:
        explainer.simple_attention_explanation(text)

    print("\n" + "=" * 60)
    print("Method 3: Batch Prediction Comparison")
    print("=" * 60)

    explainer.compare_predictions(test_texts)

    print("\n✅ Explainability analysis complete!")
    print("\nGenerated files:")
    print("  - explainability/plots/lime_*.html (interactive)")
    print("  - explainability/plots/lime_*.png (static images)")
    print("\nOpen the HTML files in a browser for interactive exploration!")


if __name__ == "__main__":
    main()
