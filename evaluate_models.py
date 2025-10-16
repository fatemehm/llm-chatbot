import json
import os

import mlflow
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from transformers import AutoTokenizer

from app.model import ChatbotModel


def evaluate_classification_model(model, tokenizer, test_data):
    """Evaluate classification model (BERT)"""
    questions = [d["question"] for d in test_data]
    true_labels = [d["label"] for d in test_data]

    predictions = []
    for question in questions:
        # Use model directly, not through generate_response
        # to avoid fuzzy matching
        inputs = tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = model.model(**inputs)
            pred_label = outputs.logits.argmax(-1).item()
            predictions.append(pred_label)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="binary", zero_division=0
    )

    print("\n=== Classification Metrics ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nDetailed Report:")
    print(
        classification_report(
            true_labels,
            predictions,
            target_names=["Non-technical", "Technical"],
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def evaluate_generation_model(model, test_data):
    """Evaluate generation models (FLAN-T5, GPT2)"""
    from rouge import Rouge

    rouge = Rouge()

    questions = [d["question"] for d in test_data]
    references = [d["answer"] for d in test_data]

    predictions = []
    for question in questions:
        response = model.generate_response(question)
        predictions.append(response)

    # Calculate ROUGE scores
    rouge_scores = rouge.get_scores(predictions, references, avg=True)

    print("\n=== Generation Metrics (ROUGE) ===")
    print(f"ROUGE-1: {rouge_scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")

    # Sample predictions
    print("\n=== Sample Predictions ===")
    for i in range(min(3, len(questions))):
        print(f"\nQ: {questions[i]}")
        print(f"Expected: {references[i]}")
        print(f"Predicted: {predictions[i]}")

    return {
        "rouge_1": rouge_scores["rouge-1"]["f"],
        "rouge_2": rouge_scores["rouge-2"]["f"],
        "rouge_l": rouge_scores["rouge-l"]["f"],
    }


def main():
    """Evaluate all trained models"""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("model-evaluation")

    # Load test data
    with open("data/tech_support_qa.json", "r") as f:
        data = json.load(f)

    # Add labels
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
    for d in data:
        d["label"] = 1 if any(kw in d["question"].lower() for kw in keywords) else 0

    # Use last 20% as test set
    split_idx = int(len(data) * 0.8)
    test_data = data[split_idx:]

    print(f"Evaluating on {len(test_data)} test samples")

    # Evaluate each model
    models = {
        "google/flan-t5-small": "generation",
        "bert-base-uncased": "classification",
        "distilgpt2": "generation",
    }

    for model_name, model_type in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        with mlflow.start_run(run_name=f"eval-{model_name.replace('/', '-')}"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_samples", len(test_data))

            try:
                # Load model
                os.environ["MODEL_NAME"] = model_name
                model = ChatbotModel()

                # Evaluate based on model type
                if model_type == "classification":
                    metrics = evaluate_classification_model(model, model.tokenizer, test_data)
                else:
                    metrics = evaluate_generation_model(model, test_data)

                # Log metrics to MLflow
                mlflow.log_metrics(metrics)

                print(f"\n✅ {model_name} evaluation complete!")

            except Exception as e:
                print(f"\n❌ Error evaluating {model_name}: {str(e)}")
                mlflow.log_param("error", str(e))

    print(f"\n{'='*60}")
    print("Evaluation complete! Check MLflow UI for detailed metrics.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
