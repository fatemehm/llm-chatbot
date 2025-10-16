import json
import os

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Setup MLflow
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tech-support-chatbot")

# Load and prepare data
with open("data/tech_support_qa.json", "r") as f:
    data = json.load(f)

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

split_idx = int(len(data) * 0.9)
train_data, val_data = data[:split_idx], data[split_idx:]


def tokenize_seq2seq(examples, tokenizer):
    inputs = tokenizer(
        examples["input"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    inputs["labels"] = tokenizer(
        examples["output"], padding="max_length", truncation=True, max_length=128
    )["input_ids"]
    return inputs


def tokenize_classification(examples, tokenizer):
    inputs = tokenizer(
        examples["input"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    inputs["labels"] = examples["label"]
    return inputs


def tokenize_causal(examples, tokenizer):
    # For GPT models, concatenate Q&A as single text
    texts = [f"Question: {q} Answer: {a}" for q, a in zip(examples["input"], examples["output"])]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)


# Model configs
MODELS = {
    "google/flan-t5-small": {
        "cls": AutoModelForSeq2SeqLM,
        "targets": ["q", "v"],
        "task_type": TaskType.SEQ_2_SEQ_LM,
        "tokenize_fn": tokenize_seq2seq,
        "labels": None,
        "data_collator": None,
    },
    "bert-base-uncased": {
        "cls": AutoModelForSequenceClassification,
        "targets": ["query", "value"],
        "task_type": TaskType.SEQ_CLS,
        "tokenize_fn": tokenize_classification,
        "labels": 2,
        "data_collator": None,
    },
    "distilgpt2": {
        "cls": AutoModelForCausalLM,
        "targets": ["c_attn"],
        "task_type": TaskType.CAUSAL_LM,
        "tokenize_fn": tokenize_causal,
        "labels": None,
        "data_collator": "causal",
    },
}

# Train each model
for model_name, cfg in MODELS.items():
    print(f"\nTraining {model_name}...")

    with mlflow.start_run(run_name=f"fine-tune-{model_name.replace('/', '-')}"):
        mlflow.log_params({"model": model_name, "lora_r": 8, "epochs": 5, "lr": 1e-4})

        # Load model and tokenizer
        model = (
            cfg["cls"].from_pretrained(model_name, num_labels=cfg["labels"])
            if cfg["labels"]
            else cfg["cls"].from_pretrained(model_name)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Setup tokenizer for causal models
        if cfg["task_type"] == TaskType.CAUSAL_LM:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id

        # Prepare datasets
        train_ds_list = [
            {"input": d["question"], "output": d["answer"], "label": d["label"]} for d in train_data
        ]
        val_ds_list = [
            {"input": d["question"], "output": d["answer"], "label": d["label"]} for d in val_data
        ]

        train_ds = Dataset.from_list(train_ds_list).map(
            lambda x: cfg["tokenize_fn"](x, tokenizer),
            batched=True,
            remove_columns=["input", "output", "label"],
        )
        val_ds = Dataset.from_list(val_ds_list).map(
            lambda x: cfg["tokenize_fn"](x, tokenizer),
            batched=True,
            remove_columns=["input", "output", "label"],
        )

        # Data collator for causal LM
        data_collator = (
            DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            if cfg["data_collator"] == "causal"
            else None
        )

        # Apply LoRA
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=cfg["targets"],
            lora_dropout=0.1,
            task_type=cfg["task_type"],
        )
        model = get_peft_model(model, lora_cfg)

        # Train
        output_dir = f"models/{model_name.replace('/', '-')}-lora"
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=4,
            learning_rate=1e-4,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
        )
        result = trainer.train()

        # Save and log
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        mlflow.log_metrics(
            {
                "train_loss": result.training_loss,
                "eval_loss": trainer.evaluate()["eval_loss"],
            }
        )
        mlflow.log_artifacts(output_dir, artifact_path="model")

        print(f" {model_name} done - Train: {result.training_loss:.4f}")

print("\n All models trained!")
