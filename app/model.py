import json
import logging
import os
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatbotModel:
    qa_pairs: List[Dict[str, str]]
    model_name: str
    model_path: str
    model: Any
    tokenizer: Any

    def __init__(self) -> None:
        dataset_path = os.path.join("data", "tech_support_qa.json")
        self.qa_pairs = []
        if os.path.exists(dataset_path):
            with open(dataset_path, "r") as f:
                self.qa_pairs = json.load(f)
            logger.info(f"Loaded {len(self.qa_pairs)} Q&A pairs")
        else:
            logger.warning(f"Dataset {dataset_path} not found")

        self.model_name = os.getenv("MODEL_NAME", "google/flan-t5-small")
        model_dir = f"{self.model_name.replace('/', '-')}-lora"
        self.model_path = os.path.join("models", model_dir)
        logger.info(f"Loading model: {self.model_name}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path {self.model_path} does not exist")

        try:
            if "flan-t5" in self.model_name:
                base = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self.model = PeftModel.from_pretrained(
                    base,
                    self.model_path,
                    is_trainable=False,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            elif "bert" in self.model_name:
                base = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, num_labels=2
                )
                self.model = PeftModel.from_pretrained(base, self.model_path, is_trainable=False)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            else:  # distilgpt2
                base = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.model = PeftModel.from_pretrained(
                    base,
                    self.model_path,
                    is_trainable=False,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if not self.tokenizer.pad_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Successfully loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100

    def _find_in_dataset(self, text: str, threshold: int = 70) -> Optional[str]:
        best_score: float = 0
        best_answer: Optional[str] = None
        for pair in self.qa_pairs:
            score = self._similarity(text, pair["question"])
            if score > best_score:
                best_score = score
                best_answer = pair["answer"]
        if best_score >= threshold:
            logger.info("Found match in dataset")
            return best_answer
        return None

    def generate_response(self, text: str) -> str:
        dataset_answer: Optional[str] = self._find_in_dataset(text)
        if dataset_answer:
            return dataset_answer

        logger.info(f"Predicting with model: {self.model_name}")
        inputs: Dict[str, Any] = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        result: str
        with torch.no_grad():
            if "flan-t5" in self.model_name:
                outputs = self.model.generate(**inputs)
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif "bert" in self.model_name:
                logits = self.model(**inputs).logits.argmax(-1).item()
                result = "Technical" if logits == 1 else "Non-technical"
            else:  # distilgpt2
                outputs = self.model.generate(**inputs, max_length=50)
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info(f"Model response: {result}")
        return result
