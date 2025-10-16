import json
import os
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
import torch

from app.model import ChatbotModel


@pytest.fixture
def mock_qa_data():
    """Mock Q&A dataset"""
    return [
        {
            "question": "How to fix blue screen error?",
            "answer": "Restart your computer and check drivers.",
        },
        {
            "question": "Why is my computer slow?",
            "answer": "Check CPU usage and close unnecessary programs.",
        },
    ]


@pytest.fixture
def setup_model_mocks():
    """Setup all mocks needed for model initialization"""
    with (
        patch("app.model.AutoTokenizer") as mock_tokenizer,
        patch("app.model.AutoModelForSeq2SeqLM") as mock_seq2seq,
        patch("app.model.AutoModelForSequenceClassification") as mock_bert,
        patch("app.model.AutoModelForCausalLM") as mock_causal,
        patch("app.model.PeftModel") as mock_peft,
        patch("os.path.exists") as mock_exists,
    ):

        # Setup tokenizer mock
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        tokenizer.decode.return_value = "Mocked response"
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "</s>"
        mock_tokenizer.from_pretrained.return_value = tokenizer

        # Setup model mocks
        model = MagicMock()
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        logits_mock = MagicMock()
        logits_mock.logits.argmax.return_value.item.return_value = 1
        model.return_value = logits_mock

        mock_seq2seq.from_pretrained.return_value = model
        mock_bert.from_pretrained.return_value = model
        mock_causal.from_pretrained.return_value = model
        mock_peft.from_pretrained.return_value = model

        # Dataset and model path exist
        mock_exists.return_value = True

        yield {"tokenizer": tokenizer, "model": model, "mock_exists": mock_exists}


def test_similarity_calculation():
    """Test similarity score calculation"""
    model = Mock(spec=ChatbotModel)
    model._similarity = ChatbotModel._similarity.__get__(model)

    assert model._similarity("test", "test") == 100.0

    score = model._similarity("hello world", "hello world!")
    assert 90 < score < 100

    score = model._similarity("cat", "dog")
    assert score < 50


def test_find_in_dataset_exact_match(mock_qa_data):
    """Test finding exact match in dataset"""
    model = Mock(spec=ChatbotModel)
    model.qa_pairs = mock_qa_data
    model._similarity = ChatbotModel._similarity.__get__(model)
    model._find_in_dataset = ChatbotModel._find_in_dataset.__get__(model)

    result = model._find_in_dataset("How to fix blue screen error?")
    assert result == "Restart your computer and check drivers."


def test_find_in_dataset_fuzzy_match(mock_qa_data):
    """Test fuzzy matching in dataset"""
    model = Mock(spec=ChatbotModel)
    model.qa_pairs = mock_qa_data
    model._similarity = ChatbotModel._similarity.__get__(model)
    model._find_in_dataset = ChatbotModel._find_in_dataset.__get__(model)

    result = model._find_in_dataset("How do I fix blue screen?")
    assert result is not None


def test_find_in_dataset_no_match(mock_qa_data):
    """Test no match in dataset"""
    model = Mock(spec=ChatbotModel)
    model.qa_pairs = mock_qa_data
    model._similarity = ChatbotModel._similarity.__get__(model)
    model._find_in_dataset = ChatbotModel._find_in_dataset.__get__(model)

    result = model._find_in_dataset("completely unrelated question")
    assert result is None


def test_find_in_dataset_custom_threshold(mock_qa_data):
    """Test custom threshold for matching"""
    model = Mock(spec=ChatbotModel)
    model.qa_pairs = mock_qa_data
    model._similarity = ChatbotModel._similarity.__get__(model)
    model._find_in_dataset = ChatbotModel._find_in_dataset.__get__(model)

    result = model._find_in_dataset("fix blue screen", threshold=95)
    assert result is None


def test_model_init_flan_t5(setup_model_mocks, mock_qa_data):
    """Test FLAN-T5 model initialization"""
    with (
        patch("builtins.open", mock_open(read_data=json.dumps(mock_qa_data))),
        patch.dict(os.environ, {"MODEL_NAME": "google/flan-t5-small"}),
    ):

        model = ChatbotModel()
        assert model.model_name == "google/flan-t5-small"
        assert len(model.qa_pairs) == 2


def test_model_init_bert(setup_model_mocks, mock_qa_data):
    """Test BERT model initialization"""
    with (
        patch("builtins.open", mock_open(read_data=json.dumps(mock_qa_data))),
        patch.dict(os.environ, {"MODEL_NAME": "bert-base-uncased"}),
    ):

        model = ChatbotModel()
        assert model.model_name == "bert-base-uncased"


def test_model_init_gpt2(setup_model_mocks, mock_qa_data):
    """Test DistilGPT2 model initialization"""
    with (
        patch("builtins.open", mock_open(read_data=json.dumps(mock_qa_data))),
        patch.dict(os.environ, {"MODEL_NAME": "distilgpt2"}),
    ):

        model = ChatbotModel()
        assert model.model_name == "distilgpt2"


def test_model_init_missing_dataset(setup_model_mocks):
    """Test model init with missing dataset"""
    setup_model_mocks["mock_exists"].side_effect = lambda x: False if "tech_support" in x else True

    with patch.dict(os.environ, {"MODEL_NAME": "google/flan-t5-small"}):
        model = ChatbotModel()
        assert model.qa_pairs == []


def test_model_init_missing_model_path(setup_model_mocks):
    """Test model init with missing model path"""
    setup_model_mocks["mock_exists"].return_value = False

    with patch.dict(os.environ, {"MODEL_NAME": "google/flan-t5-small"}):
        with pytest.raises(FileNotFoundError):
            ChatbotModel()


def test_generate_response_dataset_match(setup_model_mocks, mock_qa_data):
    """Test generate response uses dataset first"""
    with (
        patch("builtins.open", mock_open(read_data=json.dumps(mock_qa_data))),
        patch.dict(os.environ, {"MODEL_NAME": "google/flan-t5-small"}),
    ):

        model = ChatbotModel()
        response = model.generate_response("How to fix blue screen error?")
        assert response == "Restart your computer and check drivers."


def test_generate_response_flan_t5_inference(setup_model_mocks, mock_qa_data):
    """Test FLAN-T5 model inference"""
    with (
        patch("builtins.open", mock_open(read_data=json.dumps(mock_qa_data))),
        patch.dict(os.environ, {"MODEL_NAME": "google/flan-t5-small"}),
    ):

        model = ChatbotModel()
        response = model.generate_response("unknown question")
        assert isinstance(response, str)
        assert len(response) > 0


def test_generate_response_bert_inference(setup_model_mocks, mock_qa_data):
    """Test BERT model inference"""
    with (
        patch("builtins.open", mock_open(read_data=json.dumps(mock_qa_data))),
        patch.dict(os.environ, {"MODEL_NAME": "bert-base-uncased"}),
    ):

        model = ChatbotModel()
        response = model.generate_response("unknown question")
        assert response in ["Technical", "Non-technical"]


def test_generate_response_gpt2_inference(setup_model_mocks, mock_qa_data):
    """Test GPT2 model inference"""
    with (
        patch("builtins.open", mock_open(read_data=json.dumps(mock_qa_data))),
        patch.dict(os.environ, {"MODEL_NAME": "distilgpt2"}),
    ):

        model = ChatbotModel()
        response = model.generate_response("unknown question")
        assert isinstance(response, str)
