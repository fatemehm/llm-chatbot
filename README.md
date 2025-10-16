# 🤖 LLM Tech Support Chatbot

[![Tests](https://github.com/yourusername/llm-chatbot/workflows/CI/badge.svg)](https://github.com/yourusername/llm-chatbot/actions)
[![Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen)](htmlcov/index.html)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready technical support chatbot powered by fine-tuned LLMs with LoRA, featuring multiple model support, MLflow tracking, and comprehensive monitoring.

## 🌟 Features

- **Multiple Model Support**: FLAN-T5, BERT, and DistilGPT2 with LoRA fine-tuning
- **Dual Interface**: FastAPI REST API + Streamlit web UI
- **MLflow Integration**: Experiment tracking and model versioning
- **Prometheus Metrics**: Real-time monitoring and observability
- **DVC**: Data version control
- **Docker Support**: Full containerized deployment
- **Fuzzy Matching**: Intelligent fallback to dataset search
- **82% Test Coverage**: Comprehensive test suite

## 📋 Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- 4GB+ RAM
- GPU recommended but not required

## 🚀 Quick Start

### 1. Installation

```bash
git clone <your-repo-url>
cd llm-chatbot

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

make install
```

### 2. Train Models

```bash
make train
```

Training time (CPU):
- FLAN-T5: ~10-15 minutes
- BERT: ~8-12 minutes
- DistilGPT2: ~10-15 minutes

### 3. Run Application

**Local Development:**
```bash
# Terminal 1: Start API
make run-api

# Terminal 2: Start UI
make run-streamlit

# Terminal 3: MLflow UI (optional)
make mlflow-ui
```

**With Docker:**
```bash
make docker-up
make docker-logs  # View logs
```

## 🌐 Access Points

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage report
make test-coverage

# Run only unit tests
make test-unit

# Run integration tests
make test-integration

# View coverage report
open htmlcov/index.html
```

**Current Coverage: 82%**

## 🔍 Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Run all quality checks
black app/ tests/ train.py
isort app/ tests/ train.py
flake8 app/ tests/ train.py
mypy app/
```

## 📊 Project Structure

```
llm-chatbot/
├── app/
│   ├── main.py              # FastAPI application
│   ├── model.py             # Model inference logic
│   └── streamlit_app.py     # Streamlit UI
├── config/
│   ├── app_config.yaml      # App configuration
│   ├── model_config.yaml    # Model parameters
│   └── training_config.yaml # Training settings
├── data/
│   └── tech_support_qa.json # Training dataset
├── models/                   # Trained models
├── tests/                    # Test suite
│   ├── test_main.py         # API tests
│   ├── test_model.py        # Model tests
│   └── test_integration.py  # E2E tests
├── train.py                 # Training script
├── Dockerfile               # API container
├── Dockerfile.streamlit     # UI container
├── docker-compose.yml       # Multi-container setup
├── Makefile                 # Development commands
└── README.md
```

## 🔧 Configuration

### Environment Variables

```bash
MODEL_NAME=google/flan-t5-small  # or bert-base-uncased, distilgpt2
API_URL=http://localhost:8000
```

### Model Selection

Choose from three models:
- **FLAN-T5**: Best for Q&A tasks (default)
- **BERT**: Classification tasks
- **DistilGPT2**: Conversational responses

## 📝 API Usage

### cURL Example

```bash
# Simple request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How to fix blue screen error?"}'

# With model selection
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Why is my computer slow?",
    "model_name": "google/flan-t5-small"
  }'
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "How to fix error?",
        "model_name": "google/flan-t5-small"
    }
)
print(response.json())
```

## 🐳 Docker Deployment

```bash
# Build images
make docker-build

# Start all services
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down

# Restart services
make docker-restart
```

Services:
- `fastapi`: API backend (port 8000)
- `streamlit`: Web UI (port 8501)
- `mlflow`: Experiment tracking (port 5000)
- `prometheus`: Metrics (port 9090)

## 📈 Monitoring

### Prometheus Metrics

- `llm_chatbot_requests_total`: Total API requests
- `llm_chatbot_request_latency_seconds`: Request latency

Access at: http://localhost:9090

### MLflow Tracking

Track experiments, compare models, and manage versions at: http://localhost:5000

## 🛠️ Development

### Setup Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Adding New Tests

```bash
# Add test file
tests/test_new_feature.py

# Run specific test
pytest tests/test_new_feature.py -v

# Update coverage
make test-coverage
```

### Training New Models

1. Add data to `data/tech_support_qa.json`
2. Update `config/training_config.yaml`
3. Run: `python train.py`
4. Check MLflow for results

## 🚀 Production Deployment

### Best Practices

1. **Use environment variables** for configuration
2. **Enable authentication** for production APIs
3. **Set up rate limiting** to prevent abuse
4. **Configure logging** with proper log levels
5. **Monitor metrics** with Prometheus + Grafana
6. **Use GPU instances** for faster inference
7. **Implement caching** for common queries

### Scaling

- Use Kubernetes for orchestration
- Add load balancer (nginx/traefik)
- Implement Redis for caching
- Use message queue for async processing

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file

## 👥 Authors

- Mahshid

## 🙏 Acknowledgments

- HuggingFace Transformers
- FastAPI
- Streamlit
- MLflow
