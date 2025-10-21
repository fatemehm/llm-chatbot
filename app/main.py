import os
from typing import Dict

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

from app.model import ChatbotModel

app = FastAPI()
model_cache: Dict[str, ChatbotModel] = {}

# Prometheus metrics
REQUEST_COUNT = Counter("llm_chatbot_requests_total", "Total API requests")
REQUEST_LATENCY = Histogram("llm_chatbot_request_latency_seconds", "Request latency")
app.mount("/metrics", make_asgi_app())


class ChatRequest(BaseModel):
    message: str
    model_name: str = "google/flan-t5-small"


def get_model(model_name: str) -> ChatbotModel:
    if model_name not in model_cache:
        os.environ["MODEL_NAME"] = model_name
        model_cache[model_name] = ChatbotModel()
    return model_cache[model_name]


@app.get("/")
async def root():
    return {
        "message": "🚀 LLM Chatbot API is running!",
        "docs": "Visit /docs for the interactive Swagger UI",
        "health": "Visit /health for health check",
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    with REQUEST_LATENCY.time():
        REQUEST_COUNT.inc()
        if not req.message.strip():
            raise HTTPException(400, "Message cannot be empty")
        try:
            model = get_model(req.model_name)
            return {
                "response": model.generate_response(req.message),
                "model_used": req.model_name,
            }
        except Exception as e:
            raise HTTPException(500, f"Error: {str(e)}")
