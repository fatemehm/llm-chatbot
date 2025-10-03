# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from .model import ChatbotModel

app = FastAPI()
chatbot = ChatbotModel()


class Query(BaseModel):
    text: str


@app.post("/chat")
def chat(query: Query):
    response = chatbot.generate_response(query.text)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
