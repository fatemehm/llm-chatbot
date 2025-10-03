from app.model import ChatbotModel

chatbot = ChatbotModel()
query = "How do I fix a blue screen error?"
response = chatbot.generate_response(query)
print(f"Query: {query}")
print(f"Response: {response}")
