import os
from langchain_google_genai import ChatGoogleGenerativeAI

def load_llm(model_name="gemini-pro"):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
