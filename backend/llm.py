import os
#import google.genai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

#genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.3
)

# Use Gemini 1.5 Flash — fast and cost effective
#model = genai.GenerativeModel("gemini-2.5-flash")

def get_gemini_answer(prompt: str) -> str:
    """Sends the structured prompt to Gemini and returns the answer."""
    response = llm.invoke(prompt)
    return response.text