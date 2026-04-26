from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def get_client() -> Groq:
    return Groq(api_key=os.getenv("GROQ_API_KEY"))