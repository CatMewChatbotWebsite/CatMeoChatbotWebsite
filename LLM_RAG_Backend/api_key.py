from dotenv import load_dotenv
import os

def load_together_apikey():
    load_dotenv(".env")  
    api_key = os.getenv("TOGETHER_API_KEY")
    return api_key

def load_openrouter_apikey():
    load_dotenv(".env")  
    api_key = os.getenv("OPENROUTER_API_KEY")
    return api_key

def load_huggingface_token():
    load_dotenv(".env")  
    token = os.getenv("HF_TOKEN")

