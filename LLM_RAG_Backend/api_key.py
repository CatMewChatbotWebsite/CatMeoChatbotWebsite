from dotenv import load_dotenv
import os

def load_apikey():
    load_dotenv(".env")  
    api_key = os.getenv("TOGETHER_API_KEY")
    return api_key