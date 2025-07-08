import torch
import os
from api_key import load_huggingface_token, load_pinecone_apikey


def initial():
    Path = 'dataPDF.pdf'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/empty_hf_cache"  # chống dùng cache cũ
    print(os.environ['TRANSFORMERS_CACHE'])
    HF_token = load_huggingface_token()
    pincone_api = load_pinecone_apikey()
    index_name = "chatbot"
    namespace = "default"
    return Path, device, HF_token, pincone_api, index_name, namespace