from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pdf_extract import Pdf_loading, text_chunk_split
from vectorDB import get_model_embedding, get_vectorstore, prompting
import torch
import chromadb
from llama_vision import call_api_llm
import json



app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="backend/templates")

Chromadb_Path = 'Chromadb'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_embeddings = get_model_embedding()
collection = chromadb.PersistentClient(path=Chromadb_Path).get_collection(name="my_collection")

@app.post("/post")
async def ask(query: str = Form(...)):
    try:
        print("⚡ Nhận query:", query, flush=True)
        query_embedding = model_embeddings.encode(query, convert_to_tensor=False)
        results = collection.query(query_texts=[query], n_results=3)
        docs = results["documents"][0]
        prompt = prompting(docs, query)
        answer = call_api_llm(prompt)
        print("⚡ Nhận answer:", answer, flush=True)
    except Exception as e:
        print("❌ Lỗi trong xử lý POST:", e, flush=True)
        answer = f"Lỗi: {e}"

    return JSONResponse(
        {"query": query,
          "answer": answer})
    

