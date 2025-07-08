from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pdf_extract import Pdf_loading, text_chunk_split
from init import initial
from vectorDB import RAG
import torch
from deepseek_v3 import call_api_llm
import json


app = FastAPI()
Path, device, HF_token, pincone_api, index_name, namespace = initial()

Rag_class = RAG(
            Path=Path,
            index_name=index_name,
            namespace=namespace,
            HF_token=HF_token,
            api_key_pinecone=pincone_api,
            device=device    
            )


@app.get("/")
def root():
    return {"message": "CatMeo chatbot is live!"}

@app.post("/post")
async def ask(query: str = Form(...)):
    try:
        print("⚡ Nhận query:", query, flush=True)
        query_embedding = Rag_class.embedding_MiniLM_api(query)
        print("✅ Đã tạo embedding xong", flush=True)
        #results = collection.query(query_texts=[query], n_results=3)
        result = Rag_class.indexing().query(
                            vector=[query_embedding],
                            top_k=3,
                            namespace=namespace,
                            include_metadata=True 
        )
        docs = result['matches'][0]['metadata']['text'][0]
        prompt = Rag_class.prompting(docs, query)
        print("✅ Tạo prompt xong", flush=True)
        answer = call_api_llm(prompt)
        print("🚀 Gọi hàm call_api_llm", flush=True)
        print("⚡ Nhận answer:", answer, flush=True)
    except Exception as e:
        print("❌ Lỗi trong xử lý POST:", e, flush=True)
        answer = f"Lỗi: {e}"

    return JSONResponse(
        {"query": query,
          "answer": answer})

