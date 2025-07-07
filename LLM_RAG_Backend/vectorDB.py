from sentence_transformers import SentenceTransformer
from pdf_extract import Pdf_loading, text_chunk_split
import torch
import chromadb
from pprint import pprint
from deepseek_v3 import call_api_llm
from huggingface_hub import InferenceClient
import os
from api_key import load_huggingface_token
from pinecone import Pinecone
from api_key import load_pinecone_apikey

Path = 'dataPDF.pdf'
#Chromadb_Path = 'Chromadb'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TRANSFORMERS_CACHE"] = "/tmp/empty_hf_cache"  # chống dùng cache cũ
print(os.environ['TRANSFORMERS_CACHE'])
chunk_text = []
vectors = []
index = []

token = load_huggingface_token()

api_key_pinecone = load_pinecone_apikey()
pc = Pinecone(api_key=api_key_pinecone)
index_name = "chatbot"
namespace = "default"

def get_model_embedding():
    """
        Local Embedding
        return: list[embedding]
    """
    model_embeddings = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device='cpu'
)   
    
    return model_embeddings
    
def embedding_MiniLM_api(text):
    """
        API Embedding
        return: list[embedding]
    """
    client = InferenceClient(
        provider="hf-inference",
        token=token,
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    embedding = client.feature_extraction(text)
    return embedding.tolist()


def get_vectorstore(docs):
    #chroma_client = chromadb.PersistentClient(path=Chromadb_Path)
    # convert docs_split to vectors:
    for idx, doc in enumerate(docs):
        chunk_text.append(doc.page_content)
        index.append(str(idx))
        vectors.append(embedding_MiniLM_api(doc.page_content))
    
    #collection = chroma_client.get_or_create_collection(name="my_collection")
    # collection.add(
    #     documents = chunk_text,
    #     embeddings = vectors,
    #     ids = index
    # )
    return vectors, index, chunk_text

def get_index_pinecone(index_name):
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": "us-east-1"
                }
            }
            
        )
    return pc.Index(index_name)

def get_vectorstore_pinecone(index_name, ids, embedding, chunk_text):
    index = get_index_pinecone(index_name)
    data = [ {
            "id": str(ids),
            "values": embedding[0],
            "metadata": {"text": chunk_text}
        }]
    index.upsert(
        namespace=namespace,
        vectors=data
    )
    return index


def prompting(docs, user_input):
    context = "\n\n".join([doc for doc in docs])
    prompt = f"""
        Dựa vào thông tin sau, hãy trả lời ngắn gọn và tự nhiên:
        {context}
        Câu hỏi: {user_input}
        Trả lời: 
    """
    return prompt


def documents(Path):
    load_pdf = Pdf_loading(Path)
    docs_split = text_chunk_split()
    docs_spliting = docs_split.split_documents(Pdf_loading(Path))
    #model_embeddings = get_model_embedding()
    return docs_spliting

def indexing():
    docs = documents(Path)
    vectors, ids, chunk_text = get_vectorstore(docs)
    index = get_vectorstore_pinecone(index_name, ids, vectors, chunk_text)
    return index
    # while True:
    #     user_input = input("nhập câu hỏi: ")
    #     query_embedding = embedding_MiniLM_api(user_input)
    #     result = index.query(
    #                           vector=[query_embedding],
    #                           top_k=3,
    #                           namespace=namespace,
    #                           include_metadata=True 
    #     )
    #     docs = result['matches'][0]['metadata']['text'][0]
    # #     results = collection.query(
    # #         query_texts=[str(query_embedding)], # Chroma will embed this for you or embedded then push it to query_text
    # #         n_results=3 # how many results to return
    # # )
    # #     docs = results['documents'][0] # list các string context top-k
    # #     print(results)
    #     prompt = prompting(docs, user_input)
    #     if user_input != "break":
    #         print(call_api_llm(prompt))
    #     else:
    #         break
   

    

    
