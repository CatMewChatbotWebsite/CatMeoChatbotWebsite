from sentence_transformers import SentenceTransformer
from pdf_extract import Pdf_loading, text_chunk_split
from pprint import pprint
from deepseek_v3 import call_api_llm
from huggingface_hub import InferenceClient
from pinecone import Pinecone




class RAG:
    def __init__(self, Path, index_name, namespace, HF_token, api_key_pinecone, device):
        self.Path_PDF = Path
        self.index_name = index_name
        self.namespace = namespace
        self.provider_HF = "hf-inference"
        self.HF_token = HF_token
        self.pc = Pinecone(api_key=api_key_pinecone)
        self.model_embedding = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.device = device
        self.chunk_text = []
        self.vectors = []
        self.index = []

    def get_model_embedding_local(self):
        """
            Local Embedding
            return: list[embedding]
        """
        model_embeddings = SentenceTransformer(
        self.model_embedding,
        self.device
    )   
        
        return model_embeddings
        
    def embedding_MiniLM_api(self, text):
        """
            API Embedding
            return: list[embedding]
        """
        client = InferenceClient(
            provider=self.provider_HF,
            token=self.HF_token,
            model=self.model_embedding
        )

        embedding = client.feature_extraction(text)
        return embedding.tolist()


    def get_vectorstore(self, docs):
        # convert docs_split to vectors:
        for idx, doc in enumerate(docs):
            self.chunk_text.append(doc.page_content)
            self.index.append(str(idx))
            self.vectors.append(self.embedding_MiniLM_api(doc.page_content))
        
        return self.vectors, self.index, self.chunk_text

    def get_index_pinecone(self):
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric='cosine',
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
                
            )
        return self.pc.Index(self.index_name)

    def get_vectorstore_pinecone(self, ids, embedding, chunk_text):
        index = self.get_index_pinecone()
        data = [ {
                "id": str(ids),
                "values": embedding[0],
                "metadata": {"text": chunk_text}
            }]
        index.upsert(
            namespace=self.namespace,
            vectors=data
        )
        return index


    def prompting(self, docs, user_input):
        context = "\n\n".join([doc for doc in docs])
        prompt = f"""
            Dựa vào thông tin sau, hãy trả lời ngắn gọn và tự nhiên:
            {context}
            Câu hỏi: {user_input}
            Trả lời: 
        """
        return prompt


    def documents(self, Path):
        load_pdf = Pdf_loading(Path)
        docs_split = text_chunk_split()
        docs_spliting = docs_split.split_documents(Pdf_loading(Path))
        return docs_spliting

    def indexing(self):
        docs = self.documents(self.Path_PDF)
        vectors, ids, chunk_text = self.get_vectorstore(docs)
        index = self.get_vectorstore_pinecone(ids, vectors, chunk_text)
        return index
        
   

    

    
