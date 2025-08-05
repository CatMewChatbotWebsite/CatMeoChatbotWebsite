from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Path = 'dataPDF.pdf'
def Pdf_loading(pdf_file):
    """
    Load and split chunk docs each page
    """
    loader = PyPDFLoader(pdf_file)
    documents = loader.load_and_split()
    return documents

def text_chunk_split():
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=400,
    chunk_overlap=70,
    length_function=len,
    is_separator_regex=False,
    separators = ["\n•", "\n-", "\n|", "\n"]
)
    return text_splitter

#docs_split = text_chunk_split()
#print(docs_split.split_documents(Pdf_loading(Path)))

