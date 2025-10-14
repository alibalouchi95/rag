from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from database import db_name, connection_args
from langchain_milvus import Milvus
from langchain_core.documents import Document

load_dotenv()
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=model_name, huggingfacehub_api_token=hugging_face_api_key
)

file_path = "./pdfs/first_pdf.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
separators = ["\n\n", "\n", ".", "?", "!", ";", ""]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    separators=separators,
    length_function=len,
    is_separator_regex=False,
)

vector_store = Milvus(
    embedding_function=hf_embeddings,
    connection_args=connection_args,
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    consistency_level="Strong",
    drop_old=False,
)

for doc in docs:
    chunks = text_splitter.split_text(doc.page_content)
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store.add_documents(documents)
