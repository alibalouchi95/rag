from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from database import connection_args
from langchain_milvus import Milvus
from langchain_core.documents import Document
from extract_keywords import extract_keywords

# Get the Hugging Face API Key from the ".env" file in the root of the project
load_dotenv()
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")

# Create the embedding function with free hugging face model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=model_name, huggingfacehub_api_token=hugging_face_api_key
)

# Folder containing all PDFs
pdf_folder = "./pdfs"

# List all PDF files in the folder
pdf_files = [
    os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")
]

# Load all documents
docs = []
for file_path in pdf_files:
    loader = PyPDFLoader(file_path)
    docs.extend(loader.load())  # Add each file’s docs into one list

# Create the text splitter function
separators = ["\n\n", "\n", ".", "?", "!", ";", ""]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=200,
    separators=separators,
    length_function=len,
    is_separator_regex=False,
)

# Create the local vector store
vector_store = Milvus(
    embedding_function=hf_embeddings,
    connection_args=connection_args,
    index_params={
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024},
    },
    consistency_level="Strong",
    drop_old=False,
    auto_id=True,
)

# Store the embeded splited texts in the vector store
i = 0
for doc in docs:
    chunks = text_splitter.split_text(doc.page_content)
    for chunk in chunks:
        keywords = extract_keywords(chunk, hugging_face_api_key)
        document = [
            Document(
                page_content=chunk,
                metadata={
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", None),
                    "pdf_name": os.path.basename(doc.metadata.get("source", "")),
                    "keywords": ", ".join(keywords),
                },
            )
        ]
        vector_store.add_documents(document)
        if i % 10 == 0:
            print(f"Number of added data: {i}")
        i += 1
