from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_core.documents import Document
from extract_keywords import extract_keywords
from database import vector_store
import time
from utilities import flush_db

# keyword extraction model
KEYWORD_MODEL_NAME = "phi3:3.8b"

# Folder containing all PDFs
pdf_folder = "./pdfs"

# List all PDF files in the folder
pdf_files = [
    os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")
]

# Create the text splitter function
separators = ["\n\n", "\n", ".", "?", "!", ";", ""]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=separators,
    length_function=len,
    is_separator_regex=False,
)

res = []


def store_chunk(store, chunk, doc):
    MAX_RETRIES = 3

    keywords = extract_keywords(chunk, KEYWORD_MODEL_NAME)

    if len(keywords) > 60000:
        keywords = keywords[:60000]

    document = [
        Document(
            page_content=chunk,
            metadata={
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", None),
                "pdf_name": os.path.basename(doc.metadata.get("source", "")),
                "keywords": keywords,
            },
        )
    ]

    for attempt in range(MAX_RETRIES):
        try:
            store.add_documents(document)
            print(f"Stored chunk {len(res)}")
            res.append(document)
            print(document)
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"Failed to store chunk after {MAX_RETRIES} attempts: {e}")
                raise


def store_document(store, doc):
    chunks = text_splitter.split_text(doc.page_content)
    for chunk in chunks:
        store_chunk(store, chunk, doc)


# Store the embeded splited texts in the vector store
for file_path in pdf_files:
    loader = PyPDFLoader(file_path)
    print(file_path)
    docs = loader.load()
    for doc in docs:
        store_document(vector_store, doc)

flush_db()
