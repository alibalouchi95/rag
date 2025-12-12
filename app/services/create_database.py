# create_database.py (Qdrant version)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import time
from langchain_core.documents import Document

# from app.services.extract_keywords import extract_keywords
# from app.services.database import vector_store
# from app.services.translator import translate

from extract_keywords import extract_keywords
from database import vector_store
from translator import local_translate

# Keyword extraction model
KEYWORD_MODEL_NAME = "phi3:3.8b"

# PDF folder
pdf_folder = os.path.abspath("./app/services/pdfs")

# List PDF files
pdf_files = [
    os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")
]

# Text splitter
separators = ["\n\n", "\n", ".", "?", "!", ";", ""]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=400,
    separators=separators,
    length_function=len,
    is_separator_regex=False,
)


def create_chunk_document(chunk, **args):
    # Translate the Farsi text
    english_translation = local_translate(chunk)
    # Extract keywords
    keywords = extract_keywords(english_translation, KEYWORD_MODEL_NAME)

    # Qdrant payload limit safety (64k recommended)
    if len(keywords) > 60000:
        keywords = [""]

    # Prepare LangChain document object
    document = [
        Document(
            page_content=english_translation,
            metadata={
                "source": args.get("source"),
                "page": args.get("page"),
                "pdf_name": os.path.basename(args.get("source")),
                "keywords": keywords,
                "farsi-text": chunk,
            },
        )
    ]

    return document


def store_chunk(store, chunk, doc):
    MAX_RETRIES = 3

    document = create_chunk_document(
        chunk=chunk,
        source=doc.metadata.get("source", ""),
        page=doc.metadata.get("page", None),
    )

    # Retry loop
    for attempt in range(MAX_RETRIES):
        try:
            store.add_documents(document)
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


# --- Run ingestion ---
for file_path in pdf_files:
    loader = PyPDFLoader(file_path)
    print(f"Processing: {file_path}")
    docs = loader.load()
    i = 1
    for doc in docs:
        store_document(vector_store, doc)
        print(f"Chunk #{i}")
        i += 1
