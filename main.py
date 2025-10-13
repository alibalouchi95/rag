from langchain_community.document_loaders import PyPDFLoader

file_path = "./pdfs/first_pdf.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

for doc in docs:
    print("DOC", doc.metadata)
