# 📘 RAG – Local Document Question Answering with LangChain & LangGraph

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docs.docker.com/compose/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

> A local Retrieval-Augmented Generation (RAG) application built with **LangChain** and **LangGraph**, allowing you to upload PDFs, process them, and query their content using natural language.

---

## 🧭 Table of Contents

- [📘 RAG – Local Document Question Answering with LangChain \& LangGraph](#-rag--local-document-question-answering-with-langchain--langgraph)
  - [🧭 Table of Contents](#-table-of-contents)
  - [📖 About the Project](#-about-the-project)
  - [🌿 Branch Information](#-branch-information)
  - [✨ Features](#-features)
  - [🧰 Prerequisites](#-prerequisites)
  - [⚙️ Installation](#️-installation)
  - [🚀 Running the Project](#-running-the-project)
  - [🗂 Project Structure](#-project-structure)
  - [🧭 Usage Guide](#-usage-guide)
  - [🧩 Troubleshooting](#-troubleshooting)
  - [🛠 Roadmap](#-roadmap)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)

---

## 📖 About the Project

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline using:
- **LangChain** for document processing and conversational retrieval
- **LangGraph** for workflow orchestration
- **Docker Compose** for managing local services (e.g., vector databases)

It enables users to:
- Upload PDFs to a local folder
- Automatically index and embed the content
- Ask natural-language questions about the documents

---

## 🌿 Branch Information

- **Main Branch** → Local version (uses APIs and Dockerized services)
- **Without-API Branch** → Minimal version that does not depend on external APIs or online endpoints

> 💡 To switch to the no-API version, use:
> ```bash
> git checkout without-api
> ```

---

## ✨ Features

✅ Local document ingestion (PDFs)  
✅ RAG pipeline using LangChain + LangGraph  
✅ Dockerized environment for easy setup  
✅ Vector store integration  
✅ Command-line or programmatic interaction  

---

## 🧰 Prerequisites

Before starting, ensure you have installed:

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [Python 3.9+](https://www.python.org/downloads/)

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alibalouchi95/rag.git
   cd rag
   ```

2. **Add your PDF files:**
   Place all your documents in the `pdfs/` folder.

3. **Start the Docker environment:**
   ```bash
   docker-compose up -d
   ```

4. **Install Python dependencies (if needed locally):**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Project

1. Make sure Docker services are running (`docker-compose up -d`).
2. Verify that your PDFs are placed under the `/pdfs` folder.
3. Run the main script (depending on your implementation):
   ```bash
   python main.py
   ```
4. The system will:
   - Read PDFs from `pdfs/`
   - Split and embed content
   - Store embeddings in a vector database
   - Start a retrieval-based QA system (Not Implemented yet)

---

## 🗂 Project Structure

```
rag/
├── pdfs/                # Place your PDF files here
├── docker-compose.yml   # Defines services and dependencies
├── main.py              # Entry point for running the app
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── ...
```

---

## 🧭 Usage Guide

1. Place all PDF documents you wish to query inside the `pdfs/` directory.  
2. Start the containers using:
   ```bash
   docker-compose up -d
   ```
3. Run the main script or notebook to process and query the files.  
4. Ask questions in natural language and get AI-generated answers from your own documents. (Not Implemented yet)

---

## 🧩 Troubleshooting

| Issue | Possible Solution |
|-------|-------------------|
| `Database 'rag_db' does not exist.` | Ensure Docker containers are running and services initialized. |
| No data indexed | Check that PDFs are actually in `/pdfs` before running the ingestion process. |
| API key errors | Verify environment variables in `.env` file or configuration settings. |

---

## 🛠 Roadmap

- [ ] Add support for multiple document formats (TXT, DOCX, HTML)  
- [ ] Integrate a web UI (Streamlit or Gradio)  
- [ ] Add chat history and streaming responses  
- [ ] Add multi-language support  
- [ ] Add FAISS / Qdrant vector store options  

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to fork this repo and submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

**Author:** [Ali Balouchi](https://github.com/alibalouchi95)  
**Repo:** [github.com/alibalouchi95/rag](https://github.com/alibalouchi95/rag)
