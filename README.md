# RAG‑Starter / rag

A starter project for building **Retrieval‑Augmented Generation (RAG)** applications using **LangChain** + **LangGraph**.

---

## 🚀 Overview

This project is designed as a learning and scaffold tool to help you build RAG systems. It provides the building blocks for:

- Document ingestion (local files, web pages, PDFs, etc.)
- Text splitting / chunking
- Embedding & vector store abstraction
- Retrieval engines (vector search, hybrid)
- LLM wrapper / generation module
- Orchestration via LangGraph (coordinate retrieval, generation, memory)
- Example pipelines and demos

The goal is to be modular, extensible, and educational — you can adapt each component as you grow your application.

---

## 📦 Structure

A sample (planned) structure:

```
.
├── README.md
├── pyproject.toml / setup.py
├── src/
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── retriever.py
│   │   ├── embedder.py
│   │   ├── model.py
│   │   ├── orchestrator.py
│   │   └── memory.py
├── examples/
│   └── demo_notebook.ipynb
├── tests/
│   ├── test_loader.py
│   ├── test_retriever.py
│   └── ...
└── docs/
    └── (documentation site / tutorials)
```

---

## 📖 Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Ingest your documents:**

   ```python
   from rag.loader import DocumentLoader
   docs = DocumentLoader.from_directory("data/")
   ```

3. **Build embeddings & index:**

   ```python
   from rag.embedder import EmbeddingModel
   from rag.retriever import VectorRetriever

   embed = EmbeddingModel(model="openai-text-embedding")
   retriever = VectorRetriever(embedding_model=embed)
   retriever.add_documents(docs)
   ```

4. **Run a query:**

   ```python
   from rag.orchestrator import RAGOrchestrator

   orchestrator = RAGOrchestrator(retriever=retriever, llm_model="gpt‑3.5‑turbo")
   answer = orchestrator.run("What is the meaning of life?")
   print(answer)
   ```

5. **(Optional) Use in conversational / memory mode:**

   ```python
   from rag.orchestrator import StatefulRAGOrchestrator
   conv = StatefulRAGOrchestrator(...)
   conv.add_user_input("Tell me about RAG.")
   conv.add_user_input("And how LangGraph helps?")
   ```

---

## 🛠️ Features & Roadmap

- ✅ Document loading (local / web)
- ✅ Text splitting / chunking
- ✅ Embedding interface (OpenAI and local)
- ✅ Vector retrieval abstraction
- ✅ LangGraph orchestration pipeline
- 🔄 Memory / stateful RAG (planned)
- 🔄 Support for multiple vector stores (Chroma, FAISS, Pinecone)
- 🔄 Tool integration / fallback logic
- 🔄 Graph-based retrieval & advanced RAG strategies (PathRAG, LeanRAG, adaptive routing)

---

## 📂 Examples & Demos

Check out the `examples/` folder for Jupyter notebooks showing example usage:

- Q&A over local documents
- Conversational RAG example
- (Future) Hybrid RAG / fallback example

---

## 🧪 Testing & CI

We include unit tests in `tests/` for each core module (loader, retriever, orchestrator, etc.).  
CI pipelines will run linting, formatting, and tests.

---

## 📖 Further Reading & References

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Graph RAG (GraphRetriever)](https://python.langchain.com/docs/integrations/retrievers/graph_rag/)
- [Adaptive RAG with LangGraph](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/)
- [PathRAG / LeanRAG (recent research)](https://arxiv.org/abs/2502.14902)

---

## 🧩 Contributing

Contributions are welcome! Please:

1. Fork the repo  
2. Create a feature branch  
3. Run tests & ensure linting passes  
4. Submit a pull request

Add issues, label them, and we’ll triage and assign accordingly.

---

## 📄 License

This project is MIT licensed.
