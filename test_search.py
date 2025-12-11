from database import hf_embeddings, vector_store, COLLECTION_NAME, qdrant_client_raw

# Count entities
count = qdrant_client_raw.count(COLLECTION_NAME).count
print(f"Entities: {count}")

# --- Embed query ---
query = "what is SDN?"
embedded_query = hf_embeddings.embed_query(query)
print(f"Embedding length: {len(embedded_query)}")

# --- Run search using LangChain wrapper ---
print("\nRunning similarity search...")

# Updated call:
results = vector_store.similarity_search(query=query, k=1)

# Print results
print("\nSearch results:")
for i, r in enumerate(results):
    print(f"--- Result {i+1} ---")
    print("Text:", r.page_content)
    print("Metadata:", r.metadata)
