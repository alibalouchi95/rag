# # Count entities
# count = qdrant_client_raw.count(COLLECTION_NAME).count
# print(f"Entities: {count}")

# # --- Embed query ---
# query = "what is SDN?"
# embedded_query = hf_embeddings.embed_query(query)
# print(f"Embedding length: {len(embedded_query)}")

# # --- Run search using LangChain wrapper ---
# print("\nRunning similarity search...")
