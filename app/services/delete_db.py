# # delete_and_recreate.py
# from qdrant_client import QdrantClient
# import os
# from dotenv import load_dotenv

# load_dotenv()

# q_host = os.getenv("QDRANT_HOST", "localhost")
# q_port = int(os.getenv("QDRANT_PORT", 6333))
# COLLECTION_NAME = "LangChainCollection"
# FARSI_COLLECTION_NAME = "FarsiEmbeddedCollection"

# # Create client
# qdrant_client = QdrantClient(url=f"http://{q_host}:{q_port}")

# # Delete collection
# try:
#     collections = [c.name for c in qdrant_client.get_collections().collections]
#     if FARSI_COLLECTION_NAME in collections:
#         qdrant_client.delete_collection(collection_name=FARSI_COLLECTION_NAME)
#         print(f"✅ Collection '{FARSI_COLLECTION_NAME}' deleted successfully!")
#     else:
#         print(f"ℹ️  Collection '{FARSI_COLLECTION_NAME}' does not exist.")
# except Exception as e:
#     print(f"❌ Error deleting collection: {e}")
