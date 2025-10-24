from pymilvus import MilvusException, connections, db, utility, Collection
from dotenv import load_dotenv
import os
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings

EMBED_MODEL_NAME = "mxbai-embed-large"

load_dotenv()
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")
milvus_username = os.getenv("MILVUS_USER_NAME")
milvus_password = os.getenv("MILVUS_PASSWORD")
milvus_port = os.getenv("MILVUS_PORT")
milvus_host = os.getenv("MILVUS_HOST")

conn = connections.connect(host=milvus_host, port=int(milvus_port))

db_name = "rag_db"

connection_args = {
    "host": milvus_host,
    "port": milvus_port,
    "user": milvus_username,
    "password": milvus_password,
    "db_name": db_name,
}


def delete_db(name):
    existing_databases = db.list_database()
    if db_name in existing_databases:
        # Use the database context
        db.using_database(name)
        # Drop all collections in the database
        collections = utility.list_collections()
        for collection_name in collections:
            collection = Collection(name=collection_name)
            collection.drop()
            print(f"Collection '{collection_name}' has been dropped.")

        db.drop_database(name)
        print(f"Database '{name}' has been deleted.")


# Print the status of the database
try:
    existing_databases = db.list_database()
    if db_name in existing_databases:
        print(f"Database '{db_name}' already exists.")
        # <------This section is for test only so in the production this should be deleted------>
        # <------ START ------>
        delete_db(db_name)
        # <------ END ------>
    else:
        print(f"Database '{db_name}' does not exist.")
        db.create_database(db_name)
        print(f"Database '{db_name}' has been created.")
except MilvusException as e:
    print(f"An error occurred: {e}")

# Create the embedding function with nomic-embed-text model
hf_embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)

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
