from pymilvus import Collection, connections
from database import db_name
from dotenv import load_dotenv
import os

load_dotenv()
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")
milvus_username = os.getenv("MILVUS_USER_NAME")
milvus_password = os.getenv("MILVUS_PASSWORD")
milvus_port = os.getenv("MILVUS_PORT")
milvus_host = os.getenv("MILVUS_HOST")

COLLECTION_NAME = "LangChainCollection"


def flush_db(collection_name=COLLECTION_NAME):
    connections.connect(
        host=milvus_host,
        port=milvus_port,
        user=milvus_username,
        password=milvus_password,
        db_name=db_name,
    )
    collection = Collection(collection_name)
    collection.flush()
    print(
        f"✓ The database ${db_name} collection ${COLLECTION_NAME} was flushed successfully"
    )


def get_collection_details(collection_name, limit=5):
    connections.connect(
        host=milvus_host,
        port=milvus_port,
        user=milvus_username,
        password=milvus_password,
        db_name=db_name,
    )
    collection = Collection(collection_name)
    collection.load()
    return {
        "schema": collection.schema,
        "name": collection.name,
        "number": collection.num_entities,
        "limits": collection.query(
            expr="",
            output_fields=["text", "keywords"],
            limit=limit,
        ),
    }
