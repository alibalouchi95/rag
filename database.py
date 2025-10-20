from pymilvus import MilvusException, connections, db, utility, Collection

conn = connections.connect(host="127.0.0.1", port=19530)

db_name = "rag_db"

connection_args = {
    "host": "127.0.0.1",
    "port": "19530",
    "user": "root",
    "password": "Milvus",
    "db_name": db_name,
}

# Create the vector store if it doesn't exists

try:
    existing_databases = db.list_database()
    if db_name in existing_databases:
        print(f"Database '{db_name}' already exists.")

        # <------This section is for test only so in the production this should be deleted------>
        # <------ START ------>
        # # Use the database context
        # db.using_database(db_name)
        # # Drop all collections in the database
        # collections = utility.list_collections()
        # for collection_name in collections:
        #     collection = Collection(name=collection_name)
        #     collection.drop()
        #     print(f"Collection '{collection_name}' has been dropped.")

        # db.drop_database(db_name)
        # print(f"Database '{db_name}' has been deleted.")
        # <------ END ------>

    else:
        print(f"Database '{db_name}' does not exist.")
        database = db.create_database(db_name)
        print(f"Database '{db_name}' created successfully.")
except MilvusException as e:
    print(f"An error occurred: {e}")
