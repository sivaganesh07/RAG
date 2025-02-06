# from chromadb import Client

# # Create a ChromaDB client
# client = Client() 

# # Get or create a collection (replace "your_collection_name" with your actual collection name)
# collection = client.get_or_create_collection("simple-rag") 

# # Get the number of documents in the collection
# num_documents = len(collection.get(include=["documents"])) 

# print(f"Number of documents in the collection: {num_documents}/n{collection.get(include=["documents"])}") 

from langchain_chroma import Chroma

import chromadb

# Initialize the Chroma client
client = chromadb.Client()

# Create or retrieve a collection
# collection = client.get_or_create_collection("Invoice-rag")


# # Query the collection
# results = collection.query(
#     query_texts=["what is this document about"],
#     n_results=5
# )

# print(results)

client.delete_collection(name="Invoice-rag")

