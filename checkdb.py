# from chromadb import Client

# # Create a ChromaDB client
# client = Client() 

# # Get or create a collection (replace "your_collection_name" with your actual collection name)
# collection = client.get_or_create_collection("simple-rag") 

# # Get the number of documents in the collection
# num_documents = len(collection.get(include=["documents"])) 

# print(f"Number of documents in the collection: {num_documents}/n{collection.get(include=["documents"])}") 

# from langchain_chroma import Chroma

# import chromadb

# # Initialize the Chroma client
# client = chromadb.Client()

# # Create or retrieve a collection
# collection = client.get_or_create_collection("Invoice-rag")

# print(collection)


# # # Query the collection
# # results = collection.query(
# #     query_texts=["what is this document about"],
# #     n_results=5
# # )

# # print(results)

# # client.delete_collection(name="Invoice-rag")

# # Get all documents from the collection
# all_docs = collection.get() 

# print(all_docs)

# # Extract file names (assuming you stored file names in the 'metadatas' field)
# file_names = [doc['metadatas'][0]['file_name'] for doc in all_docs['documents']] 

# # Print the list of file names
# print(file_names)

from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIRECTORY = "./chroma_db_invoice"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "Invoice-rag"
embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Load the existing Chroma vectorstore
vector_db = Chroma(
    embedding_function=embedding,
    collection_name=VECTOR_STORE_NAME,
    persist_directory=PERSIST_DIRECTORY,
)

# # Now you can use the loaded vectorstore:
# query = "What is the total value of invoice 123100402?"
# results = vector_db.similarity_search(query, k=2) 
# print(results)
# # Print the retrieved documents
# for doc in results:
#     print(doc.page_content)

print(f"Number of documents in the collection: {vector_db._collection.count()}")
