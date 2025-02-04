## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages


import pymupdf4llm


doc_path = "./data/BOI.pdf" 
model = "llama3.2"


md_read = pymupdf4llm.LlamaMarkdownReader()
# Load data from the PDF file
llama_docs = md_read.load_data(doc_path)

print(llama_docs)
   
print("done loading....")



# Now you can use the 'docs' with your LangChain model

# ==== End of PDF Ingestion ====

# ==== Extract Text from PDF Files and Split into Small Chunks ====

# from langchain_ollama import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.document_loaders import TextLoader

# # Split and chunk
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)

# # Use the TextLoader to create a list of LangChain Documents
# loader = TextLoader(data=df)
# docs = loader.load()
# chunks = text_splitter.split_documents(docs)
# print("done splitting....")

# print(f"Number of chunks: {len(chunks)}")
# print(f"Example chunk: {chunks[0]}")





