## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages
import streamlit as st
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangChainDocument
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import ollama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)

#constants
DOC_PATH = "./data/Doc1.pdf" 
model = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"


def ingest_pdf(doc_path):
    # Load data from the PDF file using pymupdf4llm
    if os.path.exists(doc_path):
        md_read = pymupdf4llm.LlamaMarkdownReader()
        llama_docs = md_read.load_data(doc_path) 

        # Convert pymupdf4llm Documents to LangChain Documents
        langchain_docs = [
            LangChainDocument(page_content=doc.get_content(), metadata={"source": doc_path})
            for doc in llama_docs
        ]
        return langchain_docs
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        return None
    


def split_documents(documents):
    # Split and chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents) 
    logging.info("Documents split into chunks.")
    return chunks
    # print("done splitting....")
    # print(f"Number of chunks: {len(chunks)}")


# ===== Add to vector database ===
@st.cache_resource
def load_vector_db():
    ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        # Load and process the PDF document
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None

        # Split the documents into chunks
        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
    return vector_db

def create_retriever(vector_db, llm):
    ## === Retrieval ===
    #To get Context
    # a simple technique to generate multiple questions from a single question and then retrieve documents
    # based on those questions, getting the best of both worlds.
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever

def create_chain(retriever, llm):
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context} 
    Question: {question}
    """

    #Wrapper for langchain
    prompt = ChatPromptTemplate.from_template(template)


    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("Chain created successfully.")
    return chain

def main():
    st.title("Document Assistant")
    
    # User input
    user_input = st.text_input("Enter your question:", "")

    if user_input:
        with st.spinner("Generating response..."):
            try:
                # Load the vector database
                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return

                # set up our model to use
                llm = ChatOllama(model=model)

                # Create the retriever
                retriever = create_retriever(vector_db, llm)

                # Create the chain with preserved syntax
                chain = create_chain(retriever, llm)

                # res = chain.invoke(input=("What is this document about?"))
                response = chain.invoke(input=user_input)

                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()