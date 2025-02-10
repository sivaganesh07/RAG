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
# from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser




# Configure logging
logging.basicConfig(level=logging.INFO)

#constants
DOC_PATH = "./data/Sample_Invoice_2.pdf" 
model = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "Invoice-rag"
PERSIST_DIRECTORY = "./chroma_db_invoice"


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
    print("done splitting....")
    print(f"Number of chunks: {len(chunks)}")
    return chunks
    


# ===== Add to vector database ===
@st.cache_resource
def load_vector_db():
    ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    logging.info({os.path.exists(PERSIST_DIRECTORY)})
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
    template="""
    You are an AI Invoice Document assistant. Your tasks are:
   
    1. Generate five different versions of the given Original question to retrieve relevant documents from a vector database.

    By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search.

    Original question: {question} 

    Generated Questions:
    1. [Rephrased question 1]
    2. [Rephrased question 2]
    3. [Rephrased question 3]
    4. [Rephrased question 4]
    5. [Rephrased question 5]
    """,
)


    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever

def create_chain(retriever, llm):

    json_template = """
{
  "invoice_id": "",
  "date_issued": "",
  "payment_due": "",
  "total_payable": "",
  "currency_code": "",
  "supplier_name": "",
  "supplier_address": "",
  "buyer_name": "",
  "buyer_address": "",
  "items": [
    {
      "description": "",
      "quantity": "",
      "unit_price": "",
      "line_total": ""
    }
  ]
}"""

    template = """<|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an AI assistant that extracts invoice details and outputs them in strict JSON format. **DO NOT** include any extra text, explanations, or formatting beyond the JSON output.

    - You will be given data reterived from vector db. Your task is to transform this data into the following JSON schema:
    {json_template}

    - Follow these rules:
    1. Map the raw data fields to the schema fields. For example:
        - If the raw data contains "Invoice No", map it to "invoice_id".
        - If the raw data contains "Date",map it to "date_issued"
        - If the raw data contains "Service Description", map it to "items".
    2. If a field in the schema is missing in the raw data, use an empty string `""`.
    3. Do not invent values. Use only the data provided in the context.
    4. Ensure the output is a well-formed JSON object.

    <|eot_id|>

    <|start_header_id|>Human<|end_header_id|>
    Context: {context}
    Question: {question}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """


    #Wrapper for langchain
    prompt = ChatPromptTemplate.from_template(template)


    chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt.partial(json_template=json_template)  # Pass json_template as a static value
    | ChatOllama(model=model, stop=["<|eot_id|>"], format="json")
    | JsonOutputParser()
)

    logging.info("Chain created successfully.")
    return chain

def main():
    st.title("Invoice Assistant")
    
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
                print(response)
                st.json(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()