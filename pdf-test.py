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
DOC_PATH = "./data/Sample_Invoice_2_nonDigital.pdf" 
model = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "Invoice-rag"
PERSIST_DIRECTORY = "./chroma_db_invoice"


def ingest_pdf(doc_path):
    # Load data from the PDF file using pymupdf4llm
    if os.path.exists(doc_path):
        md_read = pymupdf4llm.LlamaMarkdownReader()
        llama_docs = md_read.load_data(doc_path) 

        print(f"Read {llama_docs}")

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

    # embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    # logging.info({os.path.exists(PERSIST_DIRECTORY)})
    # if os.path.exists(PERSIST_DIRECTORY):
    #     vector_db = Chroma(
    #         embedding_function=embedding,
    #         collection_name=VECTOR_STORE_NAME,
    #         persist_directory=PERSIST_DIRECTORY,
    #     )
    #     logging.info("Loaded existing vector database.")
    # else:
        # Load and process the PDF document
    data = ingest_pdf(DOC_PATH)
    print(data)
        # if data is None:
        #     return None

        # # Split the documents into chunks
        # chunks = split_documents(data)

        # vector_db = Chroma.from_documents(
        #     documents=chunks,
        #     embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        #     collection_name=VECTOR_STORE_NAME,
        #     persist_directory=PERSIST_DIRECTORY
        # )
        # vector_db.persist()
        # logging.info("Vector database created and persisted.")
    # return vector_db

def create_retriever(vector_db, llm):
    ## === Retrieval ===
    #To get Context
    # a simple technique to generate multiple questions from a single question and then retrieve documents
    # based on those questions, getting the best of both worlds.
    QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
    You are an AI Invoice Document assistant. Your tasks are:
   
    1. Generate five different versions of the given user question to retrieve relevant documents from a vector database.

    By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search. As an Invoice assistant, you are obliged to provide the invoice number and customer name & address to the user for each request.

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

    

    # RAG prompt
    template = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant specialized in processing invoices from chroma db. Please provide answers in the following format:
- **Answer**: [Your answer here]
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Context: {context}
Question: {question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    #Wrapper for langchain
    prompt = ChatPromptTemplate.from_template(template)


    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | ChatOllama(model=model,stop=["<|eot_id|>"])
        | StrOutputParser()
    )
    logging.info("Chain created successfully.")
    return chain

def main():
    data = ingest_pdf(DOC_PATH)
    print(data)
    st.title("Invoice Assistant")
    
    # # User input
    user_input = st.text_input("Enter your question:", """
Retrieve the relevant invoice data for 123100402  from the local Chroma DB and generate a structured JSON output in the format required for SAP's BAPI_ACC_DOCUMENT_POST.

Query Chroma DB to extract the following fields:
- DOCUMENTHEADER: Extract invoice metadata such as DOC_TYPE, DOC_DATE, PSTNG_DATE, REF_DOC_NO, COMP_CODE, CURRENCY, HEADER_TXT.
- ACCOUNTGL: Extract item-level details such as ITEMNO_ACC, GL_ACCOUNT, ITEM_TEXT, AMT_DOCCUR, CURRENCY, TAX_CODE.
- CURRENCYAMOUNT: Extract currency and amount details per item.
- TAXDATA: Extract tax breakdown details.

Ensure the extracted values are **accurate** and match exactly what is stored in Chroma. If a value is missing, leave it as an empty string.

### Expected Output:
The response should be a structured JSON with the extracted data, formatted as follows:

{
  "DOCUMENTHEADER": {
    "DOC_TYPE": "",
    "DOC_DATE": "",
    "PSTNG_DATE": "",
    "REF_DOC_NO": "",
    "COMP_CODE": "",
    "CURRENCY": "",
    "HEADER_TXT": ""
  },
  "ACCOUNTGL": [
    {
      "ITEMNO_ACC": "",
      "GL_ACCOUNT": "",
      "ITEM_TEXT": "",
      "AMT_DOCCUR": "",
      "CURRENCY": "",
      "TAX_CODE": ""
    }
  ],
  "CURRENCYAMOUNT": [
    {
      "ITEMNO_ACC": "",
      "CURRENCY": "",
      "AMT_DOCCUR": ""
    }
  ],
  "TAXDATA": [
    {
      "ITEMNO_ACC": "",
      "TAX_CODE": "",
      "TAX_AMOUNT": "",
      "TAX_BASE_AMOUNT": ""
    }
  ]
}

Use **only** the data retrieved from Chroma DB. Do not generate or assume any missing values.
The output should be **only** the JSON structure without any explanations or formatting as code.
 """)
    
    if st.button("Execute"):
        print("Yesss")

    # if user_input:
    #     with st.spinner("Generating response..."):
    #         try:
    #             # Load the vector database
    #             # vector_db = load_vector_db()
    #             # if vector_db is None:
    #             #     st.error("Failed to load or create the vector database.")
    #             #     return

    #             # # set up our model to use
    #             # llm = ChatOllama(model=model)

    #             # # Create the retriever
    #             # retriever = create_retriever(vector_db, llm)

    #             # # Create the chain with preserved syntax
    #             # chain = create_chain(retriever, llm)

    #             # # res = chain.invoke(input=("What is this document about?"))
    #             # response = chain.invoke(input=user_input)

    #             # st.markdown("**Assistant:**")
    #             # st.write(response)
    #         except Exception as e:
    #             st.error(f"An error occurred: {str(e)}")
    # else:
    #     st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()