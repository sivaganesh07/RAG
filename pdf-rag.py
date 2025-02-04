## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages

import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangChainDocument
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
# import ollama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

doc_path = "./data/BOI.pdf" 
model = "llama3.2"

# Load data from the PDF file using pymupdf4llm
md_read = pymupdf4llm.LlamaMarkdownReader()
llama_docs = md_read.load_data(doc_path) 

# Convert pymupdf4llm Documents to LangChain Documents
langchain_docs = [
    LangChainDocument(page_content=doc.get_content(), metadata={"source": doc_path})
    for doc in llama_docs
]

# print(langchain_docs[0])

# Split and chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(langchain_docs) 

print("done splitting....")
print(f"Number of chunks: {len(chunks)}")

# ===== Add to vector database ===

# ollama.pull("nomic-embed-text")


vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)
print("done adding to vector database....")

## === Retrieval ===
# set up our model to use
llm = ChatOllama(model=model)

#To get Context
# a simple technique to generate multiple questions from a single question and then retrieve documents
# based on those questions, getting the best of both worlds.
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)


# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = chain.invoke(input=("what is the document about?",))

print(res)