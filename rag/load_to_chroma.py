import os
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def Load_VD():
    # Load documents
    loader = PyPDFLoader("./data/RAG.pdf")
    documents = loader.load()
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(len(texts))
    # Select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # Create the vector store to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # Expose this index in a retriever interface
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever


Load_VD()
