
import json
from glob import glob
import sys
import boto3
import numpy as np
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from config import aws_access_key, aws_region_name, aws_secret_key

import chainlit as cl

def create_client():
    bedrock = boto3.client(service_name='bedrock-runtime',
                        region_name=aws_region_name,
                        aws_access_key_id =aws_access_key,
                        aws_secret_access_key =aws_secret_key
                        )
    return bedrock



def load_pdfs(chunk_size=3000, chunk_overlap=100):

    # load the pdf documents
    loader=PyPDFDirectoryLoader("PDF Documents")
    documents=loader.load()

    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents=documents)
    return docs

def create_vector_store(docs):

    # Set up bedrock client
    bedrock = create_client()
    bedrock_embeddings=BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=bedrock)

    # create and save the vector store
    vector_store = FAISS.from_documents(docs, bedrock_embeddings)
    vector_store.save_local("faiss_index")
    
    return None

if __name__ == "__main__":
    docs = load_pdfs()
    create_vector_store(docs=docs)