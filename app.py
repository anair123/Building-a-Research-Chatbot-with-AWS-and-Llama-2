
import json
from glob import glob
import sys
import boto3
import numpy as np
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA



# Set up bedrock client
bedrock = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings=BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=bedrock)


def load_pdfs(chunk_size=1000, chunk_overlap=100):
    loader=PyPDFDirectoryLoader("PDF Documents")
    documents=loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents=documents)
    return docs

def create_vector_store(docs):

    vector_store = chroma.from_documents(docs, bedrock_embeddings)
    vector_store.save_local("chroma_index")


def create_llm():

    llm = Bedrock(model_id='meta.llama2-13b-chat-v1')