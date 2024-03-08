
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
import chainlit as cl

def create_llm():

    llm = Bedrock(model_id='meta.llama2-13b-chat-v1', client=bedrock,
                  model_kwargs={'temperature':1})
    return llm

def create_prompt():

    prompt_template = """
    Use the provided information to answer the users questions. If you do not have the context required to answer the question, respond with "I don't know."

    {context}

    Question: {question}
    Answer: 

    """

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    return prompt 

def generate_response(llm, vector_store, prompt, query):

    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', 
                                           retriever=vector_store.as_retriever(search_type='similarity', search_kwargs={"k":3}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt':prompt})
    
    response = qa_chain({"query":query})

    return response['result']


def main():

    query = 'What are some techniques that can be uses for interacting with tabular data with LLMs?'

    llm = create_llm()
    prompt = create_prompt()
    vector_store = FAISS.load_local('chroma_index')
    response = generate_response(llm=llm, 
                                 vector_store=vector_store, 
                                 query=query,
                                 prompt=prompt)

    print(response)