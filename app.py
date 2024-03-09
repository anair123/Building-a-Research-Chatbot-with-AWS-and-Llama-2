
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

def create_client():
    bedrock = boto3.client(service_name='bedrock-runtime')
    return bedrock

def create_llm(bedrock):
    llm = Bedrock(model_id='meta.llama2-13b-chat-v1', 
                  client=bedrock,
                  streaming=True,
                  model_kwargs={'temperature':1})
    return llm

def create_prompt():

    prompt_template = """
    Use the provided information to answer the users questions. 
    If you do not have the context required to answer the question, respond with "I don't know."

    {context}

    Question: {question}
    Answer: 

    """

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    return prompt 

def create_qa_chain(llm, vector_store, prompt):

    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', 
                                           retriever=vector_store.as_retriever(search_type='similarity', search_kwargs={"k":3}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt':prompt})
    return qa_chain

def generate_response(qa_chain, query):
    response = qa_chain({"query":query})

    return response['result']


def main():

    bedrock = boto3.client(service_name='bedrock-runtime')
    llm = create_llm(bedrock=bedrock)
    prompt = create_prompt()
    bedrock_embeddings=BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=bedrock)

    query = 'What are some techniques that can be uses for interacting with tabular data with LLMs?'

    vector_store = FAISS.load_local('faiss_index', bedrock_embeddings, allow_dangerous_deserialization=True)
    response = generate_response(llm=llm, 
                                 vector_store=vector_store, 
                                 query=query,
                                 prompt=prompt)

    print(response)


### start the app 
@cl.on_chat_start
async def start():
     
    bedrock = create_client()
    llm = create_llm(bedrock=bedrock)
    prompt = create_prompt()
    bedrock_embeddings=BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=bedrock)
    vector_store = FAISS.load_local('faiss_index', bedrock_embeddings, allow_dangerous_deserialization=True)


    qa_chain = create_qa_chain(llm=llm, prompt=prompt, vector_store=vector_store)
    msg = cl.Message(content="Loading the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the Chatbot! Please ask your question."
    await msg.update()
    cl.user_session.set("chain", qa_chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens= ["Final", "Answer"]
    )
    cb.answer_reached=True

    res=await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res['source_documents']

    await cl.Message(content=answer).send()