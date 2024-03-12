
import json
import re
from glob import glob
import sys
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl

def create_client():
    bedrock = boto3.client(service_name='bedrock-runtime')
    return bedrock

def create_llm(bedrock_client):
    llm = Bedrock(model_id='meta.llama2-13b-chat-v1', 
                  client=bedrock_client,
                  streaming=True,
                  model_kwargs={'temperature':0, 'top_p':0.9})
    return llm

def create_prompt():

    prompt_template = """
    If the question is not relevant to the provided documents, respond with "I don't know" or "This question is outside the bounds of the data I am trained on".

    {context}

    Question: {question}
    Answer: 

    """

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    return prompt 

@cl.on_chat_start
async def create_qa_chain():

    bedrock_client = boto3.client(service_name='bedrock-runtime')

    llm = create_llm(bedrock_client=bedrock_client)

    bedrock_embeddings=BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=bedrock_client)
    vector_store = FAISS.load_local('faiss_index', bedrock_embeddings, allow_dangerous_deserialization=True)
    
    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    qa_chain = ConversationalRetrievalChain.from_llm(llm, 
                                           chain_type='stuff', 
                                           retriever=vector_store.as_retriever(search_type='similarity', search_kwargs={"k":3}),
                                           return_source_documents=True,
                                           #chain_type_kwargs={'prompt':prompt}
                                           memory=memory
                                           )
    msg = cl.Message(content="Loading the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the QA Chatbot! Please ask your question."
    await msg.update()
    
    cl.user_session.set('qa_chain' ,qa_chain)

@cl.on_message
async def generate_response(query):
    qa_chain = cl.user_session.get('qa_chain')
    

    res = await qa_chain.acall(query.content, callbacks=[cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, 
        #answer_prefix_tokens= ["Final", "Answer"]
        )])

    # extract results and source documents
    #result, source_documents = res['result'], res['source_documents']
    result, source_documents = res['answer'], res['source_documents']
    # Extract all values associated with the 'metadata' key
    source_documents = str(source_documents)
    metadata_values = re.findall(r"metadata={'source': '([^']*)', 'page': (\d+)}", source_documents)

    # Convert metadata_values into a single string
    pattern = r'PDF Documents|\\'

    metadata_string = "\n".join([f"Source: {re.sub(pattern, '', source)}, page: {page}" for source, page in metadata_values])

    result += f'\n\n{metadata_string}'

    print(result)

    await cl.Message(content=result).send()
