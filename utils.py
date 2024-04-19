
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from typing import Dict,Tuple
from langchain.agents import initialize_agent, Tool,AgentExecutor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import requests
import io
from dotenv import load_dotenv
import google.generativeai as genai 
from PIL import Image
load_dotenv()

import streamlit as st


os.environ['TAVILY_API_KEY'] = st.secrets['TAVILY_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']


tavily_api_key = os.environ['TAVILY_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']
google_api_key=os.environ['GOOGLE_API_KEY']

class Config():
    """
    Contains the configuration of the LLM.
    """
    model = 'gpt-3.5-turbo'
    llm = ChatOpenAI(temperature=0, model=model)


embeddings = OpenAIEmbeddings()

def encode_and_query_api(image, api_key):
    genai.configure(api_key="AIzaSyCldy4Q-Bg4wVbC6GAmiGzC9Fq27x32vPc")

    model = genai.GenerativeModel('gemini-pro-vision')

    model_response = model.generate_content(["describe this image", image])
    return model_response.text


Delphinium = Chroma(persist_directory="./Examples/Delphinium", embedding_function=embeddings)
# spider_plant = Chroma(persist_directory="C:\Users\pc\Desktop\Oasis hackathon\Examples\Spider_Plant", embedding_function=embeddings)

cfg = Config()
qa = RetrievalQA.from_chain_type(
    llm=cfg.llm,
    chain_type="stuff",
    retriever=Delphinium.as_retriever()
)
# def qa(VectorDb):
#     return RetrievalQA.from_chain_type(
#     llm=cfg.llm,
#     chain_type="stuff",
#     retriever=VectorDb.as_retriever()
#     )

def setup_memory() -> Tuple[Dict, ConversationBufferMemory]:
    """
    Sets up memory for the  agent.
    :return a tuple with the agent keyword pairs and the conversation memory.
    """
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
    return agent_kwargs, memory

def setup_agent() -> AgentExecutor:
    """
    Sets up the tools for a function based chain.
    We have here the following tools:

    """
    cfg = Config()
    tools = [
        Tool(
            name="knowledge base",
            func=qa.run,
            description="Used when you need the name , age , characteristics or something tha you need to know about the plant "
        ),
        Tool(
        name='web search',
        func=TavilySearchResults().run,
        description='''Use this tool when you can't find the content in the knowledge base and you need more advenced search functionalities that are related to the user query '''
        ),
    ]
    agent_kwargs, memory = setup_memory()

    return initialize_agent(
        tools,
        cfg.llm,
        verbose=False,
        agent_kwargs=agent_kwargs,
        memory=memory,
        handle_parsing_errors=True
    )
