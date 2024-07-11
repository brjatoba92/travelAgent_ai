import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter





llm = ChatOpenAI(model="gpt-3.5-turbo")

query = """
Vou viajar para Berlim em Novembro de 2024.
Quero que faça um roteiro de viagem para mim com os os events que irão ocorrer na data da viagem e com o preço de passagem de São Paulo para Berlim.
"""

def reserchAgent(query, llm):
    tools = load_tools(['ddg-search', 'wikipedia'], llm= llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt, verbose=True)
    webContext = agent_executor.invoke({"input": query})
    return webContext['output']

# print(reserchAgent(query, llm))


def loadData():
    loader = WebBaseLoader(
        webpaths=("https://www.dicasdeviagem.com/alemanha/"),
        bs_kwargs=dict(parse_only=bs4.SoapStrainer(class_=("postcontentwrap", "pagetitleloading background-imaged loading-dark")))
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def getRelevantDocs(query):
    retriever =loadData()
    relevant_documents = retriever.invoke(query)
    return relevant_documents

