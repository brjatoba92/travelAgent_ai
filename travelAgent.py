import os
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub


llm = ChatOpenAI(model="gpt-3.5-turbo")

tools = load_tools(['ddg-search', 'wikipedia'], llm= llm)

# print(tools[1].name, tools[1].description)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt, verbose=True)


query = """
Vou viajar para Berlim em Novembro de 2024.
Quero que faça um roteiro de viagem para mim com os os events que irão ocorrer na data da viagem e com o preço de passagem de São Paulo para Berlim.
"""

agent_executor.invoke({"input": query})

