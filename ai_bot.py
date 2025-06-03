import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import SystemMessage

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

openai_llm = ChatOpenAI(model="gpt-4o-mini")
groq_llm = ChatGroq(model="llama-3.3-70b-versatile")
search_tool = TavilySearchResults(max_results=2)

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id)

    tools = [TavilySearchResults(max_results=2)] if allow_search else []
    agent = create_react_agent(model=llm, tools=tools)
    messages = [SystemMessage(content=system_prompt)] + query
    state = {"messages": messages}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]