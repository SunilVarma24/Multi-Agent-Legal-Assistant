# src/agents.py
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Define the Agent State structure 
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

# Initialize LLM (for summarization)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Query Agent Node ---
from src.document import vector_store

def query_agent_node(state: AgentState) -> AgentState:
    # Extract user query from the last message
    user_query = state["messages"][-1].content
    # Retrieve relevant document chunks (e.g., top 5 results)
    results = vector_store.similarity_search(user_query, k=5)
    # Concatenate the retrieved sections
    retrieved_text = "\n\n".join([doc.page_content for doc in results])
    # Prefix with an indicator to trigger the next agent
    content = f"RETRIEVAL COMPLETE\n\n{retrieved_text}"
    return {
        "messages": [AIMessage(content=content, name="Query_Agent")],
        "sender": "Query_Agent"
    }

# --- Summarization Agent Node ---
legal_summarization_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Legal Summarization Assistant. "
     "Your task is to provide a **concise and neutral summary** of the legal procedure in India. "
     "Summarize the **standard** steps of the process, avoiding unnecessary focus on **specific cases** like government lawsuits, foreign state cases, or mediation unless they are the primary focus. "
     "Ensure the response represents the **general filing process** fairly and does not highlight **exceptions** unless required. "
     "Use **clear and simple language** suitable for a general audience. "
     "Limit your response to **2-3 sentences** and ensure it begins with 'SUMMARY COMPLETE'."),
    MessagesPlaceholder(variable_name="messages"),
])
summarization_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
summarization_agent = legal_summarization_prompt | summarization_llm

def summarization_agent_node(state: AgentState) -> AgentState:
    result = summarization_agent.invoke(state)
    result_message = AIMessage(content=result.content, name="Summarization_Agent")
    return {
        "messages": [result_message],
        "sender": "Summarization_Agent"
    }
