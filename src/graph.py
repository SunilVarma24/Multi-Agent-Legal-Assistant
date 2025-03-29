# src/graph.py
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from src.agents import AgentState, query_agent_node, summarization_agent_node

# Define Router Logic to direct the flow between agents
def router(state: AgentState) -> str:
    messages = state['messages']
    # Check the last message to determine next step
    last_message = messages[-1]
    # If the last message is from the Query Agent and contains retrieval results
    if isinstance(last_message, AIMessage) and "RETRIEVAL COMPLETE" in last_message.content:
        return "continue_to_summarization"
    # If we've already processed the query, end the workflow
    return "__end__"

# Build the Multi-Agent Graph
workflow = StateGraph(AgentState)

# Add agent nodes
workflow.add_node("Query_Agent", query_agent_node)
workflow.add_node("Summarization_Agent", summarization_agent_node)

# Define edges from Query Agent based on router output.
workflow.add_conditional_edges(
    "Query_Agent",
    router,
    {"continue_to_summarization": "Summarization_Agent"}
)

# Define edges from Summarization Agent.
workflow.add_conditional_edges(
    "Summarization_Agent",
    router,
    {"__end__": END}
)

# Set the entry point for the workflow.
workflow.set_entry_point("Query_Agent")
agent_workflow = workflow.compile()

# For visualization in main.py
def get_agent_workflow():
    return agent_workflow
