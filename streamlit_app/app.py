import os
import operator
import streamlit as st
from typing import Annotated, Sequence, TypedDict, Literal
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph

# Streamlit UI setup
st.set_page_config(page_title="Legal Assistant Chatbot", layout="wide")
st.title("📜 AI-Powered Legal Assistant")
st.write("Ask legal questions and receive simplified explanations.")

# Load and process legal documents
@st.cache_resource
def load_and_process_documents():
    guide_pdf_path = "data/Guide-to-Litigation-in-India.pdf"
    corporate_pdf_path = "data/Legal-Compliance-&-Corporate-Laws-by-ICAI.pdf"
    loader_guide = PyMuPDFLoader(guide_pdf_path)
    loader_corporate = PyMuPDFLoader(corporate_pdf_path)
    all_docs = loader_guide.load() + loader_corporate.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = FAISS.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH, collection_name="legal_docs")
    return vector_store

vector_store = load_and_process_documents()

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

# Query Agent
def query_agent_node(state: AgentState) -> AgentState:
    user_query = state["messages"][-1].content
    results = vector_store.similarity_search(user_query, k=5)
    retrieved_text = "\n\n".join([doc.page_content for doc in results])
    content = f"RETRIEVAL COMPLETE\n\n{retrieved_text}"
    return {"messages": [AIMessage(content=content, name="Query_Agent")], "sender": "Query_Agent"}

# Summarization Agent
legal_summarization_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Legal Summarization Assistant. "
     "Your task is to provide a **concise and neutral summary** of the legal procedure in India. "
     "Summarize the **standard** steps of the process, avoiding unnecessary focus on **specific cases** like government lawsuits, foreign state cases, or mediation unless they are the primary focus. "
     "Ensure the response represents the **general filing process** fairly and does not highlight **exceptions** unless required. "
     "Use **clear and simple language** suitable for a general audience. "
     "Limit your response to **3-4 sentences** and ensure it begins with 'SUMMARY COMPLETE'."),
    MessagesPlaceholder(variable_name="messages"),
])

summarization_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
summarization_agent = legal_summarization_prompt | summarization_llm

def summarization_agent_node(state: AgentState) -> AgentState:
    result = summarization_agent.invoke(state)
    return {"messages": [AIMessage(content=result.content, name="Summarization_Agent")], "sender": "Summarization_Agent"}

# Routing logic
def router(state: AgentState) -> str:
    messages = state['messages']

    # Check the last message to determine next step
    last_message = messages[-1]

    # If the last message is from the Query Agent and contains retrieval results
    if isinstance(last_message, AIMessage) and "RETRIEVAL COMPLETE" in last_message.content:
        return "continue_to_summarization"

    # If we've already processed the query, end the workflow
    return "__end__"

# Define the multi-agent graph
workflow = StateGraph(AgentState)
workflow.add_node("Query_Agent", query_agent_node)
workflow.add_node("Summarization_Agent", summarization_agent_node)
workflow.add_conditional_edges("Query_Agent", router, {"continue_to_summarization": "Summarization_Agent", "__end__": END})
workflow.add_conditional_edges("Summarization_Agent", router, {"__end__": END})
workflow.set_entry_point("Query_Agent")
agent_workflow = workflow.compile()

# Streamlit UI for querying
user_input = st.text_input("Enter your legal query:")
if st.button("Submit") and user_input:
    with st.spinner("Processing..."):
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        response = agent_workflow.invoke(initial_state)
        final_summary = response["messages"][-1].content.replace("SUMMARY COMPLETE: ", "", 1)
        st.markdown("## Response:")
        st.markdown(final_summary)
