# Multi-Agent Legal Chatbot

## Project Overview
The Multi-Agent Legal Chatbot leverages multi-agent architecture to fetch legal information from trusted sources and summarize complex legal topics into simple, clear responses. The system uses LangChain and LangGraph and is deployed via Streamlit for an interactive user interface.

## Table of Contents
1. Introduction
2. System Design
3. Installation
4. Usage
5. Results
6. Conclusion

## Introduction
Legal documents often contain complex language, making it difficult for users to understand the information. This chatbot automates the process of fetching relevant legal information and simplifying it into easy-to-understand summaries.

## System Design

The system is designed with two key agents:

1. Query Agent

- Retrieves relevant sections from legal documents based on user queries.

- Uses vector embeddings to search legal bge-small-en-v1.5 embedding model to generate vector embeddings for the text.

2. Summarization Agent

- Extracts key points from the retrieved legal content.

- Converts complex legal terminology into simplified, plain language using GPT-4o Mini model.

## Example Flow

User Query: "What are the steps to file a lawsuit in India?"

1. Query Agent retrieves the related sections from the documents.

2. Summarization Agent converts the legal content into:

- Step 1: Prepare necessary documents.

- Step 2: File a petition in court.

- Step 3: Serve notice to the opposing party.

- Step 4: Attend court hearings.

3. Response
Filing a lawsuit in India involves preparing legal documents, submitting a petition in court, serving a notice to the opposing party, and attending hearings. Would you like more details on any step?

## Installation

To set up the project, install the following dependencies:
```bash
pip install langchain langchain-core langchain-openai langchain-google-genai langgraph chromadb PyMuPDF streamlit
```
## Usage
1. Load the legal PDF documents.
2. Ask legal queries.
3. The Query Agent retrieves relevant sections.
4. The Summarization Agent converts the content into simple language.
5. The chatbot displays the summarized response.

## Running the App
```bash
streamlit run app.py
```
## Results
- Accurate retrieval of legal sections.
- Simplified legal language for better user understanding.
- Interactive chatbot interface.

## Conclusion

This chatbot leverages multi-agent architecture to provide legal information summarization. The system enhances legal research efficiency by breaking down complex legal jargon into understandable terms.
