# ChatGPT with Documents

## Overview

**ChatGPT with Documents** is a simple demostration on how to build a RAG retrieval and Q&A application that allows users to upload PDF files and interact with them through a conversational chatbot interface. Built with **Streamlit**, **LangChain**, and **Azure OpenAI**, this project leverages advanced NLP models to deliver accurate, context-based answers from uploaded documents.

## Features

- **PDF Upload and Management:** Upload multiple PDF documents via the sidebar.
- **Automated Document Processing:** Extracts, splits, and vectorizes text from PDFs for efficient retrieval.
- **Conversational Chatbot:** Engage in dynamic conversations with ChatGPT, which pulls relevant context from your uploaded documents.
- **Query Rewriting:** Automatically expands abbreviations and clarifies queries for improved answer accuracy.
- **Document Relevance Grading:** Uses an LLM to filter irrelevant content, enhancing the quality of retrieved answers.
- **Session Memory:** Maintains conversation history for coherent multi-turn dialogues.

## Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python, LangChain
- **LLM:** Azure OpenAI (ChatGPT & Embeddings)
- **Vector Store:** FAISS
- **Document Handling:** PyPDFLoader
- **Environment Management:** dotenv

## To Run The Application

- Convert the ".env.example" to ".env" with your environment key
- Run "streamlit run app.py" to start the application
