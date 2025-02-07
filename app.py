import streamlit as st
import os
import time
import hashlib
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from openai import RateLimitError

# Load environment variables
load_dotenv(override=True)

AZURE_OPENAI_ENDPOINT = os.environ['AZURE_OPENAI_ENDPOINT']
AZURE_OPENAI_API_KEY = os.environ['AZURE_OPENAI_API_KEY']
AZURE_OPENAI_DEPLOYMENT = os.environ['AZURE_OPENAI_DEPLOYMENT']
OPENAI_API_VERSION = os.environ['OPENAI_API_VERSION']
TEXT_EMBEDDING_MODEL_NAME = os.environ['TEXT_EMBEDDING_MODEL_NAME']

# Define paths and constants
DATA_DIR = "data"
VECTOR_STORE_DIR = "faiss_store"
EMBEDDING_CACHE_FILE = "embedding_cache.json"
BATCH_SIZE = 50  # Number of chunks per API request

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)

def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Load cached embeddings metadata
def load_embedding_cache():
    if os.path.exists(EMBEDDING_CACHE_FILE):
        with open(EMBEDDING_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

# Save cache metadata
def save_embedding_cache(cache):
    with open(EMBEDDING_CACHE_FILE, "w") as f:
        json.dump(cache, f)

# Function to handle rate limit errors and store progress
def embed_with_retry(embeddings, texts, max_retries=5, cooldown=10):
    for attempt in range(max_retries):
        try:
            return embeddings.embed_documents(texts)
        except RateLimitError:
            wait_time = cooldown * (2 ** attempt)
            print(f"Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    print("Exceeded maximum retries. Switching to local embeddings.")
    return [[0.0] * 1536 for _ in texts]  # Fake zero embeddings as fallback

# Function to load and vectorize PDFs incrementally
def load_and_vectorize_pdfs():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    cache = load_embedding_cache()
    updated_cache = cache.copy()
    new_docs = []
    
    # Scan the DATA_DIR for PDFs
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            file_hash = get_file_hash(filepath)
            
            if filename in cache and cache[filename] == file_hash:
                print(f"Skipping {filename}, already processed.")
                continue  
            
            print(f"Processing {filename}...")
            loader = PyPDFLoader(filepath)
            pdf_texts = loader.load_and_split()
            
            # Attach the filename (and possibly page info) to each document
            for doc in pdf_texts:
                doc.metadata["source"] = filename
                if "page" not in doc.metadata:
                    doc.metadata["page"] = "Unknown"
            # Split each document into chunks
            new_docs.extend(text_splitter.split_documents(pdf_texts))
            updated_cache[filename] = file_hash

    # Always load the existing vector store if it exists
    embeddings = AzureOpenAIEmbeddings(model=TEXT_EMBEDDING_MODEL_NAME)
    if os.path.exists(VECTOR_STORE_DIR):
        vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store = None
    
    # If there are new document chunks, embed and add them to the vector store.
    if new_docs:
        print(f"Embedding {len(new_docs)} new document chunks...")
        for i in range(0, len(new_docs), BATCH_SIZE):
            batch = new_docs[i:i+BATCH_SIZE]
            texts = [doc.page_content for doc in batch]
            embedded_vectors = embed_with_retry(embeddings, texts)
            
            # Save the embedding with metadata for each chunk.
            for doc, vector in zip(batch, embedded_vectors):
                doc.metadata["embedding"] = vector
            
            if vector_store:
                vector_store.add_documents(batch)
            else:
                vector_store = FAISS.from_documents(batch, embeddings)
            
            print(f"Processed batch {i//BATCH_SIZE + 1}/{-(-len(new_docs)//BATCH_SIZE)}")
            time.sleep(2)
        
        vector_store.save_local(VECTOR_STORE_DIR)
        save_embedding_cache(updated_cache)
    else:
        print("No new documents to process.")

    # If no vector_store exists yet, create an empty one.
    if vector_store is None:
        vector_store = FAISS.from_documents([], embeddings)
        vector_store.save_local(VECTOR_STORE_DIR)
    
    return vector_store

# ---------------------- Sidebar for Document Management ----------------------
with st.sidebar:
    st.header("Document Management")
    
    # File uploader widget for PDFs
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(DATA_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            st.success(f"Uploaded {file.name}")
    
    st.markdown("### Uploaded Documents:")
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if pdf_files:
        for filename in pdf_files:
            st.write(filename)
    else:
        st.write("No documents uploaded yet.")

# ---------------------- Update / Load the FAISS Vector Store ----------------------
vector_store = load_and_vectorize_pdfs()

# ---------------------- Initialize the LLM and Conversational Chain ----------------------
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    model_name=AZURE_OPENAI_DEPLOYMENT,
    api_version=OPENAI_API_VERSION,
    temperature=0
)

# Conversation memory for the chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# (The initial retrieval chain is still available, but we will later create a temporary one using filtered documents.)

# ---------------------- Custom Question-Rewriting Chain ----------------------
def rewrite_query(query: str) -> str:

    custom_prompt = (
        "You are knowledgable in REPLACE THIS TO SUIT YOUR USE CASE and rewrites queries to expand abbreviations for clarity. "
        "For example, 'TTYL' should be expanded to 'Talk To You Later'.\n"
        "Rewrite the following query accordingly:\n\n"
        f"Query: {query}\n\nRewritten Query:"
    )
    response = llm(custom_prompt)
    rewritten_text = response.content if hasattr(response, "content") else str(response)
    return rewritten_text.strip()

# ---------------------- Streamlit Chat UI ----------------------
st.title("Custom Knowledge Base Chatbot")
st.write("Upload PDFs using the sidebar to increase the chatbot's knowledge.")

# Initialize chat history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------- New Grader Agent Function ---------------
def grade_doc(query: str, doc_text: str) -> bool:
    """
    Uses an LLM prompt to grade whether a document excerpt is relevant to the given query.
    The grader agent returns True if the excerpt is relevant, False otherwise.
    """
    grader_prompt = (
        "You are knowledgable in Information Retrieval Course and a document relevance grader. Given the question and the following document excerpt, "
        "answer with a single word: 'Yes' if the document is relevant for answering the question, or 'No' if it is not.\n\n"
        f"Question: {query}\n\nDocument Excerpt:\n{doc_text}\n\nAnswer:"
    )
    grader_response = llm(grader_prompt)
    response_text = grader_response.content if hasattr(grader_response, "content") else str(grader_response)
    return "yes" in response_text.lower()

# --------------- Chat Input and Processing ---------------
if user_query := st.chat_input("Ask something..."):
    # Display the user message.
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # ---- Rewrite the query to expand abbreviations before retrieval ----
    rewritten_query = rewrite_query(user_query)
    st.write(f"**Expanded Query for Retrieval:** {rewritten_query}")
    
    # ---- Retrieve candidate context with increased context size (k=10) ----
    candidate_docs = vector_store.similarity_search(rewritten_query, k=10)
    
    # ---- Filter candidate documents using the grader agent ----
    relevant_docs = []
    for doc in candidate_docs:
        if grade_doc(user_query, doc.page_content):
            relevant_docs.append(doc)
    
    if not relevant_docs:
        st.write("No sufficiently relevant context found. Using all candidate documents as fallback.")
        relevant_docs = candidate_docs
    
    # ---- Chain Answer Generation Using Filtered Context ----
    # We create a temporary vector store from the filtered documents so that the retrieval chain uses only these docs.
    temp_embeddings = AzureOpenAIEmbeddings(model=TEXT_EMBEDDING_MODEL_NAME)
    temp_vector_store = FAISS.from_documents(relevant_docs, temp_embeddings)
    temp_retriever = temp_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": len(relevant_docs)})
    
    # Build a new conversational retrieval chain with the filtered context and existing memory.
    temp_chain = ConversationalRetrievalChain.from_llm(
        llm,
        temp_retriever,
        memory=memory
    )
    
    # Get the final answer using the original user query (you could also use rewritten_query if desired)
    final_response = temp_chain.run(user_query)
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    
    with st.chat_message("assistant"):
        st.markdown(final_response)
        
        st.markdown("### Retrieved Context (after grading):")
        for doc in relevant_docs:
            source = doc.metadata.get("source", "Unknown")
            page_number = doc.metadata.get("page", "Unknown")
            with st.expander(f"Source: {source} | Page: {page_number}"):
                st.markdown(doc.page_content)
