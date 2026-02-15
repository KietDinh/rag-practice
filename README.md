# RAG Project

## ENV

## Ingestion Pipeline with Functions (`1_1_ingestion_pipeline.py`)

This is an improved version of the ingestion pipeline with modular functions and smart caching. It performs the following steps:

1.  **Check for Existing Vector Store**:
    - **Action**: Before processing, checks if the vector database already exists in `db/chroma_db`. If it exists, loads it directly without re-processing documents to save time.

2.  **Load Text Documents from Directory**:
    - **Library**: `langchain_community.document_loaders`
    - **Action**: Uses `DirectoryLoader` and `TextLoader` to read all `.txt` files from the `docs/` directory with UTF-8 encoding. Validates that the directory exists and contains documents.

3.  **Split Documents into Smaller Chunks**:
    - **Library**: `langchain_text_splitters`
    - **Action**: Uses `CharacterTextSplitter` to break documents into chunks with configurable size and overlap. Displays sample chunks for verification.

4.  **Convert Text Chunks to Vector Embeddings**:
    - **Library**: `langchain_huggingface`
    - **Action**: Uses `HuggingFaceEmbeddings` to convert text chunks into vector representations using the sentence-transformers model.

5.  **Store Embeddings in Vector Database**:
    - **Library**: `langchain_chroma`
    - **Action**: Creates a `Chroma` vector store from the document chunks and persists it to disk for reuse.

## Test Retrieval Pipeline (`1_2_test_retrieval_pipeline.py`)

This script tests the retrieval functionality by querying the vector database and displaying relevant documents. It performs the following steps:

1.  **Load Embedding Model and Connect to Vector Database**:
    - **Library**: `langchain_huggingface`, `langchain_chroma`
    - **Action**: Initializes the `HuggingFaceEmbeddings` model and connects to the existing `Chroma` vector store from the persistent directory.

2.  **Configure Retriever for Top-K Search**:
    - **Library**: `langchain_chroma`
    - **Action**: Sets up the `Chroma` retriever to return the top $k$ (e.g., 5) most relevant document chunks based on semantic similarity.

3.  **Retrieve Relevant Context for Query**:
    - **Library**: `langchain_chroma`
    - **Action**: Searches the vector database with a natural language query and retrieves the most semantically similar document chunks.

4.  **Display Retrieved Document Chunks**:
    - **Action**: Prints the user query and the content of the retrieved documents to verify the retrieval quality. Includes sample synthetic questions for testing.

## RAG Answer Generation (`2_1_answer_generation.py`)

This script demonstrates the complete RAG workflow: retrieval + answer generation using a local LLM. It performs the following steps:

1.  **Load Embedding Model and Connect to Vector Database**:
    - **Library**: `langchain_huggingface`, `langchain_chroma`
    - **Action**: Initializes the `HuggingFaceEmbeddings` model and connects to the existing `Chroma` vector store from the persistent directory.

2.  **Configure Retriever for Top-K Search**:
    - **Library**: `langchain_chroma`
    - **Action**: Sets up the `Chroma` retriever to return the top $k$ (e.g., 5) most relevant document chunks based on semantic similarity.

3.  **Retrieve Relevant Context for Query**:
    - **Library**: `langchain_chroma`
    - **Action**: Searches the vector database with a natural language query and retrieves the most semantically similar document chunks.

4.  **Construct Prompt with Retrieved Context**:
    - **Action**: Combines the user query with the retrieved document chunks into a structured prompt that instructs the LLM to answer based only on the provided context.

5.  **Generate Answer Using Local LLM**:
    - **Library**: `langchain_ollama`
    - **Action**: Uses `ChatOllama` with configurable parameters (temperature, context window, GPU/CPU settings) to generate a factual answer grounded in the retrieved documents.

6.  **Display Query, Context, and Generated Answer**:
    - **Action**: Prints the user query, retrieved document chunks (context), and the final LLM-generated answer to demonstrate the complete RAG pipeline.
