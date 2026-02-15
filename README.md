# RAG Project

## ENV

## Ingestion Pipeline with Functions (`1_1_ingestion_pipeline.py`)

Loads text documents from the `docs/` directory, splits them into chunks, converts them to vector embeddings, and stores them in a Chroma vector database for semantic search.

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

Queries the vector database with test questions and displays the most relevant document chunks retrieved.

1.  **Load Embedding Model and Connect to Vector Database**:
    - **Library**: `langchain_huggingface`, `langchain_chroma`
    - **Action**: Initializes the `HuggingFaceEmbeddings` model and connects to the existing `Chroma` vector store from the persistent directory.

2.  **Configure Retriever for Top-K Search**:
    - **Library**: `langchain_chroma`
    - **Action**: Configures the retriever to return the top $k$ (e.g., 5) most similar documents.

3.  **Retrieve Relevant Context for Query**:
    - **Library**: `langchain_chroma`
    - **Action**: Searches the database with a question and retrieves the most similar document chunks.

4.  **Display Retrieved Document Chunks**:
    - **Action**: Prints the query and retrieved documents to check retrieval quality.

## RAG Answer Generation (`2_answer_generation.py`)

Retrieves relevant document chunks from the vector database and uses a local LLM (via Ollama) to generate answers based on those documents.

1.  **Load Embedding Model and Connect to Vector Database**:
    - **Library**: `langchain_huggingface`, `langchain_chroma`
    - **Action**: Initializes the `HuggingFaceEmbeddings` model and connects to the existing `Chroma` vector store from the persistent directory.

2.  **Configure Retriever for Top-K Search**:
    - **Library**: `langchain_chroma`
    - **Action**: Configures the retriever to return the top $k$ (e.g., 5) most similar documents.

3.  **Retrieve Relevant Context for Query**:
    - **Library**: `langchain_chroma`
    - **Action**: Searches the database with a question and retrieves the most similar document chunks.

4.  **Construct Prompt with Retrieved Context**:
    - **Action**: Combines the user query with the retrieved document chunks into a prompt that tells the LLM to answer using only the provided documents.

5.  **Generate Answer Using Local LLM**:
    - **Library**: `langchain_ollama`
    - **Action**: Sends the prompt to `ChatOllama` to generate an answer based on the retrieved documents.

6.  **Display Query, Context, and Generated Answer**:
    - **Action**: Prints the user query, retrieved documents, and the LLM's final answer.
