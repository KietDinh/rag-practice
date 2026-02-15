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

## History-Aware RAG Generation (`3_history_aware_generation.py`)

Implements conversational RAG that maintains chat history and uses it to improve question understanding and answer generation.

1.  **Load Embedding Model and Connect to Vector Database**:
    - **Library**: `langchain_huggingface`, `langchain_chroma`
    - **Action**: Initializes the `HuggingFaceEmbeddings` model and connects to the existing `Chroma` vector store.

2.  **Initialize Chat History Storage**:
    - **Action**: Sets up a list to store conversation history as message objects.

3.  **Process User Question with Context**:
    - **Action**: If there's chat history, uses the LLM to rewrite the question as standalone and searchable.

4.  **Retrieve Relevant Documents**:
    - **Library**: `langchain_chroma`
    - **Action**: Searches the vector database with the processed question to find relevant document chunks.

5.  **Generate Context-Aware Answer**:
    - **Library**: `langchain_ollama`
    - **Action**: Creates a prompt combining the question, retrieved documents, and chat history, then generates an answer.

6.  **Update and Display Conversation**:
    - **Action**: Adds the question and answer to chat history, then displays the complete interaction.

## Text Chunking Algorithms Comparison (`4_different_chunking_algorithms.py`)

Demonstrates and compares four different text chunking strategies for document preprocessing.

1.  **Character Text Splitter**:
    - **Library**: `langchain.text_splitter`
    - **Action**: Uses a single separator (like spaces) to split text. Simple but can fail on long text without separators.

2.  **Recursive Character Text Splitter**:
    - **Library**: `langchain.text_splitter`
    - **Action**: Tries multiple separators in order until finding one that works. More robust for varied text formats.

3.  **Semantic Chunking**:
    - **Library**: `langchain_experimental.text_splitter`, `langchain_huggingface`
    - **Action**: Uses embeddings to detect topic changes and group semantically related content together.

4.  **Agentic Chunking**:
    - **Library**: `langchain_ollama`
    - **Action**: Uses an LLM to intelligently decide where to split text based on context and logical boundaries.

## Retrieval Methods Comparison (`5_retrieval_methods.py`)

Compares different document retrieval strategies to find the most relevant information from the vector database.

1.  **Similarity Search**:
    - **Library**: `langchain_chroma`
    - **Action**: Returns the top-k most similar documents based on cosine similarity scores.

2.  **Similarity with Score Threshold**:
    - **Library**: `langchain_chroma`
    - **Action**: Only returns documents above a minimum similarity threshold, filtering out low-relevance results.

3.  **Maximum Marginal Relevance (MMR)**:
    - **Library**: `langchain_chroma`
    - **Action**: Balances relevance and diversity by selecting documents that are both relevant and different from each other.

## Multi-Query Retrieval (`6_multi_query_retrieval.py`)

Generates multiple variations of a query and retrieves documents for each, then combines the results.

1.  **Generate Query Variations**:
    - **Library**: `langchain_ollama`
    - **Action**: Uses the LLM to create multiple rephrased versions of the original query from different angles.

2.  **Retrieve Documents for Each Query**:
    - **Library**: `langchain_chroma`
    - **Action**: Searches the vector database with each query variation and collects the retrieved documents.

3.  **Display Results**:
    - **Action**: Shows the original query, generated variations, and retrieved documents for each variation.

## Reciprocal Rank Fusion (`7_reciprocal_rank_fusion.py`)

Combines results from multiple queries using reciprocal rank fusion to create a unified ranked list.

1.  **Generate Query Variations**:
    - **Library**: `langchain_ollama`
    - **Action**: Creates multiple query variations using the LLM to approach the question from different perspectives.

2.  **Retrieve Documents for Each Query**:
    - **Library**: `langchain_chroma`
    - **Action**: Performs similarity search for each query variation and collects ranked results.

3.  **Apply Reciprocal Rank Fusion**:
    - **Action**: Combines rankings from all queries using RRF formula (1/(k + r)) where k is a constant and r is the rank.

4.  **Display Fused Results**:
    - **Action**: Shows the final ranked list of documents that appeared across multiple query variations.
