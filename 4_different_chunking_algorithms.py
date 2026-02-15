from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from constants import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME

# Load environment variables
load_dotenv()

# Tesla text for demonstration (expanded for better semantic/agentic chunking)
tesla_text_expanded = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

# Simple text for basic/recursive comparison
tesla_text_simple = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1: CHARACTER TEXT SPLITTER
# ══════════════════════════════════════════════════════════════════════════════
# Uses a single separator to split text. Can fail if separator is not present.
# Best for simple text with consistent formatting.
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("METHOD 1: CHARACTER TEXT SPLITTER (Basic)")
print("=" * 70)

char_splitter = CharacterTextSplitter(
    separator=" ",  # Single separator - fails on long text without spaces
    chunk_size=100,
    chunk_overlap=0,
)

chunks1 = char_splitter.split_text(tesla_text_simple)
print(f"Created {len(chunks1)} chunks:\n")
for i, chunk in enumerate(chunks1, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2: RECURSIVE CHARACTER TEXT SPLITTER
# ══════════════════════════════════════════════════════════════════════════════
# Tries multiple separators in order until it finds one that works.
# Best for general text splitting with configurable chunk size.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("METHOD 2: RECURSIVE CHARACTER TEXT SPLITTER")
print("=" * 70)

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],  # Multiple separators in priority order
    chunk_size=100,
    chunk_overlap=0,
)

chunks2 = recursive_splitter.split_text(tesla_text_simple)
print(f"Created {len(chunks2)} chunks (same text, better splitting):\n")
for i, chunk in enumerate(chunks2, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3: SEMANTIC CHUNKING
# ══════════════════════════════════════════════════════════════════════════════
# Groups text by meaning using embeddings to detect topic changes.
# Best for keeping semantically related content together.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("METHOD 3: SEMANTIC CHUNKING")
print("=" * 70)

semantic_splitter = SemanticChunker(
    embeddings=HuggingFaceEmbeddings(model=EMBEDDING_MODEL_NAME),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=70,
)

chunks3 = semantic_splitter.split_text(tesla_text_expanded)
print(f"Created {len(chunks3)} chunks (grouped by meaning):\n")
for i, chunk in enumerate(chunks3, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 4: AGENTIC CHUNKING
# ══════════════════════════════════════════════════════════════════════════════
# Uses an LLM to intelligently decide where to split based on context and logic.
# Best for complex text where AI can make better decisions than rules.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("METHOD 4: AGENTIC CHUNKING (LLM-based)")
print("=" * 70)

llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0)

prompt = f"""
You are a text chunking expert. Split this text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text:
{tesla_text_expanded}

Return the text with <<<SPLIT>>> markers where you want to split:
"""

print("🤖 Asking AI to chunk the text...")
response = llm.invoke(prompt)
marked_text = response.content

# Split the text at the markers
chunks4 = marked_text.split("<<<SPLIT>>>")

# Clean up the chunks (remove extra whitespace)
clean_chunks = []
for chunk in chunks4:
    cleaned = chunk.strip()
    if cleaned:  # Only keep non-empty chunks
        clean_chunks.append(cleaned)

print(f"\nCreated {len(clean_chunks)} chunks (AI-decided splits):\n")
for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()

print("\n" + "=" * 70)
print("COMPARISON COMPLETE - 4 Chunking Methods Demonstrated")
print("=" * 70)
