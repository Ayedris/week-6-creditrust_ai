# initialize_vectorstore.py (OPTIMIZED VERSION)
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm  # For progress bars
import numpy as np
import os

# CONFIGURATION
DATA_PATH = 'data/filtered_complaints.csv'
EMBEDDING_MODEL_PATH = "C:/Users/ayedr/week-6-creditrust_ai/data/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "C:/Users/ayedr/week-6-creditrust_ai/vector_store"
CHUNK_SIZE = 1000  # Increased from 500
BATCH_SIZE = 50    # Reduced from 100

# 1. Load data FASTER
print("Loading data with optimized settings...")
df = pd.read_csv(
    DATA_PATH,
    usecols=['Consumer complaint narrative'],  # Only load needed column
    dtype={'Consumer complaint narrative': 'string'},  # Force string type
    low_memory=False
)
complaints = df['Consumer complaint narrative'].dropna().tolist()

# 2. Load embedding model ONCE
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

# 3. Initialize ChromaDB
print("Setting up vector database...")
client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
collection = client.get_or_create_collection(
    "complaint_embeddings",
    metadata={"hnsw:space": "cosine"}  # Faster similarity search
)

# 4. Process in optimized batches
print("Processing documents...")
for i in tqdm(range(0, len(complaints), BATCH_SIZE)):
    batch = complaints[i:i+BATCH_SIZE]
    
    # Encode entire batch at once
    embeddings = embedding_model.encode(
        batch,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()
    
    # Add to collection
    collection.add(
        documents=batch,
        embeddings=embeddings,
        ids=[f"doc_{i+j}" for j in range(len(batch))]
    )

print(f"\n✅ Done! Processed {len(complaints)} complaints.")