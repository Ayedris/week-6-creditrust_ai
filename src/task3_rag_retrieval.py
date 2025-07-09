import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Load persistent vector DB (MUST match Task 2)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("complaint_embeddings")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Input query
query = input("Enter your question: ")

# Embed query
query_embedding = embedding_model.encode([query])

# Search in vector DB (top 5 most similar)
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=5,
    include=['documents', 'metadatas']
)

print("\nTop 5 Relevant Complaint Chunks:")
for idx, doc in enumerate(results['documents'][0]):
    print(f"\n--- Result {idx+1} ---")
    print(doc)
