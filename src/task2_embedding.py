import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from tqdm import tqdm

# === CONFIGURATION ===
DATA_PATH = Path('data/filtered_complaints.csv')
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
BATCH_SIZE = 128
VECTOR_STORE_NAME = "complaint_embeddings"

# === STEP 1: Load Data ===
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Dataset loaded with {df.shape[0]} records.")

# === STEP 2: Text Chunking ===
print("Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

texts = df['Consumer complaint narrative'].astype(str).tolist()
chunks = []
for text in texts:
    chunks.extend(splitter.split_text(text))

print(f"Total Chunks Created Before Filtering: {len(chunks)}")

# Filter empty chunks
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
print(f"Total Chunks After Filtering: {len(chunks)}")

# === STEP 3: Embedding Model (Fast Version) ===
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # FAST CPU Model

# === STEP 4: Embedding Chunks ===
print("Generating embeddings (this may take time)...")
chunk_embeddings = embedding_model.encode(
    chunks,
    batch_size=BATCH_SIZE,
    show_progress_bar=True
)

# === STEP 5: Store Embeddings in ChromaDB (Batch Write) ===
print("Initializing persistent vector database...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(VECTOR_STORE_NAME)

print("Storing embeddings in batches...")
for start_idx in tqdm(range(0, len(chunks), BATCH_SIZE)):
    end_idx = min(start_idx + BATCH_SIZE, len(chunks))
    collection.add(
        documents=chunks[start_idx:end_idx],
        embeddings=chunk_embeddings[start_idx:end_idx].tolist(),
        metadatas=[{"chunk_id": idx} for idx in range(start_idx, end_idx)],
        ids=[f"chunk_{idx}" for idx in range(start_idx, end_idx)]
    )

print("\n✅ Embeddings stored successfully in vector DB.")
