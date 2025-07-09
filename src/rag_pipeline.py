# src/rag_pipeline.py
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Configuration
MODELS_DIR = "C:/Users/ayedr/week-6-creditrust_ai/data"
VECTOR_STORE_PATH = "C:/Users/ayedr/week-6-creditrust_ai/vector_store"

# Model paths
EMBEDDING_MODEL_PATH = f"{MODELS_DIR}/all-MiniLM-L6-v2"
LLM_MODEL_PATH = f"{MODELS_DIR}/TinyLlama-1_1B-Chat-v1"

# Initialize components
print("Loading models...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
collection = client.get_collection("complaint_embeddings")

# Load LLM with error handling
print("Loading language model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
    
    try:
        # Attempt GPU loading
        from accelerate import Accelerator
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH,
            device_map="auto",
            torch_dtype="auto"
        )
    except Exception as e:
        print(f"GPU unavailable, using CPU: {e}")
        # CPU fallback
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH,
            device_map=None,
            torch_dtype=torch.float32
        )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.5
    )
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Please verify:")
    print(f"1. The folder {LLM_MODEL_PATH} exists")
    print("2. It contains model files (config.json, pytorch_model.bin etc.)")
    exit(1)

# RAG functions
def retrieve(query: str, k: int = 5) -> list[str]:
    """Retrieve top-k relevant complaint chunks with scores."""
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=['documents', 'distances']
    )
    return results['documents'][0], results['distances'][0]

def generate_answer(question: str) -> tuple:
    """Generate answer with confidence score and sources"""
    chunks, distances = retrieve(question)
    context = "\n".join([f"- {chunk}" for chunk in chunks])
    
    # Calculate confidence (0-1 scale)
    avg_distance = sum(distances)/len(distances) if distances else 1.0
    confidence = min(1.0, 1.2 - avg_distance)  # Convert similarity to confidence
    
    prompt = f"""As a financial analyst, analyze these complaints:
    
    Complaints:
    {context}
    
    Question: {question}
    
    Provide:
    1. A 2-3 sentence summary of key issues
    2. Any notable patterns
    3. Specific customer pain points
    
    Format as:
    Summary: [concise analysis]
    Patterns: [bullet points]
    Pain Points: [bullet points]"""
    
    output = pipe(
        prompt,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    
    full_response = output[0]['generated_text']
    
    # Extract structured response
    summary = ""
    patterns = ""
    pain_points = ""
    
    if "Summary:" in full_response:
        summary = full_response.split("Summary:")[1].split("Patterns:")[0].strip()
    if "Patterns:" in full_response:
        patterns = full_response.split("Patterns:")[1].split("Pain Points:")[0].strip()
    if "Pain Points:" in full_response:
        pain_points = full_response.split("Pain Points:")[1].strip()
    
    return summary, patterns, pain_points, confidence, chunks[:3]  # Return top 3 chunks

# Interactive session
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FINANCIAL COMPLAINT ANALYST".center(60))
    print("="*60)
    print("Type questions about credit cards, loans, or other services")
    print("Enter 'quit' or 'exit' to end".center(60))
    print("="*60)
    
    while True:
        try:
            question = input("\n\x1b[1mYour question:\x1b[0m ").strip()  # Bold text
            
            if question.lower() in ['quit', 'exit']:
                print("\nSession ended. Goodbye!")
                break
                
            if not question:
                print("Please enter a valid question")
                continue
                
            print("\n\x1b[2mAnalyzing complaints...\x1b[0m")  # Dimmed text
            summary, patterns, pain_points, confidence, chunks = generate_answer(question)
            
            # Display results
            print(f"\n\x1b[1mSUMMARY (Confidence: {confidence:.0%}):\x1b[0m")
            print(summary)
            
            if patterns:
                print("\n\x1b[1mPATTERNS:\x1b[0m")
                print(patterns.replace("- ", "• ").replace("* ", "• "))
                
            if pain_points:
                print("\n\x1b[1mPAIN POINTS:\x1b[0m")
                print(pain_points.replace("- ", "• ").replace("* ", "• "))
            
            print("\n\x1b[1mTOP SUPPORTING COMPLAINTS:\x1b[0m")
            for i, chunk in enumerate(chunks, 1):
                print(f"\n[{i}] {chunk[:200]}{'...' if len(chunk) > 200 else ''}")
            
            print("\n" + "-"*60)
            
        except KeyboardInterrupt:
            print("\nSession ended by user")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            continue