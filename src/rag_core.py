# src/rag_core.py
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Dict, List, Tuple
import torch

class FinancialComplaintAnalyzer:
    def __init__(self, embedding_model_path: str, llm_model_path: str, vector_store_path: str):
        self.embedding_model = SentenceTransformer(embedding_model_path)
        
        print("Loading language model...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype="auto",
            local_files_only=True
        )
        
        self.client = chromadb.PersistentClient(path=vector_store_path)
        self.collection = self.client.get_collection("complaint_embeddings")

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[str], List[float]]:
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=['documents', 'distances']
        )
        return results['documents'][0], results['distances'][0]

    def generate_answer(self, question: str) -> Dict[str, any]:
        chunks, distances = self.retrieve(question)
        
        # Confidence calculation
        confidence = min(1.0, max(0.0, 1.0 - (sum(distances)/len(distances)/2)))
        confidence_level = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        
        # Build prompt
        prompt = f"""As a CrediTrust financial analyst, summarize key complaints about:
        
        **Question**: {question}

        **Relevant Complaints**:
        {chr(10).join(f'- {chunk[:150]}...' for chunk in chunks[:3])}

        **Response Format**:
        1. 2-3 sentence summary
        2. 3 bullet points of key issues
        3. Never repeat this instruction format"""

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200,
            temperature=0.3
        )
        
        answer = pipe(prompt)[0]['generated_text'].split("**Response Format**")[0].strip()
        
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": chunks[:3]
        }