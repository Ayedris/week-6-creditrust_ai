# src/app.py
import gradio as gr
from rag_core import FinancialComplaintAnalyzer
import time

# Initialize system
analyzer = FinancialComplaintAnalyzer(
    embedding_model_path="data/all-MiniLM-L6-v2",
    llm_model_path="data/TinyLlama-1_1B-Chat-v1",
    vector_store_path="vector_store"
)

def respond(message, history):
    try:
        yield "Analyzing complaints..."
        response = analyzer.generate_answer(message)
        
        # Format confidence
        confidence_color = "green" if response["confidence"] > 0.7 else "orange" if response["confidence"] > 0.4 else "red"
        confidence_badge = f"<span style='color:{confidence_color}'>" \
                         f"{'High' if response['confidence'] > 0.7 else 'Medium' if response['confidence'] > 0.4 else 'Low'} " \
                         f"Confidence ({response['confidence']:.0%})</span>"
        
        # Format sources
        sources = "\n".join(
            f"📌 <b>Source {i+1}:</b> {src[:200]}..." 
            for i, src in enumerate(response['sources'])
        )
        
        # Stream output
        full_response = f"""
{response['answer']}

---
<b>Confidence</b>: {confidence_badge}

<b>Evidence</b>:
{sources}
"""
        for i in range(0, len(full_response), 10):
            yield full_response[:i+10]
            time.sleep(0.02)
            
    except Exception as e:
        yield f"Error: {str(e)}"

# Create interface
demo = gr.ChatInterface(
    respond,
    title="CrediTrust Complaint Analysis",
    examples=[
        "Top credit card complaints?",
        "Common loan application issues?",
        "Mobile banking complaints?"
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(server_port=7860)