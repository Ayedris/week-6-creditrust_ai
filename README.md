CrediTrust AI Complaint Analysis Chatbot 

This project is part of the 10 Academy Artificial Intelligence Mastery Program - Week 6 Challenge.

Project Overview
We are building an AI-powered internal chatbot that transforms customer complaint data into actionable insights using 
Retrieval-Augmented Generation (RAG).

The chatbot will allow product, compliance, and support teams to:
- Ask plain-English questions about customer complaints.
- Retrieve relevant complaints from a vector database.
- Generate concise answers using an LLM.

Business Objective
CrediTrust Financial wants to:
- Reduce the time needed to detect complaint trends.
- Enable non-technical teams to get insights easily.
- Shift from reactive to proactive complaint handling.

Project Structure

week-6-creditrust_ai/
├── app.py 
├── data/ Dataset folder
├── notebooks/ Jupyter Notebooks for EDA & preprocessing
├── src/ Source Python modules for RAG pipeline
├── vector_store/ Saved vector database (FAISS/ChromaDB)
├── requirements.txt ( Python dependencies)
├── README.md  Project documentation
└── .gitignore  

Main Technologies
- Python
- FAISS or ChromaDB (Vector DB)
- Hugging Face Transformers (LLMs & Embeddings)
- Gradio or Streamlit (UI)
- Pandas, Matplotlib (EDA)
- LangChain (Optional)
