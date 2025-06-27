import os
import gradio as gr
import fitz  # PyMuPDF
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "mistralai/mistral-7b-instruct"  # Change as needed

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
dim = 384
index = faiss.IndexFlatL2(dim)
stored_chunks = []

def chunk_pdf_text(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text() for page in doc])
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def store_embeddings(chunks):
    embeddings = embed_model.encode(chunks)
    index.add(embeddings)
    stored_chunks.extend(chunks)

def search_similar_chunks(query, k=5):
    query_embedding = embed_model.encode([query])
    D, I = index.search(query_embedding, k)
    return [stored_chunks[i] for i in I[0]]

def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://your-app.com",  # Replace with your domain if needed
        "Content-Type": "application/json"
    }
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for academic research."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Error: " + response.text

def process_file_and_query(file, query):
    if file is None or query.strip() == "":
        return "Upload a PDF and enter a question."
    chunks = chunk_pdf_text(file.name)
    store_embeddings(chunks)
    top_chunks = search_similar_chunks(query)
    return generate_answer(query, top_chunks)

gr.Interface(
    fn=process_file_and_query,
    inputs=[gr.File(label="Upload PDF", type="filepath"), gr.Textbox(label="Ask a question")],
    outputs="text",
    title="📚 Academic RAG Assistant",
    description="Upload a research paper PDF and ask questions using OpenRouter LLM."
).launch()
