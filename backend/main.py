from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymupdf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# Create the FastAPI application
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Store the processed resume information
resume_chunks = []
faiss_index = None


# Request format for questions
class QuestionRequest(BaseModel):
    question: str


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "AI Interview Assistant Backend is running!"
    }


# Upload and process a resume
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    global resume_chunks, faiss_index

    # Check the file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    # Read the uploaded PDF
    pdf_bytes = await file.read()

    try:
        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        text = "\n".join(
            page.get_text() for page in document
        )

        document.close()

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF file."
        )

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text found in the PDF."
        )

    # Split the text into overlapping chunks
    chunk_size = 300
    overlap = 50

    new_chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size]
        new_chunks.append(chunk)
        start += chunk_size - overlap

    # Generate embeddings for the chunks
    embeddings = model.encode(new_chunks)

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Create a FAISS index
    dimension = embeddings.shape[1]
    new_index = faiss.IndexFlatL2(dimension)

    # Store the embeddings
    new_index.add(embeddings)

    # Save the processed resume in memory
    resume_chunks = new_chunks
    faiss_index = new_index

    return {
        "message": "Resume processed successfully!",
        "filename": file.filename,
        "total_chunks": len(resume_chunks)
    }


# Search the resume using a question
@app.post("/ask")
def ask_question(request: QuestionRequest):

    # Check whether a resume has been uploaded
    if faiss_index is None or not resume_chunks:
        raise HTTPException(
            status_code=400,
            detail="Please upload a resume first."
        )

    # Get the user's question
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    # Convert the question into an embedding
    question_embedding = model.encode([question])

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    # Retrieve the two most relevant chunks
    k = min(5, len(resume_chunks))

    distances, indices = faiss_index.search(
        question_embedding,
        k
    )

    # Collect the retrieved text
    retrieved_chunks = [
        resume_chunks[i]
        for i in indices[0]
        if i >= 0
    ]

    # Combine the retrieved chunks into context
    context = "\n\n".join(retrieved_chunks)

    # Prepare the prompt for Ollama
    prompt = f"""
You are an AI interview assistant.

Answer the user's question using the resume context provided below.

Rules:
- Use the resume context as your primary source.
- Do not invent skills, qualifications, or experience.
- If the answer is not available in the context, say that the resume
  does not provide enough information.
- Give a clear and concise answer.

Resume context:
{context}

User question:
{question}

Answer:
"""

    # Send the prompt to Ollama
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        answer = response.json()["response"]

    except requests.RequestException:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not connect to Ollama. "
                "Please check that Ollama is running."
            )
        )

    # Return the generated answer
    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks
    }