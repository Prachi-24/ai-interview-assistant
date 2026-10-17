from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Create example resume chunks
chunks = [
    "I developed a driver drowsiness detection system using Python and OpenCV.",
    "I built a news application using React and JavaScript.",
    "I created a hospital appointment booking frontend using React.",
    "I have experience with machine learning and data science."
]

# Step 3: Convert the chunks into embeddings
embeddings = model.encode(chunks)

# FAISS requires 32-bit floating-point vectors
embeddings = np.asarray(embeddings, dtype="float32")

# Step 4: Create a FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Step 5: Store the embeddings in FAISS
index.add(embeddings)

print("Total chunks stored:", index.ntotal)

# Step 6: Ask a question
question = "Which project did I develop using machine learning?"

# Convert the question into an embedding
question_embedding = model.encode([question])

question_embedding = np.asarray(
    question_embedding,
    dtype="float32"
)

# Step 7: Search for the two closest chunks
distances, indices = index.search(question_embedding, k=2)

# Step 8: Display the results
print("\nQuestion:", question)
print("\nRelevant chunks:")

for i in indices[0]:
    print("-", chunks[i])