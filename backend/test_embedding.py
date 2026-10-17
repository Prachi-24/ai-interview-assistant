from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example sentences
sentences = [
    "I know Python programming.",
    "I have experience writing Python code.",
    "I enjoy cooking food."
]

# Convert sentences into embeddings
embeddings = model.encode(sentences)

# Display the shape of the embeddings
print("Embedding shape:", embeddings.shape)

# Display the first embedding
print("First embedding:", embeddings[0])