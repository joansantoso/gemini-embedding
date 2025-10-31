from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import faiss
import numpy as np
from docs import DocumentDB as DBData
load_dotenv()
DIMENSION = 768
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

collection = DBData()

results = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents=collection.docs_to_embed,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=DIMENSION)
)

document_embeddings = [emb.values for emb in results.embeddings]
document_embeddings_np = np.array(document_embeddings).astype('float32')

index = faiss.IndexFlatL2(DIMENSION)

# Add the document embeddings to the index
index.add(document_embeddings_np)
print(f"FAISS index created with {index.ntotal} vectors.")
faiss.write_index(index, "vector.db")

def faiss_search(query: str, k: int = 2) -> str:
    """
    Searches the FAISS index for the most similar documents to a given query.
    """
    # 1. Embed the user's query
    query_embedding_response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=DIMENSION)
    )
    query_embedding = np.array([query_embedding_response.embeddings[0].values]).astype('float32')

    # 2. Perform the similarity search in the FAISS index
    distances, indices = index.search(query_embedding, k)
    
    # 3. Retrieve the corresponding documents and format the result
    retrieved_docs = []
    for i in indices[0]:
        retrieved_docs.append(collection.index_to_text_map[i])

    result_text = "Retrieved documents:\n" + "\n".join(
        [f"- {doc}" for doc in retrieved_docs]
    )
    
    return result_text


if __name__ == "__main__":
    print(faiss_search("ADK"))

