from google.adk.agents import LlmAgent
import faiss
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os 
import numpy as np
import docs
from docs import DocumentDB as DBData

collection = DBData()

index = faiss.read_index("vector.db")
load_dotenv()
DIMENSION = 768
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

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


# Define the agent with the new tool
agent = LlmAgent(
    model="gemini-2.0-flash",
    name="rag_agent",
    tools=[faiss_search],
    instruction="You are a helpful assistant. If the user asks a question that requires knowledge about large language models, vector databases, or AI development, use the 'faiss_search' tool to find relevant information before answering. If the user's question is general, you can answer directly."
)

root_agent = agent

if __name__ == "__main__":
    collection = docs.CollectionDocument()