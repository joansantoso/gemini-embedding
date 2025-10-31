from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)

text_to_embed = "What is the meaning of life?"
print(text_to_embed)
print('----------------------------------------------')
# Generate embedding for a single text
result = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents=text_to_embed,
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)
#print (result.embeddings)
print(f"Embedding : {result.embeddings[0].values[:10]}... with vector length: {len(result.embeddings[0].values)}")
print()

result = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents=text_to_embed,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
)
#print (result.embeddings)
print(f"Embedding : {result.embeddings[0].values[:10]}... with vector length: {len(result.embeddings[0].values)}")
print()


# Generate embeddings for multiple texts (batch embedding) and config with custom dimensionality 
texts = [
    "What is the meaning of life?",
    "What is the purpose of existence?",
    "How do I bake a cake?"
]
results = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents=texts,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768)
)
for i, emb in enumerate(results.embeddings):
    print(f"Embedding for '{texts[i]}': {emb.values[:10]}... with vector length: {len(emb.values)}")

"""
# Task Type           | Description                                                          | Examples
# --------------------|----------------------------------------------------------------------|-------------------------------------------------------
# SEMANTIC_SIMILARITY | Embeddings optimized to assess text similarity.                      | Recommendation systems, duplicate detection
# CLASSIFICATION      | Embeddings optimized to classify texts according to preset labels.   | Sentiment analysis, spam detection
# CLUSTERING          | Embeddings optimized to cluster texts based on their similarities.   | Document organization, market research, 
#                     |                                                                      | anomaly detection
# RETRIEVAL_DOCUMENT  | Embeddings optimized for document search.                            | Indexing articles, books, or web pages for search.
# RETRIEVAL_QUERY     | Embeddings optimized for general search queries. Use                 | Custom search
#                     | RETRIEVAL_QUERY for queries; RETRIEVAL_DOCUMENT for documents to be  |
#                     | retrieved.                                                           |
# CODE_RETRIEVAL_QUERY| Embeddings optimized for retrieval of code blocks based on natural   | Code suggestions and search
#                     | language queries. Use CODE_RETRIEVAL_QUERY for queries;              |
#                     | RETRIEVAL_DOCUMENT for code blocks to be retrieved.                  |
# QUESTION_ANSWERING  | Embeddings for questions in a question-answering system, optimized   | Chatbox
#                     | for finding documents that answer the question. Use                  |
#                     | YoutubeING for questions; RETRIEVAL_DOCUMENT for documents           |
#                     | to be retrieved.                                                     |
# FACT_VERIFICATION   | Embeddings for statements that need to be verified, optimized for    | Automated fact-checking systems
#                     | retrieving documents that contain evidence supporting or refuting    |
#                     | the statement. Use FACT_VERIFICATION for the target text;            |
#                     | RETRIEVAL_DOCUMENT for documents to be retrieved.                    |
"""