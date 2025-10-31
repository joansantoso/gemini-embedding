class DocumentDB:
    def __init__(self):
        self.docs_to_embed = [
            "Gemini is a powerful family of large language models.",
            "The Agent Development Kit (ADK) simplifies building multi-agent applications.",
            "FAISS is an open-source library for efficient similarity search.",
            "A vector database stores embeddings and allows for fast similarity queries.",
            "Retrieval Augmented Generation (RAG) improves LLM responses with external knowledge.",
            "The Eiffel Tower is in Paris, France."
        ]
        self.index_to_text_map = {i: text for i, text in enumerate(self.docs_to_embed)}