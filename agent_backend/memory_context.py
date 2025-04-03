# memory_context.py

import logging
from typing import List, Dict, Any, Optional, Protocol, Tuple
from abc import ABC, abstractmethod

# Assuming state_management is available for context
from state_management import ConversationState, Message

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Interfaces (Protocols) ---

class TextSplitter(Protocol):
    """Defines the interface for text splitting strategies."""
    def split_text(self, text: str) -> List[str]:
        ...

class EmbeddingModel(Protocol):
    """Defines the interface for embedding models."""
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...
    async def embed_query(self, text: str) -> List[float]:
        ...

# --- Placeholder Implementations ---

class SimpleCharacterTextSplitter(TextSplitter):
    """A very basic text splitter based on character count."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")

    def split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
            if start >= len(text): # Avoid infinite loop on zero-length step
                 break
        logging.debug(f"Split text into {len(chunks)} chunks.")
        return chunks

# Placeholder embedding model - Replace with actual implementation
# E.g., using sentence-transformers, openai, etc.
class DummyEmbeddingModel(EmbeddingModel):
    """A dummy embedding model for structural purposes."""
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logging.warning("Using DummyEmbeddingModel. Embeddings will not be meaningful.")
        # Generate dummy embeddings of fixed size, e.g., 5 dimensions
        return [[hash(text + str(i)) % 100 / 100.0 for i in range(5)] for text in texts]

    async def embed_query(self, text: str) -> List[float]:
        logging.warning("Using DummyEmbeddingModel. Embeddings will not be meaningful.")
        return [hash(text + str(i)) % 100 / 100.0 for i in range(5)]

# --- Vector Store Interface and In-Memory Example ---

class VectorStore(ABC):
    """Abstract base class for vector storage and retrieval."""

    @abstractmethod
    async def add_documents(self, texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Add documents and their embeddings to the store."""
        pass

    @abstractmethod
    async def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find k most similar documents to the query embedding."""
        pass

# Simple in-memory vector store for demonstration
class InMemoryVectorStore(VectorStore):
    """A basic in-memory vector store using cosine similarity."""
    def __init__(self):
        self._store: List[Dict[str, Any]] = [] # Stores {'text': str, 'embedding': list[float], 'metadata': dict}
        try:
            import numpy as np
            self._np = np
        except ImportError:
             logging.error("NumPy not found, which is required for InMemoryVectorStore similarity calculation. Install with `pip install numpy`")
             self._np = None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if not self._np:
            raise RuntimeError("NumPy is required for similarity calculation.")
        v1 = self._np.array(vec1)
        v2 = self._np.array(vec2)
        # Handle potential zero vectors
        norm1 = self._np.linalg.norm(v1)
        norm2 = self._np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return self._np.dot(v1, v2) / (norm1 * norm2)

    async def add_documents(self, texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None):
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match.")
        if metadatas and len(texts) != len(metadatas):
             raise ValueError("Number of texts and metadatas must match if metadatas are provided.")

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            self._store.append({
                "text": text,
                "embedding": embeddings[i],
                "metadata": metadata
            })
        logging.info(f"Added {len(texts)} documents to in-memory vector store. Total size: {len(self._store)}")

    async def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self._store:
            return []
        if not self._np:
            logging.error("NumPy not available. Cannot perform similarity search.")
            return []

        similarities = []
        for i, doc in enumerate(self._store):
            similarity = self._cosine_similarity(query_embedding, doc["embedding"])
            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda item: item[1], reverse=True)

        # Get top k results
        results = []
        for i in range(min(k, len(similarities))):
            doc_index, score = similarities[i]
            doc_data = self._store[doc_index]
            results.append((doc_data["text"], score, doc_data["metadata"]))

        logging.debug(f"Similarity search returned {len(results)} results.")
        return results


# --- Memory & Context Manager ---

class MemoryContextManager:
    """
    Orchestrates document processing, embedding, storage, and retrieval.
    """
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        text_splitter: TextSplitter
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.text_splitter = text_splitter
        logging.info("MemoryContextManager initialized.")

    async def add_document_to_memory(self, document_text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Processes a single document, embeds it, and adds it to the vector store.

        Args:
            document_text: The full text of the document.
            metadata: Optional metadata associated with the document (e.g., source filename).
        """
        logging.info(f"Adding document to memory (first 100 chars): {document_text[:100]}...")
        if not document_text:
            logging.warning("Attempted to add an empty document to memory.")
            return

        # 1. Split document into chunks
        chunks = self.text_splitter.split_text(document_text)
        if not chunks:
            logging.warning("Text splitting resulted in no chunks.")
            return

        # 2. Embed chunks
        try:
            embeddings = await self.embedding_model.embed_documents(chunks)
        except Exception as e:
            logging.error(f"Failed to embed document chunks: {e}", exc_info=True)
            # Decide whether to raise or just log and return
            return

        # 3. Add to vector store
        # Create metadata for each chunk, including original document info if provided
        chunk_metadatas = []
        base_metadata = metadata or {}
        for i, chunk in enumerate(chunks):
             chunk_meta = base_metadata.copy()
             chunk_meta['chunk_index'] = i
             chunk_meta['chunk_text_preview'] = chunk[:50] # Add a preview for context
             chunk_metadatas.append(chunk_meta)

        try:
            await self.vector_store.add_documents(chunks, embeddings, chunk_metadatas)
            logging.info(f"Successfully added {len(chunks)} chunks from document to vector store.")
        except Exception as e:
             logging.error(f"Failed to add document chunks to vector store: {e}", exc_info=True)


    async def retrieve_relevant_context(self, query: str, k: int = 4) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Embeds a query and retrieves the most relevant document chunks from memory.

        Args:
            query: The query string to search for.
            k: The number of relevant chunks to retrieve.

        Returns:
            A list of tuples, each containing (document_chunk_text, similarity_score, metadata).
        """
        logging.debug(f"Retrieving relevant context for query: {query}")
        try:
            query_embedding = await self.embedding_model.embed_query(query)
            results = await self.vector_store.similarity_search(query_embedding, k=k)
            logging.info(f"Retrieved {len(results)} relevant context chunks.")
            return results
        except Exception as e:
            logging.error(f"Failed to retrieve relevant context: {e}", exc_info=True)
            return [] # Return empty list on failure

    def enhance_state_with_context(self, state: ConversationState, context_results: List[Tuple[str, float, Dict[str, Any]]]):
        """
        Adds retrieved context to the short-term memory or another part of the state.
        (This is a simple example; context injection strategy can vary).
        """
        formatted_context = "\n\n".join([f"Context Chunk (Score: {score:.4f}):\n{text}" for text, score, meta in context_results])
        state.update_memory("retrieved_context", formatted_context)
        logging.debug("Added retrieved context to conversation state memory.")


# Example Usage (Illustrative)
async def main():
    # Setup components (using dummies/in-memory)
    splitter = SimpleCharacterTextSplitter()
    embedder = DummyEmbeddingModel()
    store = InMemoryVectorStore()
    memory_manager = MemoryContextManager(store, embedder, splitter)

    # Add some documents
    doc1 = "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends the LangChain Expression Language."
    doc2 = "Memory in conversational AI refers to the ability of the system to retain and recall information from previous interactions."
    doc3 = "Retrieval-Augmented Generation (RAG) combines retrieval systems with generative models to improve response factuality."

    await memory_manager.add_document_to_memory(doc1, {"source": "doc1.txt"})
    await memory_manager.add_document_to_memory(doc2, {"source": "doc2.txt"})
    await memory_manager.add_document_to_memory(doc3, {"source": "doc3.txt"})

    # Retrieve context
    query = "What is LangGraph used for?"
    relevant_chunks = await memory_manager.retrieve_relevant_context(query, k=2)

    print(f"\n--- Context retrieved for query: '{query}' ---")
    for text, score, meta in relevant_chunks:
        print(f"Score: {score:.4f}")
        print(f"Metadata: {meta}")
        print(f"Text: {text}")
        print("-" * 10)

    # Enhance state
    test_state = ConversationState(conversation_id="mem-test-1")
    memory_manager.enhance_state_with_context(test_state, relevant_chunks)
    print("\n--- State after context enhancement ---")
    print(f"Memory['retrieved_context']: {test_state.get_from_memory('retrieved_context')}")


if __name__ == "__main__":
    import asyncio
    try:
        import numpy # Check if numpy is installed for the example
        asyncio.run(main())
    except ImportError:
        logging.warning("NumPy not found. Skipping MemoryContextManager example. Install with `pip install numpy`")
    except Exception as e:
        logging.error(f"An error occurred in the memory_context example: {e}", exc_info=True) 