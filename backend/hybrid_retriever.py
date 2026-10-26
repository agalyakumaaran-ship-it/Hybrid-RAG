
from backend.vector_store import VectorStore
from backend.knowledge_graph import KnowledgeGraph


class HybridRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.knowledge_graph = KnowledgeGraph()
        self.ready = False

    def build(self, chunks: list[str]) -> None:
        """Build both retrieval systems from the same chunks."""
        if not chunks:
            raise ValueError("No document chunks provided.")

        self.vector_store.build(chunks)
        self.knowledge_graph.build(chunks)
        self.ready = True

    def search(self, question: str) -> dict:
        """Search both systems and return their results."""
        if not self.ready:
            raise ValueError("Build the retriever before searching.")

        vector_results = self.vector_store.search(question)
        graph_results = self.knowledge_graph.search(question)

        return {
            "vector_results": vector_results,
            "graph_results": graph_results,
        }
