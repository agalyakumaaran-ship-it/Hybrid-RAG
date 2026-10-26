
import re
import networkx as nx


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def build(self, chunks: list[str]) -> None:
        """Build a graph of entities found in document chunks."""
        self.graph.clear()

        # Common words that should not become graph entities.
        stop_words = {
            "this", "that", "these", "those", "there",
            "is", "are", "was", "were", "the", "a", "an",
            "and", "or", "but", "of", "to", "in", "on",
            "for", "with", "from", "by", "as", "it",
            "document", "system", "service", "services",
            "feature", "features", "including", "overall",
            "which", "what", "where", "when", "who", "why",
            "how", "their", "they", "using", "used",
        }

        # Known technical terms and service names.
        known_entities = [
            "Media Processor Service",
            "Analytics & Events Service",
            "Streaming / Media Delivery Service",
            "Recommendation & Personalization Service",
            "Social & Activity Service",
            "Subscription & Billing Service",
            "FFmpeg",
            "ClickHouse",
            "PostgreSQL",
            "Redis",
            "Elasticsearch",
            "OpenSearch",
            "S3",
            "CloudFront",
            "Fastly",
            "Cloudflare",
            "Docker",
            "Kubernetes",
            "Vault",
            "GitHub Actions",
            "GitLab CI",
            "Kafka",
            "NATS",
            "RabbitMQ",
            "gRPC",
            "Nginx",
            "Envoy",
            "Kong",
            "REST",
            "HLS",
            "API Gateway",
            "CDN",
            "GKE",
            "EKS",
            "AKS",
            "PostgreSQL",
            "Spotify",
        ]

        for chunk in chunks:
            entities = []

            # Match known terms, ignoring capitalization differences.
            for term in known_entities:
                pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
                if re.search(pattern, chunk, flags=re.IGNORECASE):
                    entities.append(term)

            # Extract additional capitalized names and technical terms.
            candidates = re.findall(
                r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*\b",
                chunk,
            )

            for term in candidates:
                if (
                    len(term) > 1
                    and term.lower() not in stop_words
                    and term not in entities
                ):
                    entities.append(term)

            # Remove duplicates while preserving order.
            entities = list(dict.fromkeys(entities))

            for entity in entities:
                self.graph.add_node(entity)

            # Connect entities appearing in the same chunk.
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    self.graph.add_edge(entities[i], entities[j])

    def search(self, question: str, top_k: int = 5) -> list[str]:
        """Find graph entities related to a user's question."""
        question_words = set(
            re.findall(r"[a-zA-Z0-9]+", question.lower())
        )

        stop_words = {
            "which", "what", "where", "when", "who", "why", "how",
            "is", "are", "was", "were", "the", "a", "an", "and",
            "or", "of", "to", "in", "on", "for", "with", "from",
            "mentioned", "document", "about", "this", "that",
            "please", "tell", "me", "list", "show", "give",
        }
        question_words -= stop_words

        print("GRAPH NODE COUNT:", self.graph.number_of_nodes())
        print("QUESTION WORDS:", question_words)

        # Rank nodes by matching question terms and graph connectivity.
        scored_nodes = []

        for node in self.graph.nodes:
            node_words = set(
                re.findall(r"[a-zA-Z0-9]+", node.lower())
            )
            overlap = question_words & node_words

            if overlap:
                score = len(overlap)
                score += min(self.graph.degree(node) / 10, 2)
                scored_nodes.append((node, score))

        scored_nodes.sort(key=lambda item: item[1], reverse=True)

        results = []

        for node, _ in scored_nodes:
            if node not in results:
                results.append(node)

            # Add connected entities to provide context.
            for neighbor in self.graph.neighbors(node):
                if neighbor not in results:
                    results.append(neighbor)

            if len(results) >= top_k:
                break

        return results[:top_k]
