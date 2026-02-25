import numpy as np
import sqlite3
from core.logging_utils import log_json

class VectorStore:
    """
    Manages semantic embeddings for content, allowing for similarity search.
    It integrates with a `ModelAdapter` to generate embeddings and persists
    vectors and content to a SQLite database via the `Brain`.
    """
    def __init__(self, model_adapter, brain):
        """
        Initializes the VectorStore.

        Args:
            model_adapter: An instance of ModelAdapter for generating embeddings.
            brain: An instance of Brain for database access.
        """
        self.model_adapter = model_adapter
        self.brain = brain # Store brain instance for database access
        self.vectors = []
        self.contents = []
        self._load_from_db()
        log_json("INFO", "vector_store_initialized")

    def _load_from_db(self):
        """
        Loads stored content and embeddings from the database.
        """
        rows = self.brain.db.execute("SELECT content, embedding FROM vector_store_data").fetchall()
        for content, embedding_blob in rows:
            self.contents.append(content)
            self.vectors.append(np.frombuffer(embedding_blob, dtype=np.float32))
        log_json("INFO", "vector_store_loaded_from_db", details={"entries_count": len(self.contents)})

        def add(self, content: str):

            """

            Adds new content to the VectorStore, generates its embedding, and persists it to the database.

    

            Args:

                content (str): The text content to add.

            """

            embedding = self.model_adapter.get_embedding(content)

            self.vectors.append(embedding)

            self.contents.append(content)

            

            # Persist to DB

            self.brain.db.execute(

                "INSERT INTO vector_store_data(content, embedding) VALUES (?, ?)",

                (content, embedding.tobytes())

            )

            self.brain.db.commit()

            log_json("INFO", "vector_store_content_added", details={"content_snippet": content[:50]})

    

        def search(self, query: str, k: int = 5) -> list[str]:

            """

            Searches for the `k` most similar content entries to the given query.

    

            Args:

                query (str): The search query.

                k (int, optional): The number of top similar results to return. Defaults to 5.

    

            Returns:

                list[str]: A list of the `k` most similar content strings.

            """

            if not self.vectors:

                log_json("INFO", "vector_store_search_empty", details={"query_snippet": query[:50]})

                return []

    

            query_embedding = self.model_adapter.get_embedding(query)

            

            similarities = []

            for i, vec in enumerate(self.vectors):

                # Calculate cosine similarity

                similarity = np.dot(query_embedding, vec) / (np.linalg.norm(query_embedding) * np.linalg.norm(vec))

                similarities.append((similarity, i))

            

            similarities.sort(key=lambda x: x[0], reverse=True)

            

            results = []

            for sim, idx in similarities[:k]:

                results.append(self.contents[idx])

            

            log_json("INFO", "vector_store_searched", details={"query_snippet": query[:50], "results_count": len(results)})

            return results

    