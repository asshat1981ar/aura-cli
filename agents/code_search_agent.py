from core.logging_utils import log_json


class CodeSearchAgent:
    """
    An agent capable of searching the project's vector store for specific
    code snippets, patterns, or similar implementations.
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def query(self, query_string: str, limit: int = 5, min_score: float = 0.5):
        """
        Queries the vector store for relevant code context.
        """
        if not self.vector_store:
            log_json("WARN", "code_search_no_vector_store", details={"query": query_string})
            return []

        cleaned_query = self._preprocess_query(query_string)
        log_json("INFO", "code_search_query_start", details={"query": cleaned_query})

        try:
            results = self.vector_store.search(cleaned_query, top_k=limit)
        except Exception as e:
            log_json("ERROR", "code_search_query_failed", details={"error": str(e)})
            return []

        valid_results = self.validate_results(results, min_score)
        return self._present_results(valid_results)

    def run(self, input_data: dict) -> dict:
        """Uniform execution interface for the orchestrator loop."""
        query_str = input_data.get("query", "")
        limit = input_data.get("limit", 5)
        min_score = input_data.get("min_score", 0.5)

        results = self.query(query_str, limit=limit, min_score=min_score)
        return {"status": "success", "query": query_str, "results": results}

    def refine_query(self, failure_data: str) -> str:
        """
        Generates a refined query based on a failure context.
        """
        return f"related code for {failure_data}"

    def validate_results(self, results: list, min_score: float) -> list:
        """
        Filters out low-confidence results from the search.
        """
        if not results:
            return []

        valid_snippets = []
        for result in results:
            score = result.get("score", 0)
            if score >= min_score:
                valid_snippets.append(result)

        return valid_snippets

    def _preprocess_query(self, query_string: str) -> str:
        """
        Cleans and formats the incoming search query.
        """
        return query_string.strip()

    def _present_results(self, results: list) -> list:
        """
        Formats the results for consumption by other agents.
        """
        formatted = []
        for res in results:
            formatted.append({"file": res.get("metadata", {}).get("file", "unknown"), "content": res.get("content", ""), "score": res.get("score", 0.0)})
        return formatted
