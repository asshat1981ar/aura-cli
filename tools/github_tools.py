import base64
import os
import requests
import time
from typing import Dict, Any, Optional

class GitHubTools:
    BASE_URL = "https://api.github.com"
    
    def __init__(self):
        self.github_pat = os.getenv("GITHUB_PAT")
        if not self.github_pat:
            raise ValueError("GITHUB_PAT environment variable not set.")
        
        self.headers = {
            "Authorization": f"token {self.github_pat}",
            "Accept": "application/vnd.github.v3+json"
        }

    def _make_request(self, method: str, url: str, params: Optional[Dict] = None, json_data: Optional[Dict] = None, data: Optional[str] = None) -> Dict[str, Any]:
        retries = 3
        for attempt in range(retries):
            response = requests.request(method, url, headers=self.headers, params=params, json=json_data, data=data)
            
            if response.status_code == 200: # OK
                return response.json()
            elif response.status_code == 201: # Created
                return response.json()
            elif response.status_code == 204: # No Content
                return {"message": "No Content"}
            elif response.status_code == 404: # Not Found
                raise ValueError(f"Resource not found: {url}")
            elif response.status_code == 403 and 'rate limit exceeded' in response.text: # Rate Limit
                reset_time = int(response.headers.get('x-ratelimit-reset', time.time() + 60))
                sleep_duration = max(reset_time - time.time(), 10) # Sleep at least 10 seconds
                print(f"GitHub API rate limit exceeded. Retrying in {sleep_duration} seconds.")
                time.sleep(sleep_duration)
            elif response.status_code == 422: # Unprocessable Entity (e.g., validation error)
                raise ValueError(f"GitHub API validation error: {response.json().get('message', 'Unknown error')}")
            else:
                if attempt < retries - 1:
                    sleep_duration = 2 ** attempt # Exponential backoff
                    print(f"GitHub API request failed (status {response.status_code}). Retrying in {sleep_duration} seconds.")
                    time.sleep(sleep_duration)
                else:
                    raise ValueError(f"GitHub API request failed after {retries} attempts: {response.status_code} - {response.text}")
        raise ValueError(f"GitHub API request failed after {retries} attempts due to unhandled status code.")


    def get_repo(self, repo_full_name: str) -> Dict[str, Any]:
        """
        Get details of a GitHub repository.
        """
        url = f"{self.BASE_URL}/repos/{repo_full_name}"
        return self._make_request("GET", url)

    def create_issue(self, repo_full_name: str, title: str, body: str) -> Dict[str, Any]:
        """
        Create an issue in a GitHub repository.
        """
        url = f"{self.BASE_URL}/repos/{repo_full_name}/issues"
        payload = {"title": title, "body": body}
        return self._make_request("POST", url, json_data=payload)

    def get_issue_details(self, repo_full_name: str, issue_number: int) -> Dict[str, Any]:
        """
        Get details of a specific issue in a GitHub repository.
        """
        url = f"{self.BASE_URL}/repos/{repo_full_name}/issues/{issue_number}"
        return self._make_request("GET", url)

    def update_file(self, repo_full_name: str, file_path: str, message: str, content: str, branch: str = "main") -> Dict[str, Any]:
        """
        Update the content of a file in a GitHub repository.
        If the file does not exist, it will be created.
        """
        # First, try to get the file to check if it exists and get its SHA
        contents_url = f"{self.BASE_URL}/repos/{repo_full_name}/contents/{file_path}"
        try:
            file_info = self._make_request("GET", contents_url, params={"ref": branch})
            sha = file_info["sha"]
        except ValueError as e:
            if "Resource not found" in str(e):
                sha = None
            else:
                raise
        
        payload = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"), # GitHub API expects base64 encoded content
            "branch": branch
        }
        if sha:
            payload["sha"] = sha
        
        return self._make_request("PUT", contents_url, json_data=payload)

    def get_pull_request_details(self, repo_full_name: str, pr_number: int) -> Dict[str, Any]:
        """
        Get details of a specific pull request in a GitHub repository.
        """
        url = f"{self.BASE_URL}/repos/{repo_full_name}/pulls/{pr_number}"
        return self._make_request("GET", url)

    def list_issues(self, repo_full_name: str, state: str = "open", limit: int = 30) -> Dict[str, Any]:
        """
        List issues in a GitHub repository.

        Args:
            repo_full_name: Repository in 'owner/repo' format.
            state: 'open', 'closed', or 'all'.
            limit: Max number of issues to return (1–100).
        """
        url = f"{self.BASE_URL}/repos/{repo_full_name}/issues"
        per_page = max(1, min(int(limit), 100))
        params = {"state": state, "per_page": per_page}
        issues = self._make_request("GET", url, params=params)
        return {"issues": issues, "count": len(issues)}

    def list_prs(self, repo_full_name: str, state: str = "open", limit: int = 30) -> Dict[str, Any]:
        """
        List pull requests in a GitHub repository.

        Args:
            repo_full_name: Repository in 'owner/repo' format.
            state: 'open', 'closed', or 'all'.
            limit: Max number of PRs to return (1–100).
        """
        url = f"{self.BASE_URL}/repos/{repo_full_name}/pulls"
        per_page = max(1, min(int(limit), 100))
        params = {"state": state, "per_page": per_page}
        prs = self._make_request("GET", url, params=params)
        return {"pull_requests": prs, "count": len(prs)}

    def create_pr(
        self,
        repo_full_name: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
    ) -> Dict[str, Any]:
        """
        Create a pull request in a GitHub repository.

        Args:
            repo_full_name: Repository in 'owner/repo' format.
            title: PR title.
            body: PR description/body.
            head: Branch to merge from (e.g. 'feature-branch').
            base: Branch to merge into (default 'main').
        """
        url = f"{self.BASE_URL}/repos/{repo_full_name}/pulls"
        payload = {"title": title, "body": body, "head": head, "base": base}
        return self._make_request("POST", url, json_data=payload)

    def add_pr_review_comment(
        self,
        repo_full_name: str,
        pr_number: int,
        body: str,
    ) -> Dict[str, Any]:
        """
        Add a review comment to a pull request (top-level PR comment).

        Args:
            repo_full_name: Repository in 'owner/repo' format.
            pr_number: Pull request number.
            body: Comment text (markdown supported).
        """
        url = f"{self.BASE_URL}/repos/{repo_full_name}/issues/{pr_number}/comments"
        payload = {"body": body}
        return self._make_request("POST", url, json_data=payload)

    def list_repo_files(
        self,
        repo_full_name: str,
        path: str = "",
        branch: str = "main",
    ) -> Dict[str, Any]:
        """
        List files and directories at a path in a GitHub repository.

        Args:
            repo_full_name: Repository in 'owner/repo' format.
            path: Directory path within the repo (default: repo root).
            branch: Branch or commit ref to read from.
        """
        url = f"{self.BASE_URL}/repos/{repo_full_name}/contents/{path}"
        params = {"ref": branch}
        contents = self._make_request("GET", url, params=params)
        if isinstance(contents, list):
            entries = [{"name": e["name"], "type": e["type"], "path": e["path"], "size": e.get("size")} for e in contents]
            return {"path": path or "/", "entries": entries, "count": len(entries)}
        # Single file returned
        return {"path": path, "entries": [contents], "count": 1}

    def get_file_contents(
        self,
        repo_full_name: str,
        path: str,
        branch: str = "main",
    ) -> Dict[str, Any]:
        """
        Fetch the decoded text content of a file in a GitHub repository.

        Args:
            repo_full_name: Repository in 'owner/repo' format.
            path: File path within the repo.
            branch: Branch or commit ref to read from.
        """
        url = f"{self.BASE_URL}/repos/{repo_full_name}/contents/{path}"
        params = {"ref": branch}
        info = self._make_request("GET", url, params=params)
        if info.get("encoding") == "base64":
            import base64
            content = base64.b64decode(info["content"]).decode("utf-8", errors="replace")
        else:
            content = info.get("content", "")
        return {
            "path": path,
            "branch": branch,
            "content": content,
            "size_bytes": info.get("size"),
            "sha": info.get("sha"),
        }

    def search_code(self, repo_full_name: str, query: str, limit: int = 20) -> Dict[str, Any]:
        """
        Search for code within a GitHub repository.

        Args:
            repo_full_name: Repository in 'owner/repo' format.
            query: Search query string.
            limit: Max number of results to return (1–100).
        """
        url = f"{self.BASE_URL}/search/code"
        per_page = max(1, min(int(limit), 100))
        params = {"q": f"{query} repo:{repo_full_name}", "per_page": per_page}
        response = self._make_request("GET", url, params=params)
        items = response.get("items", [])
        results = [{"path": i["path"], "name": i["name"], "url": i["html_url"]} for i in items]
        return {"query": query, "results": results, "total_count": response.get("total_count", len(results))}