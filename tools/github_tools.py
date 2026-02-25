import base64
import os
import requests
import json
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
            action = "updated"
        except ValueError as e:
            if "Resource not found" in str(e):
                sha = None
                action = "created"
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