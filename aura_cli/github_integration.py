"""GitHub App integration for AURA.

Handles webhooks, PR creation, and comment commands.
"""

from __future__ import annotations

import hmac
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime

from core.logging_utils import log_json

# Optional GitHub SDK
try:
    from github import Github, Auth
    from github.Repository import Repository
    from github.PullRequest import PullRequest
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    Github = None
    Repository = None
    PullRequest = None


class GitHubIntegrationError(Exception):
    """GitHub integration error."""
    pass


class GitHubApp:
    """GitHub App integration handler."""
    
    def __init__(
        self,
        app_id: str,
        private_key: str,
        webhook_secret: Optional[str] = None,
    ):
        self.app_id = app_id
        self.private_key = private_key
        self.webhook_secret = webhook_secret
        
        # App-level client (for app operations) - lazy initialization
        self._app_auth = None
        self._app_client = None
    
    def _ensure_client(self):
        """Ensure GitHub client is initialized."""
        if not GITHUB_AVAILABLE:
            raise ImportError(
                "GitHub integration requires PyGithub. "
                "Install with: pip install PyGithub"
            )
        if self._app_client is None:
            self._app_auth = Auth.AppAuth(self.app_id, self.private_key)
            self._app_client = Github(auth=self._app_auth)
    
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify GitHub webhook signature."""
        if not self.webhook_secret:
            log_json("WARN", "github_webhook_no_secret")
            return True
        
        expected = hmac.new(
            self.webhook_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()
        
        return hmac.compare_digest(f"sha256={expected}", signature)
    
    def get_installation_client(self, installation_id: int):
        """Get GitHub client for specific installation."""
        self._ensure_client()
        auth = self._app_auth.get_installation_auth(installation_id)
        return Github(auth=auth)
    
    def handle_webhook(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GitHub webhook event."""
        handlers = {
            "pull_request": self._handle_pull_request,
            "pull_request_review": self._handle_pull_request_review,
            "pull_request_review_comment": self._handle_pr_comment,
            "issue_comment": self._handle_issue_comment,
            "push": self._handle_push,
            "installation": self._handle_installation,
        }
        
        handler = handlers.get(event_type)
        if handler:
            return handler(payload)
        
        log_json("DEBUG", "github_unhandled_event", {"event": event_type})
        return {"status": "ignored", "event": event_type}
    
    def _handle_pull_request(self, payload: Dict) -> Dict:
        """Handle pull request events."""
        action = payload.get("action")
        pr_data = payload.get("pull_request", {})
        repo_data = payload.get("repository", {})
        
        log_json("INFO", "github_pr_event", {
            "action": action,
            "pr_number": pr_data.get("number"),
            "repo": repo_data.get("full_name"),
        })
        
        # Handle opened PR
        if action == "opened":
            return self._on_pr_opened(pr_data, repo_data)
        
        # Handle synchronize (new commits pushed)
        if action == "synchronize":
            return self._on_pr_synchronize(pr_data, repo_data)
        
        return {"status": "processed", "action": action}
    
    def _handle_pull_request_review(self, payload: Dict) -> Dict:
        """Handle pull request review events."""
        action = payload.get("action")
        review = payload.get("review", {})
        pr_data = payload.get("pull_request", {})
        
        log_json("INFO", "github_pr_review", {
            "action": action,
            "state": review.get("state"),
            "pr_number": pr_data.get("number"),
        })
        
        return {"status": "processed"}
    
    def _handle_pr_comment(self, payload: Dict) -> Dict:
        """Handle PR comment events."""
        action = payload.get("action")
        comment = payload.get("comment", {})
        pr_data = payload.get("pull_request", {})
        
        if action == "created":
            body = comment.get("body", "")
            return self._process_command(body, pr_data, "pr_comment")
        
        return {"status": "ignored"}
    
    def _handle_issue_comment(self, payload: Dict) -> Dict:
        """Handle issue comment events."""
        action = payload.get("action")
        comment = payload.get("comment", {})
        issue = payload.get("issue", {})
        
        # Only process PR comments (issues with pull_request field)
        if not issue.get("pull_request"):
            return {"status": "ignored", "reason": "not_a_pr"}
        
        if action == "created":
            body = comment.get("body", "")
            return self._process_command(body, issue, "issue_comment")
        
        return {"status": "ignored"}
    
    def _handle_push(self, payload: Dict) -> Dict:
        """Handle push events."""
        ref = payload.get("ref", "")
        repo = payload.get("repository", {})
        
        # Only process main/master branch pushes
        if not ref.endswith(("/main", "/master")):
            return {"status": "ignored", "reason": "not_main_branch"}
        
        log_json("INFO", "github_push_event", {
            "ref": ref,
            "repo": repo.get("full_name"),
            "commits": len(payload.get("commits", [])),
        })
        
        return {"status": "processed"}
    
    def _handle_installation(self, payload: Dict) -> Dict:
        """Handle installation events."""
        action = payload.get("action")
        installation = payload.get("installation", {})
        
        log_json("INFO", "github_installation", {
            "action": action,
            "installation_id": installation.get("id"),
            "account": installation.get("account", {}).get("login"),
        })
        
        return {"status": "processed"}
    
    def _on_pr_opened(self, pr_data: Dict, repo_data: Dict) -> Dict:
        """Handle new PR opened."""
        log_json("INFO", "github_pr_opened", {
            "title": pr_data.get("title"),
            "number": pr_data.get("number"),
            "repo": repo_data.get("full_name"),
        })
        
        # Trigger AURA analysis on the PR
        return self._trigger_pr_analysis(pr_data, repo_data)
    
    def _on_pr_synchronize(self, pr_data: Dict, repo_data: Dict) -> Dict:
        """Handle PR updated with new commits."""
        log_json("INFO", "github_pr_updated", {
            "number": pr_data.get("number"),
            "repo": repo_data.get("full_name"),
        })
        
        return {"status": "processed"}
    
    def _process_command(self, body: str, context: Dict, source: str) -> Dict:
        """Process slash commands in comments."""
        commands = {
            "/aura review": self._cmd_review,
            "/aura fix": self._cmd_fix,
            "/aura help": self._cmd_help,
        }
        
        for cmd, handler in commands.items():
            if body.strip().startswith(cmd):
                return handler(context, source)
        
        return {"status": "ignored", "reason": "no_command"}
    
    def _cmd_review(self, context: Dict, source: str) -> Dict:
        """Handle /aura review command."""
        log_json("INFO", "github_cmd_review", {"source": source})
        
        # Trigger AURA review
        pr_number = context.get("number")
        repo_full_name = context.get("repository", {}).get("full_name")
        
        if pr_number and repo_full_name:
            # Schedule review asynchronously
            import asyncio
            asyncio.create_task(self._perform_review(pr_number, repo_full_name))
        
        return {
            "status": "accepted",
            "command": "review",
            "message": "🤖 AURA is analyzing this PR. Results will be posted shortly.",
        }
    
    def _cmd_fix(self, context: Dict, source: str) -> Dict:
        """Handle /aura fix command."""
        log_json("INFO", "github_cmd_fix", {"source": source})
        
        # TODO: Trigger AURA fix attempt
        return {
            "status": "accepted",
            "command": "fix",
            "message": "AURA will attempt to fix issues in this PR.",
        }
    
    def _cmd_help(self, context: Dict, source: str) -> Dict:
        """Handle /aura help command."""
        return {
            "status": "success",
            "command": "help",
            "message": """Available commands:
- `/aura review` - Request AURA to review this PR
- `/aura fix` - Request AURA to fix issues
- `/aura help` - Show this help message""",
        }
    
    def create_pr(
        self,
        repo_name: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        installation_id: Optional[int] = None,
    ) -> Dict:
        """Create a pull request."""
        self._ensure_client()
        if installation_id:
            client = self.get_installation_client(installation_id)
        else:
            client = self._app_client
        
        try:
            repo = client.get_repo(repo_name)
            pr = repo.create_pull(
                title=title,
                body=body,
                head=head_branch,
                base=base_branch,
            )
            
            log_json("INFO", "github_pr_created", {
                "repo": repo_name,
                "pr_number": pr.number,
                "url": pr.html_url,
            })
            
            return {
                "success": True,
                "pr_number": pr.number,
                "url": pr.html_url,
                "title": pr.title,
            }
        except Exception as e:
            log_json("ERROR", "github_pr_create_failed", {"error": str(e)})
            return {"success": False, "error": str(e)}
    
    def create_pr_from_goal(
        self,
        repo_name: str,
        goal_description: str,
        files_changed: List[str],
        installation_id: Optional[int] = None,
    ) -> Dict:
        """Create a PR from an AURA goal completion."""
        title = f"AURA: {goal_description[:50]}{'...' if len(goal_description) > 50 else ''}"
        
        body = f"""## AURA Automated Implementation

**Goal:** {goal_description}

### Changes
This PR was automatically generated by AURA.

### Files Modified
{chr(10).join(f"- `{f}`" for f in files_changed)}

### Verification
- [ ] Tests pass
- [ ] Code review completed
- [ ] Changes verified

---
*Generated by AURA CLI*
"""
        
        # Generate unique branch name
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        branch_name = f"aura/{timestamp}-{hashlib.md5(goal_description.encode()).hexdigest()[:8]}"
        
        return self.create_pr(
            repo_name=repo_name,
            title=title,
            body=body,
            head_branch=branch_name,
            installation_id=installation_id,
        )
    
    def _trigger_pr_analysis(self, pr_data: Dict, repo_data: Dict) -> Dict:
        """Trigger AURA analysis on a PR.
        
        This schedules an asynchronous review of the PR.
        """
        pr_number = pr_data.get("number")
        repo_full_name = repo_data.get("full_name")
        
        log_json("INFO", "github_trigger_analysis", {
            "pr_number": pr_number,
            "repo": repo_full_name,
        })
        
        # Schedule review asynchronously
        import asyncio
        asyncio.create_task(self._perform_review(pr_number, repo_full_name))
        
        return {
            "status": "processing",
            "action": "analyze_pr",
            "pr_number": pr_number,
            "message": "AURA review scheduled",
        }
    
    async def _perform_review(self, pr_number: int, repo_full_name: str) -> Dict:
        """Perform actual PR review using PRReviewAgent.
        
        Args:
            pr_number: The PR number
            repo_full_name: Repository full name (owner/repo)
            
        Returns:
            Review result dict
        """
        try:
            # Import here to avoid circular imports
            from agents.pr_review import PRReviewAgent
            
            log_json("INFO", "github_perform_review_start", {
                "pr_number": pr_number,
                "repo": repo_full_name,
            })
            
            # Get PR details from GitHub
            client = self._app_client
            if not client:
                return {"error": "GitHub client not initialized"}
            
            repo = client.get_repo(repo_full_name)
            pr = repo.get_pull(pr_number)
            
            # Get changed files
            files = []
            for file in pr.get_files():
                files.append({
                    "filename": file.filename,
                    "status": file.status,
                    "additions": file.additions,
                    "deletions": file.deletions,
                    "changes": file.changes,
                    "patch": file.patch,
                    "content": "",  # Would need to fetch actual content
                })
            
            # Perform review
            agent = PRReviewAgent()
            result = await agent.review_pr(
                pr_number=pr_number,
                pr_title=pr.title,
                diff_content=pr.diff_url,
                files=files,
            )
            
            # Post review to GitHub
            github_review = agent.format_for_github(result)
            
            # Create review
            pr.create_review(
                body=github_review["body"],
                event=github_review["event"],
                comments=[
                    {
                        "path": c["path"],
                        "line": c["line"],
                        "body": c["body"],
                    }
                    for c in github_review["comments"]
                ],
            )
            
            log_json("INFO", "github_review_posted", {
                "pr_number": pr_number,
                "comments_count": len(result.comments),
                "approved": result.approved,
            })
            
            return {
                "status": "completed",
                "pr_number": pr_number,
                "comments_count": len(result.comments),
                "approved": result.approved,
            }
            
        except Exception as e:
            log_json("ERROR", "github_review_failed", {
                "pr_number": pr_number,
                "error": str(e),
            })
            return {
                "status": "failed",
                "pr_number": pr_number,
                "error": str(e),
            }


# Global instance
_github_app: Optional[GitHubApp] = None


def init_github_app(app_id: str, private_key: str, webhook_secret: Optional[str] = None) -> GitHubApp:
    """Initialize global GitHub App instance."""
    global _github_app
    _github_app = GitHubApp(app_id, private_key, webhook_secret)
    return _github_app


def get_github_app() -> Optional[GitHubApp]:
    """Get global GitHub App instance."""
    return _github_app
