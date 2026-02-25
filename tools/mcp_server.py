from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, Any
from .github_tools import GitHubTools
from core.logging_utils import log_json
import uvicorn
import os

app = FastAPI(title="AURA MCP GitHub Tools Server", version="0.1.0")

# Initialize GitHubTools. This will raise ValueError if GITHUB_PAT is not set.
# This should ideally be handled during application startup with proper error logging.
try:
    github_tools = GitHubTools()
    log_json("INFO", "github_tools_initialized")
except ValueError as e:
    log_json("CRITICAL", "github_pat_missing", details={"error": str(e)})
    github_tools = None


# --- Generic Tool Request Model ---
class ToolRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any]

# --- Helper to check if GitHubTools is available ---
def _check_github_tools_available():
    if github_tools is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub tools are not initialized. GITHUB_PAT might be missing."
        )

# --- Generic Tool Endpoint ---
@app.post("/tool/{tool_name}")
async def execute_tool(tool_name: str, request: ToolRequest):
    _check_github_tools_available()

    if tool_name != request.tool_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tool name in path ('{tool_name}') does not match tool_name in body ('{request.tool_name}')."
        )

    # Map tool names to GitHubTools methods
    github_tool_map = {
        "get_repo": github_tools.get_repo,
        "create_issue": github_tools.create_issue,
        "get_issue_details": github_tools.get_issue_details,
        "update_file": github_tools.update_file,
        "get_pull_request_details": github_tools.get_pull_request_details,
    }

    tool_function = github_tool_map.get(tool_name)

    if not tool_function:
        log_json("ERROR", "mcp_tool_not_found", details={"tool_name": tool_name})
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tool '{tool_name}' not found.")

    try:
        result = tool_function(**request.args)
        log_json("INFO", f"mcp_tool_execution_success", details={"tool_name": tool_name, "args": request.args})
        return result
    except ValueError as e:
        log_json("ERROR", f"mcp_tool_execution_failed", details={"tool_name": tool_name, "args": request.args, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        log_json("CRITICAL", f"mcp_tool_execution_unexpected_error", details={"tool_name": tool_name, "args": request.args, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred during tool execution.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)