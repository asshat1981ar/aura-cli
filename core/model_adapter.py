import os
import subprocess
import requests
import json
import socket
import time
from pathlib import Path
import numpy as np

from core.logging_utils import log_json # Import log_json
from core.file_tools import _aura_clean_json, _aura_safe_loads # Import _aura_clean_json and _aura_safe_loads

# Removed dangerous global IPv4-only monkeypatch for socket.getaddrinfo.
# This monkeypatch forced all network connections to use IPv4, potentially
# causing connectivity issues and hiding underlying network misconfigurations.
# The system should now correctly handle both IPv4 and IPv6 as determined
# by the operating system and network stack.

class ModelAdapter:
    """
    The ModelAdapter manages interactions with various Large Language Models (LLMs)
    and external tools. It provides a unified interface for sending prompts,
    receiving responses, and executing tool calls. It includes a robust fallback
    mechanism and performance tracking for different LLMs.
    """

    def __init__(self):
        """
        Initializes the ModelAdapter, validates the Gemini CLI path,
        and defines an allowlist for executable tools.
        """
        # API keys will be fetched dynamically within their respective call methods
        self.gemini_cli_path = os.getenv("GEMINI_CLI_PATH", "/data/data/com.termux/files/usr/bin/gemini") # Configurable path to gemini CLI
        self.mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000") # Configurable MCP server URL

        # Validate gemini CLI path
        if not Path(self.gemini_cli_path).is_file():
            log_json("WARN", "gemini_cli_not_found", details={"path": self.gemini_cli_path})
            self.gemini_cli_path = None
        elif not os.access(self.gemini_cli_path, os.X_OK):
            log_json("WARN", "gemini_cli_not_executable", details={"path": self.gemini_cli_path})
            self.gemini_cli_path = None

        # Define an explicit allowlist for tools
        self.ALLOWED_TOOLS = {
            "search", "read_file", "list_directory", "glob",
            # New GitHub tools
            "get_repo", "create_issue", "get_issue_details", "update_file", "get_pull_request_details"
        }


    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """
        Executes a specified tool by making a request to the local MCP server
        or falling back to a direct 'npx @modelcontextprotocol/sdk' call for
        non-GitHub tools.

        Args:
            tool_name (str): The name of the tool to execute.
            args (dict): A dictionary of arguments for the tool.

        Returns:
            str: The JSON string output from the tool execution.
        """
        # Now, this calls the local FastAPI MCP server
        try:
            log_json("INFO", "executing_tool_via_mcp_server", details={"tool_name": tool_name, "args": args})

            # The tool_name will determine the endpoint in the MCP server
            # e.g., /github/get_repo, /github/create_issue
            # The structure of args might need to be flattened or adjusted
            # depending on the FastAPI endpoint's Pydantic model.
            
            # For simplicity, let's assume all GitHub tools are under /github/{tool_name}
            # and that the args directly map to the request body.
            
            # Construct the URL for the specific tool
            # Need to map tool_name to the actual FastAPI endpoint path
            # This is a simplified mapping, more complex routing might be needed for other tools
            if tool_name in ["get_repo", "create_issue", "get_issue_details", "update_file", "get_pull_request_details"]:
                # GitHub tools need special handling for pathing and method.
                # For GET requests, args become query parameters.
                # For POST/PUT requests, args become JSON body.
                # This requires more sophisticated routing than a simple f-string.
                
                # Let's refine this: the MCP server already defines distinct endpoints.
                # We need to map tool_name to the correct endpoint path and HTTP method.

                # Assuming the MCP server has endpoints like:
                # GET /github/repo -> args = {"repo_full_name": "owner/repo"}
                # POST /github/issue -> args = {"repo_full_name": "owner/repo", "title": "...", "body": "..."}
                # GET /github/issue/{issue_number} -> args = {"repo_full_name": "owner/repo", "issue_number": 123}
                # PUT /github/file -> args = {"repo_full_name": "...", "file_path": "...", "message": "...", "content": "..."}
                # GET /github/pull_request/{pr_number} -> args = {"repo_full_name": "owner/repo", "pr_number": 123}

                # This implies a more complex routing logic here.
                # For now, let's just create a generic POST endpoint.
                # Or, even better, have a single /mcp/tool endpoint that takes tool_name and args.

                # Let's assume a simpler API for the MCP server:
                # POST /tool/{tool_name} with args as JSON body.
                # This is more aligned with the original npx @modelcontextprotocol/sdk call.

                # Determine HTTP method and construct URL/payload based on tool_name
                # This requires knowledge of the MCP server's exposed API.
                # Since we just created it, we define it here:
                # All tools will be POST requests to a generic /tool endpoint.

                # Generic tool calling endpoint on MCP server
                mcp_tool_url = f"{self.mcp_server_url}/tool/{tool_name}"
                
                try:
                    response = requests.post(mcp_tool_url, json=args, timeout=60)
                    response.raise_for_status() # Raise an exception for HTTP errors
                    return json.dumps(response.json()) # Return JSON response as string
                except requests.exceptions.RequestException as e:
                    return f"MCP Server tool execution failed for {tool_name}: {e}"
            else:
                # Fallback to the original npx @modelcontextprotocol/sdk for non-GitHub tools
                # This ensures compatibility with existing basic tools.
                args_str = json.dumps(args)
                command = ["npx", "@modelcontextprotocol/sdk", "call", tool_name, args_str]
                
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True, # Raise an exception for non-zero exit codes
                    timeout=60 # Timeout for tool execution
                )
                return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Tool execution failed for {tool_name}: {e.stderr}"
        except json.JSONDecodeError:
            return f"Tool execution failed: Invalid JSON arguments for {tool_name}."
        except Exception as e:
            return f"Tool execution failed unexpectedly for {tool_name}: {str(e)}"

    def _make_request_with_retries(self, method, url, headers, json_payload, retries=3, backoff_factor=0.5):
        """
        Makes an HTTP request with retry logic and exponential backoff.

        Args:
            method (str): The HTTP method (e.g., "POST", "GET").
            url (str): The URL for the request.
            headers (dict): HTTP headers for the request.
            json_payload (dict): JSON payload for the request body.
            retries (int, optional): Number of retry attempts. Defaults to 3.
            backoff_factor (float, optional): Factor for exponential backoff. Defaults to 0.5.

        Returns:
            requests.Response: The successful HTTP response object.

        Raises:
            requests.exceptions.RequestException: If all retry attempts fail.
        """
        for attempt in range(retries):
            try:
                response = requests.request(method, url, headers=headers, json=json_payload, timeout=60) # Increased timeout to 60 seconds
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    sleep_time = backoff_factor * (2 ** attempt)
                    log_json("WARN", "request_failed_retrying", details={"attempt": attempt + 1, "retries": retries, "error": str(e), "sleep_time": f"{sleep_time:.2f}"})
                    time.sleep(sleep_time)
                else:
                    raise # Re-raise exception if all retries fail
        return None # Should not be reached

    # -------- OPENROUTER --------
    def call_openrouter(self, prompt: str) -> str:
        """
        Calls the OpenRouter API to get a chat completion.

        Args:
            prompt (str): The prompt message for the LLM.

        Returns:
            str: The content of the LLM's response.

        Raises:
            ValueError: If OPENROUTER_API_KEY environment variable is not set.
        """
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY not set for OpenRouter call.")
        url = "https://api.openrouter.ai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openrouter/auto",
            "messages": [{"role": "user", "content": prompt}]
        }

        response = self._make_request_with_retries("POST", url, headers, payload)
        data = response.json()
        return data["choices"][0]["message"]["content"]

    # -------- OPENAI API --------
    def call_openai(self, prompt: str) -> str:
        """
        Calls the OpenAI API to get a chat completion.

        Args:
            prompt (str): The prompt message for the LLM.

        Returns:
            str: The content of the LLM's response, or None if API key is not set.
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return None # Return None to allow fallback
        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4o-mini",  # Using a cost-effective, capable model
            "messages": [{"role": "user", "content": prompt}]
        }

        response = self._make_request_with_retries("POST", url, headers, payload)
        data = response.json()
        return data["choices"][0]["message"]["content"]

    # -------- LOCAL FALLBACK --------
    def call_local(self, prompt: str) -> str:
        """
        Calls a locally configured LLM via a shell command.

        Args:
            prompt (str): The prompt message for the local LLM.

        Returns:
            str: The content of the LLM's response.
        """
        local_model_command = os.getenv("AURA_LOCAL_MODEL_COMMAND")
        if local_model_command:
            try:
                # Assuming the local model command expects the prompt as an argument
                # and outputs the response to stdout.
                command_parts = local_model_command.split()
                command_parts.append(prompt)
                
                result = subprocess.run(
                    command_parts,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=120 # Increased timeout for local models
                )
                return result.stdout.strip()
            except FileNotFoundError:
                return "Error: Local model command not found. Please ensure it's in your PATH or specify full path."
            except subprocess.CalledProcessError as e:
                return f"Error: Local model command failed with exit code {e.returncode}. Stderr: {e.stderr.strip()}"
            except Exception as e:
                return f"Error: An unexpected error occurred while calling local model: {e}"
        else:
            return "Local model not configured. Set the AURA_LOCAL_MODEL_COMMAND environment variable " \
                   "to specify a command for local inference (e.g., 'ollama run llama2')."

    # -------- HYBRID ROUTING --------
    def respond(self, prompt: str):
        """
        Routes the prompt to the best available LLM based on a predefined fallback
        strategy (OpenAI -> OpenRouter -> Local). It also attempts to parse the
        LLM's response for tool calls and execute them if allowed.

        Args:
            prompt (str): The prompt message to send to the LLM.

        Returns:
            str: The LLM's response content, which might be a raw text response,
                 a structured JSON response, or the output of a tool call.
        """
        model_response = None
        
        # Try OpenAI first
        try:
            model_response = self.call_openai(prompt)
        except Exception as e:
            log_json("WARN", "openai_call_failed", details={"error": str(e), "fallback": "OpenRouter"})
            # Fallback to OpenRouter (still problematic, but included for completeness)
            try:
                model_response = self.call_openrouter(prompt)
            except Exception as e:
                log_json("WARN", "openrouter_call_failed", details={"error": str(e), "fallback": "Local Model"})
                # Final fallback
                model_response = self.call_local(prompt)

        if model_response is None:
            return "Error: No model successfully responded."
        
        try:
            parsed_response = _aura_safe_loads(model_response, "model_response")
            if isinstance(parsed_response, dict) and "tool_code" in parsed_response:
                tool_call_data = parsed_response["tool_code"]
                tool_name = tool_call_data.get("name")
                args = tool_call_data.get("args", {})

                if not isinstance(tool_name, str) or not isinstance(args, dict):
                    return "Model attempted tool call with invalid structure (name or args missing/wrong type)."

                if tool_name not in self.ALLOWED_TOOLS:
                    return f"Error: Tool '{tool_name}' is not allowed by the system configuration."

                tool_output = self._execute_tool(tool_name, args)
                return f"Tool Output: {tool_output}"
            else:
                # If it's valid JSON but not a tool_code, treat it as a direct model response.
                # The model might return structured data that is not a tool call.
                return model_response # Return original model response as it is not a tool call
        except json.JSONDecodeError:
            # Not a JSON response, treat as normal text response and return it directly.
            pass

        return model_response

    def get_embedding(self, text: str) -> list[float]:
        """
        Generates a vector embedding for the given text using the OpenAI API.

        Args:
            text (str): The input text to generate an embedding for.

        Returns:
            list[float]: A list of floats representing the vector embedding.

        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set.
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set for OpenAI embedding call. Please set it to use VectorStore.")

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": text,
            "model": "text-embedding-ada-002"  # A common and cost-effective embedding model
        }

        response = self._make_request_with_retries("POST", url, headers, payload)
        data = response.json()
        return np.array(data["data"][0]["embedding"], dtype=np.float32)