import re # Added for path validation
from typing import Dict
import json
from pathlib import Path

from core.file_tools import _aura_safe_loads
from core.logging_utils import log_json



class ScaffolderAgent:
    """
    The ScaffolderAgent is responsible for generating new project structures and
    basic file content based on an LLM-provided blueprint. It includes robust
    path validation to prevent malicious file system operations and ensures
    all generated paths are confined to the intended project root.
    """
    def __init__(self, brain, model):
        """
        Initializes the ScaffolderAgent with access to the system's brain and model.

        Args:
            brain: An instance of the system's memory (Brain).
            model: An instance of the model adapter for LLM interactions.
        """
        self.brain = brain
        self.model = model

    def _validate_path_component(self, component: str, context: str = "") -> str:
        """
        Validates an individual path component (file or directory name) to prevent
        path traversal vulnerabilities and ensure it's not an absolute path.

        Args:
            component (str): The path component string to validate.
            context (str, optional): Additional context for error messages. Defaults to "".

        Returns:
            str: The validated path component.

        Raises:
            ValueError: If the path component is invalid (empty, absolute, or contains '..').
        """
        """
        Validates an individual path component (file or directory name).
        Raises ValueError if the component is invalid (e.g., contains '..', is absolute).
        """
        if not component or component.strip() == '':
            raise ValueError(f"Path component cannot be empty. Context: {context}")
        
        path_obj = Path(component)

        # Disallow absolute paths or paths starting with separator
        if path_obj.is_absolute() or str(path_obj).startswith(('/', '\\')):
            raise ValueError(f"Path component '{component}' is absolute or starts with a separator. Context: {context}")

        # Disallow path traversal
        if '..' in path_obj.parts:
            raise ValueError(f"Path component '{component}' contains path traversal ('..'). Context: {context}")
        
        # Disallow Windows drive letters if applicable (though unlikely in current env)
        if re.match(r'^[a-zA-Z]:[/\\]', component):
            raise ValueError(f"Path component '{component}' contains Windows drive letter. Context: {context}")

        return component

    def scaffold_project(self, project_name: str, description: str) -> str:
        """
        Generates and creates a new project's directory structure and basic files
        based on an LLM-generated JSON blueprint. Includes validation for
        `project_name` and all file/directory paths within the structure.

        Args:
            project_name (str): The desired name for the new project.
            description (str): A description of the project, used to guide the LLM.

        Returns:
            str: A message indicating the success or failure of the scaffolding process.
        """
        prompt = f"""
You are an autonomous project scaffolding agent. Your task is to generate a suitable directory structure and basic file content for a new software project.

Project Name: {project_name}
Project Description: {description}

Previous memory:
{self.brain.recall_all()}

Generate a JSON object representing the project structure. Each key in the JSON should be a file or directory path relative to the project root.
For directories, the value should be an empty dictionary. For files, the value should be the content of the file.
Example:
{{
    "README.md": "# {project_name}

{description}",
    "src/": {{}},
    "src/main.py": "def main():
    print('Hello, {project_name}!')

if __name__ == '__main__':
    main()",
    "tests/": {{}}
}}
"""
        response = self.model.respond(prompt)
        self.brain.remember(f"Scaffolded project '{project_name}' with description: {description}. Structure: {response}")

        try:
            project_structure = _aura_safe_loads(response, "model_response")
            if not isinstance(project_structure, dict):
                raise ValueError("Response is not a valid JSON dictionary.")

            # Validate project_name
            validated_project_name = self._validate_path_component(project_name, "project_name")
            project_root = Path(f"projects/{validated_project_name}")
            project_root.mkdir(parents=True, exist_ok=True)
            
            self._create_from_structure(project_root, project_structure)
            return f"Project '{validated_project_name}' scaffolded successfully at {project_root.resolve()}"

        except ValueError as ve:
            log_json("ERROR", "scaffolder_validation_error", details={"error": str(ve), "project_name": project_name, "raw_response_snippet": response[:200]})
            return f"Error in scaffolding: {ve}. Check logs for details."
        except json.JSONDecodeError:
            log_json("ERROR", "scaffolder_json_decode_error", details={"raw_response_snippet": response[:200]})
            return f"Failed to parse project structure JSON. Raw response: {response}"
        except Exception as e:
            log_json("CRITICAL", "scaffolder_unexpected_error", details={"error": str(e), "raw_response_snippet": response[:200]})
            return f"An unexpected error occurred during scaffolding: {e}"

    def _create_from_structure(self, current_path: Path, structure: Dict):
        """
        Recursively creates directories and writes files based on a dictionary
        representing the project structure. All paths are validated to ensure
        they remain within the `project_root`.

        Args:
            current_path (Path): The current base path for creation (initially the project root).
            structure (Dict): A dictionary where keys are file/directory names and values
                              are either content (for files) or nested dictionaries (for directories).
        """
        project_root = current_path.resolve() # This is the actual project root for validation
        for name, content in structure.items():
            try:
                # Validate individual component
                validated_name = self._validate_path_component(name, f"structure key: {name}")

                path = current_path / validated_name
                
                # Ensure the resolved path remains within the project_root
                resolved_path = path.resolve()
                if not resolved_path.is_relative_to(project_root):
                    log_json("ERROR", "scaffolder_path_outside_root", details={"invalid_path": str(resolved_path), "project_root": str(project_root), "original_name": name})
                    continue # Skip this entry

                if isinstance(content, dict): # It's a directory
                    path.mkdir(parents=True, exist_ok=True)
                    self._create_from_structure(path, content)
                elif isinstance(content, str): # It's a file
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                else:
                    log_json("WARN", "scaffolder_unknown_content_type", details={"name": name, "content_type": str(type(content))})
            except ValueError as ve:
                log_json("ERROR", "scaffolder_structure_validation_error", details={"error": str(ve), "name": name})
            except Exception as e:
                log_json("ERROR", "scaffolder_create_from_structure_unexpected_error", details={"error": str(e), "name": name})