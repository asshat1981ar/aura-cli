import json
import warnings
from pathlib import Path # Import Path for _validate_file_path
import re # Import re for _validate_file_path

from core.git_tools import GitToolsError
from core.logging_utils import log_json
from core.file_tools import (
    FileToolsError,
    MISMATCH_OVERWRITE_BLOCK_EVENT,
    MismatchOverwriteBlockedError,
    OldCodeNotFoundError,
    _aura_safe_loads,
    apply_change_with_explicit_overwrite_policy,
    mismatch_overwrite_block_log_details,
)
from agents.debugger import DebuggerAgent

class HybridClosedLoop:
    """
    Implements the Hybrid Closed Loop for autonomous development, orchestrating
    LLM interactions, code generation, critique, and Git operations. It features
    a robust scoring mechanism, regression detection, and stable convergence logic.
    """

    _ABSOLUTE_PASS_THRESHOLD = 8.5
    _REGRESSION_THRESHOLD = 3
    _STABLE_CONVERGENCE_THRESHOLD = 3

    _bootstrap_prompt_template = """You are AURA, a closed-loop autonomous development workflow.

You must follow exactly these phases:

1. DEFINE
2. PLAN
3. IMPLEMENT
4. TEST
5. CRITIQUE
6. IMPROVE
7. VERSION
8. SUMMARY

You must optimize across four axes:

A: Performance
B: Stability
C: Security
D: Code Elegance

Evaluation Rules:
- Score each axis from 1–10.
- Identify measurable weaknesses.
- Improvements must remain within the existing architecture.
- Do not introduce new workflow phases.
- Do not redesign the loop structure.

Objective:
{GOAL}

System Snapshot:
{STATE}

Return output strictly structured as a JSON object with the following keys.
The "IMPLEMENT" section should describe file changes using a 'file_path', 'old_code', and 'new_code' structure.
The "CRITIQUE" section should be a JSON object with specific score keys.

{{
  "DEFINE": "A clear definition of the current problem or task.",
  "PLAN": "A detailed step-by-step plan to address the DEFINE stage, focusing on the current iteration.",
  "IMPLEMENT": {{
    "file_path": "path/to/file.py",
    "old_code": "Existing code snippet to be replaced.",
    "new_code": "New code snippet that replaces old_code."
  }},
  "TEST": "Specific test commands or steps to verify the IMPLEMENTATION. Always include expected output or success criteria.",
  "CRITIQUE": {{
    "performance_score": "Score from 1-10, focusing on efficiency and resource usage.",
    "stability_score": "Score from 1-10, focusing on reliability and error handling.",
    "security_score": "Score from 1-10, focusing on vulnerability and secure practices.",
    "elegance_score": "Score from 1-10, focusing on code clarity, maintainability, and design patterns.",
    "weaknesses": ["List identified weaknesses or areas for improvement."]
  }},
  "IMPROVE": "Based on CRITIQUE, a plan for the next iteration's improvements.",
  "VERSION": "A concise description of the changes made in this iteration, suitable for a git commit message.",
  "SUMMARY": "A brief overall summary of this iteration's progress and outcomes."
}}
"""

    def __init__(self, model, brain, git_tools, prompt_template=None):
        """
        Initializes the HybridClosedLoop with model, brain, and Git tools.

        Args:
            model: An instance of the ModelAdapter for LLM interactions.
            brain: An instance of the system's memory (Brain).
            git_tools: An instance of GitTools for repository operations.
            prompt_template (str, optional): A custom prompt template for the loop.
        """
        self.model = model
        self.brain = brain
        self.git = git_tools
        self.debugger = DebuggerAgent(brain, model)
        self.prompt_template = prompt_template or self._bootstrap_prompt_template
        warnings.warn(
            "HybridClosedLoop is deprecated. Use LoopOrchestrator (core/orchestrator.py) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.weights = {
            "performance": 0.30,
            "stability": 0.30,
            "security": 0.25,
            "elegance": 0.15
        }

        self.previous_score = 0
        self.current_score = 0.0
        self.current_goal = None
        self.regression_count = 0
        self.stable_convergence_count = 0 # Added for Robust Confirmation

    def snapshot(self):
        """
        Captures a snapshot of the current system state, primarily memory entries.

        Returns:
            str: A string representation of the system snapshot.
        """
        return f"Memory entries: {len(self.brain.recall_all())}"

    def extract_scores(self, critique_dict: dict, goal: str):
        """
        Extracts performance, stability, security, and elegance scores from a critique dictionary.
        Logs a warning if a score is missing or invalid.

        Args:
            critique_dict (dict): The dictionary containing critique scores.
            goal (str): The current goal being processed, for logging context.

        Returns:
            dict: A dictionary of extracted scores, defaulting to 0 for invalid entries.
        """
        scores = {}
        for key in self.weights:
            dict_key = f"{key}_score"
            if dict_key in critique_dict and isinstance(critique_dict[dict_key], int):
                scores[key] = critique_dict[dict_key]
            else:
                log_json("WARN", "score_missing_or_invalid", goal=goal, details={"score_key": dict_key, "critique": critique_dict})
                scores[key] = 0
        return scores

    def weighted_score(self, scores):
        """
        Calculates the weighted sum of the given scores based on predefined weights.

        Args:
            scores (dict): A dictionary of scores (performance, stability, security, elegance).

        Returns:
            float: The calculated weighted score.
        """
        return sum(scores[k] * self.weights[k] for k in scores)

    def absolute_pass(self, score):
        """
        Determines if a given score meets the absolute passing threshold.

        Args:
            score (float): The score to evaluate.

        Returns:
            bool: True if the score meets or exceeds the threshold, False otherwise.
        """
        return score >= self._ABSOLUTE_PASS_THRESHOLD

    def _validate_file_path(self, file_path: str, current_goal: str) -> bool:
        # New strict validation for code patterns in file_path
        code_pattern = re.compile(r'\b(def|class|import|from|while|for|if|else|elif|try|except|finally)\b|\S+\s*\(.*\)\s*\{|\S+\s*\(.*\)\s*:', re.DOTALL) # Basic patterns to detect code
        if code_pattern.search(file_path):
            log_json("ERROR", "invalid_file_path_contains_code", goal=current_goal, details={"original_file_path": file_path[:500], "reason": "file_path contains code patterns"})
            return False

        if '\\n' in file_path or len(file_path) > 255: # Arbitrary length limit to catch clearly erroneous paths
            log_json("ERROR", "invalid_file_path_from_hybrid_loop", goal=current_goal, details={"original_file_path": file_path[:500], "reason": "contains newline or too long"})
            return False
        
        return True

    def _log_and_diagnose_error(
        self,
        error_message: str,
        current_goal: str,
        file_path: str,
        change_idx: int,
        log_level: str,
        log_event: str,
        result_json: dict,
        change: dict,
        context_message: str = "",
        extra_details: dict | None = None,
    ):
        diagnosis = self.debugger.diagnose( # Changed from debugger_instance to self.debugger
            error_message=error_message,
            current_goal=current_goal,
            context=f"File: {file_path}, {context_message}",
            improve_plan=result_json.get("IMPROVE", ""),
            implement_details=change
        )
        details = {
            "error": error_message,
            "file": file_path,
            "change_idx": change_idx,
            "diagnosis": diagnosis
        }
        if extra_details:
            details.update(extra_details)

        log_json(
            log_level,
            log_event,
            goal=current_goal,
            details=details
        )
        return False # Indicates that changes were not applied successfully

    def _apply_change_with_debug(
        self,
        project_root: Path,
        sanitized_file_path: str,
        old_code: str,
        new_code: str,
        overwrite_file: bool,
        current_goal: str,
        change_idx: int,
        result_json: dict,
        change: dict
    ) -> bool:
        changes_applied_successfully = True
        try:
            log_json("INFO", "applying_code_change", goal=current_goal, details={"file": sanitized_file_path, "change_idx": change_idx, "overwrite": overwrite_file})
            apply_change_with_explicit_overwrite_policy(
                project_root,
                sanitized_file_path,
                old_code,
                new_code,
                overwrite_file=overwrite_file,
            )
        except MismatchOverwriteBlockedError as e:
            changes_applied_successfully = self._log_and_diagnose_error(
                str(e), current_goal, sanitized_file_path, change_idx,
                "ERROR", MISMATCH_OVERWRITE_BLOCK_EVENT, result_json, change,
                context_message="Policy blocked mismatch-overwrite fallback; explicit overwrite_file with empty old_code required.",
                extra_details=mismatch_overwrite_block_log_details(e, sanitized_file_path),
            )
        except OldCodeNotFoundError as e:
            changes_applied_successfully = self._log_and_diagnose_error(
                str(e), current_goal, sanitized_file_path, change_idx,
                "ERROR", "old_code_not_found", result_json, change, context_message=f"Change: {change.get('old_code')}"
            )
        except FileNotFoundError as e:
            changes_applied_successfully = self._log_and_diagnose_error(
                str(e), current_goal, sanitized_file_path, change_idx,
                "ERROR", "file_not_found_for_replace", result_json, change
            )
        except FileToolsError as e:
            changes_applied_successfully = self._log_and_diagnose_error(
                str(e), current_goal, sanitized_file_path, change_idx,
                "ERROR", "file_tools_error", result_json, change
            )
        except Exception as e:
            changes_applied_successfully = self._log_and_diagnose_error(
                str(e), current_goal, sanitized_file_path, change_idx,
                "CRITICAL", "unexpected_error_applying_change", result_json, change, context_message=f"Change: {change}"
            )
        return changes_applied_successfully

    def run(self, goal, dry_run: bool = False): # Added dry_run parameter
        """
        Executes a single iteration of the Hybrid Closed Loop. This involves:
        1. Prompting the LLM with the goal and system state.
        2. Parsing the LLM's structured response (DEFINE, PLAN, IMPLEMENT, etc.).
        3. Evaluating the CRITIQUE scores.
        4. Managing Git operations (stash, commit, stash_pop) based on outcomes.
        5. Detecting regression and managing stable convergence.

        Args:
            goal (str): The current goal for this iteration.
            dry_run (bool, optional): If True, Git operations and file changes are skipped. Defaults to False.

        Returns:
            str: A JSON string representing the structured response from the LLM
                 with added status information, or a termination message.
        """

        self.current_goal = goal
        state = self.snapshot()
        prompt = self.prompt_template.format(GOAL=goal, STATE=state)

        structured_response = {
            "DEFINE": "Error: Model response failed.",
            "PLAN": "Error: Model response failed.",
            "IMPLEMENT": {},
            "TEST": "Error: Model response failed.",
            "CRITIQUE": {
                "performance_score": 0,
                "stability_score": 0,
                "security_score": 0,
                "elegance_score": 0,
                "weaknesses": ["Model response failed due to an exception or invalid JSON."]
            },
            "IMPROVE": "Error: Model response failed.",
            "VERSION": "Error: Model response failed.",
            "SUMMARY": "Error: Model response failed."
        }

        raw_response = "" # Initialize raw_response for potential error logging

        try:
            raw_response = self.model.respond(prompt)
            parsed_response = _aura_safe_loads(raw_response, "hybrid_loop_run")
            # Basic validation to ensure it's a dict and has expected keys
            expected_keys = ["DEFINE", "PLAN", "IMPLEMENT", "TEST", "CRITIQUE", "IMPROVE", "VERSION", "SUMMARY"]
            if not isinstance(parsed_response, dict) or not all(key in parsed_response for key in expected_keys):
                raise ValueError("Model response is not a valid structured JSON or missing required keys.")
            structured_response.update(parsed_response) # Update default with actual response
            log_json("INFO", "model_response_parsed", goal=goal, details={"dry_run": dry_run, "summary": structured_response.get("SUMMARY")})
        except json.JSONDecodeError as e:
            log_json("ERROR", "model_json_decode_error", goal=goal, details={"error": str(e), "raw_response_snippet": raw_response[:200]})
            diagnosis = self.debugger.diagnose(
                error_message=str(e),
                current_goal=goal,
                context=raw_response[:500],
                improve_plan=structured_response.get("IMPROVE", "No IMPROVE plan available."),
                implement_details=structured_response.get("IMPLEMENT", {})
            )
            structured_response["CRITIQUE"]["weaknesses"].append(f"JSONDecodeError: {e}. Diagnosis: {diagnosis.get('summary', 'N/A')}. Fix: {diagnosis.get('fix_strategy', 'N/A')}. Raw: {raw_response[:200]}...")
            self.stable_convergence_count = 0
            self.regression_count += 1
        except Exception as e:
            log_json("ERROR", "model_response_error", goal=goal, details={"error": str(e)})
            diagnosis = self.debugger.diagnose(
                error_message=str(e),
                current_goal=goal,
                context=raw_response[:500],
                improve_plan=structured_response.get("IMPROVE", "No IMPROVE plan available."),
                implement_details=structured_response.get("IMPLEMENT", {})
            )
            structured_response["CRITIQUE"]["weaknesses"].append(f"Unexpected Error during model response parsing: {e}. Diagnosis: {diagnosis.get('summary', 'N/A')}. Fix: {diagnosis.get('fix_strategy', 'N/A')}.")
            self.stable_convergence_count = 0
            self.regression_count += 1

        # Now, extract scores from the structured_response
        scores = self.extract_scores(structured_response["CRITIQUE"], goal=goal)
        current_score = self.weighted_score(scores)
        self.current_score = current_score

        # Store the current state as a stash before making changes for potential rollback
        if dry_run:
            log_json("INFO", "git_stash_skipped", goal=goal, details={"reason": "dry_run"})
        else:
            try:
                self.git.stash(message=f"Pre-iteration stash for goal: {goal}")
                log_json("INFO", "git_stashed", goal=goal)
            except GitToolsError as e:
                # Log the error but don't terminate the loop immediately, as stashing might fail if no changes
                log_json("WARN", "git_stash_failed", goal=goal, details={"error": str(e)})

        # Regression detection (after changes are applied and scores are re-evaluated)
        # This part needs to be revisited as the "IMPLEMENT" changes are applied externally.
        # The current score is for the *previous* state if this is the first evaluation after model response.
        # However, for now, let's keep the existing logic and assume external agent will re-evaluate.

        if current_score < self.previous_score:
            self.regression_count += 1
            self.stable_convergence_count = 0 # Reset stable convergence on regression
            log_json("INFO", "regression_detected", goal=goal, details={"current_score": current_score, "previous_score": self.previous_score, "regression_count": self.regression_count})
        else:
            self.regression_count = 0
            # If the score improved or stayed same, and there was a stash, pop it.
            # If the score decreased, keep the stash for potential rollback.
            # This logic needs refinement with external application of changes.

        self.previous_score = current_score

        self.brain.remember(goal)
        self.brain.remember(structured_response) # Remember the structured response
        log_json("INFO", "brain_remembered_goal_and_response", goal=goal, details={"dry_run": dry_run, "summary_snippet": structured_response.get("SUMMARY", "")[:100]})

        # Git commit logic:
        # Only commit if we have a stable convergence or a significant improvement
        # This is a temporary rule for now. A better rule would involve testing.
        if self.stable_convergence_count >= 1 or current_score > self.previous_score: # Commit on any improvement or single stable pass
            if dry_run:
                log_json("INFO", "git_commit_skipped", goal=goal, details={"reason": "dry_run", "version_message": structured_response.get('VERSION')})
                log_json("INFO", "git_stash_pop_skipped", goal=goal, details={"reason": "dry_run_after_commit"})
            else:
                try:
                    self.git.commit_all(f"AURA evolution: {structured_response.get('VERSION', f'Iteration for {goal}')}")
                    log_json("INFO", "git_committed", goal=goal, details={"version_message": structured_response.get('VERSION')})
                    # If committed successfully, we can pop the stash if it exists
                    try:
                        self.git.stash_pop()
                        log_json("INFO", "git_stash_popped", goal=goal)
                    except GitToolsError as e:
                        log_json("WARN", "git_stash_pop_failed_after_commit", goal=goal, details={"error": str(e)})
                except GitToolsError as e:
                    log_json("ERROR", "git_commit_failed", goal=goal, details={"error": str(e), "version_message": structured_response.get('VERSION')})
                    try:
                        # self.git.rollback_last_commit(f"Rollback failed commit for goal: {goal}") # This might rollback a prior commit, which is dangerous.
                        # Instead of rolling back, if commit fails, leave it for manual inspection/stash pop.
                        log_json("ERROR", "git_commit_failed_no_rollback", goal=goal, details={"message": "Commit failed, manual intervention needed. Stash might still be present."})
                    except GitToolsError as rb_e:
                        log_json("CRITICAL", "git_rollback_failed_after_commit_failure", goal=goal, details={"error": str(rb_e)})
                    self.regression_count += 1 # Treat failed commit as a regression
        else: # If no improvement or regression, then implicitly changes are not applied/reverted
            # If there was a stash, and no commit, pop the stash to revert changes from this iteration
            if dry_run:
                log_json("INFO", "git_stash_pop_skipped", goal=goal, details={"reason": "dry_run_no_improvement"})
            else:
                try:
                    self.git.stash_pop()
                    log_json("INFO", "git_stash_popped_reverted_changes", goal=goal)
                except GitToolsError as e:
                    log_json("WARN", "git_stash_pop_failed_to_revert", goal=goal, details={"error": str(e)})

        if self.regression_count >= self._REGRESSION_THRESHOLD:
            log_json("WARN", "repeated_regression_detected", goal=goal, details={"regression_count": self.regression_count})
            final_output = {"FINAL_STATUS": "Terminated: Repeated regression detected."}
            if dry_run:
                final_output["DRY_RUN"] = True
            return json.dumps(final_output)

        # Robust Confirmation Logic:
        if self.absolute_pass(current_score) and (current_score >= self.previous_score): # Changed relative_pass
            self.stable_convergence_count += 1
            log_json("INFO", "stable_convergence_count_incremented", goal=goal, details={"current_score": current_score, "stable_convergence_count": self.stable_convergence_count})
            if self.stable_convergence_count >= self._STABLE_CONVERGENCE_THRESHOLD:
                # Assuming external agent will process the structured_response
                final_output = structured_response
                final_output["FINAL_STATUS"] = f"Optimization converged at {current_score} with Robust Confirmation."
                if dry_run:
                    final_output["DRY_RUN"] = True
                log_json("INFO", "optimization_converged", goal=goal, details={"final_score": current_score, "dry_run": dry_run})
                return json.dumps(final_output)
        else:
            self.stable_convergence_count = 0 # Reset if conditions are not met
            log_json("INFO", "stable_convergence_count_reset", goal=goal, details={"current_score": current_score})


        # If not converged, continue evolution
        current_status = f"Continuing evolution (Score: {current_score}, Stable Convergence Count: {self.stable_convergence_count})"
        current_output = structured_response
        current_output["STATUS"] = current_status
        if dry_run:
            current_output["DRY_RUN"] = True
        log_json("INFO", "continuing_evolution", goal=goal, details={"current_score": current_score, "stable_convergence_count": self.stable_convergence_count, "dry_run": dry_run})
        return json.dumps(current_output)
