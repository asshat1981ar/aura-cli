"""Unit tests for the Fleet Dispatcher n8n workflow JSON definitions.

Tests verify:
- All WF-0..6 JSON files parse and have the expected structure
- Required nodes exist in each workflow
- WF-0 has a webhook trigger
- WF-6 has a quality gate check, PR creation, escalation, and close-issue nodes
- fleet-trigger.yml GitHub Actions workflow has the correct event trigger
- fleet-labels-setup.sh creates all required fleet labels
- n8n-workflows/README.md covers all workflows and the webhook URL secret
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
N8N_DIR = ROOT / "n8n-workflows"


def load_wf(stem: str) -> dict:
    path = N8N_DIR / f"{stem}.json"
    with path.open() as fh:
        return json.load(fh)


class TestWorkflowFilesExist:
    def test_wf0_master_dispatcher_exists(self):
        assert (N8N_DIR / "WF-0-master-dispatcher.json").exists()

    def test_wf1_bug_fix_handler_exists(self):
        assert (N8N_DIR / "WF-1-bug-fix-handler.json").exists()

    def test_wf2_feature_handler_exists(self):
        assert (N8N_DIR / "WF-2-feature-handler.json").exists()

    def test_wf3_refactor_handler_exists(self):
        assert (N8N_DIR / "WF-3-refactor-handler.json").exists()

    def test_wf4_security_handler_exists(self):
        assert (N8N_DIR / "WF-4-security-handler.json").exists()

    def test_wf5_docs_handler_exists(self):
        assert (N8N_DIR / "WF-5-docs-handler.json").exists()

    def test_wf6_code_gen_pr_push_exists(self):
        assert (N8N_DIR / "WF-6-code-gen-pr-push.json").exists()

    def test_fleet_trigger_yml_exists(self):
        assert (ROOT / ".github" / "workflows" / "fleet-trigger.yml").exists()

    def test_fleet_labels_setup_script_exists(self):
        assert (ROOT / "docs" / "fleet-labels-setup.sh").exists()

    def test_readme_exists(self):
        assert (N8N_DIR / "README.md").exists()


class TestWF0MasterDispatcher:
    WF = "WF-0-master-dispatcher"

    def test_parses_as_valid_json(self):
        wf = load_wf(self.WF)
        assert isinstance(wf, dict)

    def test_has_nodes(self):
        wf = load_wf(self.WF)
        assert len(wf.get("nodes", [])) > 0

    def test_has_connections(self):
        wf = load_wf(self.WF)
        assert len(wf.get("connections", {})) > 0

    def test_has_webhook_or_trigger_node(self):
        wf = load_wf(self.WF)
        names = [n.get("name", "").lower() for n in wf.get("nodes", [])]
        types = [n.get("type", "").lower() for n in wf.get("nodes", [])]
        assert any("webhook" in name or "webhook" in t for name, t in zip(names, types)), \
            f"No webhook node found in WF-0. Names: {names}"


class TestWF1To5Handlers:
    @pytest.mark.parametrize("stem", [
        "WF-1-bug-fix-handler",
        "WF-2-feature-handler",
        "WF-3-refactor-handler",
        "WF-4-security-handler",
        "WF-5-docs-handler",
    ])
    def test_handler_parses_and_has_nodes(self, stem):
        wf = load_wf(stem)
        assert isinstance(wf, dict)
        assert len(wf.get("nodes", [])) > 0

    @pytest.mark.parametrize("stem", [
        "WF-1-bug-fix-handler",
        "WF-2-feature-handler",
        "WF-3-refactor-handler",
        "WF-4-security-handler",
        "WF-5-docs-handler",
    ])
    def test_handler_has_connections(self, stem):
        wf = load_wf(stem)
        assert len(wf.get("connections", {})) > 0


class TestWF6CodeGenPRPush:
    WF = "WF-6-code-gen-pr-push"

    def test_parses_as_valid_json(self):
        wf = load_wf(self.WF)
        assert isinstance(wf, dict)

    def test_has_substantial_node_count(self):
        wf = load_wf(self.WF)
        assert len(wf.get("nodes", [])) >= 10, "WF-6 should have quality gate, PR, merge, escalation nodes"

    def test_has_quality_gate_node(self):
        wf = load_wf(self.WF)
        names = [n.get("name", "").lower() for n in wf.get("nodes", [])]
        assert any("quality" in name for name in names), \
            f"Quality gate node missing. Found: {names}"

    def test_has_pr_creation_node(self):
        wf = load_wf(self.WF)
        names = [n.get("name", "").lower() for n in wf.get("nodes", [])]
        assert any("pr" in name or "pull request" in name or "create pr" in name
                   for name in names), f"PR creation node missing. Found: {names}"

    def test_has_escalation_or_blocked_node(self):
        wf = load_wf(self.WF)
        names = [n.get("name", "").lower() for n in wf.get("nodes", [])]
        assert any("block" in name or "escalat" in name for name in names), \
            f"Escalation/blocked node missing. Found: {names}"

    def test_has_close_issue_node(self):
        wf = load_wf(self.WF)
        names = [n.get("name", "").lower() for n in wf.get("nodes", [])]
        assert any("close" in name for name in names), \
            f"Close issue node missing. Found: {names}"

    def test_has_connections(self):
        wf = load_wf(self.WF)
        assert len(wf.get("connections", {})) > 0


class TestFleetTriggerWorkflow:
    def test_triggers_on_issues_labeled(self):
        content = (ROOT / ".github" / "workflows" / "fleet-trigger.yml").read_text()
        assert "issues:" in content
        assert "labeled" in content

    def test_filters_fleet_trigger_label(self):
        content = (ROOT / ".github" / "workflows" / "fleet-trigger.yml").read_text()
        assert "fleet:trigger" in content

    def test_posts_to_n8n_webhook_url_secret(self):
        content = (ROOT / ".github" / "workflows" / "fleet-trigger.yml").read_text()
        assert "N8N_FLEET_WEBHOOK_URL" in content


class TestFleetLabelsSetupScript:
    def test_creates_required_fleet_labels(self):
        content = (ROOT / "docs" / "fleet-labels-setup.sh").read_text()
        for label in ["fleet:trigger", "fleet:in-progress", "fleet:done", "fleet:blocked"]:
            assert label in content, f"Label '{label}' missing from setup script"

    def test_is_bash_script(self):
        content = (ROOT / "docs" / "fleet-labels-setup.sh").read_text()
        assert "bash" in content.split("\n")[0]


class TestN8NWorkflowsReadme:
    def test_readme_mentions_all_workflows(self):
        readme = (N8N_DIR / "README.md").read_text()
        for i in range(7):
            assert f"WF-{i}" in readme, f"README missing WF-{i} reference"

    def test_readme_mentions_webhook_url_secret(self):
        readme = (N8N_DIR / "README.md").read_text()
        assert "N8N_FLEET_WEBHOOK_URL" in readme

    def test_readme_has_setup_section(self):
        readme = (N8N_DIR / "README.md").read_text()
        assert "Setup" in readme or "setup" in readme
