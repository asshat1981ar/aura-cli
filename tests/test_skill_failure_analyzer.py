import json
import pytest
from pathlib import Path
from agents.skills.skill_failure_analyzer import SkillFailureAnalyzerSkill

@pytest.fixture
def temp_summaries(tmp_path):
    p = tmp_path / "cycle_summaries.json"
    data = [
        {
            "cycle_id": "c1",
            "failures": ["security_scanner", {"skill": "lint"}]
        },
        {
            "cycle_id": "c2",
            "failures": ["security_scanner"]
        }
    ]
    p.write_text(json.dumps(data))
    return p

def test_skill_failure_analyzer_basic(temp_summaries):
    skill = SkillFailureAnalyzerSkill()
    results = skill.run({"summaries_path": str(temp_summaries)})
    
    assert results["total_cycles_analyzed"] == 2
    assert results["most_problematic"] == "security_scanner"
    
    failing = {item["skill"]: item for item in results["failing_skills"]}
    assert failing["security_scanner"]["failure_count"] == 2
    assert failing["lint"]["failure_count"] == 1
    assert "project_root is accessible" in failing["security_scanner"]["remediation"]

def test_skill_failure_analyzer_file_not_found():
    skill = SkillFailureAnalyzerSkill()
    results = skill.run({"summaries_path": "non_existent_file.json"})
    assert "error" in results
    assert results["total_cycles_analyzed"] == 0

def test_skill_failure_analyzer_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json")
    skill = SkillFailureAnalyzerSkill()
    results = skill.run({"summaries_path": str(p)})
    assert "error" in results
    assert results["total_cycles_analyzed"] == 0
