"""Unit tests for the reporter module."""

import json
import pytest
from retrieval_guard.benchmark.scorer import GeneralizationReport, RegressionAlert
from retrieval_guard.reporter import export


MOCK_REPORT = GeneralizationReport(
    model_name="test-model",
    overall_score=0.82,
    per_category={"negation": 0.80, "role_reversal": 0.85, "spatial": 0.90, "binding": 0.75},
    failure_ids=["neg_002", "binding_003"],
    total_pairs=15,
    passed_pairs=13,
)

MOCK_ALERT = RegressionAlert(
    fired=True,
    delta=0.12,
    before_score=0.82,
    after_score=0.70,
    threshold=0.05,
    model_name="fine-tuned-model",
    new_failures=["neg_001", "spatial_002"],
    recommendation="Regression detected. Enable two-stage verification.",
)


def test_export_json():
    content = export(MOCK_REPORT, format="json")
    data = json.loads(content)
    assert data["overall_score"] == 0.82
    assert data["model_name"] == "test-model"


def test_export_json_with_alert():
    content = export(MOCK_REPORT, MOCK_ALERT, format="json")
    data = json.loads(content)
    assert "regression_alert" in data
    assert data["regression_alert"]["fired"] is True


def test_export_markdown():
    content = export(MOCK_REPORT, format="markdown")
    assert "retrieval-guard Report" in content
    assert "test-model" in content
    assert "negation" in content


def test_export_html():
    content = export(MOCK_REPORT, format="html")
    assert "<!DOCTYPE html>" in content
    assert "test-model" in content
    assert "negation" in content


def test_export_unknown_format():
    with pytest.raises(ValueError, match="Unknown format"):
        export(MOCK_REPORT, format="csv")


def test_export_writes_file(tmp_path):
    output = tmp_path / "report.json"
    export(MOCK_REPORT, format="json", path=output)
    assert output.exists()
    data = json.loads(output.read_text())
    assert data["overall_score"] == 0.82
