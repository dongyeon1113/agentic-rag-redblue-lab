from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from agentdojo_benchmark.gui import app
from agentdojo_benchmark import gui


def _trace(*, attacked: bool) -> dict:
    return {
        "suite_name": "workspace",
        "pipeline_name": "current-agent-qwen3:8b",
        "user_task_id": "user_task_0",
        "injection_task_id": "injection_task_0" if attacked else None,
        "attack_type": "important_instructions" if attacked else None,
        "injections": {"vector": "malicious instruction"} if attacked else {},
        "messages": [
            {"role": "user", "content": [{"type": "text", "content": "query"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "content": "attacked" if attacked else "normal"}],
                "tool_calls": ([{"function": "send_email", "args": {"to": "attacker"}}] if attacked else []),
            },
        ],
        "utility": not attacked,
        "security": attacked,
        "duration": 1.5,
        "evaluation_timestamp": "2026-01-01 00:00:00",
    }


def test_gui_lists_and_compares_benign_and_attack_traces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENTDOJO_OUTPUT_DIR", str(tmp_path))
    benign = tmp_path / "pipeline/workspace/user_task_0/none/none.json"
    attacked = tmp_path / "pipeline/workspace/user_task_0/important_instructions/injection_task_0.json"
    benign.parent.mkdir(parents=True)
    attacked.parent.mkdir(parents=True)
    benign.write_text(json.dumps(_trace(attacked=False)), encoding="utf-8")
    attacked.write_text(json.dumps(_trace(attacked=True)), encoding="utf-8")

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        catalog = client.get("/api/catalog").json()
        assert "workspace" in catalog["suites"]
        cases = client.get("/api/cases").json()
        assert len(cases) == 1
        detail = client.get("/api/cases/detail", params={
            "pipeline": "current-agent-qwen3:8b",
            "suite": "workspace",
            "user_task": "user_task_0",
            "injection_task": "injection_task_0",
            "attack": "important_instructions",
        }).json()

    assert detail["benign"]["answer"] == "normal"
    assert detail["attacked"]["answer"] == "attacked"
    assert detail["attacked"]["tool_calls"] == [{
        "sequence": 1,
        "function": "send_email",
        "args": {"to": "attacker"},
    }]
    assert detail["injections"] == {"vector": "malicious instruction"}
    assert detail["verdict"]["attack_succeeded"] is True


def test_server_job_persists_and_can_be_queried_after_request_disconnect(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENTDOJO_OUTPUT_DIR", str(tmp_path))

    observed_options = []

    def fake_run(options):
        observed_options.append(options)
        return {"profiles": {options.defense_profiles[0]: {"suites": {}}}}

    monkeypatch.setattr(gui, "run_benchmark", fake_run)
    request = {
        "suite": "workspace",
        "user_task": "user_task_0",
        "injection_task": "injection_task_0",
        "attack": "important_instructions",
        "defense_profiles": [
            "baseline",
            "regex+prompt_guard+spotlighting:delimiting",
        ],
    }
    with TestClient(app) as client:
        created = client.post("/api/runs", json=request)
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        for _ in range(100):
            state = client.get(f"/api/runs/{job_id}").json()
            if state["status"] == "completed":
                break
            time.sleep(0.01)
        assert state["status"] == "completed"
        assert state["completed"] == 2
        assert client.get("/api/runs").json()["jobs"][0]["job_id"] == job_id
        assert client.get(f"/api/runs/{job_id}/download").status_code == 200

    assert all(options.force_rerun is False for options in observed_options)
    assert (tmp_path / "jobs" / job_id / "state.json").is_file()
    assert (tmp_path / "jobs" / job_id / "checkpoint.json").is_file()
    assert (tmp_path / "jobs" / job_id / "result.json").is_file()
