"""FastAPI server — OpenEnv standard + competition endpoints (/tasks, /grader, /baseline)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from openenv.core.env_server import create_fastapi_app
from pydantic import BaseModel, Field

from contract_risk_env.models import ClauseFlag, ContractAction, ContractObservation, ContractState
from .corpus import get_labels, list_contracts, get_contract
from .environment import ContractRiskEnvironment
from .graders import grade

# ── Standard OpenEnv app ────────────────────────────────────────────────────

app = create_fastapi_app(ContractRiskEnvironment, ContractAction, ContractObservation)


# ── Competition endpoints ───────────────────────────────────────────────────

@app.get("/tasks")
def get_tasks() -> List[Dict[str, Any]]:
    """List all tasks with their action schemas."""
    tasks = [
        {
            "task_id": "easy",
            "description": "SaaS subscription agreement — 6 explicit risks with clear headings. Expected F1 ~0.85 for GPT-4o baseline.",
            "difficulty": "easy",
            "expected_f1_baseline": 0.85,
            "action_schema": ContractAction.model_json_schema(),
        },
        {
            "task_id": "medium",
            "description": "IP licensing agreement — risks buried in cross-references between definitions and operative clauses. Expected F1 ~0.51.",
            "difficulty": "medium",
            "expected_f1_baseline": 0.51,
            "action_schema": ContractAction.model_json_schema(),
        },
        {
            "task_id": "hard",
            "description": "Enterprise MSA — risks hidden inside clauses that appear protective on first reading. Expected F1 ~0.29.",
            "difficulty": "hard",
            "expected_f1_baseline": 0.29,
            "action_schema": ContractAction.model_json_schema(),
        },
    ]
    return tasks


class GraderRequest(BaseModel):
    episode_id: str = ""
    action: Dict[str, Any] = Field(default_factory=dict)
    contract_id: str = ""


@app.post("/grader")
def grade_episode(req: GraderRequest) -> Dict[str, Any]:
    """Score a completed episode given the action and contract_id."""
    labels = get_labels(req.contract_id)
    flagged = req.action.get("flagged_clauses", [])
    result = grade(flagged, labels)
    result["episode_id"] = req.episode_id
    result["contract_id"] = req.contract_id
    return result


# ── Baseline heuristic (keyword matching, no LLM) ──────────────────────────

_KEYWORD_RULES: List[Dict[str, Any]] = [
    {
        "pattern": r"(?i)auto(?:matic(?:ally)?)?[\s\-]*renew",
        "risk_type": "auto_renewal",
        "severity": 2,
    },
    {
        "pattern": r"(?i)unlimited\s+liability|liability.*?(?:shall\s+be\s+unlimited|no\s+cap|without\s+limit)",
        "risk_type": "unlimited_liability",
        "severity": 3,
    },
    {
        "pattern": r"(?i)unilateral(?:ly)?\s+(?:amend|modif|change)|reserves?\s+the\s+right\s+to\s+modify",
        "risk_type": "unilateral_amendment",
        "severity": 3,
    },
    {
        "pattern": r"(?i)(?:irrevocabl[ye]\s+assign|hereby\s+assigned\s+to|shall\s+(?:automatically\s+)?vest\s+in\s+and\s+be\s+assigned)",
        "risk_type": "ip_assignment_overreach",
        "severity": 3,
    },
    {
        "pattern": r"(?i)indemnif.*?(?:not\s+subject\s+to\s+any\s+cap|unlimited|any\s+and\s+all\s+(?:claims|losses|damages))",
        "risk_type": "indemnification_asymmetry",
        "severity": 2,
    },
    {
        "pattern": r"(?i)(?:perpetual|irrevocable).*?(?:royalty[\s\-]*free)?.*?license\s+to\s+use\s+(?:Customer|Client)\s+Data|(?:Usage\s+Data|Derived\s+Models?).*?(?:own|sell|sublicense|commercial)",
        "risk_type": "data_ownership_ambiguity",
        "severity": 2,
    },
    {
        "pattern": r"(?i)terminat(?:e|ion)\s+(?:for\s+convenience|at\s+any\s+time\s+for\s+any\s+reason|without\s+cause)",
        "risk_type": "termination_without_cause",
        "severity": 2,
    },
    {
        "pattern": r"(?i)evergreen\s+term",
        "risk_type": "auto_renewal",
        "severity": 3,
    },
    {
        "pattern": r"(?i)notwithstanding\s+subsections?\s+\(a\).*?(?:exclusive|perpetual|irrevocable).*?(?:commercialize|exploit|sublicense)",
        "risk_type": "ip_assignment_overreach",
        "severity": 3,
    },
    {
        "pattern": r"(?i)(?:mutual\s+liability\s+cap|exceptions?\s+to\s+(?:mutual\s+)?liability).*?(?:shall\s+not\s+apply|no\s+cap|unlimited)",
        "risk_type": "unlimited_liability",
        "severity": 3,
    },
    {
        "pattern": r"(?i)service\s+continuity\s+provision.*?(?:automatically\s+extend|renewal\s+trigger)",
        "risk_type": "auto_renewal",
        "severity": 3,
    },
]


import re as _re


def _run_baseline_heuristic(task_id: str) -> Dict[str, Any]:
    """Keyword-based baseline — no API key required."""
    contract = get_contract(task_id)
    labels = contract["labels"]
    text = contract["text"]

    # Split text into sections for clause_id extraction
    flagged: List[Dict[str, Any]] = []
    seen_clause_ids: set = set()

    for rule in _KEYWORD_RULES:
        for m in _re.finditer(rule["pattern"], text):
            # Try to find nearest section number
            preceding = text[max(0, m.start() - 300): m.start()]
            section_matches = list(_re.finditer(r"(\d+)\.(\d+)", preceding))
            if section_matches:
                last = section_matches[-1]
                clause_id = f"clause_{last.group(1)}_{last.group(2)}"
            else:
                # try in the match itself
                in_match = _re.search(r"(\d+)\.(\d+)", text[m.start(): m.end() + 100])
                if in_match:
                    clause_id = f"clause_{in_match.group(1)}_{in_match.group(2)}"
                else:
                    clause_id = f"clause_unknown_{m.start()}"

            if clause_id not in seen_clause_ids:
                seen_clause_ids.add(clause_id)
                span = text[m.start(): min(m.end() + 100, len(text))]
                flagged.append({
                    "clause_id": clause_id,
                    "risk_type": rule["risk_type"],
                    "severity": rule["severity"],
                    "span_text": span[:200],
                })

    result = grade(flagged, labels)
    result["flagged_count"] = len(flagged)
    result["task_id"] = task_id
    return result


@app.get("/baseline")
def run_baseline() -> Dict[str, Any]:
    """Run keyword-matching heuristic against all 3 tasks. No API key needed."""
    scores: Dict[str, Any] = {}
    total = 0.0
    for tid in ("easy", "medium", "hard"):
        r = _run_baseline_heuristic(tid)
        scores[tid] = {
            "reward": r["reward"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "flagged_count": r["flagged_count"],
        }
        total += r["reward"]
    scores["mean"] = round(total / 3, 4)
    return scores

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
