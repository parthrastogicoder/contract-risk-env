#!/usr/bin/env python3
"""LLM-based inference script — calls GPT-4o (or BASELINE_MODEL) to flag risky clauses.

Uses /reset to get the contract, then /grader to score (stateless).
This avoids the WebSocket session requirement for /step.

Usage:
    OPENAI_API_KEY=sk-... python -m contract_risk_env.inference
    BASE_URL=http://localhost:8000 BASELINE_MODEL=gpt-4o python -m contract_risk_env.inference
"""

from __future__ import annotations

import json
import os
import sys

import requests
from openai import OpenAI

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
MODEL = os.environ.get("BASELINE_MODEL", "gpt-4o")

SYSTEM_PROMPT = """You are a legal contract risk analyst. You will be given a contract and must identify ALL risky clauses.

For each risky clause, return a JSON object with:
- clause_id: string matching the section numbering, e.g. "clause_4_2" for Section 4.2, "clause_6_3b" for a sub-part of Section 6.3
- risk_type: one of ["unlimited_liability", "auto_renewal", "unilateral_amendment", "ip_assignment_overreach", "indemnification_asymmetry", "data_ownership_ambiguity", "termination_without_cause"]
- severity: integer 1 (medium), 2 (high), or 3 (critical)
- span_text: the exact text from the contract that contains the risk (quote directly, ≤300 chars)

Return your analysis as a JSON object with this exact schema:
{
  "flagged_clauses": [ { "clause_id": "...", "risk_type": "...", "severity": N, "span_text": "..." }, ... ],
  "confidence": 0.0 to 1.0,
  "reasoning": "your chain-of-thought analysis"
}

Be thorough. Missing a real risk is worse than a false alarm. Pay special attention to:
- Clauses titled protectively but containing traps (e.g. "Service Continuity" hiding auto-renewal)
- Definitions sections that contain operative risk terms
- "Notwithstanding" clauses that negate preceding protections
- Asymmetric liability caps (mutual heading, one-sided exceptions)
- IP assignment buried in derivative work definitions

Return ONLY valid JSON, no markdown fences."""


def run_episode(task_id: str) -> dict:
    """Reset env to get contract, send to LLM, score via /grader."""
    # 1. Reset to get contract text and contract_id
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    r.raise_for_status()
    reset_data = r.json()
    obs = reset_data.get("observation", reset_data)
    contract_text = obs.get("contract_text", "")

    # Extract contract_id from state endpoint or infer from task mapping
    # Use the /grader endpoint which is stateless
    task_to_contracts = {"easy": "saas_001", "medium": "ip_license_002", "hard": "enterprise_msa_003"}
    contract_id = task_to_contracts.get(task_id, task_id)

    # 2. Call LLM
    client = OpenAI()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this contract for risky clauses:\n\n{contract_text}"},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content
    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [!] Failed to parse LLM JSON for {task_id}")
        action = {"flagged_clauses": [], "confidence": 0.0, "reasoning": raw}

    # 3. Score via /grader (stateless, no session needed)
    r = requests.post(f"{BASE_URL}/grader", json={
        "episode_id": f"inference_{task_id}",
        "contract_id": contract_id,
        "action": action,
    })
    r.raise_for_status()
    grade = r.json()

    return {
        "task_id": task_id,
        "reward": grade.get("reward", 0.0),
        "precision": grade.get("precision", 0.0),
        "recall": grade.get("recall", 0.0),
        "f1": grade.get("f1", 0.0),
        "flagged": len(action.get("flagged_clauses", [])),
        "missed": grade.get("missed_clauses", []),
        "false_pos": grade.get("false_positives", []),
    }


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Export it and re-run.")
        sys.exit(1)

    print(f"ContractRiskEnv Inference — model={MODEL}, base_url={BASE_URL}\n")
    print(f"{'Task':<10} {'Reward':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Flagged':>8}")
    print("-" * 60)

    results = []
    for tid in ("easy", "medium", "hard"):
        r = run_episode(tid)
        results.append(r)
        print(f"{r['task_id']:<10} {r['reward']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['flagged']:>8d}")
        if r["missed"]:
            print(f"  missed: {r['missed']}")
        if r["false_pos"]:
            print(f"  false+: {r['false_pos']}")

    mean_reward = sum(r["reward"] for r in results) / len(results)
    print("-" * 60)
    print(f"{'MEAN':<10} {mean_reward:>8.4f}")


if __name__ == "__main__":
    main()
