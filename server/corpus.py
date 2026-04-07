"""Corpus loader — reads contracts_corpus.json from data/ directory."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_CORPUS: Optional[Dict[str, Any]] = None
_ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_corpus() -> Dict[str, Any]:
    global _CORPUS
    if _CORPUS is None:
        corpus_path = _ROOT_DIR / "contracts_corpus.json"
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found at {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as f:
            _CORPUS = json.load(f)
    return _CORPUS


def get_contract(task_id: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Return one contract dict for the given task_id (difficulty).

    task_id maps directly to difficulty: "easy" -> easy, "medium" -> medium, "hard" -> hard.
    If multiple contracts exist per difficulty, seed selects deterministically.
    """
    corpus = _load_corpus()
    difficulty = task_id  # direct mapping
    candidates = [c for c in corpus["contracts"] if c["difficulty"] == difficulty]
    if not candidates:
        raise ValueError(f"No contract found for task_id='{task_id}' (difficulty='{difficulty}')")
    if seed is not None:
        idx = seed % len(candidates)
    else:
        idx = 0
    return candidates[idx]


def get_labels(contract_id: str) -> List[Dict[str, Any]]:
    """Return the ground-truth label list for a given contract_id."""
    corpus = _load_corpus()
    for c in corpus["contracts"]:
        if c["contract_id"] == contract_id:
            return c["labels"]
    raise ValueError(f"No contract with contract_id='{contract_id}'")


def list_contracts() -> List[Dict[str, Any]]:
    """Return a summary list of all contracts (no full text)."""
    corpus = _load_corpus()
    result = []
    for c in corpus["contracts"]:
        result.append({
            "contract_id": c["contract_id"],
            "difficulty": c["difficulty"],
            "title": c["title"],
            "parties": c["parties"],
            "num_labels": len(c["labels"]),
        })
    return result
