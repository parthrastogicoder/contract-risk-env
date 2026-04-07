"""ContractRiskEnvironment — 1-step RL env for legal clause risk detection."""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

from contract_risk_env.models import ContractAction, ContractObservation, ContractState
from . import corpus
from .graders import grade


def _count_clauses(text: str) -> int:
    """Rough clause count — count numbered sections like '1.1', '2.3', etc."""
    return max(len(re.findall(r"\n\d+\.\d+\s", text)), 1)


class ContractRiskEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._state = ContractState()
        self._contract: Dict[str, Any] = {}
        self._labels: List[Dict[str, Any]] = []
        self._done = False

    def reset(
        self,
        task_id: str = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ContractObservation:
        contract = corpus.get_contract(task_id, seed=seed)
        self._contract = contract
        self._labels = contract["labels"]
        self._done = False

        self._state = ContractState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            contract_id=contract["contract_id"],
            task_id=task_id,
            total_risk_clauses=len(self._labels),
        )

        return ContractObservation(
            done=False,
            reward=None,
            contract_text=contract["text"],
            task_id=task_id,
            difficulty=contract["difficulty"],
            clause_count=_count_clauses(contract["text"]),
            feedback={},
        )

    def step(self, action: ContractAction, **kwargs: Any) -> ContractObservation:
        self._state.step_count += 1
        self._done = True

        flagged_dicts = [fc.model_dump() for fc in action.flagged_clauses]
        result = grade(flagged_dicts, self._labels)

        feedback = {
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "reward": result["reward"],
            "missed_clauses": result["missed_clauses"],
            "false_positives": result["false_positives"],
            "severity_mismatches": result["severity_mismatches"],
        }

        return ContractObservation(
            done=True,
            reward=result["reward"],
            contract_text=self._contract["text"],
            task_id=self._state.task_id,
            difficulty=self._contract["difficulty"],
            clause_count=_count_clauses(self._contract["text"]),
            feedback=feedback,
        )

    @property
    def state(self) -> ContractState:
        return self._state
