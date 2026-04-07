"""Typed Pydantic models for ContractRiskEnv."""

from typing import Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


RISK_TYPES = [
    "unlimited_liability",
    "auto_renewal",
    "unilateral_amendment",
    "ip_assignment_overreach",
    "indemnification_asymmetry",
    "data_ownership_ambiguity",
    "termination_without_cause",
]


class ClauseFlag(Action):
    """A single flagged clause within a contract."""

    clause_id: str = Field(..., description="Must match a clause_id in ground truth, e.g. 'clause_4_2'")
    risk_type: str = Field(..., description="One of the 7 risk types")
    severity: int = Field(..., ge=1, le=3, description="Severity 1-3")
    span_text: str = Field(..., description="Exact quoted text from the contract")


class ContractAction(Action):
    """Agent submits flagged clauses for a single contract."""

    flagged_clauses: List[ClauseFlag] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Agent self-reported confidence")
    reasoning: str = Field("", description="Chain-of-thought, logged but not graded")


class ContractObservation(Observation):
    """Returned by reset() and step()."""

    # done: bool and reward: Optional[float] inherited from Observation
    contract_text: str = ""
    task_id: str = ""
    difficulty: str = ""
    clause_count: int = 0
    feedback: Dict = Field(default_factory=dict)


class ContractState(State):
    """Episode metadata — revealed via state(), not observation."""

    # episode_id: Optional[str] and step_count: int inherited from State
    contract_id: str = ""
    task_id: str = ""
    total_risk_clauses: int = 0
