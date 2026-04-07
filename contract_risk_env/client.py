"""Client — pip-installable OpenEnv client for ContractRiskEnv."""

from __future__ import annotations

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import ContractAction, ContractObservation, ContractState


class ContractRiskEnv(EnvClient[ContractAction, ContractObservation, ContractState]):
    """WebSocket-based client for ContractRiskEnv."""

    def _step_payload(self, action: ContractAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=ContractObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                contract_text=obs_data.get("contract_text", ""),
                task_id=obs_data.get("task_id", ""),
                difficulty=obs_data.get("difficulty", ""),
                clause_count=obs_data.get("clause_count", 0),
                feedback=obs_data.get("feedback", {}),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> ContractState:
        return ContractState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            contract_id=payload.get("contract_id", ""),
            task_id=payload.get("task_id", ""),
            total_risk_clauses=payload.get("total_risk_clauses", 0),
        )
