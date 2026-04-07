# ContractRiskEnv

An [OpenEnv](https://huggingface.co/collections/openenv/environment-hub)-compliant reinforcement learning environment for **legal contract risk clause detection**.

An agent reads synthetic legal contracts and flags risky clauses. The grader is fully deterministic — precision/recall/F1 against ground-truth clause labels. No LLM calls inside the grader.

## Tasks

| Task ID  | Difficulty | Description | Expected Baseline F1 |
|----------|-----------|-------------|---------------------|
| `easy`   | Easy      | SaaS agreements — explicit risks with clear headings | ~0.85 |
| `medium` | Medium    | IP licensing — risks buried in cross-references | ~0.51 |
| `hard`   | Hard      | Enterprise MSA — protective-looking clauses with negating exceptions | ~0.29 |

## Risk Types

`unlimited_liability` · `auto_renewal` · `unilateral_amendment` · `ip_assignment_overreach` · `indemnification_asymmetry` · `data_ownership_ambiguity` · `termination_without_cause`

## Reward Function

```
reward = 0.35 × precision + 0.45 × recall − 0.10 × fp_penalty − 0.10 × severity_miss_penalty
```

Recall is weighted higher — missing a real risk is worse than a false alarm.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run server
uvicorn contract_risk_env.server.app:app --host 0.0.0.0 --port 8000 --reload

# Check health
curl http://localhost:8000/health

# Run LLM baseline (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python -m contract_risk_env.inference
```

## Client Usage

```python
from contract_risk_env.client import ContractRiskEnv
from contract_risk_env.models import ContractAction, ClauseFlag

with ContractRiskEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="easy")
    obs = result.observation
    print(obs.contract_text[:200])

    action = ContractAction(
        flagged_clauses=[
            ClauseFlag(clause_id="clause_4_2", risk_type="auto_renewal", severity=2, span_text="..."),
        ],
        confidence=0.8,
        reasoning="Found auto-renewal clause in Section 4.2",
    )
    result = env.step(action)
    print(f"Reward: {result.reward}")
    print(f"Feedback: {result.observation.feedback}")
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit action |
| `/state` | GET | Get episode state |
| `/tasks` | GET | List all tasks with schemas |
| `/grader` | POST | Score a completed episode |
| `/baseline` | GET | Run keyword heuristic baseline |
| `/docs` | GET | OpenAPI documentation |

## Docker

```bash
docker build -t contract-risk-env -f server/Dockerfile ..
docker run -p 8000:8000 contract-risk-env
```

## Project Structure

```
contract_risk_env/
├── models.py          — Pydantic models (Action, Observation, State)
├── client.py          — OpenEnv WebSocket client
├── inference.py       — GPT-4o baseline inference script
├── server/
│   ├── app.py         — FastAPI app + competition endpoints
│   ├── environment.py — RL environment (reset/step/state)
│   ├── corpus.py      — Contract corpus loader
│   ├── graders.py     — Deterministic grader
│   └── Dockerfile
├── data/
│   └── contracts_corpus.json
├── openenv.yaml
├── pyproject.toml
└── requirements.txt
```

## License

MIT
