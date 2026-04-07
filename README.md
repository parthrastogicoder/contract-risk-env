# ContractRiskEnv

ContractRiskEnv is a robust reinforcement learning environment built on the OpenEnv specification for training and evaluating LLM agents on legal contract risk detection.

## Overview
The task involves reading a synthetic legal contract and identifying risky clauses across three difficulty levels:
- **Easy:** Explicit risks (e.g., "Term extends automatically for 10 years").
- **Medium:** Risks hidden across sections or via broad definitions.
- **Hard:** Clauses that appear protective but contain negating "notwithstanding" exceptions.

Agents are graded based on precision, recall, and F1-score compared directly to a set of ground-truth annotations.

## Repository Structure

```
├── contract_risk_env/     # Python package for the environment
│   ├── server/            # FastAPI Server (OpenEnv runner)
│   ├── models.py          # Pydantic Schemas
│   └── client.py          # WebSocket client
├── inference.py           # Competition-compliant LLM runner
├── openenv.yaml           # Deployment manifest
├── Dockerfile             # Docker environment definition
├── requirements.txt       # Environment dependencies
└── contracts_corpus.json  # Corpus of contracts + ground truth labels
```

## Running Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the OpenEnv server:**
   ```bash
   uvicorn contract_risk_env.server.app:app --host 0.0.0.0 --port 8000
   ```

3. **Run inference:**
   Provide your OpenAI/HF credentials via environment variables:
   ```bash
   export HF_TOKEN="sk-proj-your-key"
   export API_BASE_URL="https://api.openai.com/v1"
   export MODEL_NAME="gpt-4o"
   python inference.py
   ```

## Output Format
The `inference.py` script emits strictly formatted OpenEnv outputs required for leaderboard evaluation:
```
[START] task=easy env=contract_risk_env model=gpt-4o
[STEP] step=1 action=flagged_5_clauses reward=0.72 done=true error=null
[END] success=true steps=1 score=0.725 rewards=0.72
```

## Validation & Evaluation
You can validate the environment natively using:
```bash
openenv validate
```
Or execute the HF validation script:
```bash
./validation.sh https://<your-hf-space-url>.hf.space
```
