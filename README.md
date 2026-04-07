---
title: Contract Risk Env
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# ContractRiskEnv ⚖️

ContractRiskEnv is a robust, highly-structured reinforcement learning environment built on the **OpenEnv** specification. It is designed to train and strictly evaluate LLM agents on complex legal risk detection tasks.

---

## 🌍 Real-World Usecases
Automating legal review is one of the highest ROI applications for modern AI. **ContractRiskEnv** bridges the gap between raw LLM generation and actionable legal triage by providing a structured benchmark for:

1. **M&A Due Diligence:** During mergers or acquisitions, lawyers must review thousands of legacy contracts. Agents trained in this environment can automatically flag hidden IP assignment overreaches or evergreen terms that dramatically impact company valuations.
2. **Automated Playbook Compliance:** In-house legal teams at enterprises maintain "playbooks" defining acceptable contract terms. This environment trains agents to scan incoming vendor agreements and immediately flag clauses that breach the playbook (e.g., unlimited liability caps or unilateral amendment rights).
3. **Legal Triage & Cost Reduction:** Before escalating standard SaaS agreements or NDAs to costly external counsel, an AI system can pre-screen documents. By correctly identifying risks early, companies drastically reduce external legal billing hours.

---

## 🎯 The Environment Challenge

The core task involves a legal AI agent reading a synthetic legal contract and precisely identifying risky clauses. Because legal text is notoriously dense, evaluating models objectively requires moving beyond basic text generation.

Agents are graded on their ability to detect risks across three tier levels:
- **Easy:** Explicit risks with clear, standard legal phrasing (e.g., "This Term shall extend automatically for 10 successive years").
- **Medium:** Risks obfuscated across multiple sections, relying on cross-references between a "Definitions" section and an operative clause.
- **Hard:** Broad protective clauses that seem harmless but contain devastating `notwithstanding` exceptions or subtle legal loopholes deeply buried in the text.

### Penalties & Strict Evaluation
- Agents are graded deterministically using **Precision**, **Recall**, and **F1-Score** against a set of expert ground-truth annotations.
- The environment penalizes hallucinated clauses and over-flagging (false positives). Agents must be both exhaustively thorough (high recall) and extremely accurate (high precision).

---

## 🛠 Repository Structure

```
├── server/                # FastAPI Server (OpenEnv compliant target)
│   ├── app.py             # Server endpoints & baseline heuristic
│   ├── environment.py     # Reset/Step logic interface
│   ├── corpus.py          # Data ingestion and target sampling
│   └── graders.py         # Precision/Recall penalty calculation 
├── contract_risk_env/     # Shared packages
│   ├── models.py          # Pydantic schemas (State, Action, Observation)
│   └── client.py          # WebSocket / API Client logic
├── contracts_corpus.json  # Corpus of contracts + master ground truth labels
├── inference.py           # Competition-compliant LLM inference runner
├── openenv.yaml           # Deployment manifest
├── Dockerfile             # Container definition for HF Spaces deployment
├── uv.lock                # Deterministic dependency locking
└── pyproject.toml         # Packaging configuration and entry points
```

---

## 🚀 Running Locally

1. **Install dependencies using `uv` (Recommended):**
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```

2. **Start the OpenEnv server:**
   ```bash
   uvicorn server.app:app --host 0.0.0.0 --port 8000
   ```

3. **Run AI Inference:**
   Provide your OpenAI and HuggingFace credentials via environment variables to run the full OpenEnv submission runner:
   ```bash
   export HF_TOKEN="your-hf-token"
   export API_BASE_URL="https://api.openai.com/v1"
   export MODEL_NAME="gpt-4o"
   python inference.py
   ```

## 📊 Leaderboard Output Format

The `inference.py` script emits strictly formatted OpenEnv outputs required for automated leaderboard evaluation. The log parsing constraints are extremely rigid to prevent score manipulation.

```
[START] task=easy env=contract_risk_env model=gpt-4o
[STEP] step=1 action=flagged_5_clauses reward=0.72 done=true error=null
[END] success=true steps=1 score=0.725 rewards=0.72
```

## ✅ Validation & Deployment

Before submitting or deploying, validate the environment locally:
```bash
openenv validate
```

You can execute the automated remote validation script against a live HuggingFace Space deployment:
```bash
./validation.sh https://<your-hf-space-url>.hf.space
```
