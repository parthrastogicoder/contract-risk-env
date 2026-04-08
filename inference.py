"""
ContractRiskEnv Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from contract_risk_env.client import ContractRiskEnv
from contract_risk_env.models import ContractAction, ClauseFlag

# ── Environment variables (competition-mandated) ────────────────────────────
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o"

BENCHMARK = "contract_risk_env"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 1  # 1-step environment: read contract → flag clauses
TEMPERATURE = 0.0
MAX_TOKENS = 4096

SYSTEM_PROMPT = textwrap.dedent("""
You are a legal contract risk analyst. You will be given a contract and must identify ALL risky clauses.

For each risky clause, return a JSON object with:
- clause_id: string matching the section numbering, e.g. "clause_4_2" for Section 4.2, "clause_6_3b" for a sub-part
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

Return ONLY valid JSON, no markdown fences.
""").strip()


# ── Structured stdout logging ───────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call ────────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, contract_text: str) -> dict:
    """Send contract to LLM, return parsed action dict."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this contract for risky clauses:\n\n{contract_text}"},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Clean markdown fences in case universal model emits them without JSON mode
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return {"flagged_clauses": [], "confidence": 0.0, "reasoning": str(exc)}


# ── Episode runner ──────────────────────────────────────────────────────────

async def run_episode(env, llm_client: OpenAI, task_id: str) -> float:
    """Run one episode (one task). Returns score in [0, 1]."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation
        contract_text = obs.contract_text

        # Step 1: Call LLM to analyze contract, then submit action
        action_dict = call_llm(llm_client, contract_text)

        flagged = []
        for fc in action_dict.get("flagged_clauses", []):
            try:
                flagged.append(ClauseFlag(
                    clause_id=fc.get("clause_id", "unknown"),
                    risk_type=fc.get("risk_type", "unlimited_liability"),
                    severity=max(1, min(3, int(fc.get("severity", 1)))),
                    span_text=fc.get("span_text", "")[:300],
                ))
            except Exception:
                continue

        action = ContractAction(
            flagged_clauses=flagged,
            confidence=float(action_dict.get("confidence", 0.5)),
            reasoning=str(action_dict.get("reasoning", ""))[:500],
        )

        result = await env.step(action)
        reward = result.reward or 0.0
        done = result.done
        error = None

        rewards.append(reward)
        steps_taken = 1

        action_summary = f"flagged_{len(flagged)}_clauses"
        log_step(step=1, action=action_summary, reward=reward, done=done, error=error)

        score = reward  # reward is already in [0, 1]
        success = score > 0.0

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        if not rewards:
            rewards.append(0.0)
            steps_taken = 1
            log_step(step=1, action="error", reward=0.0, done=True, error=str(exc))

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ────────────────────────────────────────────────────────────────────

async def main() -> None:
    # Log all relevant env vars so we can diagnose what the validator injects
    print(f"[DEBUG] IMAGE_NAME={IMAGE_NAME!r}", flush=True)
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL!r}", flush=True)
    print(f"[DEBUG] API_KEY={'set' if API_KEY else 'NOT SET'}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME!r}", flush=True)
    print(f"[DEBUG] ENV_BASE_URL={os.getenv('ENV_BASE_URL')!r}", flush=True)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to environment — try strategies in order
    env = None
    if IMAGE_NAME:
        print(f"[DEBUG] Strategy: from_docker_image({IMAGE_NAME})", flush=True)
        env = await ContractRiskEnv.from_docker_image(IMAGE_NAME)
    else:
        env_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")
        print(f"[DEBUG] Strategy: URL-based connection to {env_url}", flush=True)
        env = ContractRiskEnv(base_url=env_url)
        await env.connect()
    
    print("[DEBUG] Environment connected successfully", flush=True)

    try:
        scores = []
        for task_id in TASKS:
            score = await run_episode(env, llm_client, task_id)
            scores.append(score)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        print(f"\n[SUMMARY] mean_score={mean_score:.3f} easy={scores[0]:.3f} medium={scores[1]:.3f} hard={scores[2]:.3f}", flush=True)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BaseException as exc:
        # Emit valid structured output even on fatal crash so the validator sees something
        print(f"[DEBUG] Fatal error ({type(exc).__name__}): {exc}", flush=True)
        err_str = str(exc).replace('\n', ' ')[:200]
        for task_id in TASKS:
            print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)
            print(f"[STEP] step=1 action=error reward=0.01 done=true error={err_str}", flush=True)
            print(f"[END] success=false steps=1 score=0.01 rewards=0.01", flush=True)

