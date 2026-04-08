"""Deterministic grader — pure Python, no LLM calls.

reward = 0.35 * precision + 0.45 * recall - 0.10 * fp_penalty - 0.10 * severity_miss_penalty
clipped to [0.0, 1.0], all floats rounded to 4 decimal places.
"""

from __future__ import annotations

from typing import Any, Dict, List


def grade(flagged_clauses: List[Dict[str, Any]], labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score predicted clause flags against ground-truth labels.

    Parameters
    ----------
    flagged_clauses : list of dicts with at least {clause_id, risk_type, severity, span_text}
    labels          : list of ground-truth dicts  {clause_id, risk_type, severity, ...}

    Returns
    -------
    dict with: reward, precision, recall, f1, fp_penalty, severity_miss_penalty,
               missed_clauses, false_positives, severity_mismatches
    """
    gold_ids = {lbl["clause_id"] for lbl in labels}
    gold_map = {lbl["clause_id"]: lbl for lbl in labels}

    pred_ids = {fc["clause_id"] for fc in flagged_clauses}
    pred_map = {fc["clause_id"]: fc for fc in flagged_clauses}

    tp_ids = pred_ids & gold_ids
    fp_ids = pred_ids - gold_ids
    fn_ids = gold_ids - pred_ids

    tp = len(tp_ids)
    fp = len(fp_ids)
    fn = len(fn_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    fp_penalty = len(fp_ids) / max(len(pred_ids), 1)

    severity_mismatches: List[str] = []
    for cid in tp_ids:
        pred_sev = pred_map[cid].get("severity", 0)
        gold_sev = gold_map[cid].get("severity", 0)
        if abs(pred_sev - gold_sev) > 1:
            severity_mismatches.append(cid)

    severity_miss_penalty = len(severity_mismatches) / max(tp, 1)

    reward = 0.35 * precision + 0.45 * recall - 0.10 * fp_penalty - 0.10 * severity_miss_penalty
    # CLAMP BETWEEN (0.001, 0.999) strictly to satisfy validator
    reward = round(max(0.001, min(0.999, reward)), 4)

    return {
        "reward": reward,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fp_penalty": round(fp_penalty, 4),
        "severity_miss_penalty": round(severity_miss_penalty, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "missed_clauses": sorted(fn_ids),
        "false_positives": sorted(fp_ids),
        "severity_mismatches": sorted(severity_mismatches),
    }
