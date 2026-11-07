"""
TruthLens Telegram Message Formatter
Formats analysis reports from FastAPI /api/analyze into structured Telegram messages
conforming strictly to the project specification and respecting message length limits.
"""

from typing import Dict, Any, List

LABEL_ORDER = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]

LABEL_DISPLAY_NAMES = {
    "pants-fire": "Pants-fire",
    "false": "False",
    "barely-true": "Barely-true",
    "half-true": "Half-true",
    "mostly-true": "Mostly-true",
    "true": "True",
}

OVERALL_ASSESSMENT_DISPLAY = {
    "supports": "SUPPORTS CLAIM",
    "contradicts": "CONTRADICTS CLAIM",
    "mixed": "MIXED / CONFLICTING EVIDENCE",
    "insufficient_evidence": "INSUFFICIENT EVIDENCE",
}

MAX_TELEGRAM_MESSAGE_LENGTH = 3900  # Well below Telegram's 4096 character limit


def format_analysis_message(claim_text: str, data: Dict[str, Any]) -> str:
    """
    Constructs a readable analysis message adhering to TruthLens Section 10 specifications:
    1. Model Prediction (Predicted label, confidence)
    2. Evidence Assessment (Overall assessment, supporting/contradicting/neutral counts, summary)
    3. Sources (Article title, source URL, relationship)
    4. Limitations (Passage depth, conflicting reports, or missing data)
    """
    prediction_obj = data.get("prediction", {})
    raw_label = prediction_obj.get("label", "unknown")
    pred_label_upper = str(raw_label).upper()
    confidence_val = prediction_obj.get("confidence", 0.0)
    confidence_pct = round(confidence_val * 100)

    model_name = data.get("model", "DeBERTa-v3 + Metadata")
    class_scores = prediction_obj.get("class_scores", {})

    lines: List[str] = [
        "TRUTHLENS ANALYSIS",
        "",
        "Claim:",
        f'"{claim_text}"',
        "",
        "1. MODEL PREDICTION",
        f"Prediction: {pred_label_upper}",
        f"Confidence: {confidence_pct}%",
        f"Model: {model_name}",
        "",
        "CLASS SCORES",
    ]

    for label in LABEL_ORDER:
        score = class_scores.get(label, 0.0)
        pct = round(score * 100)
        display = LABEL_DISPLAY_NAMES.get(label, label.capitalize())
        lines.append(f"- {display}: {pct}%")

    lines.append("")
    lines.append("2. EVIDENCE ASSESSMENT")

    evidence_obj = data.get("evidence")
    results = []
    limitations: List[str] = []

    if evidence_obj and isinstance(evidence_obj, dict):
        status = evidence_obj.get("status", "unavailable")
        results = evidence_obj.get("results", [])
        overall_assessment = evidence_obj.get("overall_assessment")
        analysis_status = evidence_obj.get("analysis_status")
        analysis_summary = evidence_obj.get("analysis_summary")
        limitations = evidence_obj.get("limitations", [])

        # Count relationships
        supports_count = sum(1 for r in results if r.get("classification") == "supports")
        contradicts_count = sum(1 for r in results if r.get("classification") == "contradicts")
        neutral_count = sum(1 for r in results if r.get("classification") in ("neutral", "unclassified"))

        if overall_assessment:
            verdict_text = OVERALL_ASSESSMENT_DISPLAY.get(
                overall_assessment, overall_assessment.replace("_", " ").upper()
            )
            lines.append(f"Overall Assessment: {verdict_text}")
            lines.append(f"Supporting Sources: {supports_count}")
            lines.append(f"Contradicting Sources: {contradicts_count}")
            lines.append(f"Neutral Sources: {neutral_count}")
            if analysis_summary:
                lines.append(f"Synthesis: {analysis_summary}")
        elif analysis_status == "failed":
            lines.append("Overall Assessment: Analysis Failed")
            lines.append("LLM evidence evaluation encountered a temporary service disruption.")
        elif status == "no_results":
            lines.append("Overall Assessment: INSUFFICIENT EVIDENCE")
            lines.append("No external web articles found for this claim.")
        elif status == "unavailable":
            lines.append("Overall Assessment: Retrieval Unavailable")
            lines.append("External search service is currently unconfigured or unreachable.")
        else:
            lines.append(f"Overall Assessment: Unclassified ({len(results)} sources retrieved)")
            lines.append("LLM evidence analysis is not enabled on this server.")
    else:
        lines.append("Evidence Status: Not requested or unavailable.")

    lines.append("")
    lines.append("3. SOURCES & EXTERNAL EVIDENCE")
    lines.append("")

    if results:
        for idx, item in enumerate(results, start=1):
            title = item.get("title", "Untitled Source").strip()
            url = item.get("url", "").strip()
            domain = item.get("domain", "unknown").strip()
            relevance = item.get("relevance_score", 0.0)
            classification = item.get("classification", "unclassified").capitalize()
            reasoning = item.get("reasoning")

            lines.append(f"{idx}. {title}")
            lines.append(f"   Domain: {domain}")
            lines.append(f"   Relationship: {classification}")
            lines.append(f"   Relevance: {relevance:.2f}")
            if reasoning:
                short_reason = reasoning[:120] + "..." if len(reasoning) > 120 else reasoning
                lines.append(f"   Reasoning: {short_reason}")
            lines.append(f"   Link: {url}")
            lines.append("")

        lines.append("Evidence status: Retrieved.")
    else:
        if evidence_obj and isinstance(evidence_obj, dict) and evidence_obj.get("status") == "no_results":
            lines.append("Evidence status: No relevant external sources found.")
        elif evidence_obj and isinstance(evidence_obj, dict) and evidence_obj.get("status") == "unavailable":
            lines.append("Evidence status: Search unavailable or temporarily disabled.")
        else:
            lines.append("Evidence status: Not requested or unavailable.")
        lines.append("")

    if limitations:
        lines.append("4. LIMITATIONS & CAVEATS")
        for lim in limitations:
            lines.append(f"- {lim}")
        lines.append("")

    lines.append("DISCLAIMER:")
    lines.append(
        "TruthLens provides neural predictions and LLM-assisted evidence assessments. "
        "Neither constitutes independent factual verification. Consult original sources."
    )

    full_message = "\n".join(lines)

    if len(full_message) > MAX_TELEGRAM_MESSAGE_LENGTH:
        full_message = full_message[: MAX_TELEGRAM_MESSAGE_LENGTH - 60] + "\n\n... [Message truncated for length]"

    return full_message
