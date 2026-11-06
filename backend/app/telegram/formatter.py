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

MAX_TELEGRAM_MESSAGE_LENGTH = 4000  # Below Telegram's 4096 character limit


def format_analysis_message(claim_text: str, data: Dict[str, Any]) -> str:
    """
    Constructs a readable analysis message adhering to TruthLens Section 11 specifications.
    """
    prediction_obj = data.get("prediction", {})
    raw_label = prediction_obj.get("label", "unknown")
    pred_label_upper = str(raw_label).upper()
    confidence_val = prediction_obj.get("confidence", 0.0)
    confidence_pct = round(confidence_val * 100)

    model_name = data.get("model", "RoBERTa + Metadata")
    class_scores = prediction_obj.get("class_scores", {})

    lines: List[str] = [
        "TRUTHLENS ANALYSIS",
        "",
        "Claim:",
        f'"{claim_text}"',
        "",
        "MODEL PREDICTION",
        "",
        f"Prediction: {pred_label_upper}",
        f"Confidence: {confidence_pct}%",
        "",
        f"Model: {model_name}",
        "",
        "CLASS SCORES",
        "",
    ]

    for label in LABEL_ORDER:
        score = class_scores.get(label, 0.0)
        pct = round(score * 100)
        display = LABEL_DISPLAY_NAMES.get(label, label.capitalize())
        lines.append(f"- {display}: {pct}%")

    lines.append("")
    lines.append("EXTERNAL EVIDENCE")
    lines.append("")

    evidence_obj = data.get("evidence")
    if evidence_obj and isinstance(evidence_obj, dict):
        status = evidence_obj.get("status", "unavailable")
        results = evidence_obj.get("results", [])

        if status == "success" and results:
            for idx, item in enumerate(results, start=1):
                title = item.get("title", "Untitled Source").strip()
                domain = item.get("domain", "unknown").strip()
                relevance = item.get("relevance_score", 0.0)
                url = item.get("url", "").strip()

                lines.append(f"{idx}. {title}")
                lines.append(f"   Domain: {domain}")
                lines.append(f"   Relevance: {relevance:.2f}")
                lines.append(f"   Link: {url}")
                lines.append("")

            lines.append("Evidence status: Retrieved.")
        elif status == "no_results":
            lines.append("Evidence status: No relevant external sources found.")
        else:
            lines.append("Evidence status: Search unavailable or temporarily disabled.")
    else:
        lines.append("Evidence status: Not requested or unavailable.")

    lines.append("")
    lines.append("DISCLAIMER:")
    lines.append(
        "This is an automated model prediction, not confirmed factual verification. "
        "External search results must be independently evaluated."
    )

    full_message = "\n".join(lines)

    # Ensure message does not exceed Telegram's length limit
    if len(full_message) > MAX_TELEGRAM_MESSAGE_LENGTH:
        full_message = full_message[: MAX_TELEGRAM_MESSAGE_LENGTH - 50] + "\n\n... [Message truncated for length]"

    return full_message
