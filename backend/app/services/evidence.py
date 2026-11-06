"""
TruthLens External Evidence Retrieval Service
Integrates Tavily Search API to retrieve web evidence relevant to submitted claims.
Preserves strict separation between RoBERTa neural predictions and retrieved external sources.
Does not make automated fact-checking determinations from search relevance scores.
"""

import re
import logging
import urllib.parse
from typing import Dict, Any, List, Optional

from app.core.config import settings

logger = logging.getLogger("truthlens.evidence")

DEFAULT_EVIDENCE_SUMMARY = (
    "External search results retrieved. Their factual relevance has not been independently verified. "
    "Results are presented as unclassified context for human review."
)


class EvidenceService:
    """
    Service for querying external web evidence using Tavily Search API.
    Handles query normalization, URL deduplication, timeout management,
    and graceful error recovery when the search service is unavailable.
    """

    _instance: Optional["EvidenceService"] = None

    def __init__(self):
        self.api_key = settings.TAVILY_API_KEY
        self.default_depth = settings.TAVILY_SEARCH_DEPTH
        self.max_results = settings.TAVILY_MAX_RESULTS
        self.timeout = settings.TAVILY_TIMEOUT_SECONDS

    @classmethod
    def get_instance(cls) -> "EvidenceService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def normalize_query(claim: str) -> str:
        """
        Normalizes a claim into a concise, information-dense search query.
        Strips conversational framing, fact-checking dataset prefixes (e.g. 'Says that', 'Claims'),
        and redundant quotation marks without altering core semantic entities.
        """
        if not claim:
            return ""

        text = claim.strip()

        # Remove surrounding quotation marks
        text = text.strip("\"'“”‘’")

        # Strip common LIAR prefix patterns that hinder web search engines
        prefix_pattern = re.compile(
            r"^(says\s*:\s*|says\s+that\s+|says\s+|claimed\s+that\s+|claims\s+that\s+|claims\s+|stated\s+that\s+|states\s+)",
            re.IGNORECASE,
        )
        text = prefix_pattern.sub("", text).strip()

        # Remove trailing attribution tags often appended in debates (e.g. ', during the debate')
        text = re.sub(r"\s+", " ", text)

        # Truncate overly long queries to the first 25 words to keep search focused
        words = text.split()
        if len(words) > 25:
            text = " ".join(words[:25])

        return text.strip()

    @staticmethod
    def extract_domain(url: str) -> str:
        """Extracts the clean host domain from a URL."""
        try:
            parsed = urllib.parse.urlparse(url)
            netloc = parsed.netloc.lower()
            # Remove 'www.' prefix for cleaner display
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc or "unknown"
        except Exception:
            return "unknown"

    async def search_evidence(
        self,
        claim: str,
        max_results: Optional[int] = None,
        search_depth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves relevant external web evidence for a submitted claim via Tavily.

        Args:
            claim: Raw statement text.
            max_results: Optional override for result count (default: 5).
            search_depth: Optional search depth ('basic' or 'advanced').

        Returns:
            Dictionary matching the EvidencePayload schema specification:
            {
                "status": "success" | "unavailable" | "no_results",
                "query": normalized_query,
                "results": [...],
                "summary": "...",
                "error": Optional[str]
            }
        """
        query = self.normalize_query(claim)

        # 1. Verify API Key availability
        api_key = settings.TAVILY_API_KEY
        if not api_key or "your_tavily_api_key" in api_key.lower():
            logger.info("Tavily API key is not configured. Returning evidence unavailable status.")
            return {
                "status": "unavailable",
                "query": query,
                "results": [],
                "summary": "Tavily Search API key is not configured. External evidence retrieval is currently unavailable.",
                "error": "TAVILY_API_KEY environment variable is missing or placeholder.",
            }

        if not query:
            return {
                "status": "no_results",
                "query": "",
                "results": [],
                "summary": "Search query could not be constructed from the provided text.",
                "error": None,
            }

        limit = max_results or self.max_results
        depth = search_depth or self.default_depth

        try:
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=api_key)
            logger.info("Dispatching Tavily search: query='%s', depth='%s', limit=%d", query, depth, limit)

            response = await client.search(
                query=query,
                search_depth=depth,
                max_results=limit,
                timeout=self.timeout,
            )

            raw_results = response.get("results", []) if isinstance(response, dict) else []

            # 2. Deduplicate URLs & Structure Results
            seen_urls = set()
            structured_results: List[Dict[str, Any]] = []

            for item in raw_results:
                if not isinstance(item, dict):
                    continue

                url = item.get("url", "").strip()
                if not url:
                    continue

                # URL deduplication based on normalized URL (lowercase without trailing slash)
                norm_url = url.rstrip("/").lower()
                if norm_url in seen_urls:
                    continue
                seen_urls.add(norm_url)

                title = item.get("title", "").strip() or "Untitled Web Source"
                snippet = item.get("content", "").strip()
                domain = self.extract_domain(url)

                raw_score = item.get("score", 0.0)
                try:
                    relevance_score = round(max(0.0, min(1.0, float(raw_score))), 4)
                except (ValueError, TypeError):
                    relevance_score = 0.0

                published_date = item.get("published_date")
                pub_clean = str(published_date).strip() if published_date else None

                structured_results.append({
                    "title": title,
                    "url": url,
                    "domain": domain,
                    "snippet": snippet,
                    "relevance_score": relevance_score,
                    "published_date": pub_clean,
                    "classification": "unclassified",
                })

                if len(structured_results) >= limit:
                    break

            if not structured_results:
                return {
                    "status": "no_results",
                    "query": query,
                    "results": [],
                    "summary": "No external web sources were returned for this query.",
                    "error": None,
                }

            return {
                "status": "success",
                "query": query,
                "results": structured_results,
                "summary": DEFAULT_EVIDENCE_SUMMARY,
                "error": None,
            }

        except Exception as exc:
            # Mask API key if accidentally included in error string
            err_msg = str(exc)
            if api_key in err_msg:
                err_msg = err_msg.replace(api_key, "[REDACTED]")

            logger.warning("Tavily evidence search failed: %s", err_msg)
            return {
                "status": "unavailable",
                "query": query,
                "results": [],
                "summary": "External evidence retrieval encountered a temporary error or timeout.",
                "error": err_msg,
            }


# Global singleton instance
evidence_service = EvidenceService.get_instance()
