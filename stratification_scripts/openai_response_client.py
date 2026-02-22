"""
OpenAI GPT response tracker for agency response tracking.

Drop-in replacement for GeminiResponseTracker (Tier 1 only) using
OpenAI's Responses API with web search grounding and structured outputs.

Uses GPT-5-mini by default with `client.responses.parse()` for structured
JSON output and `tools=[{"type": "web_search"}]` for web search.

Example:
    >>> from stratification_scripts.openai_response_client import OpenAIResponseTracker
    >>> tracker = OpenAIResponseTracker(api_key="sk-...")
    >>> result = await tracker.track_response(
    ...     "I am concerned about this rule...",
    ...     {"comment_id": "ABC-123", "agency": "EPA"}
    ... )
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError
from tqdm import tqdm

from stratification_scripts.logging_utils import get_logger
from stratification_scripts.gemini_client import (
    AgencyResponse,
    RESPONSE_TRACKING_PROMPT,
)

logger = get_logger(__name__)


class OpenAIResponseTracker:
    """
    Async-compatible OpenAI client for tracking agency responses.

    Uses GPT-5-mini with the Responses API for structured output and
    web search grounding. Sync calls are wrapped in asyncio.to_thread()
    for async compatibility with the existing pipeline.

    Attributes:
        api_key: OpenAI API key
        model: Model to use (default: gpt-5-mini)
        max_retries: Maximum retry attempts per request
        enable_search: Enable web search tool
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini",
        max_retries: int = 5,
        enable_search: bool = True,
    ) -> None:
        from openai import OpenAI

        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.enable_search = enable_search
        self._client = OpenAI(api_key=api_key)
        self._tools = [{"type": "web_search"}] if enable_search else []

        logger.debug(
            f"Initialized OpenAIResponseTracker model={model} search={enable_search}"
        )

    def _is_retryable_error(self, e: Exception) -> bool:
        """Determine if an error is retryable (same logic as Gemini tracker)."""
        msg = str(e).lower()
        if "429" in msg or "rate limit" in msg or "quota" in msg:
            return True
        if "timeout" in msg or "timed out" in msg:
            return True
        if "500" in msg or "502" in msg or "503" in msg or "504" in msg:
            return True
        if "400" in msg or "invalid" in msg or "bad request" in msg:
            return False
        if "401" in msg or "403" in msg or "permission" in msg:
            return False
        if "404" in msg or "not found" in msg:
            return False
        return True

    def _track_response_sync(
        self,
        comment_text: str,
        comment_metadata: Dict[str, str],
    ) -> Tuple[str, Dict[str, str], str]:
        """
        Synchronous tracking call. Invoked via asyncio.to_thread().

        Uses OpenAI Responses API with web search and structured output.
        """
        max_chars = 20000
        if len(comment_text) > max_chars:
            comment_text = comment_text[:max_chars] + "\n\n[... truncated ...]"

        prompt = RESPONSE_TRACKING_PROMPT.format(
            comment_id=comment_metadata.get("comment_id", "N/A"),
            document_number=comment_metadata.get("document_number", "N/A"),
            agency=comment_metadata.get("agency", "N/A"),
            commenter_type=comment_metadata.get("commenter_type", "N/A"),
            submission_date=comment_metadata.get("submission_date", "N/A"),
            full_comment_text=comment_text,
        )

        comment_id = comment_metadata.get("comment_id", "unknown")
        backoff = 2.0

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"OpenAI API call attempt {attempt + 1}/{self.max_retries} "
                    f"for comment_id={comment_id}, model={self.model}"
                )

                response = self._client.responses.parse(
                    model=self.model,
                    instructions=(
                        "You are analyzing a public comment submitted to a U.S. federal agency. "
                        "Search the web to find if the agency responded. "
                        "Return structured JSON matching the required schema."
                    ),
                    input=prompt,
                    tools=self._tools,
                    text_format=AgencyResponse,
                )

                # Try parsed output first
                parsed = response.output_parsed
                if parsed is not None:
                    result = parsed.normalized()
                    logger.debug(f"Successfully parsed response for {comment_id}")
                    raw_preview = (response.output_text or "")[:500]
                    return (comment_id, result, raw_preview or "OK_JSON")

                # Fallback: try to parse output_text as JSON
                if response.output_text:
                    try:
                        manual = AgencyResponse.model_validate_json(response.output_text)
                        return (comment_id, manual.normalized(), response.output_text[:500])
                    except (ValidationError, Exception) as parse_err:
                        logger.warning(
                            f"OpenAI: output_parsed=None, manual parse failed for {comment_id}: {parse_err}"
                        )

                # Empty response
                logger.warning(f"OpenAI: empty response for {comment_id}")
                return (
                    comment_id,
                    AgencyResponse(
                        response_found="uncertain",
                        agency_decision="uncertain",
                        response_text="N/A",
                        response_location="N/A",
                        reasoning="Empty model response",
                    ).normalized(),
                    "EMPTY_RESPONSE",
                )

            except ValidationError as ve:
                logger.warning(
                    f"VALIDATION_ERROR: schema validation failed for {comment_id}: {ve}"
                )
                return (
                    comment_id,
                    AgencyResponse(
                        response_found="uncertain",
                        agency_decision="uncertain",
                        response_text="N/A",
                        response_location="N/A",
                        reasoning="VALIDATION_ERROR: Structured output validation failed",
                    ).normalized(),
                    "ERROR: schema_validation_failed",
                )

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)

                logger.error(
                    f"OpenAI API error for {comment_id} "
                    f"(attempt {attempt + 1}/{self.max_retries}): {error_type}: {error_msg}"
                )

                retryable = self._is_retryable_error(e)

                if attempt >= self.max_retries - 1 or not retryable:
                    logger.warning(
                        f"OpenAI call failed (retryable={retryable}) after attempt {attempt + 1}: "
                        f"{error_type}: {error_msg}"
                    )
                    return (
                        comment_id,
                        AgencyResponse(
                            response_found="uncertain",
                            agency_decision="uncertain",
                            response_text="N/A",
                            response_location="N/A",
                            reasoning=f"API error: {error_type}",
                        ).normalized(),
                        f"ERROR: {error_msg}",
                    )

                jitter = random.uniform(0.5, 2.0)
                sleep_time = backoff + jitter
                if "429" in str(e).lower():
                    sleep_time += 10.0

                logger.debug(
                    f"Retrying after error (attempt {attempt + 1}): "
                    f"{error_type}: {error_msg} (sleep {sleep_time:.1f}s)"
                )
                time.sleep(sleep_time)
                backoff = min(backoff * 2, 60.0)

        # Should never reach here
        return (
            comment_id,
            AgencyResponse(
                response_found="uncertain",
                agency_decision="uncertain",
                response_text="N/A",
                response_location="N/A",
                reasoning="API retries exhausted",
            ).normalized(),
            "ERROR: retries_exhausted",
        )

    async def track_response(
        self,
        comment_text: str,
        comment_metadata: Dict[str, str],
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Tuple[str, Dict[str, str], str]:
        """
        Track agency response for a single comment (async wrapper).

        Args:
            comment_text: Full comment text
            comment_metadata: Dict with comment_id, document_number, agency, etc.
            semaphore: Optional semaphore for concurrency control

        Returns:
            Tuple of (comment_id, parsed_response_dict, raw_model_response)
        """
        async def do_request():
            return await asyncio.to_thread(
                self._track_response_sync, comment_text, comment_metadata
            )

        if semaphore:
            async with semaphore:
                return await do_request()
        return await do_request()

    async def track_batch(
        self,
        comments: List[Tuple[str, Dict[str, str]]],
        max_concurrency: int = 10,
    ) -> List[Tuple[str, Dict[str, str], str]]:
        """
        Track agency responses for a batch of comments concurrently.

        Args:
            comments: List of (comment_text, comment_metadata) tuples
            max_concurrency: Maximum concurrent API calls (default: 10)

        Returns:
            List of (comment_id, parsed_response_dict, raw_response) tuples
        """
        if not comments:
            return []

        semaphore = asyncio.Semaphore(max_concurrency)

        async def run_single(text: str, metadata: Dict[str, str]):
            return await self.track_response(text, metadata, semaphore)

        tasks = [asyncio.create_task(run_single(text, meta)) for text, meta in comments]

        results: List[Tuple[str, Dict[str, str], str]] = []
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Tracking responses (OpenAI)",
        ):
            results.append(await task)

        return results

    async def close(self) -> None:
        """Close the client (placeholder for compatibility)."""
        pass
