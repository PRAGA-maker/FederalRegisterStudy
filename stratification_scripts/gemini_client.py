"""
Gemini API client for agency response tracking.

This module provides an async wrapper for Google's Gemini API with:
- Retry logic with exponential backoff
- Rate limit handling  
- Batch response tracking support
- Google Search grounding integration
- Structured outputs using Pydantic schemas

Example:
    >>> from stratification_scripts.gemini_client import GeminiResponseTracker
    >>> from stratification_scripts.config import get_gemini_api_key
    >>> 
    >>> api_key = get_gemini_api_key(required=True)
    >>> tracker = GeminiResponseTracker(api_key)
    >>> 
    >>> result = await tracker.track_response(
    ...     comment_text="I am concerned about this rule...",
    ...     comment_metadata={"comment_id": "ABC-123", "agency": "EPA"}
    ... )
"""

from __future__ import annotations

import asyncio
import random
from typing import Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from stratification_scripts.logging_utils import get_logger

logger = get_logger(__name__)


# =========================
# Structured Output Schema
# =========================

ResponseFound = Literal["yes", "no", "uncertain"]
AgencyDecision = Literal["accept", "reject", "uncertain"]


class AgencyResponse(BaseModel):
    """Structured response schema for agency response tracking."""
    
    response_found: ResponseFound = Field(
        description='Whether a response exists: "yes" | "no" | "uncertain"'
    )
    agency_decision: AgencyDecision = Field(
        description='Only meaningful if response_found="yes": "accept" | "reject" | "uncertain"'
    )
    response_text: str = Field(
        description='Agency response text, or "N/A" if none found'
    )
    response_location: str = Field(
        description='URL or location description, or "N/A"'
    )
    reasoning: str = Field(
        description="Brief explanation of determination (1-2 sentences)"
    )
    
    def normalized(self) -> Dict[str, str]:
        """Normalize output for CSV storage."""
        d = self.model_dump()
        
        # Enforce decision consistency
        if d["response_found"] != "yes":
            d["agency_decision"] = "uncertain"
            if not d["response_text"] or d["response_text"].lower() in ["", "none", "null"]:
                d["response_text"] = "N/A"
            if not d["response_location"] or d["response_location"].lower() in ["", "none", "null"]:
                d["response_location"] = "N/A"
        
        # Cap response_text length to reduce CSV bloat (keep at ~4000 chars)
        if len(d["response_text"]) > 4000:
            d["response_text"] = d["response_text"][:4000] + " ...[truncated]"
        
        return d


# Response tracking prompt template (no XML, no JSON examples)
RESPONSE_TRACKING_PROMPT = """You are analyzing a public comment submitted to a U.S. federal agency.

COMMENT INFORMATION:
- Comment ID: {comment_id}
- Document Number: {document_number}
- Agency: {agency}
- Commenter Type: {commenter_type}
- Submission Date: {submission_date}

COMMENT TEXT (truncated if very long):
{full_comment_text}

TASK:
1. Use web grounding (search) to find whether the agency responded to THIS specific comment.
   Look for Federal Register notices, response-to-comments PDFs, agency docket materials, or official response documents.
2. Decide if a response exists.
3. If a response exists, classify whether the agency accepted the comment's suggestion, rejected it, or the disposition is unclear.
4. Provide the response text (can be detailed, up to several paragraphs) and where you found it.

IMPORTANT:
- If you cannot confidently determine, mark response_found="uncertain".
- Keep response_text informative but concise (excerpt or summary is fine).
- Be thorough in your search - check Federal Register, agency websites, and docket materials.
- If multiple responses exist, summarize the primary/most relevant one.
"""


class GeminiResponseTracker:
    """
    Async Gemini client for tracking agency responses to comments.
    
    Uses Gemini 3 Flash with structured outputs, Google Search grounding,
    and thinking mode support.
    
    Attributes:
        api_key: Gemini API key
        model: Model to use for tracking
        max_retries: Maximum retry attempts per request
        enable_search: Enable Google Search grounding
        thinking_level: Optional thinking level for Gemini 3
    
    Example:
        >>> tracker = GeminiResponseTracker(api_key, model="gemini-3-flash-preview")
        >>> 
        >>> # Single tracking
        >>> result = await tracker.track_response(
        ...     "As a concerned citizen...",
        ...     {"comment_id": "ABC-123", "agency": "EPA"}
        ... )
        >>> 
        >>> # Batch tracking
        >>> results = await tracker.track_batch(comments, max_concurrency=10)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3-flash-preview",
        max_retries: int = 5,
        enable_search: bool = True,
        thinking_level: Optional[str] = None,  # "minimal"|"low"|"medium"|"high"
    ) -> None:
        """
        Initialize the tracker.
        
        Args:
            api_key: Gemini API key
            model: Model to use (default: gemini-3-flash-preview)
            max_retries: Maximum retry attempts (default: 5)
            enable_search: Enable Google Search grounding (default: True)
            thinking_level: Optional thinking level for Gemini 3
        """
        from google import genai
        from google.genai import types
        
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.enable_search = enable_search
        
        # Async client
        self._client = genai.Client(api_key=api_key)
        self._types = types
        
        # Configure tools
        tools = None
        if enable_search:
            # Correct Search tool wiring for new SDK
            tools = [types.Tool(google_search=types.GoogleSearch())]
        
        # Configure thinking (Gemini 3 feature)
        thinking_config = None
        if thinking_level:
            thinking_config = types.ThinkingConfig(thinking_level=thinking_level)
        
        # Generation config with structured outputs
        self._base_config = types.GenerateContentConfig(
            temperature=0.2,  # Lower temperature for extraction tasks
            top_p=0.95,
            max_output_tokens=30000,  # Allow long responses
            tools=tools,
            thinking_config=thinking_config,
            response_mime_type="application/json",
            response_schema=AgencyResponse,
        )
        
        logger.debug(
            f"Initialized GeminiResponseTracker model={model} search={enable_search} thinking={thinking_level}"
        )
    
    def _is_retryable_error(self, e: Exception) -> bool:
        """
        Determine if an error is retryable.
        
        Don't retry: 400/401/403/404 (config/auth/permission errors)
        Do retry: 429/5xx/timeout (rate limits, transient errors)
        """
        msg = str(e).lower()
        
        # Retry: rate limits, transient, server errors, timeouts
        if "429" in msg or "rate limit" in msg or "quota" in msg:
            return True
        if "timeout" in msg or "timed out" in msg:
            return True
        if "500" in msg or "502" in msg or "503" in msg or "504" in msg:
            return True
        
        # DO NOT retry: invalid arguments / auth / permission issues
        if "400" in msg or "invalid argument" in msg or "bad request" in msg:
            return False
        if "401" in msg or "403" in msg or "permission" in msg or "api key" in msg:
            return False
        if "404" in msg or "not found" in msg:
            return False
        
        # Default conservative: retry
        return True

    async def track_response(
        self,
        comment_text: str,
        comment_metadata: Dict[str, str],
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Tuple[str, Dict[str, str], str]:
        """
        Track agency response for a single comment.
        
        Args:
            comment_text: Full comment text to analyze (truncated to ~20k chars)
            comment_metadata: Dict with comment_id, document_number, agency, 
                            commenter_type, submission_date
            semaphore: Optional semaphore for concurrency control
        
        Returns:
            Tuple of (comment_id, parsed_response_dict, raw_model_response)
        """
        # Truncate very long comments
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
        
        async def do_request() -> Tuple[str, Dict[str, str], str]:
            backoff = 2.0
            
            for attempt in range(self.max_retries):
                try:
                    response = await self._client.aio.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=self._base_config,
                    )
                    
                    raw_text = (response.text or "").strip()
                    
                    # Preferred: parsed structured output
                    parsed_obj = getattr(response, "parsed", None)
                    if parsed_obj is not None:
                        if isinstance(parsed_obj, AgencyResponse):
                            parsed = parsed_obj.normalized()
                        else:
                            # Sometimes SDK may return dict-like; validate via Pydantic
                            parsed = AgencyResponse.model_validate(parsed_obj).normalized()
                    else:
                        # Fallback: validate JSON string from response.text
                        parsed = AgencyResponse.model_validate_json(raw_text).normalized()
                    
                    return (
                        comment_metadata.get("comment_id", "unknown"),
                        parsed,
                        raw_text or "OK_JSON",
                    )
                    
                except ValidationError as ve:
                    # Schema validation failed - not retryable
                    logger.warning(f"Schema validation failed for {comment_metadata.get('comment_id')}: {ve}")
                    return (
                        comment_metadata.get("comment_id", "unknown"),
                        AgencyResponse(
                            response_found="uncertain",
                            agency_decision="uncertain",
                            response_text="N/A",
                            response_location="N/A",
                            reasoning="Structured output validation failed",
                        ).normalized(),
                        "ERROR: schema_validation_failed",
                    )
                    
                except Exception as e:
                    retryable = self._is_retryable_error(e)
                    
                    if attempt >= self.max_retries - 1 or not retryable:
                        logger.warning(
                            f"Gemini call failed (retryable={retryable}) after attempt {attempt+1}: {e}"
                        )
                        return (
                            comment_metadata.get("comment_id", "unknown"),
                            AgencyResponse(
                                response_found="uncertain",
                                agency_decision="uncertain",
                                response_text="N/A",
                                response_location="N/A",
                                reasoning=f"API error: {type(e).__name__}",
                            ).normalized(),
                            f"ERROR: {e}",
                        )
                    
                    jitter = random.uniform(0.5, 2.0)
                    sleep_time = backoff + jitter
                    if "429" in str(e).lower():
                        sleep_time += 30.0
                    
                    logger.debug(f"Retrying after error (attempt {attempt+1}): {e} (sleep {sleep_time:.1f}s)")
                    await asyncio.sleep(sleep_time)
                    backoff = min(backoff * 2, 120.0)
            
            # Should never hit due to returns above, but keep safe
            return (
                comment_metadata.get("comment_id", "unknown"),
                AgencyResponse(
                    response_found="uncertain",
                    agency_decision="uncertain",
                    response_text="N/A",
                    response_location="N/A",
                    reasoning="API retries exhausted",
                ).normalized(),
                "ERROR: retries_exhausted",
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
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Tracking responses"):
            results.append(await task)

        return results

    async def close(self) -> None:
        """Close the client (placeholder for compatibility)."""
        pass
