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
import json
import random
import re
from typing import Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from stratification_scripts.logging_utils import get_logger

logger = get_logger(__name__)


# =========================
# Structured Output Schema
# =========================

ResponseFound = Literal["yes", "no", "uncertain"]
AgencyDecision = Literal["accept", "reject", "partial", "uncertain"]
Tier2AcceptanceStatus = Literal["ACCEPTED", "REJECTED", "PARTIAL", "UNCLEAR"]


class AgencyResponse(BaseModel):
    """Structured response schema for agency response tracking."""

    response_found: ResponseFound = Field(
        description='Whether a response exists: "yes" | "no" | "uncertain"'
    )
    agency_decision: AgencyDecision = Field(
        description='Only meaningful if response_found="yes": "accept" | "reject" | "partial" | "uncertain"'
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


class Tier2Response(BaseModel):
    """Structured response schema for Tier 2 NPRM vs Final Rule text comparison."""

    acceptance_status: Tier2AcceptanceStatus = Field(
        description=(
            'Whether the comment concern was addressed by changes between proposed and final rule: '
            '"ACCEPTED" | "REJECTED" | "PARTIAL" | "UNCLEAR"'
        )
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    text_change_summary: str = Field(
        description=(
            "1-2 sentence summary describing what changed between proposed and final rule "
            "relevant to the comment, or why the determination was made"
        )
    )

    def normalized(self) -> Dict[str, str]:
        """Normalize output for CSV storage."""
        d = self.model_dump()
        # Cap summary length
        if len(d["text_change_summary"]) > 2000:
            d["text_change_summary"] = d["text_change_summary"][:2000] + " ...[truncated]"
        # Ensure confidence is a string for CSV
        d["confidence"] = str(d["confidence"])
        return d


# Response tracking prompt template (optimized for structured JSON output)
RESPONSE_TRACKING_PROMPT = """You are a researcher studying the U.S. federal notice-and-comment rulemaking process.

BACKGROUND:
Under the Administrative Procedure Act, federal agencies publish proposed rules in the Federal Register
and invite public comments. After the comment period closes, agencies must consider and respond to
substantive comments before issuing a final rule. Responses typically appear in the preamble of the
final rule (often under headings like "Discussion of Comments", "Response to Comments", or
"Summary of Comments"), in separate "response to comments" documents in the docket, or in
Federal Register notices. Agencies may also post response-to-comments PDFs on regulations.gov
alongside the final rule.

You have TWO jobs for each comment:
1. FIND whether the agency published a response to this comment (or the class of comments it belongs to).
2. CLASSIFY the agency's decision: did they accept, reject, or split the commenter's request?

COMMENT DETAILS:
- Comment ID: {comment_id}
- Submitted to Federal Register Document: {document_number}
- Agency: {agency}
- Commenter Type: {commenter_type}
- Submission Date: {submission_date}

COMMENT TEXT:
{full_comment_text}

SEARCH STRATEGY:
Search the web for the agency's response. Productive searches include:
- The comment ID or docket number on regulations.gov
- "{agency} response to comments {document_number}" on the Federal Register website
- The agency name + key phrases from the comment + "final rule" or "response to comments"
- The docket page on regulations.gov for related final rules or response documents
Do NOT fabricate URLs. If you find a relevant document, cite its actual URL.

CLASSIFICATION RULES:
- response_found: "yes" if the agency published any response addressing this comment or its topic.
  "no" if you searched thoroughly and found nothing. "uncertain" ONLY if search results are
  ambiguous (e.g., you found a final rule but cannot tell if it addresses this comment).
  Prefer "yes" or "no" over "uncertain" -- make a call.
- agency_decision (only when response_found="yes"):
  - "accept" = the agency adopted the commenter's suggestion, or changed the rule in the direction
    the commenter requested. If the agency mostly accepted with minor caveats, this is still "accept".
  - "reject" = the agency kept the provision as proposed or explicitly declined the suggestion.
    If the agency mostly rejected but acknowledged a small point, this is still "reject".
  - "partial" = ONLY when the comment raised multiple separable issues and the agency accepted
    some while rejecting others. This must be a genuinely split outcome on distinct sub-issues.
    Do not use "partial" for hedged language or soft acceptances.
  - "uncertain" = the response exists but you cannot determine the disposition.
- If response_found is "no" or "uncertain", set agency_decision to "uncertain".

OUTPUT: Return valid JSON matching the schema exactly.
- response_found: "yes" | "no" | "uncertain"
- agency_decision: "accept" | "reject" | "partial" | "uncertain"
- response_text: the relevant excerpt from the agency's response (up to a few paragraphs), or "N/A"
- response_location: the URL where you found it, or "N/A"
- reasoning: 1-2 sentence explanation of your determination
"""


# Tier 2: NPRM vs Final Rule text comparison prompt
TIER2_COMPARISON_PROMPT = """You are comparing a public comment against the proposed rule (NPRM) and the final rule to determine whether the comment's concern was addressed.

COMMENT INFORMATION:
- Comment ID: {comment_id}
- Document Number: {document_number}
- Agency: {agency}

COMMENT TEXT:
{comment_text}

PROPOSED RULE (NPRM) TEXT (may be truncated):
{nprm_text}

FINAL RULE TEXT (may be truncated):
{final_rule_text}

TASK:
1. Read the public comment and understand what concern, suggestion, or objection it raises.
2. Compare the proposed rule text (NPRM) with the final rule text.
3. Determine whether the changes between proposed and final rule address the comment's concern.
4. Consider the final rule preamble -- agencies typically discuss and respond to public comments in the preamble section of the final rule (often under headings like "Discussion of Comments" or "Response to Comments").

DETERMINATION CRITERIA:
- ACCEPTED: The final rule made changes that align with what the comment suggested or requested. The concern was addressed favorably.
- REJECTED: The final rule explicitly or implicitly did not adopt the comment's suggestion. The agency kept the provision as proposed, or the preamble explains why the suggestion was not adopted.
- PARTIAL: Some aspects of the comment were addressed but not all, or the change was a compromise.
- UNCLEAR: Cannot determine from the available text whether the comment was addressed. The text may be truncated, the comment may be off-topic, or the relevant section may not be present.

OUTPUT REQUIREMENTS:
- acceptance_status: Exactly one of "ACCEPTED", "REJECTED", "PARTIAL", or "UNCLEAR"
- confidence: A float from 0.0 to 1.0 indicating how confident you are in the determination
- text_change_summary: 1-2 sentences describing what specifically changed (or didn't change) between proposed and final rule relevant to this comment

IMPORTANT:
- Focus on substantive changes, not formatting or numbering changes
- If the comment is very general or off-topic relative to the rule, mark as UNCLEAR with low confidence
- If the final rule preamble explicitly discusses and responds to the comment's concern, that is strong evidence
- Be conservative: if unsure, prefer UNCLEAR over guessing
"""


def _parse_gemini_response(response, schema_class, logger):
    """
    Extract and validate structured output from a Gemini response.

    Returns (raw_text, parsed_dict) where parsed_dict is the normalized
    schema output, or (raw_text, None) if parsing failed but raw_text exists,
    or ("", None) if the response is completely empty.

    Raises TypeError or AttributeError if the SDK response object has
    None internals (e.g., None candidates/parts). The caller should catch
    these and route to the empty-response handler.
    """
    # Extract raw text
    raw_text = ""
    if hasattr(response, 'text'):
        try:
            raw_text = (response.text or "").strip()
        except (TypeError, AttributeError, ValueError):
            raw_text = ""

    if not raw_text and hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and candidate.content:
            parts = getattr(candidate.content, 'parts', None)
            if parts is not None:
                text_parts = [p.text for p in parts if hasattr(p, 'text') and p.text]
                raw_text = " ".join(text_parts).strip()

    # Try parsed structured output
    parsed_obj = None
    if hasattr(response, 'parsed'):
        parsed_obj = response.parsed  # May raise TypeError if SDK internals are None
    elif hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and candidate.content:
            parts = getattr(candidate.content, 'parts', None)
            if parts is not None:
                for part in parts:
                    if hasattr(part, 'struct_data'):
                        parsed_obj = part.struct_data
                        break

    if parsed_obj is not None:
        if isinstance(parsed_obj, schema_class):
            return raw_text, parsed_obj.normalized()
        elif isinstance(parsed_obj, dict):
            return raw_text, schema_class.model_validate(parsed_obj).normalized()
        else:
            try:
                if hasattr(parsed_obj, '__dict__'):
                    return raw_text, schema_class.model_validate(parsed_obj.__dict__).normalized()
                elif hasattr(parsed_obj, 'model_dump'):
                    return raw_text, schema_class.model_validate(parsed_obj.model_dump()).normalized()
                else:
                    return raw_text, schema_class.model_validate_json(str(parsed_obj)).normalized()
            except Exception as parse_err:
                logger.warning(f"Failed to parse parsed_obj: {parse_err}, falling back to raw_text")

    # No parsed object — try raw text as JSON
    if raw_text:
        try:
            return raw_text, schema_class.model_validate_json(raw_text).normalized()
        except Exception:
            # Try regex extraction
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_text, re.DOTALL)
            if json_match:
                try:
                    return raw_text, schema_class.model_validate_json(json_match.group(0)).normalized()
                except Exception:
                    pass
            # Has text but couldn't parse — raise so caller gets the error
            raise ValueError(f"Failed to parse JSON from response text: {raw_text[:200]}")

    # Completely empty
    return "", None


def _diagnose_empty_response(response) -> str:
    """
    Inspect a Gemini response that produced no text/parsed output.

    Returns a short diagnostic string describing what the response contains
    (e.g., finish_reason, part types). Used for logging and error messages.
    """
    parts_info = []
    try:
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason:
                parts_info.append(f"finish_reason={finish_reason}")
            safety = getattr(candidate, 'safety_ratings', None)
            if safety:
                blocked = [r for r in safety if getattr(r, 'blocked', False)]
                if blocked:
                    parts_info.append(f"safety_blocked={len(blocked)}")
            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                part_types = []
                for p in candidate.content.parts:
                    ptype = type(p).__name__
                    # Check for common non-text part types
                    if hasattr(p, 'thought') and p.thought:
                        ptype = "thought"
                    elif hasattr(p, 'text') and p.text:
                        ptype = f"text({len(p.text)})"
                    elif hasattr(p, 'executable_code'):
                        ptype = "executable_code"
                    elif hasattr(p, 'code_execution_result'):
                        ptype = "code_execution_result"
                    part_types.append(ptype)
                parts_info.append(f"parts={part_types}")
            else:
                parts_info.append("no_content_parts")
        else:
            parts_info.append("no_candidates")
    except Exception as diag_err:
        parts_info.append(f"diag_error={type(diag_err).__name__}")

    return "; ".join(parts_info) if parts_info else "unknown"


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
        
        # Validate model name
        if not model or not isinstance(model, str):
            raise ValueError(f"Invalid model name: {model}")
        
        # Warn if using old model name
        if "gemini-2" in model.lower() or "thinking-exp" in model.lower():
            logger.warning(
                f"Using potentially outdated model: {model}. "
                f"Consider using 'gemini-3-flash-preview' for better compatibility."
            )
        
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

    # Maximum retries specifically for empty-response errors (thought_signature
    # only, no text/parsed output). These are NOT transient: the model simply
    # didn't produce JSON output. Retrying 10 times with exponential backoff
    # wastes ~486s per comment. Cap at 2 retries instead.
    MAX_EMPTY_RESPONSE_RETRIES = 2

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
            empty_response_count = 0  # Track empty-response retries separately

            for attempt in range(self.max_retries):
                try:
                    # Log request details for debugging
                    logger.debug(
                        f"Gemini API call attempt {attempt+1}/{self.max_retries} for comment_id={comment_metadata.get('comment_id')}, "
                        f"model={self.model}, prompt_length={len(prompt)}"
                    )

                    # Use Content list format (standard for current SDK)
                    contents_input = [self._types.Content(
                        role="user",
                        parts=[self._types.Part(text=prompt)]
                    )]
                    response = await self._client.aio.models.generate_content(
                        model=self.model,
                        contents=contents_input,
                        config=self._base_config,
                    )

                    if response is None:
                        raise ValueError("API returned None response")

                    # ---- RESPONSE PARSING ----
                    # The entire parsing block is wrapped to catch
                    # TypeError/AttributeError from SDK internals (e.g.,
                    # iterating over None candidates/parts). These errors
                    # are routed to the empty-response handler instead of
                    # the general retry loop with exponential backoff.
                    try:
                        raw_text, parsed = _parse_gemini_response(
                            response, AgencyResponse, logger
                        )
                    except (TypeError, AttributeError) as sdk_err:
                        # SDK returned a response we can't parse — treat as empty
                        logger.debug(
                            f"SDK error parsing response for "
                            f"{comment_metadata.get('comment_id')}: {sdk_err}"
                        )
                        raw_text = ""
                        parsed = None

                    if parsed is not None:
                        logger.debug(f"Successfully parsed response for comment_id={comment_metadata.get('comment_id')}")
                        return (
                            comment_metadata.get("comment_id", "unknown"),
                            parsed,
                            raw_text or "OK_JSON",
                        )

                    # ---- EMPTY RESPONSE HANDLING ----
                    # Gemini returned a response with no usable text/JSON.
                    # Often caused by thought_signature-only parts or
                    # SDK TypeError from None candidates.
                    # Cap retries to avoid exponential backoff storms.
                    empty_response_count += 1
                    diag = _diagnose_empty_response(response)

                    if empty_response_count >= self.MAX_EMPTY_RESPONSE_RETRIES:
                        logger.warning(
                            f"Empty response for {comment_metadata.get('comment_id')} "
                            f"after {empty_response_count} attempts (capped). "
                            f"Diagnosis: {diag}"
                        )
                        return (
                            comment_metadata.get("comment_id", "unknown"),
                            AgencyResponse(
                                response_found="uncertain",
                                agency_decision="uncertain",
                                response_text="N/A",
                                response_location="N/A",
                                reasoning=f"Empty model response after {empty_response_count} attempts: {diag}",
                            ).normalized(),
                            f"EMPTY_RESPONSE: {diag}",
                        )

                    logger.warning(
                        f"Empty response for {comment_metadata.get('comment_id')} "
                        f"(attempt {empty_response_count}/{self.MAX_EMPTY_RESPONSE_RETRIES}). "
                        f"Diagnosis: {diag}. Retrying..."
                    )
                    await asyncio.sleep(2.0 + random.uniform(0.5, 1.5))
                    continue

                except ValidationError as ve:
                    # Schema validation failed - not retryable.
                    logger.warning(
                        f"VALIDATION_ERROR: Pydantic schema validation failed for "
                        f"{comment_metadata.get('comment_id')}: {ve}. "
                        f"Response text preview: {raw_text[:500] if 'raw_text' in locals() and raw_text else 'not found'}"
                    )
                    return (
                        comment_metadata.get("comment_id", "unknown"),
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
                        f"Gemini API error for comment_id={comment_metadata.get('comment_id')} "
                        f"(attempt {attempt+1}/{self.max_retries}): {error_type}: {error_msg}"
                    )

                    retryable = self._is_retryable_error(e)

                    if attempt >= self.max_retries - 1 or not retryable:
                        logger.warning(
                            f"Gemini call failed (retryable={retryable}) after attempt {attempt+1}: "
                            f"{error_type}: {error_msg}"
                        )
                        return (
                            comment_metadata.get("comment_id", "unknown"),
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
                        sleep_time += 30.0

                    logger.debug(f"Retrying after error (attempt {attempt+1}): {error_type}: {error_msg} (sleep {sleep_time:.1f}s)")
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

    async def track_tier2_response(
        self,
        comment_text: str,
        nprm_text: str,
        final_rule_text: str,
        comment_metadata: Dict[str, str],
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Tuple[str, Dict[str, str], str]:
        """
        Tier 2: Compare NPRM vs Final Rule text to determine if a comment was addressed.

        This method does NOT use Google Search grounding (unlike Tier 1). Instead,
        it compares the provided document texts directly.

        Args:
            comment_text: Full comment text
            nprm_text: Full text of the proposed rule (NPRM)
            final_rule_text: Full text of the final rule
            comment_metadata: Dict with comment_id, document_number, agency
            semaphore: Optional semaphore for concurrency control

        Returns:
            Tuple of (comment_id, parsed_tier2_dict, raw_model_response)
        """
        # Truncate comment text
        max_comment_chars = 10000
        if len(comment_text) > max_comment_chars:
            comment_text = comment_text[:max_comment_chars] + "\n\n[... truncated ...]"

        # NPRM and final rule texts are pre-truncated by the caller, but enforce a safety limit
        max_doc_chars = 55000
        if len(nprm_text) > max_doc_chars:
            nprm_text = nprm_text[:max_doc_chars] + "\n\n[... truncated ...]"
        if len(final_rule_text) > max_doc_chars:
            final_rule_text = final_rule_text[:max_doc_chars] + "\n\n[... truncated ...]"

        prompt = TIER2_COMPARISON_PROMPT.format(
            comment_id=comment_metadata.get("comment_id", "N/A"),
            document_number=comment_metadata.get("document_number", "N/A"),
            agency=comment_metadata.get("agency", "N/A"),
            comment_text=comment_text,
            nprm_text=nprm_text,
            final_rule_text=final_rule_text,
        )

        # Build a Tier 2-specific config WITHOUT search grounding
        tier2_config = self._types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=10000,
            tools=None,  # No Google Search for Tier 2
            thinking_config=self._base_config.thinking_config,
            response_mime_type="application/json",
            response_schema=Tier2Response,
        )

        async def do_request() -> Tuple[str, Dict[str, str], str]:
            backoff = 2.0
            comment_id = comment_metadata.get("comment_id", "unknown")
            empty_response_count = 0

            for attempt in range(self.max_retries):
                try:
                    logger.debug(
                        f"Tier 2 API call attempt {attempt+1}/{self.max_retries} for {comment_id}, "
                        f"prompt_length={len(prompt)}"
                    )

                    contents_input = [self._types.Content(
                        role="user",
                        parts=[self._types.Part(text=prompt)]
                    )]

                    response = await self._client.aio.models.generate_content(
                        model=self.model,
                        contents=contents_input,
                        config=tier2_config,
                    )

                    # ---- RESPONSE PARSING ----
                    # Use shared parser, same pattern as Tier 1.
                    # TypeError/AttributeError from SDK internals (None
                    # candidates/parts) are routed to empty-response handler.
                    try:
                        raw_text, parsed = _parse_gemini_response(
                            response, Tier2Response, logger
                        )
                    except (TypeError, AttributeError) as sdk_err:
                        logger.debug(
                            f"Tier 2: SDK error parsing response for {comment_id}: {sdk_err}"
                        )
                        raw_text = ""
                        parsed = None

                    if parsed is not None:
                        logger.debug(f"Tier 2: Successfully parsed response for {comment_id}")
                        return (comment_id, parsed, raw_text or "OK_JSON")

                    # ---- EMPTY RESPONSE HANDLING ----
                    empty_response_count += 1
                    diag = _diagnose_empty_response(response)

                    if empty_response_count >= self.MAX_EMPTY_RESPONSE_RETRIES:
                        logger.warning(
                            f"Tier 2: Empty response for {comment_id} "
                            f"after {empty_response_count} attempts (capped). "
                            f"Diagnosis: {diag}"
                        )
                        return (
                            comment_id,
                            Tier2Response(
                                acceptance_status="UNCLEAR",
                                confidence=0.0,
                                text_change_summary=f"Empty model response after {empty_response_count} attempts: {diag}",
                            ).normalized(),
                            f"EMPTY_RESPONSE: {diag}",
                        )

                    logger.warning(
                        f"Tier 2: Empty response for {comment_id} "
                        f"(attempt {empty_response_count}/{self.MAX_EMPTY_RESPONSE_RETRIES}). "
                        f"Diagnosis: {diag}. Retrying..."
                    )
                    await asyncio.sleep(2.0 + random.uniform(0.5, 1.5))
                    continue

                except ValidationError as ve:
                    logger.warning(
                        f"Tier 2: Schema validation failed for {comment_id}: {ve}"
                    )
                    return (
                        comment_id,
                        Tier2Response(
                            acceptance_status="UNCLEAR",
                            confidence=0.0,
                            text_change_summary="Schema validation failed",
                        ).normalized(),
                        "ERROR: tier2_schema_validation_failed",
                    )

                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)

                    logger.error(
                        f"Tier 2 API error for {comment_id} "
                        f"(attempt {attempt+1}/{self.max_retries}): {error_type}: {error_msg}"
                    )

                    retryable = self._is_retryable_error(e)

                    if attempt >= self.max_retries - 1 or not retryable:
                        logger.warning(
                            f"Tier 2: call failed (retryable={retryable}) after attempt {attempt+1}: "
                            f"{error_type}: {error_msg}"
                        )
                        return (
                            comment_id,
                            Tier2Response(
                                acceptance_status="UNCLEAR",
                                confidence=0.0,
                                text_change_summary=f"API error: {error_type}",
                            ).normalized(),
                            f"ERROR: {error_msg}",
                        )

                    jitter = random.uniform(0.5, 2.0)
                    sleep_time = backoff + jitter
                    if "429" in str(e).lower():
                        sleep_time += 30.0

                    await asyncio.sleep(sleep_time)
                    backoff = min(backoff * 2, 120.0)

            # Should not reach here
            return (
                comment_metadata.get("comment_id", "unknown"),
                Tier2Response(
                    acceptance_status="UNCLEAR",
                    confidence=0.0,
                    text_change_summary="API retries exhausted",
                ).normalized(),
                "ERROR: tier2_retries_exhausted",
            )

        if semaphore:
            async with semaphore:
                return await do_request()
        return await do_request()

    async def track_tier2_batch(
        self,
        items: List[Tuple[str, str, str, Dict[str, str]]],
        max_concurrency: int = 10,
    ) -> List[Tuple[str, Dict[str, str], str]]:
        """
        Tier 2: Process a batch of comment+document comparisons concurrently.

        Args:
            items: List of (comment_text, nprm_text, final_rule_text, metadata) tuples
            max_concurrency: Maximum concurrent API calls

        Returns:
            List of (comment_id, parsed_tier2_dict, raw_response) tuples
        """
        if not items:
            return []

        semaphore = asyncio.Semaphore(max_concurrency)

        async def run_single(
            comment_text: str,
            nprm_text: str,
            final_rule_text: str,
            metadata: Dict[str, str],
        ):
            return await self.track_tier2_response(
                comment_text, nprm_text, final_rule_text, metadata, semaphore
            )

        tasks = [
            asyncio.create_task(run_single(ct, nt, ft, meta))
            for ct, nt, ft, meta in items
        ]

        results: List[Tuple[str, Dict[str, str], str]] = []
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Tier 2 comparisons",
        ):
            results.append(await task)

        return results

    async def close(self) -> None:
        """Close the client (placeholder for compatibility)."""
        pass
