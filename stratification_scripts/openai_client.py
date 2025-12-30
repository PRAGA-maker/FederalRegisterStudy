"""
OpenAI API client for comment classification.

This module provides an async wrapper for OpenAI's API with:
- Retry logic with exponential backoff
- Rate limit handling (429 responses)
- Batch classification support

Example:
    >>> from stratification_scripts.openai_client import OpenAIClassifier
    >>> from stratification_scripts.config import get_openai_api_key
    >>> 
    >>> api_key = get_openai_api_key(required=True)
    >>> classifier = OpenAIClassifier(api_key)
    >>> 
    >>> category = await classifier.classify_comment(
    ...     "I am concerned about this rule as a citizen...",
    ...     {"organization": None, "first_name": "John"}
    ... )
"""

from __future__ import annotations

import asyncio
import random
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from stratification_scripts.logging_utils import get_logger

logger = get_logger(__name__)

# Category mapping
LABEL_MAP = {
    "undecided": "Undecided/Anonymous",
    "citizen": "Ordinary Citizen",
    "org": "Organization/Corporation",
    "expert": "Academic/Industry/Expert (incl. small/local business)",
    "lobbyist": "Political Consultant/Lobbyist",
}

# Classification prompt template
PROMPT_TEMPLATE = """Comment to classify:
{comment_text}

Metadata:
- Organization: {organization}
- Submitter Type: {submitter_type}
- Name: {first_name} {last_name}

---

Classify the AUTHOR of the above public comment as exactly ONE of these categories:

**org** - Large corporations, industry associations, trade groups, government agencies, or organizational entities
**citizen** - Individual ordinary citizens, residents, or community members without professional affiliation
**expert** - Academics, researchers, small/local businesses, industry experts, technical specialists
**lobbyist** - Political consultants, registered lobbyists, advocacy groups, or professional campaigners
**undecided** - ONLY if truly anonymous with no identifying info or completely unclear authorship

Decision guidance:
- Use ALL available context clues (tone, language, content, structure, signatures, letterheads)
- Look for: organizational letterhead, academic titles, business names, lobbying language, personal stories, signatures
- Large company or trade association or group → org
- Generic "concerned citizen" with personal story/opinion → citizen (NOT undecided)
- Individual small business owners → expert (not org)
- Professional advocates or political operatives → lobbyist
- AVOID over-classifying as citizen. Citizens are ordinary people with no professional affiliation; citizens whom clearly demonstrate or have realized subject-matter competence (ie. via formal training, corporate experience, etc.) from a technical standpoint are considered experts. Those acting in their own personal self interest (or for an issue/area they care about) are citizen, those not acting in a personal capacity (such as writing for a business or organization) and acting in the interests of the greater organization are not citizens. This is the difference between a hunter (citizen) and a farmer (expert) or small business owner (expert).
- AVOID undecided unless genuinely impossible to determine - make your best inference from the text

Be decisive. Most comments CAN be classified if you look at the language, tone, and context. Only use "undecided" as a last resort.

Note: Text may be from inline comment or extracted from PDF attachment (first 2 pages).

Reply with ONLY ONE WORD from: org, citizen, expert, lobbyist, undecided"""


def classify_from_metadata(row: Dict) -> Optional[str]:
    """
    Classify comment author from metadata fields before resorting to OpenAI.
    
    This provides a fast path for clear-cut cases based on structured metadata,
    saving API calls and cost.
    
    Args:
        row: Dict with metadata fields (organization, first_name, last_name,
             submitter_type, gov_agency)
    
    Returns:
        Category label if confident, None if metadata insufficient.
    """
    organization = row.get("organization")
    first_name = row.get("first_name")
    last_name = row.get("last_name")
    submitter_type = row.get("submitter_type")
    gov_agency = row.get("gov_agency")
    
    # Organization present without personal name → Organization/Corporation
    if organization and not first_name and not last_name:
        org_lower = str(organization).lower()
        if any(term in org_lower for term in [
            "inc", "llc", "corp", "company", "association",
            "coalition", "group", "council"
        ]):
            return LABEL_MAP["org"]
    
    # Government agency → Academic/Industry/Expert
    if gov_agency:
        return LABEL_MAP["expert"]
    
    # Submitter type hints
    if submitter_type:
        sub_lower = str(submitter_type).lower()
        if "organization" in sub_lower or "company" in sub_lower or "business" in sub_lower:
            return LABEL_MAP["org"]
        if "individual" in sub_lower and not organization:
            return LABEL_MAP["citizen"]
    
    # Organization + name suggests small business or expert affiliation
    if organization and (first_name or last_name):
        org_lower = str(organization).lower()
        if any(term in org_lower for term in [
            "university", "college", "institute", "research",
            "lab", "dept", "department"
        ]):
            return LABEL_MAP["expert"]
        if any(term in org_lower for term in [
            "consulting", "consultancy", "partners", "solutions", "services"
        ]):
            return LABEL_MAP["expert"]
    
    # Insufficient metadata
    return None


class OpenAIClassifier:
    """
    Async OpenAI client for comment author classification.
    
    This class provides methods for classifying public comment authors
    into categories (citizen, org, expert, lobbyist, undecided).
    
    Attributes:
        api_key: OpenAI API key
        model: Model to use for classification
        max_retries: Maximum retry attempts per request
    
    Example:
        >>> classifier = OpenAIClassifier(api_key, model="gpt-4o-mini")
        >>> 
        >>> # Single classification
        >>> category, prompt, response = await classifier.classify_comment(
        ...     "As a concerned citizen...",
        ...     {"organization": None}
        ... )
        >>> 
        >>> # Batch classification
        >>> results = await classifier.classify_batch(comments, max_concurrency=50)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 8,
    ) -> None:
        """
        Initialize the classifier.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
            max_retries: Maximum retry attempts (default: 8)
        """
        from openai import AsyncOpenAI
        
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self._client = AsyncOpenAI(api_key=api_key)
        
        logger.debug(f"Initialized OpenAIClassifier with model={model}")

    async def classify_comment(
        self,
        comment_text: str,
        metadata: Dict[str, str],
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Tuple[Optional[str], str, str]:
        """
        Classify a single comment.
        
        Args:
            comment_text: Comment text to classify (truncated to 3000 chars)
            metadata: Dict with organization, submitter_type, first_name, last_name
            semaphore: Optional semaphore for concurrency control
        
        Returns:
            Tuple of (category, prompt_used, model_response)
        """
        prompt = PROMPT_TEMPLATE.format(
            comment_text=comment_text[:3000],
            organization=metadata.get("organization", "N/A"),
            submitter_type=metadata.get("submitter_type", "N/A"),
            first_name=metadata.get("first_name", ""),
            last_name=metadata.get("last_name", ""),
        )
        
        async def do_request() -> Tuple[Optional[str], str, str]:
            backoff = 2.0
            
            for attempt in range(self.max_retries):
                try:
                    response = await self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=20,
                    )
                    
                    # Extract response
                    if response.choices and response.choices[0].message.content:
                        raw_result = response.choices[0].message.content.strip()
                    else:
                        raw_result = ""
                    
                    if not raw_result:
                        logger.debug("Empty LLM response")
                        return LABEL_MAP["undecided"], prompt, "EMPTY_RESPONSE"
                    
                    result = raw_result.lower()
                    
                    # Validate response
                    if result in LABEL_MAP:
                        return LABEL_MAP[result], prompt, raw_result
                    
                    # Try to extract a valid token
                    for token in LABEL_MAP.keys():
                        if token in result:
                            return LABEL_MAP[token], prompt, raw_result
                    
                    logger.debug(f"Unexpected LLM response: {raw_result}")
                    return LABEL_MAP["undecided"], prompt, raw_result
                    
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = "429" in error_str or "rate limit" in error_str
                    
                    if attempt < self.max_retries - 1:
                        jitter = random.uniform(0.5, 2.0)
                        sleep_time = backoff + jitter
                        
                        if is_rate_limit:
                            sleep_time += 20.0
                            logger.debug(f"Rate limit hit, retrying in {sleep_time:.1f}s")
                        else:
                            logger.debug(f"API error (attempt {attempt+1}): {e}")
                        
                        await asyncio.sleep(sleep_time)
                        backoff = min(backoff * 2, 60.0)
                    else:
                        logger.warning(f"API call failed after {self.max_retries} attempts: {e}")
            
            return LABEL_MAP["undecided"], prompt, "ERROR: retries_exhausted"
        
        if semaphore:
            async with semaphore:
                return await do_request()
        else:
            return await do_request()

    async def classify_batch(
        self,
        comments: List[Tuple[str, str, Dict[str, str]]],
        max_concurrency: int = 100,
    ) -> List[Tuple[str, Optional[str], str, str]]:
        """
        Classify a batch of comments concurrently.
        
        Args:
            comments: List of (comment_id, text, metadata) tuples
            max_concurrency: Maximum concurrent API calls
        
        Returns:
            List of (comment_id, category, prompt, model_response) tuples
        """
        if not comments:
            return []

        semaphore = asyncio.Semaphore(max_concurrency)

        async def run_single(
            comment_id: str,
            text: str,
            metadata: Dict[str, str],
        ) -> Tuple[str, Optional[str], str, str]:
            category, prompt, model_response = await self.classify_comment(
                text, metadata, semaphore
            )
            return comment_id, category, prompt, model_response

        tasks = [
            asyncio.create_task(run_single(comment_id, text, metadata))
            for comment_id, text, metadata in comments
        ]

        results: List[Tuple[str, Optional[str], str, str]] = []
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Classifying",
        ):
            results.append(await task)

        return results

    async def close(self) -> None:
        """Close the client."""
        await self._client.close()

