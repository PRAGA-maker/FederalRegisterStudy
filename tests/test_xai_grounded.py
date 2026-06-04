import asyncio

from stratification_scripts.gemini_client import GROUNDED_RESPONSE_PROMPT, AgencyResponse
from stratification_scripts.xai_response_client import XAIResponseTracker


def test_grounded_prompt_includes_evidence_and_comment():
    p = GROUNDED_RESPONSE_PROMPT.format(
        comment_id="C1", document_number="2024-1", agency="EPA",
        commenter_type="industry", submission_date="2024-01-01",
        full_comment_text="please change rule X",
        grounded_text="We agree with commenters on X and adopt the change.")
    assert "please change rule X" in p
    assert "We agree with commenters on X" in p
    assert "this comment" in p.lower()


def test_grounded_tracker_disables_search_and_returns_schema(monkeypatch):
    tracker = XAIResponseTracker(api_key="x", enable_search=True)  # search on for fallback

    class FakeParsed:
        output_parsed = AgencyResponse(response_found="yes", agency_decision="accept",
                                       response_text="We agree.", response_location="N/A",
                                       reasoning="grounded")
        output_text = "{}"

    captured = {}

    def fake_parse(**kwargs):
        captured.update(kwargs)
        return FakeParsed()

    monkeypatch.setattr(tracker._client.responses, "parse", fake_parse)

    cid, parsed, raw = asyncio.run(tracker.track_response_grounded(
        "comment text", "We agree with commenters.", {"comment_id": "C1"}))
    assert cid == "C1"
    assert parsed["agency_decision"] == "accept"
    assert captured["tools"] == []          # web search disabled on grounded path
