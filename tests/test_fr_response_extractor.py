from stratification_scripts.makeup.fr_response_extractor import (
    extract_response_section,
    is_rtc_title,
    GROUND_CAP,
)

# A small final-rule XML with an explicit, substantial response section.
XML_RESP_HD = """<RULE><PREAMB>
<HD SOURCE="HED">SUPPLEMENTARY INFORMATION:</HD>
<SUPLINF>
<HD SOURCE="HD1">I. Background</HD>
<P>The agency proposed X.</P>
<HD SOURCE="HD1">III. Response to Comments</HD>
<P>We received 500 comments. We agree with commenters that X should change. """ + ("blah " * 600) + """</P>
<HD SOURCE="HD1">IV. Regulatory Analysis</HD>
<P>Executive Order 12866 review.</P>
</SUPLINF></PREAMB></RULE>"""

# Comment-by-topic: response head followed by same-level "Comments on X" heads.
# Blocks are sized realistically (> POINTER_MIN) so the section-slicing path runs.
XML_BY_TOPIC = """<SUPLINF>
<HD SOURCE="HD1">Response to Comments</HD>
<P>intro.</P>
<HD SOURCE="HD1">Comments on Scope</HD><P>""" + ("a " * 1500) + """We adopt this.</P>
<HD SOURCE="HD1">Comments on Cost</HD><P>""" + ("b " * 1500) + """We decline.</P>
<HD SOURCE="HD1">Regulatory Flexibility Act</HD><P>not a comment.</P>
</SUPLINF>"""

# Tiny pointer head -> should NOT be used as the section (falls to whole preamble).
XML_POINTER = """<SUPLINF>
<HD SOURCE="HD1">Public Comments Received</HD><P>We received 3,500 comments; responses appear under the relevant headings below.</P>
<HD SOURCE="HD1">Payment Rates</HD><P>""" + ("c " * 500) + """We agree with the commenter on rates.</P>
</SUPLINF>"""

# No SUPLINF, but PREAMB present (DOE/FHWA style).
XML_PREAMB_ONLY = """<PREAMB>
<HD SOURCE="HED">AGENCY:</HD><P>DOE.</P>
<HD SOURCE="HED">SUPPLEMENTARY INFORMATION:</HD>
<HD SOURCE="HD1">Discussion of Comments</HD><P>""" + ("d " * 600) + """We disagree with the commenter.</P>
</PREAMB>"""

# No comment discussion at all (technical correction).
XML_NO_RESPONSE = """<SUPLINF><HD SOURCE="HD1">Need for Correction</HD><P>This corrects a typo.</P></SUPLINF>"""


def test_explicit_response_section_is_isolated():
    r = extract_response_section(XML_RESP_HD)
    assert r.found_response_hd is True
    assert r.matched_header.startswith("III. Response to Comments")
    assert "We agree with commenters" in r.grounded_text
    assert "Executive Order 12866" not in r.grounded_text  # stops at next section
    assert r.method == "response_hd"


def test_comment_by_topic_block_is_kept_whole():
    r = extract_response_section(XML_BY_TOPIC)
    assert "We adopt this." in r.grounded_text
    assert "We decline." in r.grounded_text          # same-level comment heads kept
    assert "Regulatory Flexibility Act" not in r.grounded_text  # non-comment head ends it


def test_pointer_head_falls_through_to_whole_preamble():
    r = extract_response_section(XML_POINTER)
    # tiny pointer section (< POINTER_MIN) must not be the grounded_text alone
    assert "We agree with the commenter on rates" in r.grounded_text
    assert r.method in ("suplinf_full", "response_hd_to_full")


def test_preamb_fallback_when_no_suplinf():
    r = extract_response_section(XML_PREAMB_ONLY)
    assert r.suplinf_len > 0
    assert "We disagree with the commenter" in r.grounded_text


def test_no_response_returns_preamble_but_short():
    r = extract_response_section(XML_NO_RESPONSE)
    assert r.found_response_hd is False
    assert "corrects a typo" in r.grounded_text


def test_grounded_text_never_exceeds_cap():
    big = "<SUPLINF>" + "".join(
        f'<HD SOURCE="HD1">Comments on Topic {i}</HD><P>{"word " * 2000} we agree.</P>' for i in range(80)
    ) + "</SUPLINF>"
    r = extract_response_section(big)
    assert r.grounded_len <= GROUND_CAP


def test_missing_xml_is_safe():
    assert extract_response_section("").grounded_text == ""
    assert extract_response_section("<RULE><REGTEXT>no preamble</REGTEXT></RULE>").method == "no_preamble"


def test_rtc_title_matching():
    assert is_rtc_title("Response to Comments (22-4.5e)")
    assert is_rtc_title("Summary of Public Comments and Responses")
    assert is_rtc_title("RTC Document")
    assert not is_rtc_title("Regulatory Impact Analysis")
    assert not is_rtc_title("Economic Analysis")
