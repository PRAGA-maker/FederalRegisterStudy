from __future__ import annotations

from stratification_scripts.rtc_parser import exhibit2

# Mirrors the real fitz extraction: number / doc-id / First / Last / Org(1+ lines);
# blank name cells render as ' '; the table header repeats mid-table at page breaks.
REGION = """Exhibit 2: List of Public Commenters
Comment Information
Submitter Information
Commenter
Number
Document ID
First Name
Last Name
Organization Name
51
EPA-HQ-OW-2018-0594-0051


The Ranger Leadership
and Policy Center
56
EPA-HQ-OW-2018-0594-0056
Brian
Callahan
Private Citizen


Exhibit 2: List of Public Commenters
Comment Information
Submitter Information
Commenter
Number
Document ID
First Name
Last Name
Organization Name
54
EPA-HQ-OW-2018-0594-0054


Anonymous
2. Comments and EPA Responses by Topic
"""


def test_parses_positional_first_last_org():
    rows = {c.number: c for c in exhibit2.parse_commenters(REGION)}
    assert set(rows) == {51, 54, 56}
    # wrapped org, blank names
    assert rows[51].first_name == "" and rows[51].last_name == ""
    assert rows[51].organization == "The Ranger Leadership and Policy Center"
    # named person
    assert rows[56].first_name == "Brian" and rows[56].last_name == "Callahan"
    assert rows[56].organization == "Private Citizen"
    assert rows[56].document_id == "EPA-HQ-OW-2018-0594-0056"
    # anonymous after a repeated mid-table header
    assert rows[54].first_name == "" and rows[54].organization == "Anonymous"


def test_ignores_document_ids_outside_the_exhibit2_region():
    # A doc-id mentioned in prose before Exhibit 2 must not become a row.
    text = "See EPA-HQ-OW-2018-0594-0031 in prose.\n" + REGION
    rows = {c.number: c for c in exhibit2.parse_commenters(text)}
    assert 31 not in rows
    assert set(rows) == {51, 54, 56}
