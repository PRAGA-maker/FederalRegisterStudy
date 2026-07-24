from pathlib import Path

from stratification_scripts.reginfo.client import (
    RegInfoClient,
    has_undetermined_final_rule,
)

FIXTURE = Path(__file__).parent / "fixtures" / "reginfo_2060-AV81_timetable.html"

LONG_TERM_PAGE = (
    "<html><body>Long-Term Actions"
    + FIXTURE.read_text()
    + "</body></html>"
)


def _parse(html: str) -> dict:
    client = RegInfoClient.__new__(RegInfoClient)  # no network in __init__ path
    return client._parse_html_response(html, "2060-AV81")


def test_tbd_final_rule_row_is_captured():
    result = _parse(LONG_TERM_PAGE)
    actions = [e["action"] for e in result["timetable"]]
    assert "NPRM" in actions
    assert "FINAL RULE" in actions
    tbd = [e for e in result["timetable"] if e["action"] == "FINAL RULE"][0]
    assert tbd["date"] == ""                       # not parseable -> stays empty
    assert tbd["date_raw"] == "To Be Determined"   # ...but no longer invisible


def test_dated_row_still_parses_with_citation():
    result = _parse(LONG_TERM_PAGE)
    nprm = [e for e in result["timetable"] if e["action"] == "NPRM"][0]
    assert nprm["date"] == "2024-03-01"
    assert nprm["citation"] == "89 FR 15101"
    assert nprm["date_raw"] == ""


def test_every_entry_carries_date_raw():
    result = _parse(LONG_TERM_PAGE)
    assert all("date_raw" in e for e in result["timetable"])


def test_timetable_action_count_not_inflated_by_tbd_row():
    result = _parse(LONG_TERM_PAGE)
    timeline = RegInfoClient.extract_structured_timeline(result["timetable"])
    assert timeline["timetable_action_count"] == 1   # only the NPRM has a real date


def test_long_term_synthetic_entry_not_suppressed_by_a_dated_row():
    # The old gate was `not result["timetable"]`, so a page with an NPRM row
    # lost the undetermined signal entirely. It must survive now.
    result = _parse(LONG_TERM_PAGE)
    assert has_undetermined_final_rule(result) is True


def test_duplicate_tbd_rows_dedupe():
    doubled = LONG_TERM_PAGE.replace("</table>", """
               <tr>
                  <td headers='TimetableAction' bgcolor="#EFEFEF">Final Rule&nbsp;</td>
                  <td headers='TimetableDate' bgcolor="#EFEFEF">To Be Determined&nbsp;</td>
                  <td headers='FRC' bgcolor="#EFEFEF"></td>
               </tr>
    </table>""")
    result = _parse(doubled)
    finals = [e for e in result["timetable"] if e["action"] == "FINAL RULE"]
    assert len(finals) == 1


def test_fallback_parser_still_handles_pages_without_a_timetable_block():
    legacy = (
        "<html>Proposed Rule Stage"
        "<td>NPRM</td><td>03/01/2024</td><td>89 FR 15101</td>"
        "</html>"
    )
    result = _parse(legacy)
    assert [e["action"] for e in result["timetable"]] == ["NPRM"]
    assert result["timetable"][0]["date"] == "2024-03-01"
    assert result["timetable"][0]["date_raw"] == ""


def test_has_undetermined_final_rule_false_for_scheduled_final():
    agenda = {"timetable": [
        {"action": "NPRM", "date": "2024-03-01", "date_raw": "", "citation": ""},
        {"action": "FINAL RULE", "date": "2025-06-01", "date_raw": "", "citation": ""},
    ]}
    assert has_undetermined_final_rule(agenda) is False


def test_has_undetermined_final_rule_handles_none():
    assert has_undetermined_final_rule(None) is False
