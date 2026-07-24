# Resolution acceptance fixtures

Recorded live on 2026-07-24 by `_record.py`. Each directory is one of the six
observed topologies; together they are the regression suite for the layer's
ontology of where a response can live.

The FBI `1110-AA36` agenda was re-traced on 2026-07-24: it now carries an
undetermined final-rule signal, so its expected reason is
`NO_FINAL_RULE_PLANNED` rather than the original trace's
`RESPONSE_NOT_YET_PUBLISHED`.

- `input.json` — the CommentRef, taken from frozen snapshot `2026-07-15-ce44ac5`
  and the goldset packet.
- `expected.json` — hand-written expectations, not recorded output.
- `fr_*.json` / `agenda_*.json` — verbatim API responses.
- `xml_*.xml.gz` — full-text XML, gzipped. Documents over 600 KB raw are trimmed
  to their `<SUPLINF>` container.

Re-record only when a fixture's expected behavior genuinely changes. `.csv` is
Git-LFS-filtered in this repo — never add a `.csv` fixture here.
