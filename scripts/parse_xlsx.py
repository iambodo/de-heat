"""
Stdlib-only xlsx reader. No openpyxl/pandas required.
Parses xlsx files as ZIP+XML (OOXML format).
"""

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def load_workbook(path):
    """Return a dict of {sheet_name: [[row], [row], ...]}."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        strings = _load_shared_strings(z)
        id_to_target = _load_relationships(z, "xl/_rels/workbook.xml.rels")
        wb_xml = ET.fromstring(z.read("xl/workbook.xml"))

        sheets = {}
        for s in wb_xml.findall(f".//{{{NS}}}sheet"):
            name = s.get("name")
            rid = s.get(f"{{{REL_NS}}}id")
            target = id_to_target.get(rid)
            if not target:
                continue
            try:
                ws_xml = ET.fromstring(z.read(f"xl/{target}"))
            except KeyError:
                continue
            sheets[name] = _parse_worksheet(ws_xml, strings)
    return sheets


def _load_shared_strings(z):
    try:
        ss_xml = ET.fromstring(z.read("xl/sharedStrings.xml"))
        return [
            "".join(t.text or "" for t in si.findall(f".//{{{NS}}}t"))
            for si in ss_xml.findall(f"{{{NS}}}si")
        ]
    except KeyError:
        return []


def _load_relationships(z, rel_path):
    try:
        rels_xml = ET.fromstring(z.read(rel_path))
        return {r.get("Id"): r.get("Target") for r in rels_xml}
    except KeyError:
        return {}


def _cell_value(cell, strings):
    t = cell.get("t", "")
    v = cell.find(f"{{{NS}}}v")
    if v is None or v.text is None:
        return ""
    if t == "s":
        idx = int(v.text)
        return strings[idx] if idx < len(strings) else ""
    if t == "b":
        return bool(int(v.text))
    return v.text


def _parse_worksheet(ws_xml, strings):
    rows = []
    for row_el in ws_xml.findall(f".//{{{NS}}}row"):
        row = [_cell_value(c, strings) for c in row_el.findall(f"{{{NS}}}c")]
        rows.append(row)
    return rows
