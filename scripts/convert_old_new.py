import re
import json
from pathlib import Path

CHUNK_DIR = Path("data/chunks")
OUT_JSON  = Path("data/chunks.jsonl")

# =========================
# 1) Robust replacement set
# =========================
# Goals:
# - Catch "Department of Defense" even if line-broken ("Department\nof Defense") or extra spaces.
# - Catch abbreviations/prefixes: DoD, DOD, DoDI, DoDD, DoDM, DoD-Issuances, etc.
# - Catch "Secretary of Defense" / "SecDef" (optional but you asked for it)
# - Preserve case style where reasonable (upper vs title vs lower).

def _case_like(src: str, dst_title: str, dst_upper: str, dst_lower: str):
    """Return dst in a casing style similar to src."""
    if src.isupper():
        return dst_upper
    if src.islower():
        return dst_lower
    # Title-ish / mixed defaults to title version
    return dst_title

# --- Key phrase matchers (space/newline tolerant) ---
# Matches:
#   Department of Defense
#   Department\nof Defense
#   Department  of   Defense
#   DEPARTMENT OF DEFENSE (all caps)
DEPT_DEF_RE = re.compile(
    r"\bDEPARTMENT\s+OF\s+DEFENSE\b|\bDepartment\s+of\s+Defense\b|\bdepartment\s+of\s+defense\b",
    re.MULTILINE,
)

# Possessive variant (with optional line breaks)
DEPT_DEF_POS_RE = re.compile(
    r"\bDEPARTMENT\s+OF\s+DEFENSE\s*'S\b|\bDepartment\s+of\s+Defense\s*'s\b|\bdepartment\s+of\s+defense\s*'s\b",
    re.MULTILINE,
)

# Secretary of Defense (space/newline tolerant)
SECDEF_RE = re.compile(
    r"\bSECRETARY\s+OF\s+DEFENSE\b|\bSecretary\s+of\s+Defense\b|\bsecretary\s+of\s+defense\b",
    re.MULTILINE,
)

# Common abbreviation "SecDef"
SECDEF_ABBR_RE = re.compile(r"\bSecDef\b|\bSECDEF\b|\bsecdef\b")

# DoD-prefixed issuance abbreviations:
# DoDI, DoDD, DoDM, DoDG, DoDAAF, etc (and uppercase variants)
DOD_PREFIX_RE = re.compile(r"\b(DoD)([A-Za-z]{1,5})\b|\b(DOD)([A-Za-z]{1,5})\b")

# Plain DoD (word boundary)
DOD_RE = re.compile(r"\bDoD\b|\bDOD\b|\bdod\b")

# If you also want to transform "Defense" in other role names/titles,
# you can add more patterns, but beware of meaning-changing replacements.

def apply_rules(text: str):
    counts = {}
    out = text

    # 1) Department of Defense's -> Department of War's (case-aware)
    def repl_dept_pos(m):
        src = m.group(0)
        rep = _case_like(src, "Department of War's", "DEPARTMENT OF WAR'S", "department of war's")
        counts["Department of Defense's (flex)"] = counts.get("Department of Defense's (flex)", 0) + 1
        return rep

    out = DEPT_DEF_POS_RE.sub(repl_dept_pos, out)

    # 2) Department of Defense -> Department of War (case-aware)
    def repl_dept(m):
        src = m.group(0)
        rep = _case_like(src, "Department of War", "DEPARTMENT OF WAR", "department of war")
        counts["Department of Defense (flex)"] = counts.get("Department of Defense (flex)", 0) + 1
        return rep

    out = DEPT_DEF_RE.sub(repl_dept, out)

    # 3) Secretary of Defense -> Secretary of War (case-aware)
    def repl_sec(m):
        src = m.group(0)
        rep = _case_like(src, "Secretary of War", "SECRETARY OF WAR", "secretary of war")
        counts["Secretary of Defense (flex)"] = counts.get("Secretary of Defense (flex)", 0) + 1
        return rep

    out = SECDEF_RE.sub(repl_sec, out)

    # 4) SecDef -> SecWar (optional; remove if you don't want this)
    def repl_secabbr(m):
        src = m.group(0)
        rep = _case_like(src, "SecWar", "SECWAR", "secwar")
        counts["SecDef"] = counts.get("SecDef", 0) + 1
        return rep

    out = SECDEF_ABBR_RE.sub(repl_secabbr, out)

    # 5) DoD + suffix -> DoW + suffix (DoDI -> DoWI, DoDM -> DoWM, etc.)
    # This catches the "DoDI/DoDM/DoDD" issue you mentioned.
    def repl_dodprefix(m):
        # groups: (DoD)(suffix) OR (DOD)(suffix)
        if m.group(1) is not None:
            prefix = m.group(1)  # "DoD"
            suffix = m.group(2)
        else:
            prefix = m.group(3)  # "DOD"
            suffix = m.group(4)

        # Preserve casing of prefix: DoD->DoW, DOD->DOW
        new_prefix = "DoW" if prefix == "DoD" else "DOW" if prefix == "DOD" else "dow"
        counts["DoD* prefix"] = counts.get("DoD* prefix", 0) + 1
        return f"{new_prefix}{suffix}"

    out = DOD_PREFIX_RE.sub(repl_dodprefix, out)

    # 6) Plain DoD -> DoW (case-aware)
    def repl_dod(m):
        src = m.group(0)
        rep = _case_like(src, "DoW", "DOW", "dow")
        counts["DoD"] = counts.get("DoD", 0) + 1
        return rep

    out = DOD_RE.sub(repl_dod, out)

    total = sum(counts.values())
    return out, counts, total

# =========================
# 2) Forbidden checks (debug)
# =========================
FORBIDDEN_PATTERNS = {
    "DoD": re.compile(r"\bDoD\b|\bDOD\b|\bdod\b"),
    "DoD-prefix": re.compile(r"\bDoD[A-Za-z]{1,5}\b|\bDOD[A-Za-z]{1,5}\b"),
    # catches any remaining "Department of Defense" even split by whitespace/newlines:
    "Department of Defense (flex)": re.compile(r"\bDepartment\s+of\s+Defense\b|\bDEPARTMENT\s+OF\s+DEFENSE\b|\bdepartment\s+of\s+defense\b", re.MULTILINE),
    "Secretary of Defense (flex)": re.compile(r"\bSecretary\s+of\s+Defense\b|\bSECRETARY\s+OF\s+DEFENSE\b|\bsecretary\s+of\s+defense\b", re.MULTILINE),
}

def count_forbidden(s: str):
    details = {k: len(p.findall(s)) for k, p in FORBIDDEN_PATTERNS.items()}
    total = sum(details.values())
    return total, {k: v for k, v in details.items() if v}

# =========================
# 3) Run conversion + write
# =========================
chunk_files = sorted(CHUNK_DIR.glob("chunk_*.txt"))
if not chunk_files:
    raise SystemExit(f"No chunk_*.txt files found in {CHUNK_DIR}")

with OUT_JSON.open("w", encoding="utf-8") as f:
    for i, path in enumerate(chunk_files):
        old = path.read_text(encoding="utf-8", errors="ignore")

        new, counts_by_name, total = apply_rules(old)
        forbid_total, forbid_details = count_forbidden(new)

        # Per-chunk counts
        print(f"{path.name}: total_replacements={total}")
        for k, v in sorted(counts_by_name.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {k}: {v}")

        if forbid_total:
            print(f"  [WARN] leftover_forbidden={forbid_total} {forbid_details}")

        rec = {
            "id": i,
            "chunk_id": path.stem,
            "OLD": old,
            "NEW": new,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"\nWrote {len(chunk_files)} records to {OUT_JSON}")

