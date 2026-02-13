import re
from pathlib import Path

raw = Path("data/520001m_vol1.txt").read_text(encoding="utf-8", errors="ignore")

# 1) Normalize page breaks
raw = raw.replace("\r", "")
raw = raw.replace("\f", "\n")  # convert page breaks to newline

lines = raw.splitlines()
out = []

HEADER_RE = re.compile(r'^\s*DoDM\s+5200\.01-V1,\s+February\s+24,\s+2012\s*$')
FOOTER_RE = re.compile(r'^\s*Change\s+\d+,\s+\d{2}/\d{2}/\d{4}\s+\d+.*$')
CONTENTS_PAGE_RE = re.compile(r'^\s*TABLE OF CONTENTS\s*$', re.IGNORECASE)
#CONTENTS_LINE_RE = re.compile(r'\.{5,}\s*\d+\s*$')  # "....8" style

for line in lines:
    s = line.rstrip()

    # Drop known header/footer patterns
    if HEADER_RE.match(s):
        continue
    if FOOTER_RE.match(s):
        continue

    # Drop “CONTENTS” footer tags if they appear
    if s.strip().upper() == "CONTENTS":
        continue

    """
    # Drop TOC leader lines (optional—keeps doc content cleaner)
    if CONTENTS_LINE_RE.search(s) and ("ENCLOSURE" in s.upper() or "GLOSSARY" in s.upper()):
        continue
    """
    out.append(s)

text = "\n".join(out)

# 2) Whitespace cleanup: collapse excessive spaces but keep paragraphs
text = re.sub(r'[ \t]+', ' ', text)
text = re.sub(r'\n{3,}', '\n\n', text).strip()

Path("data/520001m_vol1_clean.txt").write_text(text, encoding="utf-8")
print("Wrote 520001m_vol1_clean.txt")
