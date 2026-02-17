import re
from pathlib import Path

INFILE = "data/520001m_vol1_clean.txt"
OUTDIR = "data/chunks"

# For 1-shot OLD+NEW+OLD in Phi-3-mini-4k, start conservative:
MAX_CHARS = 2600
MIN_CHARS = 1900

text = Path(INFILE).read_text(encoding="utf-8", errors="ignore")

# IMPORTANT: do NOT collapse whitespace/newlines here.
# We keep the "PDF-like" wrapping exactly as-is.

# Split into (paragraph, separator) pairs while preserving separators.
# Paragraph boundary = blank line(s): "\n" + optional spaces + "\n" (and repeats)
SEP_RE = re.compile(r"\n[ \t]*\n+")

parts = SEP_RE.split(text)
seps  = SEP_RE.findall(text)

# Rebuild units that represent "a paragraph plus the separator after it" (except last).
units = []
for i, p in enumerate(parts):
    if i < len(seps):
        units.append(p + seps[i])  # keep exact separator
    else:
        units.append(p)            # last paragraph no trailing sep

def pack_units(units, max_chars, min_chars):
    chunks = []
    cur = ""
    for u in units:
        if not u:
            continue

        # If a single paragraph unit is larger than max_chars, we cannot split it
        # without violating "no paragraph split". Put it alone.
        if len(u) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(u)
            continue

        # If adding would overflow, flush current chunk
        if cur and (len(cur) + len(u) > max_chars):
            chunks.append(cur)
            cur = ""

        cur += u

        # If we already have enough, and next add might create a tiny remainder,
        # we just keep going until we hit max_chars. (No early flush here.)

    if cur:
        chunks.append(cur)

    # Merge small chunks forward/backward if possible (still no paragraph splits)
    out = []
    for c in chunks:
        if out and len(c) < min_chars and len(out[-1]) + len(c) <= max_chars:
            out[-1] += c
        else:
            out.append(c)

    return out

chunks = pack_units(units, MAX_CHARS, MIN_CHARS)

outdir = Path(OUTDIR)
outdir.mkdir(exist_ok=True)
for i, c in enumerate(chunks):
    (outdir / f"chunk_{i:04d}.txt").write_text(c, encoding="utf-8")

sizes = [len(c) for c in chunks]
print("Wrote", len(chunks), "chunks to", OUTDIR)
print("min/max/avg chars:", min(sizes), max(sizes), sum(sizes)//len(sizes))

# Optional sanity check: concatenation should equal original exactly
recon = "".join(chunks)
print("Exact reconstruction:", recon == text)


