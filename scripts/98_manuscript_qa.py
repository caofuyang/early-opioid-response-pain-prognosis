"""Systematic QA for the manuscript markdown.
Checks: stale wording, citation integrity, abstract numbers, placeholders,
word count, section completeness, ordering of tables/figures, terminology.
Exit code 0 only when every check passes.
"""
import re, sys, pathlib

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
body, refs = text.split("## References", 1)
issues = []

# 1 stale wording
for pat in [r"three[- ]database", r"three databases"]:
    for m in re.finditer(pat, body, re.I):
        issues.append(f"stale wording: '{m.group(0)}' near ...{body[max(0,m.start()-40):m.end()+40]}...")

# 2 citations
nums = set()
for m in re.finditer(r"\[([0-9,\s–-]+)\]", body):
    for part in m.group(1).replace("–", "-").split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); nums.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            nums.add(int(part))
ref_items = [int(x) for x in re.findall(r"^(\d+)\. ", refs, re.M)]
if nums != set(ref_items):
    issues.append(f"citation mismatch: cited {sorted(nums)}; listed {sorted(ref_items)}")

# 3 abstract numbers must appear elsewhere
abs_txt = body.split("## Abstract")[1].split("## Introduction")[0]
pcts = set(re.findall(r"\d+\.\d%", abs_txt))
for p in pcts:
    if body.count(p) < 2:
        issues.append(f"abstract number {p} appears only once")

# 4 placeholders
for pat in ["TODO", "TBD", "xx", "to be assembled", "??"]:
    if pat in text:
        issues.append(f"placeholder found: {pat}")

# 5 word count
mc = re.search(r"\*\*Word count\*\* ([\d,]+)", body)
main = body.split("## Introduction")[1].split("## Tables")[0]
actual = len(re.findall(r"\b[\w’'-]+\b", main))
if mc and int(mc.group(1).replace(",", "")) != actual:
    issues.append(f"word count stated {mc.group(1)} but computed {actual:,}")

# 6 required sections
for sec in ["## Abstract", "## Introduction", "## Methods", "## Results",
            "## Discussion", "### Conclusions", "## Declarations", "## References"]:
    if sec not in text:
        issues.append(f"missing section: {sec}")
for sub in ["Ethics approval", "Consent for publication", "Availability of data",
            "Code availability", "Competing interests", "Funding",
            "contributions", "Use of artificial intelligence"]:
    if sub not in text:
        issues.append(f"missing declaration: {sub}")

# 7 abstract length
abs_words = len(re.findall(r"\b[\w’'-]+\b", abs_txt))
if abs_words > 250:
    issues.append(f"abstract length {abs_words} > 250")

# 8 ordering of first citation of tables/figures
def first_pos(label):
    m = re.search(re.escape(label), body)
    return m.start() if m else 10**9
for i in range(1, 5):
    if first_pos(f"Table {i}") > first_pos(f"Table {i+1}") and f"Table {i+1}" in body:
        issues.append(f"Table {i+1} first cited before Table {i}")
for i in range(1, 3):
    if first_pos(f"Figure {i}") > first_pos(f"Figure {i+1}") and f"Figure {i+1}" in body:
        issues.append(f"Figure {i+1} first cited before Figure {i}")

# 9 terminology
if "eICU " in body.replace("eICU-CRD", ""):
    issues.append("inconsistent 'eICU' vs 'eICU-CRD'")

if issues:
    for i in issues:
        print("FAIL:", i)
    print(f"{len(issues)} issue(s)")
    sys.exit(1)
print("ALL CHECKS PASSED")
