"""Cross-reference audit: every table/figure cited in the text must exist."""
import re, sys, pathlib

def audit(manuscript: str, supplementary: str | None = None, figure_dir: str | None = None) -> int:
    text = pathlib.Path(manuscript).read_text(encoding="utf-8")
    body = text.split("## References")[0]
    cites = set(re.findall(r"(Table|Figure)\s+(S?\d+)", body))
    issues = []
    for kind, num in sorted(cites):
        if kind == "Table" and num.startswith("S"):
            if supplementary is None or not pathlib.Path(supplementary).exists():
                issues.append(f"Supplementary Table {num} cited but no supplementary file")
            else:
                sup = pathlib.Path(supplementary).read_text(encoding="utf-8")
                if f"Supplementary Table {num}" not in sup:
                    issues.append(f"Supplementary Table {num} cited but not present in supplementary file")
        elif kind == "Table":
            if f"**Table {num}.**" not in text:
                issues.append(f"Table {num} cited but no table heading found")
        elif kind == "Figure":
            if f"Figure {num}." not in text and f"**Figure {num}.**" not in text:
                issues.append(f"Figure {num} cited but no figure legend found")
            if figure_dir:
                hits = list(pathlib.Path(figure_dir).glob(f"Figure_{num}*"))
                if not hits:
                    issues.append(f"Figure {num} cited but no image file in {figure_dir}")
    for i in issues:
        print("FAIL:", i)
    print("PASS" if not issues else f"{len(issues)} issue(s)")
    return 0 if not issues else 1

if __name__ == "__main__":
    sys.exit(audit(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
                    sys.argv[3] if len(sys.argv) > 3 else None))
