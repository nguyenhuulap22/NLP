from pathlib import Path
import csv

VERSION = "PAPER_SELECTIVE_FORCE_K1_FIXED_20260629"
print(f"OK: {VERSION}")

path = Path("resources/it.csv")
if not path.exists():
    raise SystemExit("MISSING resources/it.csv")

watch = {
    "API",
    "JSON response",
    "server",
    "deploy",
    "database query",
    "decoder",
    "logits",
    "next token",
    "Transformer",
    "attention",
    "beam search",
    "bug",
    "code",
    "model",
    "dataset",
    "accuracy",
}

with path.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

by_source = {row["source"]: row for row in rows}

for source in sorted(watch):
    row = by_source.get(source)
    if not row:
        print(f"MISSING: {source}")
        continue
    print(
        f"{source} -> {row['target']} "
        f"[{row['constraint_type']} | force={row['force']} | protect={row['protect']}]"
    )

forced = [r for r in rows if str(r.get("force", "0")).strip() == "1"]
soft = [r for r in rows if str(r.get("force", "0")).strip() != "1"]
print(f"FORCED={len(forced)} SOFT={len(soft)} TOTAL={len(rows)}")
