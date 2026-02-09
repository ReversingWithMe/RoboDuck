#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table


# ---------------------------
# Helpers to parse description
# ---------------------------
_DESC_FIELD_RE = re.compile(r"^(category|reason|source):\s*(.*)$", re.IGNORECASE)


def parse_description(desc: str) -> Dict[str, str]:
    """
    Parses a multi-line description formatted like:
      category: ...
      reason: ...
      source: ...
    Returns dict with keys among {category, reason, source} when present.
    Always includes 'raw' as a fallback.
    """
    out: Dict[str, str] = {"raw": (desc or "").strip()}
    current_key: Optional[str] = None
    buf: List[str] = []

    def flush() -> None:
        nonlocal current_key, buf
        if current_key is not None:
            out[current_key] = "\n".join(buf).strip()
        current_key, buf = None, []

    for line in (desc or "").splitlines():
        m = _DESC_FIELD_RE.match(line.strip())
        if m:
            flush()
            current_key = m.group(1).lower()
            buf = [m.group(2)]
        else:
            if current_key is None:
                continue
            buf.append(line)
    flush()
    return out


def shorten_path(path: str, keep_tail_parts: int = 4) -> str:
    if not path:
        return ""
    p = Path(path)
    parts = p.parts
    if len(parts) <= keep_tail_parts:
        return str(p)
    return str(Path(*parts[-keep_tail_parts:]))


# ---------------------------
# PoV note pretty printing
# ---------------------------
def render_pov_note(pov_note: Optional[str]) -> Optional[Panel]:
    """
    Render pov_note as:
      - JSON pretty + syntax highlight if it's valid JSON
      - else Markdown panel as fallback
    """
    if not pov_note:
        return None

    text = pov_note.strip()
    if not text:
        return None

    # Try JSON first
    try:
        parsed = json.loads(text)
        pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
        syntax = Syntax(pretty, "json", line_numbers=False, word_wrap=True)
        return Panel(syntax, title="🧪 PoV Note", border_style="magenta", box=box.ROUNDED)
    except Exception:
        # Fallback: render as markdown-ish text
        return Panel(Markdown(text), title="🧪 PoV Note", border_style="magenta", box=box.ROUNDED)


# ---------------------------
# Data structures
# ---------------------------
@dataclass
class TraceEntry:
    file: str
    function: str
    description: str
    likely_confidence: float
    likely_above_threshold: Optional[bool] = None
    pov_note: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceEntry":
        return cls(
            file=d.get("file", "") or "",
            function=d.get("function", "") or "",
            description=d.get("description", "") or "",
            likely_confidence=float(d.get("likely_confidence", 0.0) or 0.0),
            likely_above_threshold=d.get("likely_above_threshold"),
            pov_note=d.get("pov_note"),
        )


# ---------------------------
# Loading / extraction
# ---------------------------
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_stage3_entries(data: Dict[str, Any]) -> List[TraceEntry]:
    stage3 = data.get("stage3_trace", [])
    if not isinstance(stage3, list):
        return []
    out: List[TraceEntry] = []
    for item in stage3:
        if isinstance(item, dict):
            out.append(TraceEntry.from_dict(item))
    return out


# ---------------------------
# Summary stats
# ---------------------------
def compute_stats(entries: List[TraceEntry]) -> Dict[str, Any]:
    files = [e.file for e in entries if e.file]
    funcs = [e.function for e in entries if e.function]

    per_file = Counter(files)
    per_func = Counter(funcs)

    categories: List[str] = []
    for e in entries:
        p = parse_description(e.description)
        cat = p.get("category")
        if cat:
            categories.append(cat)
    per_category = Counter(categories)

    above = sum(1 for e in entries if e.likely_above_threshold is True)
    below = sum(1 for e in entries if e.likely_above_threshold is False)
    unknown = sum(1 for e in entries if e.likely_above_threshold is None)

    return {
        "total_entries": len(entries),
        "unique_files": len(per_file),
        "unique_functions": len(per_func),
        "per_file": per_file,
        "per_func": per_func,
        "per_category": per_category,
        "threshold": {"above": above, "below": below, "unknown": unknown},
    }


def print_summary(console: Console, stats: Dict[str, Any], top_n: int = 10) -> None:
    console.print(
        Panel.fit(
            f"[b]stage3_trace summary[/b]\n"
            f"entries: [b]{stats['total_entries']}[/b]\n"
            f"unique files: [b]{stats['unique_files']}[/b]\n"
            f"unique functions: [b]{stats['unique_functions']}[/b]\n"
            f"above threshold: [b]{stats['threshold']['above']}[/b], "
            f"below: [b]{stats['threshold']['below']}[/b], "
            f"unknown: [b]{stats['threshold']['unknown']}[/b]",
            box=box.ROUNDED,
        )
    )

    t_files = Table(title=f"Top files by stage3_trace entries (top {top_n})", box=box.SIMPLE_HEAVY)
    t_files.add_column("Count", justify="right", style="bold")
    t_files.add_column("File", overflow="fold")

    for file, cnt in stats["per_file"].most_common(top_n):
        t_files.add_row(str(cnt), shorten_path(file, keep_tail_parts=4))
    console.print(t_files)

    if stats["per_category"]:
        t_cat = Table(title=f"Top categories (top {top_n})", box=box.SIMPLE_HEAVY)
        t_cat.add_column("Count", justify="right", style="bold")
        t_cat.add_column("Category", overflow="fold")
        for cat, cnt in stats["per_category"].most_common(top_n):
            t_cat.add_row(str(cnt), cat)
        console.print(t_cat)


# ---------------------------
# Detailed printing
# ---------------------------
def print_entries(console: Console, entries: List[TraceEntry], limit: Optional[int] = None) -> None:
    """
    Prints entries sorted by likely_confidence desc.
    Uses a compact per-entry table (better than stuffing huge text into one big table),
    then prints PoV note panel (if present).
    """
    entries_sorted = sorted(entries, key=lambda e: e.likely_confidence, reverse=True)
    if limit is not None:
        entries_sorted = entries_sorted[:limit]

    for idx, e in enumerate(entries_sorted, 1):
        parsed = parse_description(e.description)

        title = f"[{idx}] {shorten_path(e.file, 4)} :: {e.function or '(no function)'}"
        table = Table(title=title, box=box.SIMPLE, show_header=False)
        table.add_column("Field", style="bold", width=12)
        table.add_column("Value", overflow="fold")

        table.add_row("Score", f"{e.likely_confidence:.6f}")
        table.add_row(
            "Above?",
            "✅" if e.likely_above_threshold is True else ("❌" if e.likely_above_threshold is False else "—"),
        )

        if e.file:
            table.add_row("File", e.file)
        if e.function:
            table.add_row("Function", e.function)

        if parsed.get("category"):
            table.add_row("Category", parsed["category"])
        if parsed.get("reason"):
            table.add_row("Reason", parsed["reason"])
        if parsed.get("source"):
            table.add_row("Source", parsed["source"])

        # If none of the structured fields existed, show raw description
        if not any(parsed.get(k) for k in ("category", "reason", "source")) and parsed.get("raw"):
            table.add_row("Desc", parsed["raw"])

        console.print(table)

        pov_panel = render_pov_note(e.pov_note)
        if pov_panel:
            console.print(pov_panel)

        console.print()  # spacing


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize and pretty-print stage3_trace with Rich.")
    ap.add_argument("json_path", type=Path, help="Path to JSON file")
    ap.add_argument("--limit", type=int, default=None, help="Limit how many entries to print in detail")
    ap.add_argument("--top", type=int, default=10, help="Top-N for summary tables")
    args = ap.parse_args()

    console = Console()

    data = load_json(args.json_path)
    entries = get_stage3_entries(data)

    if not entries:
        console.print("[yellow]No stage3_trace entries found.[/yellow]")
        return

    stats = compute_stats(entries)
    print_summary(console, stats, top_n=args.top)
    print_entries(console, entries, limit=args.limit)


if __name__ == "__main__":
    main()

