#!/usr/bin/env python3
"""Extract plaintext transcript from a mikupad/multipad JSON export.

Mikupad's export stores ``prompt`` as a JSON-encoded array of segments.
Each segment is either:
  * ``{"type": "user", "content": "..."}``        — human-inserted prefill text
  * ``{"content": "...", "prob": ..., ...}``      — model-generated token

Plaintext recovery = concat all ``content`` fields in order.

Usage:
    extract_transcript.py INPUT.json [-o OUTPUT.txt] [--raw]
    extract_transcript.py --batch DIR [-o OUTDIR]

Default behavior strips ChatML control tokens (``<|im_start|>``, ``<|im_end|>``)
to produce a clean reading transcript. ``--raw`` preserves them verbatim.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ChatML control markers we strip in cleaned mode. We collapse
# ``<|im_start|>role\n`` into ``role:\n`` and drop ``<|im_end|>`` entirely.
_IM_START = re.compile(r"<\|im_start\|>([^\n]*)\n")
_IM_END = re.compile(r"<\|im_end\|>\n?")


def extract(path: Path, *, raw: bool = False) -> str:
    data = json.loads(path.read_text())
    prompt = data.get("prompt")
    if prompt is None:
        raise ValueError(f"{path}: no 'prompt' field")
    if isinstance(prompt, str):
        segments = json.loads(prompt)
    elif isinstance(prompt, list):
        segments = prompt
    else:
        raise ValueError(f"{path}: unexpected 'prompt' type {type(prompt).__name__}")

    out = "".join(seg.get("content", "") for seg in segments)
    if raw:
        return out
    out = _IM_START.sub(lambda m: f"{m.group(1)}:\n", out)
    out = _IM_END.sub("\n", out)
    return out


def _outfile_for(infile: Path, outdir: Path | None) -> Path:
    target_dir = outdir if outdir is not None else infile.parent
    return target_dir / (infile.stem + ".txt")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    ap.add_argument("input", type=Path, help="Input .json file, or directory if --batch")
    ap.add_argument("-o", "--output", type=Path, help="Output file (single) or directory (--batch)")
    ap.add_argument("--raw", action="store_true", help="Preserve ChatML markers verbatim")
    ap.add_argument("--batch", action="store_true", help="Treat input as a directory of .json files")
    args = ap.parse_args(argv)

    if args.batch:
        if not args.input.is_dir():
            print(f"error: --batch requires a directory, got {args.input}", file=sys.stderr)
            return 2
        outdir = args.output
        if outdir is not None:
            outdir.mkdir(parents=True, exist_ok=True)
        files = sorted(args.input.glob("*.json"))
        if not files:
            print(f"error: no .json files in {args.input}", file=sys.stderr)
            return 1
        for f in files:
            try:
                text = extract(f, raw=args.raw)
            except Exception as exc:
                print(f"skip {f.name}: {exc}", file=sys.stderr)
                continue
            dest = _outfile_for(f, outdir)
            dest.write_text(text)
            print(f"{f.name} -> {dest} ({len(text)} chars)")
        return 0

    text = extract(args.input, raw=args.raw)
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text)
        print(f"{args.input.name} -> {args.output} ({len(text)} chars)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
