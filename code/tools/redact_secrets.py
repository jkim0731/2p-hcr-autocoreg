#!/usr/bin/env python3
"""Redact secrets from a directory before publishing it as a data asset.

Why this exists
---------------
Session transcripts under `.claude/projects/` can capture secrets that were
printed to a shell during a session (e.g. an `env` dump that exposed a GitHub
token). Before copying `.claude/` (or any transcripts) into a read-only data
asset under `/results`, run this to scrub known credential patterns and to
refuse to publish live credential files.

Design choices
--------------
- DRY-RUN by default. Nothing is written unless you pass --apply.
- High-precision, provider-prefixed patterns only (ghp_, github_pat_, glpat-,
  xox*-, hf_, sk-ant-, AKIA/ASIA, PEM blocks, url basic-auth). Bare `AIza...`
  (Google) and bare `sk-...` (OpenAI) are intentionally OMITTED: transcripts
  embed base64 images, and those short prefixes collide with base64, producing
  false positives that would corrupt images without any security benefit.
  Add them with --extra only if you know the target has no base64 blobs.
- Reports counts per pattern per file; never prints the secret value.
- Refuses to leave live credential files (.credentials.json, *.pem, id_rsa,
  .env) inside the publish target: with --apply it deletes them and warns.

Usage
-----
    # scan only (safe, default)
    python redact_secrets.py /results/<asset>/.claude

    # actually redact in place
    python redact_secrets.py /results/<asset>/.claude --apply
"""
from __future__ import annotations
import argparse, os, re, sys

# (label, compiled pattern). Replacement is f"<REDACTED:{label}>".
PATTERNS = [
    ("github_token",   re.compile(r"gh[pousr]_[A-Za-z0-9]{30,255}")),
    ("github_pat",     re.compile(r"github_pat_[A-Za-z0-9_]{22,255}")),
    ("gitlab_pat",     re.compile(r"glpat-[A-Za-z0-9_-]{20,}")),
    ("slack_token",    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("hf_token",       re.compile(r"hf_[A-Za-z0-9]{34,}")),
    ("anthropic_key",  re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("aws_access_key", re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}")),
    ("private_key",    re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----.*?-----END[A-Z ]*PRIVATE KEY-----", re.DOTALL)),
    ("url_basic_auth", re.compile(r"(https?://[^/\s:@]+:)[^/\s:@]+(@)")),
]

# Opt-in extras (false-positive prone in base64 transcripts).
EXTRA_PATTERNS = {
    "google_api_key": re.compile(r"AIza[A-Za-z0-9_-]{35}"),
    "openai_key":     re.compile(r"sk-[A-Za-z0-9]{20,}"),
}

# Live credential files that must never be published.
FORBIDDEN_NAMES = {".credentials.json", "id_rsa", "id_ed25519", ".env"}
FORBIDDEN_SUFFIXES = (".pem", ".key")

MAX_BYTES = 200 * 1024 * 1024  # skip absurdly large files


def is_probably_text(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(8192)
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    return True


def url_auth_sub(m: re.Match) -> str:
    # keep scheme/host, drop the password
    return m.group(1) + "<REDACTED:url_basic_auth>" + m.group(2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Redact secrets before publishing a data asset.")
    ap.add_argument("target", help="Directory to scan/redact (e.g. /results/<asset>/.claude)")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    ap.add_argument("--extra", action="store_true", help="Also use false-positive-prone patterns (google/openai)")
    args = ap.parse_args()

    target = os.path.abspath(args.target)
    if not os.path.isdir(target):
        print(f"ERROR: not a directory: {target}", file=sys.stderr)
        return 2

    patterns = list(PATTERNS)
    if args.extra:
        patterns += list(EXTRA_PATTERNS.items())

    mode = "APPLY (writing)" if args.apply else "DRY-RUN (no writes)"
    print(f"# redact_secrets - {mode}\n# target: {target}\n")

    forbidden_found, total_by_label, files_changed = [], {}, 0

    for root, _dirs, files in os.walk(target):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, target)

            if name in FORBIDDEN_NAMES or name.endswith(FORBIDDEN_SUFFIXES):
                forbidden_found.append(rel)
                if args.apply:
                    os.remove(path)
                continue

            try:
                if os.path.getsize(path) > MAX_BYTES or not is_probably_text(path):
                    continue
                text = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue

            hits, new_text = {}, text
            for label, pat in patterns:
                repl = url_auth_sub if label == "url_basic_auth" else f"<REDACTED:{label}>"
                new_text, n = pat.subn(repl, new_text)
                if n:
                    hits[label] = n
                    total_by_label[label] = total_by_label.get(label, 0) + n
            if hits:
                files_changed += 1
                summary = ", ".join(f"{k}={v}" for k, v in sorted(hits.items()))
                print(f"  {rel}: {summary}")
                if args.apply:
                    open(path, "w", encoding="utf-8").write(new_text)

    print("\n# ---- summary ----")
    if forbidden_found:
        verb = "DELETED" if args.apply else "WOULD DELETE (forbidden to publish)"
        print(f"  {verb}: {len(forbidden_found)} credential file(s):")
        for f in forbidden_found:
            print(f"      {f}")
    if total_by_label:
        verb = "redacted" if args.apply else "would redact"
        print(f"  {verb} {sum(total_by_label.values())} secret(s) across {files_changed} file(s):")
        for k, v in sorted(total_by_label.items()):
            print(f"      {k}: {v}")
    else:
        print("  no secret patterns matched.")
    if not args.apply and (total_by_label or forbidden_found):
        print("\n  re-run with --apply to write these changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
