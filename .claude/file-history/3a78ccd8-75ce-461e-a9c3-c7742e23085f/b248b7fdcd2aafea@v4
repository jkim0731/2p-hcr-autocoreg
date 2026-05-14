"""Emit a minimal BigWarp project JSON pairing the exported TIFFs.

The HCR volumes are already in CZ frame, so the project just lists the
four files with no transform — open in BigWarp/BDV-Fiji and toggle
overlay/contrast. The CZ stack is the fixed image; the warped HCR
volumes are also fixed (already aligned). seg is optional moving.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT_ROOT = Path("/root/capsule/code/sessions/13_pairwise_unmix_gfp/outputs/bigwarp_export")


def project_dict(sid: str) -> dict:
    """All volumes are in CZ frame and identity-aligned. HCR sources are
    marked moving (so BigWarp can lay down landmarks for any residual
    refinement); CZ sources are fixed (target)."""
    d = OUT_ROOT / sid
    src = [
        ("hcr_405_in_cz",       True ),
        ("hcr_488_in_cz",       True ),
        ("hcr_seg_coreg_in_cz", True ),
        ("hcr_seg_missed_in_cz",True ),
        ("cz_stack",            False),
        ("cz_seg_all",          False),
        ("cz_seg_coreg",        False),
    ]
    sources = {
        str(i): {"uri": str(d / f"{name}.tif"), "name": name, "isMoving": moving}
        for i, (name, moving) in enumerate(src)
    }
    return {"Sources": sources}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="+")
    args = ap.parse_args()
    for sid in args.subjects:
        d = OUT_ROOT / sid
        if not d.exists():
            print(f"  {sid}: SKIP (no export dir)")
            continue
        proj = project_dict(sid)
        out = d / "bigwarp-project.json"
        out.write_text(json.dumps(proj, indent=2))
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
