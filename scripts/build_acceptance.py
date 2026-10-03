#!/usr/bin/env python3
"""Build fail-closed production acceptance metadata from completed edit runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SUMMARY_FILES = {
    "generation": "generation_summary.json",
    "conditioning": "conditioning/multimodal_conditioning_summary.json",
    "transformer": "transformer/multimodal_edit_summary.json",
    "vae": "vae/vae_stage_summary.json",
}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_case(name: str, run_dir: str | Path) -> dict:
    run = Path(run_dir)
    result = {"name": name, "run_dir": str(run), "status": "FAIL", "errors": []}
    summaries = {}
    for stage, rel in SUMMARY_FILES.items():
        path = run / rel
        if not path.is_file():
            result["errors"].append(f"missing:{rel}")
            continue
        try:
            summaries[stage] = _read_json(path)
        except Exception as exc:
            result["errors"].append(f"invalid_json:{rel}:{type(exc).__name__}")
    if len(summaries) != len(SUMMARY_FILES):
        return result
    for stage, data in summaries.items():
        if data.get("status") != "PASS":
            result["errors"].append(f"stage_not_pass:{stage}")
    ref = run / "reference/reference_latent.npy"
    if not ref.is_file():
        result["errors"].append("missing:reference/reference_latent.npy")
    pngs = [Path(p) for p in summaries["generation"].get("pngs", [])]
    if len(pngs) != 4:
        result["errors"].append(f"png_count:{len(pngs)}")
    missing_pngs = [str(p) for p in pngs if not p.is_file()]
    if missing_pngs:
        result["errors"].append(f"missing_pngs:{len(missing_pngs)}")
    hashes = [_sha256(p) for p in pngs if p.is_file()]
    unique = len(set(hashes))
    if len(hashes) == 4 and unique != 4:
        result["errors"].append(f"duplicate_png_sha256:{unique}")
    vae = summaries["vae"]
    if vae.get("deterministic_bit_exact") is not True:
        result["errors"].append("vae_not_deterministic_bit_exact")
    if vae.get("outputs_unique") is not True:
        result["errors"].append("vae_outputs_not_unique")
    transformer = summaries["transformer"]
    if transformer.get("steps") != 4:
        result["errors"].append(f"transformer_steps:{transformer.get('steps')}")
    generation = summaries["generation"]
    result.update({
        "resolution": generation.get("resolution"),
        "topology": generation.get("topology"),
        "seeds": generation.get("seeds"),
        "conditioning_tokens": summaries["conditioning"].get("conditioning_tokens"),
        "png_count": len(pngs),
        "unique_png_sha256": unique,
        "png_sha256": hashes,
        "transformer_wall_s": transformer.get("wall_s"),
        "peak_hbm_bytes": max((x.get("peak_bytes_in_use") or 0) for x in transformer.get("hbm", [])) if transformer.get("hbm") else None,
    })
    if not result["errors"]:
        result["status"] = "PASS"
    return result


def build_acceptance(cases: list[tuple[str, str | Path]]) -> dict:
    evaluated = [evaluate_case(name, path) for name, path in cases]
    return {
        "format_version": 1,
        "project": "Mage-Flow-Edit-Turbo JAX TPU v5e-8",
        "status": "PASS" if evaluated and all(c["status"] == "PASS" for c in evaluated) else "FAIL",
        "case_count": len(evaluated),
        "cases": evaluated,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", required=True, help="NAME=/absolute/run/path")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cases = []
    for raw in args.case:
        if "=" not in raw:
            raise SystemExit("--case must use NAME=PATH")
        name, path = raw.split("=", 1)
        cases.append((name, path))
    acceptance = build_acceptance(cases)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, indent=2))
    return 0 if acceptance["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
