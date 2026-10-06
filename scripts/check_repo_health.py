#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("REPO_HEALTH_ROOT", Path(__file__).resolve().parents[1])).resolve()
errors: list[str] = []

REQUIRED = [
    "README.md", "README.vi.md", "CHANGELOG.md", "CHANGELOG.vi.md",
    "CONTRIBUTING.md", "CONTRIBUTING.vi.md", "CODE_OF_CONDUCT.md", "CODE_OF_CONDUCT.vi.md",
    "SECURITY.md", "SECURITY.vi.md", "SUPPORT.md", "SUPPORT.vi.md",
    "NOTICE.md", "NOTICE.vi.md", "MODEL_LICENSE.md", "MODEL_LICENSE.vi.md",
    "ASSET_LICENSE.md", "ASSET_LICENSE.vi.md", "LICENSES.md", "LICENSES.vi.md",
    "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.vi.md",
    "RELEASE_NOTES_v1.0.0.md", "RELEASE_NOTES_v1.0.0.vi.md",
    "LICENSE", "LICENSE.vi.md", "CITATION.cff",
    "acceptance/PRODUCTION_ACCEPTANCE.json",
    "acceptance/PRODUCTION_ACCEPTANCE_CLOSEOUT.md",
    "acceptance/PRODUCTION_ACCEPTANCE_CLOSEOUT.vi.md",
    "bootstrap/08_run_tpu_edit.py", "bootstrap/08_tpu_stage_multimodal_conditioning.py",
    "bootstrap/08_tpu_stage_reference_vae.py", "bootstrap/08_tpu_stage_edit_transformer.py",
    "bootstrap/06_tpu_stage_vae.py", "runtime/experiment/qwen3vl_vision_runtime.py",
    "runtime/experiment/qwen3vl_multimodal_runtime.py", "scripts/build_acceptance.py",
    "requirements-ci.txt", "notebooks/kaggle-production-demo.ipynb",
    "release-evidence/EXTENDED_VALIDATION.json", "release-evidence/PUBLISH_MANIFEST.json",
    "release-evidence/BENCHMARKS.md", "release-evidence/BENCHMARKS.vi.md",
    "licenses/README.md", "licenses/README.vi.md",
    "licenses/Mage-Flow-MIT.txt", "licenses/Qwen3-VL-Apache-2.0.txt",
    ".github/PULL_REQUEST_TEMPLATE.md", ".github/PULL_REQUEST_TEMPLATE.vi.md",
]
REQUIRED += [
    f"docs/{name}{suffix}"
    for name in (
        "ARCHITECTURE", "INFERENCE-GUIDE", "BENCHMARKS", "REPRODUCIBILITY", "KAGGLE",
        "ARTIFACTS", "LIMITATIONS", "TROUBLESHOOTING", "RELEASE-EVIDENCE-v1.0.0",
    )
    for suffix in (".md", ".vi.md")
]

for rel in REQUIRED:
    if not (ROOT / rel).is_file():
        errors.append(f"missing:{rel}")

for p in ROOT.rglob("*"):
    if not p.is_file() or ".git" in p.parts or ".pytest_cache" in p.parts or "__pycache__" in p.parts:
        continue
    rel = p.relative_to(ROOT).as_posix()
    try:
        size = p.stat().st_size
    except OSError:
        continue
    if size >= 50 * 1024 * 1024:
        errors.append(f"oversize:{rel}:{size}")
    if p.suffix.lower() in {".safetensors", ".whl", ".ckpt", ".tar", ".gz"}:
        errors.append(f"forbidden_artifact:{rel}")
    low = rel.lower()
    if "hf_token" in low or low.endswith("/token"):
        errors.append(f"sensitive_name:{rel}")
    if rel.startswith("bootstrap/") and "pilot" in p.name.lower():
        errors.append(f"pilot_name:{rel}")
    if p.is_symlink():
        errors.append(f"symlink:{rel}")

EXEMPT_MD = {"LICENSE.vi.md"}
for p in ROOT.rglob("*.md"):
    if not p.is_file() or ".git" in p.parts or ".pytest_cache" in p.parts:
        continue
    rel = p.relative_to(ROOT).as_posix()
    if rel in EXEMPT_MD or rel.startswith("docs/superpowers/"):
        continue
    if p.name.endswith(".vi.md"):
        peer = p.with_name(p.name[:-6] + ".md")
        if not peer.is_file():
            errors.append(f"bilingual_peer_missing:{rel}->{peer.relative_to(ROOT).as_posix()}")
        elif peer.name not in p.read_text(encoding="utf-8", errors="ignore"):
            errors.append(f"language_switch_missing:{rel}->{peer.name}")
    else:
        peer = p.with_name(p.stem + ".vi.md")
        if not peer.is_file():
            errors.append(f"bilingual_peer_missing:{rel}->{peer.relative_to(ROOT).as_posix()}")
        elif peer.name not in p.read_text(encoding="utf-8", errors="ignore"):
            errors.append(f"language_switch_missing:{rel}->{peer.name}")

license_path = ROOT / "LICENSE"
license_text = license_path.read_text(encoding="utf-8") if license_path.is_file() else ""
exact_author = "Copyright (c) 2026 Đăng Khoa <i.am@dangkhoa.dev>"
if exact_author not in license_text:
    errors.append("license_author_line_missing")

model_en = (ROOT / "MODEL_LICENSE.md").read_text(encoding="utf-8") if (ROOT / "MODEL_LICENSE.md").is_file() else ""
model_vi = (ROOT / "MODEL_LICENSE.vi.md").read_text(encoding="utf-8") if (ROOT / "MODEL_LICENSE.vi.md").is_file() else ""
for marker, text, label in [
    ("Mage-Flow / Mage-Flow-Turbo / Mage-Flow-Edit-Turbo components — MIT;", model_en, "MODEL_LICENSE.md summary"),
    ("Mage-Flow / Mage-Flow-Turbo / Mage-Flow-Edit-Turbo — MIT;", model_vi, "MODEL_LICENSE.vi.md summary"),
    ("Mage-Flow-Edit-Turbo, Microsoft", model_en, "MODEL_LICENSE.md trademark boundary"),
    ("Mage-Flow-Edit-Turbo, Microsoft", model_vi, "MODEL_LICENSE.vi.md trademark boundary"),
]:
    if marker not in text:
        errors.append(f"license_marker_missing:{label}")

asset_en = (ROOT / "ASSET_LICENSE.md").read_text(encoding="utf-8") if (ROOT / "ASSET_LICENSE.md").is_file() else ""
licenses_en = (ROOT / "LICENSES.md").read_text(encoding="utf-8") if (ROOT / "LICENSES.md").is_file() else ""
notice_en = (ROOT / "NOTICE.md").read_text(encoding="utf-8") if (ROOT / "NOTICE.md").is_file() else ""
third_party_en = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8") if (ROOT / "THIRD_PARTY_NOTICES.md").is_file() else ""
for marker, text, label in [
    ("does not independently assert source-image ownership", asset_en.lower(), "ASSET_LICENSE.md source-image rights boundary"),
    ("Mage-Flow-MIT.txt", licenses_en, "LICENSES.md Mage license authority"),
    ("Qwen3-VL-Apache-2.0.txt", licenses_en, "LICENSES.md Qwen license authority"),
    ("ASSET_LICENSE.md", notice_en, "NOTICE.md visual provenance link"),
    ("standalone model artifact", third_party_en.lower(), "THIRD_PARTY_NOTICES.md distribution notice"),
]:
    if marker not in text:
        errors.append(f"distribution_license_marker_missing:{label}")

acc_path = ROOT / "acceptance/PRODUCTION_ACCEPTANCE.json"
if acc_path.is_file():
    try:
        acc = json.loads(acc_path.read_text(encoding="utf-8"))
        if acc.get("status") != "PASS" or acc.get("case_count") != 3:
            errors.append("acceptance:not_three_case_pass")
        for case in acc.get("cases", []):
            if case.get("status") != "PASS" or case.get("png_count") != 4 or case.get("unique_png_sha256") != 4:
                errors.append(f"acceptance:case_invalid:{case.get('name')}")
    except Exception as exc:
        errors.append(f"acceptance:invalid_json:{type(exc).__name__}")

manifest_path = ROOT / "release-evidence/PUBLISH_MANIFEST.json"
if manifest_path.is_file():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = [(x.get("seed"), x.get("mode")) for x in manifest.get("selected", [])]
        if selected != [(52, "768 q256"), (43, "768 q256"), (43, "768 q256"), (43, "768 q256")]:
            errors.append("release_manifest:canonical_showcase_mismatch")
        tech = manifest.get("technical_evidence", {})
        if tech.get("production_qualified_baseline", {}).get("resolution") != 512:
            errors.append("release_manifest:production_baseline_not_512")
        if tech.get("experimental_high_resolution_1024", {}).get("query_chunk") != 128:
            errors.append("release_manifest:experimental_1024_not_q128")
        if tech.get("query_chunk_cross_parity", {}).get("512_q128_vs_q256_bit_exact") is not False:
            errors.append("release_manifest:q128_q256_cross_parity_claim_invalid")
    except Exception as exc:
        errors.append(f"release_manifest:invalid_json:{type(exc).__name__}")

readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").is_file() else ""
for marker in (
    "Production-qualified baseline",
    "Validated scaling",
    "Experimental high resolution",
    "not bit-identical at 512px",
    "does **not** require a separate Turbo-JAX model",
    "ASSET_LICENSE.md",
    "LICENSES.md",
):
    if marker not in readme:
        errors.append(f"readme_claim_missing:{marker}")

attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8") if (ROOT / ".gitattributes").is_file() else ""
for rel in (
    "qwen3vl-processor-assets/tokenizer.json",
    "qwen3vl-processor-assets/merges.txt",
    "qwen3vl-processor-assets/vocab.json",
):
    if f"{rel} -diff" not in attrs:
        errors.append(f"opaque_asset_rule_missing:{rel}")

for stale in ("docs/architecture.md", "docs/inference-guide.md", "docs/limitations.md"):
    if (ROOT / stale).exists():
        errors.append(f"stale_doc_path:{stale}")
for duplicate in (
    ".github/CONTRIBUTING.md", ".github/CODE_OF_CONDUCT.md", ".github/SECURITY.md", ".github/SUPPORT.md",
):
    if (ROOT / duplicate).exists():
        errors.append(f"duplicate_community_doc:{duplicate}")

if errors:
    for error in sorted(set(errors)):
        print("[FAIL]", error)
    raise SystemExit(1)

print("[PASS] required production and governance files")
print("[PASS] bilingual public Markdown contract")
print("[PASS] lightweight source-only package and exact MIT boundary")
print("[PASS] mixed-origin model and asset licensing boundary")
print("[PASS] 512 production / 768 validated / 1024 experimental claim boundary")
print("[PASS] three-case acceptance and canonical showcase metadata")
print("REPO_HEALTH_CHECK=PASS")
