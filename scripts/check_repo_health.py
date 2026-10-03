#!/usr/bin/env python3
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
errors = []
required = [
    'README.md','README.vi.md','LICENSE','NOTICE.md','MODEL_LICENSE.md','MODEL_LICENSE.vi.md',
    'acceptance/PRODUCTION_ACCEPTANCE.json','bootstrap/08_run_tpu_edit.py',
    'bootstrap/08_tpu_stage_multimodal_conditioning.py','bootstrap/08_tpu_stage_reference_vae.py',
    'bootstrap/08_tpu_stage_edit_transformer.py','bootstrap/06_tpu_stage_vae.py',
    'runtime/experiment/qwen3vl_vision_runtime.py','runtime/experiment/qwen3vl_multimodal_runtime.py',
    'scripts/build_acceptance.py','requirements-ci.txt',
]
for rel in required:
    if not (ROOT/rel).is_file():
        errors.append(f'missing:{rel}')

for p in ROOT.rglob('*'):
    if not p.is_file() or '.git' in p.parts:
        continue
    rel=str(p.relative_to(ROOT))
    if p.stat().st_size >= 50*1024*1024:
        errors.append(f'oversize:{rel}:{p.stat().st_size}')
    if p.suffix.lower() in {'.safetensors','.whl','.ckpt','.tar','.gz'}:
        errors.append(f'forbidden_artifact:{rel}')
    low=rel.lower()
    if 'hf_token' in low or '/tmp/' in low or low.endswith('/token'):
        errors.append(f'sensitive_name:{rel}')
    if rel.startswith('bootstrap/') and 'pilot' in p.name.lower():
        errors.append(f'pilot_name:{rel}')
    if p.is_symlink():
        errors.append(f'symlink:{rel}')

license_en=(ROOT/'MODEL_LICENSE.md').read_text(encoding='utf-8') if (ROOT/'MODEL_LICENSE.md').is_file() else ''
license_vi=(ROOT/'MODEL_LICENSE.vi.md').read_text(encoding='utf-8') if (ROOT/'MODEL_LICENSE.vi.md').is_file() else ''
for marker, text, label in [
    ('Mage-Flow / Mage-Flow-Turbo / Mage-Flow-Edit-Turbo components — MIT;', license_en, 'MODEL_LICENSE.md summary'),
    ('Mage-Flow / Mage-Flow-Turbo / Mage-Flow-Edit-Turbo — MIT;', license_vi, 'MODEL_LICENSE.vi.md summary'),
    ('Mage-Flow-Edit-Turbo, Microsoft', license_en, 'MODEL_LICENSE.md trademark boundary'),
    ('Mage-Flow-Edit-Turbo, Microsoft', license_vi, 'MODEL_LICENSE.vi.md trademark boundary'),
]:
    if marker not in text:
        errors.append(f'license_marker_missing:{label}')

acc_path=ROOT/'acceptance/PRODUCTION_ACCEPTANCE.json'
if acc_path.is_file():
    try:
        acc=json.loads(acc_path.read_text(encoding='utf-8'))
        if acc.get('status')!='PASS' or acc.get('case_count',0)<3:
            errors.append('acceptance:not_three_case_pass')
        for case in acc.get('cases',[]):
            if case.get('status')!='PASS' or case.get('png_count')!=4 or case.get('unique_png_sha256')!=4:
                errors.append(f"acceptance:case_invalid:{case.get('name')}")
    except Exception as exc:
        errors.append(f'acceptance:invalid_json:{type(exc).__name__}')

if errors:
    for error in errors:
        print('[FAIL]',error)
    raise SystemExit(1)
print('[PASS] required production files')
print('[PASS] lightweight source-only package')
print('[PASS] three-case acceptance metadata')
print('REPO_HEALTH_CHECK=PASS')
