import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_repo_health.py"


def run_health(root: Path):
    env = os.environ.copy()
    env["REPO_HEALTH_ROOT"] = str(root)
    return subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True, capture_output=True)


def copy_candidate(tmp_path: Path) -> Path:
    dst = tmp_path / "repo"
    shutil.copytree(
        REPO,
        dst,
        ignore=shutil.ignore_patterns(".git", ".pytest_cache", "__pycache__", "*.pyc"),
    )
    return dst


def test_repo_health_honors_root_override(tmp_path):
    result = run_health(tmp_path)
    assert result.returncode != 0
    assert "missing:README.md" in result.stdout


def test_repo_health_requires_bilingual_peer(tmp_path):
    root = copy_candidate(tmp_path)
    (root / "README.vi.md").unlink()
    result = run_health(root)
    assert result.returncode != 0
    assert "bilingual_peer_missing:README.md->README.vi.md" in result.stdout


def test_repo_health_requires_exact_mit_author(tmp_path):
    root = copy_candidate(tmp_path)
    p = root / "LICENSE"
    p.write_text(p.read_text().replace("Đăng Khoa <i.am@dangkhoa.dev>", "Wrong Author"))
    result = run_health(root)
    assert result.returncode != 0
    assert "license_author_line_missing" in result.stdout


def test_repo_health_rejects_weight_blob(tmp_path):
    root = copy_candidate(tmp_path)
    (root / "accidental.safetensors").write_bytes(b"not-a-real-weight")
    result = run_health(root)
    assert result.returncode != 0
    assert "forbidden_artifact:accidental.safetensors" in result.stdout


def test_candidate_repository_health_passes():
    result = run_health(REPO)
    assert result.returncode == 0, result.stdout + result.stderr
