from __future__ import annotations
import subprocess, sys, pathlib

def run_once(tmp: pathlib.Path, out_name: str) -> pathlib.Path:
    out = tmp / out_name
    cmd = [sys.executable, "-m", "jpr.run_repro", "--steps", "20", "--batch-size", "32", "--seed", "7", "--out", str(out)]
    subprocess.run(cmd, check=True)
    return out

def test_repeatability(tmp_path: pathlib.Path) -> None:
    a = run_once(tmp_path, "a.json")
    b = run_once(tmp_path, "b.json")
    cmp_cmd = [sys.executable, "-m", "jpr.compare", str(a), str(b)]
    subprocess.run(cmp_cmd, check=True)

def test_resume_equivalence(tmp_path: pathlib.Path) -> None:
    full = tmp_path / "full.json"
    subprocess.run([sys.executable, "-m", "jpr.run_repro", "--steps", "30", "--batch-size", "32", "--seed", "9", "--out", str(full)], check=True)

    ckpt = tmp_path / "ckpt"
    phase1 = tmp_path / "phase1.json"
    subprocess.run([sys.executable, "-m", "jpr.run_repro", "--steps", "20", "--batch-size", "32", "--seed", "9", "--out", str(phase1), "--checkpoint-dir", str(ckpt), "--save-at", "20"], check=True)

    resumed = tmp_path / "resumed.json"
    subprocess.run([sys.executable, "-m", "jpr.run_repro", "--steps", "30", "--batch-size", "32", "--seed", "9", "--out", str(resumed), "--checkpoint-dir", str(ckpt), "--restore-from", "20"], check=True)

    cmp_cmd = [sys.executable, "-m", "jpr.compare", str(full), str(resumed)]
    subprocess.run(cmp_cmd, check=True)
