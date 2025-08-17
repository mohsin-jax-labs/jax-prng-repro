#!/usr/bin/env python
"""Verification script to ensure the JAX reproducibility setup works correctly.

Run this script after installation to verify:
1. All dependencies are correctly installed
2. Basic reproducibility works
3. Checkpoint/resume functionality works
4. Parameter hashing works
"""

import sys
import subprocess
import tempfile
import json
from pathlib import Path
import shutil

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"Testing: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"  ✅ PASS: {description}")
            return True
        else:
            print(f"  ❌ FAIL: {description}")
            print(f"     Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ❌ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"  ❌ ERROR: {description} - {e}")
        return False

def test_basic_imports():
    """Test that all required packages can be imported."""
    print("\n=== Testing Package Imports ===")
    
    imports = [
        ("import jax", "JAX"),
        ("import jax.numpy as jnp", "JAX NumPy"),
        ("import flax", "Flax"),
        ("import optax", "Optax"),
        ("import orbax.checkpoint", "Orbax Checkpoint"),
        ("import numpy as np", "NumPy"),
    ]
    
    all_passed = True
    for import_cmd, name in imports:
        try:
            exec(import_cmd)
            print(f"  ✅ {name} imported successfully")
        except ImportError as e:
            print(f"  ❌ {name} import failed: {e}")
            all_passed = False
    
    return all_passed

def test_reproducibility():
    """Test basic reproducibility by running the same config twice."""
    print("\n=== Testing Basic Reproducibility ===")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Run same configuration twice
        cmd1 = f"python -m jpr.run_repro --steps 10 --batch-size 32 --seed 42 --out {tmp_path}/run1.json"
        cmd2 = f"python -m jpr.run_repro --steps 10 --batch-size 32 --seed 42 --out {tmp_path}/run2.json"
        
        success1 = run_command(cmd1, "First training run")
        success2 = run_command(cmd2, "Second training run")
        
        if not (success1 and success2):
            return False
        
        # Compare results
        compare_cmd = f"python -m jpr.compare {tmp_path}/run1.json {tmp_path}/run2.json"
        success3 = run_command(compare_cmd, "Comparing identical runs")
        
        return success3

def test_checkpoint_resume():
    """Test checkpoint and resume functionality."""
    print("\n=== Testing Checkpoint/Resume ===")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ckpt_dir = tmp_path / "checkpoints"
        
        # Full run
        full_cmd = f"python -m jpr.run_repro --steps 20 --batch-size 32 --seed 123 --out {tmp_path}/full.json"
        success1 = run_command(full_cmd, "Full 20-step run")
        
        if not success1:
            return False
        
        # Interrupted run + resume
        phase1_cmd = f"python -m jpr.run_repro --steps 15 --batch-size 32 --seed 123 --out {tmp_path}/phase1.json --checkpoint-dir {ckpt_dir} --save-at 15"
        success2 = run_command(phase1_cmd, "Phase 1: Run 15 steps and checkpoint")
        
        if not success2:
            return False
        
        resume_cmd = f"python -m jpr.run_repro --steps 20 --batch-size 32 --seed 123 --out {tmp_path}/resumed.json --checkpoint-dir {ckpt_dir} --restore-from 15"
        success3 = run_command(resume_cmd, "Phase 2: Resume from checkpoint")
        
        if not success3:
            return False
        
        # Compare full vs resumed
        compare_cmd = f"python -m jpr.compare {tmp_path}/full.json {tmp_path}/resumed.json"
        success4 = run_command(compare_cmd, "Comparing full vs resumed runs")
        
        return success4

def test_parameter_hashing():
    """Test that parameter hashing works for verification."""
    print("\n=== Testing Parameter Hashing ===")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Run training
        cmd = f"python -m jpr.run_repro --steps 5 --batch-size 32 --seed 999 --out {tmp_path}/hash_test.json"
        success = run_command(cmd, "Training run for hash test")
        
        if not success:
            return False
        
        # Check that output contains a hash
        try:
            with open(tmp_path / "hash_test.json") as f:
                result = json.load(f)
            
            if "hash" not in result:
                print("  ❌ Output JSON missing 'hash' field")
                return False
            
            hash_value = result["hash"]
            if not isinstance(hash_value, str) or len(hash_value) != 64:
                print(f"  ❌ Invalid hash format: {hash_value}")
                return False
            
            print(f"  ✅ Valid parameter hash: {hash_value[:16]}...")
            return True
            
        except Exception as e:
            print(f"  ❌ Error reading output JSON: {e}")
            return False

def test_examples():
    """Test that tutorial examples run without errors."""
    print("\n=== Testing Tutorial Examples ===")
    
    examples = [
        "examples/reproducibility_basics.py",
        "examples/jax_prng_tutorial.py",
        "examples/prng_pattern.py",
        "examples/model_explained.py",
        "examples/data_explained.py",
        "examples/hashing_explained.py",
        "examples/complete_flow.py",
        "examples/pitfalls_and_best_practices.py",
    ]
    
    all_passed = True
    for example in examples:
        if Path(example).exists():
            success = run_command(f"python {example}", f"Running {example}")
            all_passed = all_passed and success
        else:
            print(f"  ⚠️  SKIP: {example} not found")
    
    return all_passed

def main():
    """Run all verification tests."""
    print("🧪 JAX PRNG Reproducibility Setup Verification")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_basic_imports),
        ("Basic Reproducibility", test_reproducibility),
        ("Checkpoint/Resume", test_checkpoint_resume),
        ("Parameter Hashing", test_parameter_hashing),
        ("Tutorial Examples", test_examples),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your JAX reproducibility setup is working correctly.")
        print("\nNext steps:")
        print("- Read TUTORIAL.md for a complete learning guide")
        print("- Try: python -m jpr.run_repro --help")
        print("- Explore examples/ directory for detailed tutorials")
        sys.exit(0)
    else:
        print("💥 SOME TESTS FAILED!")
        print("Please check the error messages above.")
        print("Common issues:")
        print("- Missing dependencies: pip install -e '.[dev]'")
        print("- Python version: Requires Python 3.12+")
        print("- JAX installation: May need platform-specific install")
        sys.exit(1)

if __name__ == "__main__":
    main()