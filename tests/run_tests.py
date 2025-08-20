#!/usr/bin/env python3
"""
Simple test runner to verify tests can execute.
Run with: python tests/run_tests.py
"""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run pytest with appropriate flags."""
    test_dir = Path(__file__).parent
    
    # Try to import pytest
    try:
        import pytest
    except ImportError:
        print("❌ pytest not installed. Install with: pip install pytest")
        return 1
    
    print("=" * 60)
    print("Running Spectrum Optimizer Test Suite")
    print("=" * 60)
    
    # Run tests with verbose output
    args = [
        "-v",  # Verbose
        "-s",  # Show print statements
        "--tb=short",  # Short traceback format
        "-x",  # Stop on first failure
        str(test_dir)
    ]
    
    print(f"\nRunning: pytest {' '.join(args)}\n")
    
    # Run pytest
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(run_tests())