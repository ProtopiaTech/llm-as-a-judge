#!/usr/bin/env python3
"""
Simple runner script for LLM-as-a-Judge evaluation
Uses pytest + DeepEval for native JUnit XML integration
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Run the LLM evaluation with proper pytest configuration"""

    # Ensure results directory exists
    results_dir = Path("test-results")
    results_dir.mkdir(exist_ok=True)

    # Build pytest command
    pytest_cmd = [
        "python", "-m", "pytest",
        "test_llm_evaluation.py",
        "--junitxml=test-results/evaluation.xml",
        "--html=test-results/report.html",
        "--self-contained-html",
        "-v",
        "--tb=short",
        "--maxfail=5",  # Stop after 5 failures to save costs
        "-x"  # Stop on first failure for quick debugging
    ]

    print("🚀 Starting LLM-as-a-Judge Evaluation")
    print("=" * 50)
    print("Framework: DeepEval + pytest")
    print("Models: gpt-5-mini-2025-08-07, claude-3-5-haiku-20241022, gpt-4o-mini-2024-07-18")
    print("Judge: gpt-5")
    print("Output: JUnit XML + HTML reports")
    print("=" * 50)

    try:
        # Run pytest
        result = subprocess.run(pytest_cmd, check=False)

        print("\n" + "=" * 50)
        if result.returncode == 0:
            print("✅ Evaluation completed successfully!")
        else:
            print("⚠️  Evaluation completed with some issues")

        print("\n📊 Results available in:")
        print(f"  - test-results/evaluation.xml (JUnit XML for CI/CD)")
        print(f"  - test-results/report.html (HTML report)")

        return result.returncode

    except KeyboardInterrupt:
        print("\n\n🛑 Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error running evaluation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())