#!/usr/bin/env python3

import subprocess
import sys
import os

def run_coverage():
    """Run comprehensive coverage analysis and testing."""
    print("Running comprehensive test coverage analysis...")
    
    print("Installing test dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "coverage", "pytest-mock"], check=True)
    
    print("\nRunning tests with coverage measurement...")
    try:
        result = subprocess.run([
            "coverage", "run", "--source=evaluation,inference", 
            "-m", "pytest", "tests/", "-v", "--tb=short"
        ], capture_output=True, text=True, check=False)
        
        print("Test execution output:")
        print(result.stdout)
        if result.stderr:
            print("Test errors:")
            print(result.stderr)
        
        print("\nGenerating coverage report...")
        coverage_result = subprocess.run([
            "coverage", "report", "--show-missing"
        ], capture_output=True, text=True, check=False)
        
        print("Coverage Report:")
        print(coverage_result.stdout)
        
        print("\nGenerating HTML coverage report...")
        subprocess.run(["coverage", "html"], check=False)
        
        lines = coverage_result.stdout.split('\n')
        for line in lines:
            if 'TOTAL' in line:
                parts = line.split()
                if len(parts) >= 4:
                    coverage_pct = parts[3].rstrip('%')
                    print(f"\nFinal Coverage: {coverage_pct}%")
                    if float(coverage_pct) >= 90:
                        print("✅ SUCCESS: Achieved 90%+ test coverage!")
                        return True
                    else:
                        print(f"❌ Coverage {coverage_pct}% is below 90% target")
                        return False
        
        print("Could not determine coverage percentage")
        return False
        
    except Exception as e:
        print(f"Error running coverage: {e}")
        return False

if __name__ == "__main__":
    success = run_coverage()
    sys.exit(0 if success else 1)
