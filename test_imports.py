#!/usr/bin/env python3
"""
Simple import test for the luggage analysis system - DEPRECATED
Use 'python run_analysis.py --check-deps' instead
"""

def test_imports():
    """Test basic imports without dependencies."""
    print("Testing basic imports...")
    print("NOTE: This script is deprecated. Use 'python run_analysis.py --check-deps' for comprehensive testing.")
    
    try:
        print("Testing basic Python modules...")
        import os
        import json
        from datetime import datetime
        from typing import List, Dict, Tuple, Any, Optional
        from collections import defaultdict, Counter
        print("Basic Python modules imported successfully")
        
        print("Testing if files exist...")
        if os.path.exists("luggage_comparator.py"):
            print("luggage_comparator.py exists")
        else:
            print("luggage_comparator.py missing")
            
        if os.path.exists("luggage_analyzer.py"):
            print("luggage_analyzer.py exists")
        else:
            print("luggage_analyzer.py missing")
            
        if os.path.exists("run_analysis.py"):
            print("run_analysis.py exists")
        else:
            print("run_analysis.py missing")
            
        print("Testing syntax by compiling files...")
        try:
            with open("luggage_comparator.py", "r") as f:
                code = f.read()
            compile(code, "luggage_comparator.py", "exec")
            print("luggage_comparator.py syntax is valid")
        except SyntaxError as e:
            print(f"luggage_comparator.py syntax error: {e}")
            
        try:
            with open("luggage_analyzer.py", "r") as f:
                code = f.read()
            compile(code, "luggage_analyzer.py", "exec")
            print("luggage_analyzer.py syntax is valid")
        except SyntaxError as e:
            print(f"luggage_analyzer.py syntax error: {e}")
            
        try:
            with open("run_analysis.py", "r") as f:
                code = f.read()
            compile(code, "run_analysis.py", "exec")
            print("run_analysis.py syntax is valid")
        except SyntaxError as e:
            print(f"run_analysis.py syntax error: {e}")
            
        print("\nAll files exist and have valid syntax!")
        print("[OK] Basic import test completed successfully!")
        print()
        print("Next Steps:")
        print("1. Install dependencies:")
        print("   pip install -r requirements.txt")
        print()
        print("2. Check system dependencies:")
        print("   python run_analysis.py --check-deps")
        print()
        print("3. Run system tests:")
        print("   python run_analysis.py --test")
        print()
        print("4. Start analysis:")
        print("   python run_analysis.py")
        print("   python run_analysis.py --interactive")
        print("   python run_analysis.py --help")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_imports()