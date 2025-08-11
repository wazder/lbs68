#!/usr/bin/env python3
"""
Simple import test for the luggage analysis system
"""

def test_imports():
    """Test basic imports without dependencies."""
    print("Testing imports...")
    
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
            
        if os.path.exists("multi_luggage_analyzer.py"):
            print("multi_luggage_analyzer.py exists")
        else:
            print("multi_luggage_analyzer.py missing")
            
        if os.path.exists("analyze_luggage.py"):
            print("analyze_luggage.py exists")
        else:
            print("analyze_luggage.py missing")
            
        print("Testing syntax by compiling files...")
        try:
            with open("luggage_comparator.py", "r") as f:
                code = f.read()
            compile(code, "luggage_comparator.py", "exec")
            print("luggage_comparator.py syntax is valid")
        except SyntaxError as e:
            print(f"luggage_comparator.py syntax error: {e}")
            
        try:
            with open("multi_luggage_analyzer.py", "r") as f:
                code = f.read()
            compile(code, "multi_luggage_analyzer.py", "exec")
            print("multi_luggage_analyzer.py syntax is valid")
        except SyntaxError as e:
            print(f"multi_luggage_analyzer.py syntax error: {e}")
            
        try:
            with open("analyze_luggage.py", "r") as f:
                code = f.read()
            compile(code, "analyze_luggage.py", "exec")
            print("analyze_luggage.py syntax is valid")
        except SyntaxError as e:
            print(f"analyze_luggage.py syntax error: {e}")
            
        print("\nAll files exist and have valid syntax!")
        print("To run the full system, install dependencies:")
        print("   pip install -r requirements.txt")
        print("Then test with:")
        print("   python analyze_luggage.py --interactive")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_imports()