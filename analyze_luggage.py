#!/usr/bin/env python3
"""
Luggage Analysis with Manual Grouping Integration
Integrates manual grouping with system results
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from luggage_analyzer import LuggageAnalyzer
from utils import get_image_files, setup_logging

def analyze_luggage_simple(folder_path: str, threshold: float = 87.0, output_dir: str = "output") -> Dict[str, Any]:
    """Simple luggage analysis without manual integration."""
    
    print("LUGGAGE ANALYSIS STARTING...")
    
    # Get photos from input folder
    image_files = get_image_files(folder_path)
    image_paths = [str(f) for f in image_files]
    
    print(f"{len(image_paths)} photos found")
    
    # System analysis
    analyzer = LuggageAnalyzer(similarity_threshold=threshold)
    results = analyzer.analyze_images(image_paths, threshold)
    
    # Save results
    json_file, summary_file = analyzer.save_ultra_results(output_dir)
    
    # Combined results format for compatibility
    combined_results = {
        'groups': results['groups'],
        'total_photos': results['total_photos'],
        'analysis_date': results['analysis_date'],
        'accuracy_score': 0.0  # No manual comparison available
    }
    
    return combined_results

# Unnecessary functions removed - only simple analysis remains

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple Luggage Analysis")
    parser.add_argument("--folder", required=True, help="Input folder path")
    parser.add_argument("--threshold", type=float, default=87.0, help="Similarity threshold")
    parser.add_argument("--output", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    # Logging setup
    setup_logging()
    
    print(f"Luggage Analysis System")
    print(f"Input: {args.folder}")
    print(f"Threshold: {args.threshold}%")
    print(f"Output: {args.output}")
    print("=" * 50)
    
    try:
        results = analyze_luggage_simple(args.folder, args.threshold, args.output)
        
        print(f"\nANALYSIS COMPLETED!")
        print(f"Total Photos: {results['total_photos']}")
        print(f"Groups Found: {len(results['groups'])}")
        
        for i, group in enumerate(results['groups'], 1):
            print(f"   Group {i}: {len(group['images'])} images (Confidence: {group['confidence']:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()