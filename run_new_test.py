#!/usr/bin/env python3
"""
Test the improved grouping algorithm
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        from luggage_analyzer import LuggageAnalyzer
        from utils import get_image_files
        
        print("TESTING IMPROVED LUGGAGE ANALYSIS")
        print("=" * 50)
        
        # Get images
        image_files = [str(f) for f in get_image_files('input')]
        print(f"Found {len(image_files)} images: {[os.path.basename(f) for f in image_files]}")
        
        if not image_files:
            print("No images found in input folder!")
            return
        
        # Test with 92% threshold for better precision
        threshold = 92.0
        print(f"\nRunning analysis with {threshold}% threshold...")
        
        analyzer = LuggageAnalyzer(similarity_threshold=threshold)
        results = analyzer.analyze_images(image_files, threshold)
        
        print(f"\nRESULTS:")
        print(f"Total groups: {len(results['groups'])}")
        
        # Show each group
        for i, group in enumerate(results['groups'], 1):
            filenames = [os.path.basename(img.split('_')[-1]) for img in group['images']]
            print(f"Group {i}: {len(group['images'])} photos")
            print(f"  Photos: {filenames}")
            print(f"  Confidence: {group['confidence']:.1f}%")
            print(f"  Features: {group['common_features']}")
            print()
        
        # Verify ground truth
        expected_groups = {
            'A': ['a01.jpeg', 'a02.jpeg', 'a03.jpeg', 'a04.jpeg'],
            'B': ['b01.jpeg', 'b02.jpeg', 'b03.jpeg', 'b04.jpeg'], 
            'C': ['c01.jpeg', 'c02.jpeg', 'c03.jpeg', 'c04.jpeg'],
            'D': ['d01.jpeg', 'd02.jpeg', 'd03.jpeg', 'd04.jpeg'],
            'E': ['e01.jpeg', 'e02.jpeg', 'e03.jpeg', 'e04.jpeg']
        }
        
        print("GROUND TRUTH COMPARISON:")
        print("-" * 30)
        
        for group_name, expected_photos in expected_groups.items():
            # Find which group contains most of these photos
            best_match = None
            best_count = 0
            
            for i, group in enumerate(results['groups']):
                group_files = [os.path.basename(img.split('_')[-1]) for img in group['images']]
                matches = len(set(expected_photos) & set(group_files))
                if matches > best_count:
                    best_count = matches
                    best_match = i + 1
            
            accuracy = (best_count / len(expected_photos)) * 100 if expected_photos else 0
            print(f"Group {group_name}: {best_count}/{len(expected_photos)} correct -> Group {best_match} ({accuracy:.1f}%)")
        
        # Save results
        json_file, summary_file = analyzer.save_ultra_results('output')
        print(f"\nResults saved to: {summary_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()