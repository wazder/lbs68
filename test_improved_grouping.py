#!/usr/bin/env python3
"""
Test script for improved grouping algorithm
"""

from luggage_analyzer import LuggageAnalyzer
from utils import get_image_files

def test_with_different_thresholds():
    """Test with different threshold values to find optimal setting."""
    
    # Get images from input folder
    image_files = [str(f) for f in get_image_files('input')]
    
    if not image_files:
        print("No images found in 'input' folder!")
        return
    
    print(f"Testing with {len(image_files)} images")
    print("=" * 60)
    
    # Test different thresholds
    thresholds = [90.0, 92.0, 94.0, 96.0, 98.0]
    
    for threshold in thresholds:
        print(f"\n🧪 TESTING THRESHOLD: {threshold}%")
        print("-" * 40)
        
        # Initialize analyzer with test threshold
        analyzer = LuggageAnalyzer(similarity_threshold=threshold)
        
        # Run analysis
        results = analyzer.analyze_images(image_files, threshold)
        
        print(f"Groups found: {len(results['groups'])}")
        
        for i, group in enumerate(results['groups'], 1):
            photos = [img.split('_')[-1] for img in group['images']]  # Extract filenames
            print(f"  Group {i}: {len(group['images'])} photos - {photos} (Confidence: {group['confidence']:.1f}%)")
        
        # Count single-image groups
        single_groups = sum(1 for group in results['groups'] if len(group['images']) == 1)
        multi_groups = sum(1 for group in results['groups'] if len(group['images']) > 1)
        
        print(f"  Multi-image groups: {multi_groups}")
        print(f"  Single-image groups: {single_groups}")
        
        # Save results for this threshold
        json_file, summary_file = analyzer.save_ultra_results(f'output/threshold_{threshold}')
        print(f"  Results saved to: {json_file}")

if __name__ == "__main__":
    test_with_different_thresholds()