#!/usr/bin/env python3
"""
Simple script to run luggage analysis
Usage: python run_analysis.py
"""

from luggage_analyzer import LuggageAnalyzer
from utils import get_image_files

def load_ground_truth():
    """Load ground truth groupings."""
    ground_truth = {
        'a01.jpeg': 'A', 'a02.jpeg': 'A', 'a03.jpeg': 'A', 'a04.jpeg': 'A',
        'b01.jpeg': 'B', 'b02.jpeg': 'B', 'b03.jpeg': 'B', 'b04.jpeg': 'B',
        'c01.jpeg': 'C', 'c02.jpeg': 'C', 'c03.jpeg': 'C', 'c04.jpeg': 'C',
        'd01.jpeg': 'D', 'd02.jpeg': 'D', 'd03.jpeg': 'D', 'd04.jpeg': 'D',
        'e01.jpeg': 'E', 'e02.jpeg': 'E', 'e03.jpeg': 'E', 'e04.jpeg': 'E'
    }
    return ground_truth

def calculate_accuracy(results, ground_truth):
    """Calculate accuracy against ground truth."""
    correct_pairs = 0
    total_pairs = 0
    
    for group in results['groups']:
        group_images = [img.split('/')[-1] for img in group['images']]
        
        # Check all pairs in this group
        for i in range(len(group_images)):
            for j in range(i+1, len(group_images)):
                total_pairs += 1
                img1, img2 = group_images[i], group_images[j]
                
                if ground_truth.get(img1) == ground_truth.get(img2):
                    correct_pairs += 1
    
    return (correct_pairs / total_pairs * 100) if total_pairs > 0 else 0

def main():
    print("LUGGAGE ANALYSIS SYSTEM")
    print("=" * 40)
    
    # Initialize analyzer with optimal settings
    analyzer = LuggageAnalyzer(similarity_threshold=86.0)
    
    # Get images from input folder
    image_files = [str(f) for f in get_image_files('input')]
    
    if not image_files:
        print("No images found in 'input' folder!")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Run analysis
    print("Running analysis...")
    results = analyzer.analyze_images(image_files)
    
    # Save results
    json_file, summary_file = analyzer.save_ultra_results('output')
    
    # Calculate accuracy
    ground_truth = load_ground_truth()
    accuracy = calculate_accuracy(results, ground_truth)
    
    # Show results
    print(f"\nAnalysis completed!")
    print(f"Total images: {results['total_photos']}")
    print(f"Groups found: {len(results['groups'])} (Ground Truth: 5)")
    print(f"Accuracy: {accuracy:.1f}%")
    
    for i, group in enumerate(results['groups'], 1):
        print(f"   Group {i}: {len(group['images'])} images (Confidence: {group['confidence']:.1f}%)")
    
    print(f"\nResults saved:")
    print(f"   - {json_file}")
    print(f"   - {summary_file}")

if __name__ == "__main__":
    main()