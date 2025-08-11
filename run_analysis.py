#!/usr/bin/env python3
"""
Simple script to run luggage analysis
Usage: python run_analysis.py
"""

from luggage_analyzer import LuggageAnalyzer
from utils import get_image_files

def main():
    print("🎯 LUGGAGE ANALYSIS SYSTEM")
    print("=" * 40)
    
    # Initialize analyzer with optimal settings
    analyzer = LuggageAnalyzer(similarity_threshold=87.0)
    
    # Get images from input folder
    image_files = [str(f) for f in get_image_files('input')]
    
    if not image_files:
        print("❌ No images found in 'input' folder!")
        return
    
    print(f"📁 Found {len(image_files)} images")
    
    # Run analysis
    print("🔄 Running analysis...")
    results = analyzer.analyze_images(image_files)
    
    # Save results
    json_file, summary_file = analyzer.save_ultra_results('output')
    
    # Show results
    print(f"\n✅ Analysis completed!")
    print(f"📊 Total images: {results['total_photos']}")
    print(f"🎯 Groups found: {len(results['groups'])}")
    
    for i, group in enumerate(results['groups'], 1):
        print(f"   Group {i}: {len(group['images'])} images (Confidence: {group['confidence']:.1f}%)")
    
    print(f"\n📁 Results saved:")
    print(f"   - {json_file}")
    print(f"   - {summary_file}")

if __name__ == "__main__":
    main()