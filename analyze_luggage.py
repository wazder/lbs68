#!/usr/bin/env python3
"""
Luggage Photo Analysis and Grouping - Easy Usage Script

This script allows you to analyze multiple luggage photos,
group those belonging to the same luggage, and get detailed reports.

Usage:
python analyze_luggage.py --folder /path/to/photos --threshold 75
python analyze_luggage.py --files photo1.jpg photo2.jpg photo3.jpg
"""

import os
import argparse
import sys
from pathlib import Path
from multi_luggage_analyzer import MultiLuggageAnalyzer


def get_image_files(folder_path: str) -> list:
    """Find all image files in the folder."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"ERROR: Folder not found: {folder_path}")
        return []
    
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    return sorted(image_files)


def print_results_summary(analyzer: MultiLuggageAnalyzer):
    """Print results to console."""
    if not analyzer.groups:
        print("No groups found. All photos appear to belong to different luggage.")
        return
    
    print(f"\nRESULTS")
    print("=" * 50)
    print(f"Total {len(analyzer.processed_images)} photos analyzed")
    print(f"{len(analyzer.groups)} different luggage groups found")
    
    for i, group in enumerate(analyzer.groups, 1):
        print(f"\nLUGGAGE {i} - Confidence: {group['confidence']}%")
        print("-" * 30)
        print(f"Photo count: {len(group['images'])}")
        
        # Show common features
        features = group['common_features']
        print("Features:")
        print(f"   - Color: {features.get('dominant_color', 'Unknown')}")
        print(f"   - Size: {features.get('size_category', 'Unknown')}")
        print(f"   - Texture: {features.get('texture_type', 'Unknown')}")
        print(f"   - Material: {features.get('material_type', 'Unknown')}")
        
        print("Photos:")
        for img_id in group['images']:
            img_path = analyzer.processed_images[img_id]['path']
            img_name = os.path.basename(img_path)
            print(f"   - {img_name}")
        
        # Show highest similarity
        if group['similarities']:
            max_similarity = max(group['similarities'].values())
            print(f"Highest similarity: {max_similarity}%")
    
    # Show single photos
    grouped_images = set()
    for group in analyzer.groups:
        grouped_images.update(group['images'])
    
    single_images = [img_id for img_id in analyzer.processed_images.keys() 
                     if img_id not in grouped_images]
    
    if single_images:
        print(f"\nINDIVIDUAL PHOTOS ({len(single_images)} items)")
        print("-" * 20)
        for img_id in single_images:
            img_path = analyzer.processed_images[img_id]['path']
            img_name = os.path.basename(img_path)
            print(f"   - {img_name}")


def interactive_mode():
    """Interactive mode - get input from user."""
    print("LUGGAGE ANALYSIS SYSTEM - Interactive Mode")
    print("=" * 40)
    
    # Ask for photo source
    print("\nHow would you like to select photos?")
    print("1. Select folder (all images in folder)")
    print("2. Select files individually")
    
    choice = input("Your choice (1 or 2): ").strip()
    
    image_paths = []
    
    if choice == "1":
        folder_path = input("Enter photo folder path: ").strip()
        image_paths = get_image_files(folder_path)
        
        if not image_paths:
            print("ERROR: No image files found in folder!")
            return
            
        print(f"Found {len(image_paths)} image files")
        
    elif choice == "2":
        print("Enter photo paths (empty line to finish):")
        while True:
            path = input("File path: ").strip()
            if not path:
                break
            if os.path.exists(path):
                image_paths.append(path)
                print(f"Added: {os.path.basename(path)}")
            else:
                print(f"ERROR: File not found: {path}")
    
    else:
        print("ERROR: Invalid choice!")
        return
    
    if len(image_paths) < 2:
        print("ERROR: At least 2 photos required!")
        return
    
    # Ask for similarity threshold
    print(f"\nWhat is the similarity threshold? (default: 75%)")
    threshold_input = input("Threshold value (0-100): ").strip()
    
    try:
        threshold = float(threshold_input) if threshold_input else 75.0
        if not 0 <= threshold <= 100:
            threshold = 75.0
            print("WARNING: Invalid value, using default: 75%")
    except ValueError:
        threshold = 75.0
        print("WARNING: Invalid value, using default: 75%")
    
    # Start analysis
    print(f"\nStarting analysis... ({len(image_paths)} photos, threshold: {threshold}%)")
    
    analyzer = MultiLuggageAnalyzer(similarity_threshold=threshold)
    
    try:
        # Process photos
        analyzer.process_images(image_paths)
        
        # Group luggage
        analyzer.group_similar_luggage()
        
        # Show results
        print_results_summary(analyzer)
        
        # Ask about saving report
        save_report = input("\nWould you like to save detailed report to file? (y/n): ").strip().lower()
        
        if save_report in ['y', 'yes']:
            output_dir = input("Output folder (default: luggage_results): ").strip()
            if not output_dir:
                output_dir = "luggage_results"
            
            results = analyzer.save_results(output_dir)
            print(f"\nReports saved:")
            for key, path in results.items():
                print(f"   - {os.path.basename(path)}")
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"ERROR: Error during analysis: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze luggage photos and group those belonging to the same luggage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --interactive                              # Interactive mode
  %(prog)s --folder photos/                           # Analyze all images in folder
  %(prog)s --files photo1.jpg photo2.jpg photo3.jpg  # Analyze specific files
  %(prog)s --folder photos/ --threshold 80 --output results/  # Detailed settings
        """
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive mode"
    )
    
    parser.add_argument(
        "--folder", "-f",
        type=str,
        help="Folder containing photos to analyze"
    )
    
    parser.add_argument(
        "--files",
        nargs="+",
        help="Photo files to analyze"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=75.0,
        help="Similarity threshold percentage (0-100, default: 75)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="luggage_results",
        help="Output folder (default: luggage_results)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save report, only print to console"
    )
    
    args = parser.parse_args()
    
    # Argument validation
    if args.interactive:
        interactive_mode()
        return
    
    if not args.folder and not args.files:
        print("ERROR: You must specify at least one photo source!")
        print("Help: python analyze_luggage.py --help")
        print("Interactive mode: python analyze_luggage.py --interactive")
        return
    
    # Prepare photo list
    image_paths = []
    
    if args.folder:
        folder_images = get_image_files(args.folder)
        image_paths.extend(folder_images)
        print(f"Found {len(folder_images)} photos from {args.folder}")
    
    if args.files:
        for file_path in args.files:
            if os.path.exists(file_path):
                image_paths.append(file_path)
            else:
                print(f"WARNING: File not found: {file_path}")
    
    if len(image_paths) < 2:
        print("ERROR: At least 2 photos required!")
        return
    
    print(f"{len(image_paths)} photos will be analyzed...")
    
    # Start analysis
    analyzer = MultiLuggageAnalyzer(similarity_threshold=args.threshold)
    
    try:
        # Process photos
        analyzer.process_images(image_paths)
        
        # Group luggage
        analyzer.group_similar_luggage()
        
        # Show results
        print_results_summary(analyzer)
        
        # Save report (unless disabled)
        if not args.no_save:
            results = analyzer.save_results(args.output)
            print(f"\nDetailed reports saved to: {args.output}/")
            
    except Exception as e:
        print(f"ERROR: Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()