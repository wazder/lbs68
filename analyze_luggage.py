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
from utils import setup_logging, validate_directory, get_image_files, validate_image_file
from config import get_config_manager, get_config

# Initialize config first
config_manager = get_config_manager()
config = get_config()

# Setup logging based on config
logger = setup_logging(
    level=config.logging.level,
    log_file=config.logging.log_file if config.logging.enable_file_logging else None,
    format_string=config.logging.log_format
)


def get_image_files_with_validation(folder_path: str) -> list:
    """Find and validate all image files in the folder."""
    logger.info(f"Searching for images in: {folder_path}")
    
    if not validate_directory(folder_path):
        logger.error(f"Directory not found or not accessible: {folder_path}")
        return []
    
    # Use utility function that validates images
    image_files = get_image_files(folder_path)
    
    if not image_files:
        logger.warning(f"No valid image files found in {folder_path}")
        return []
    
    logger.info(f"Found {len(image_files)} valid image files")
    for img_file in image_files:
        logger.debug(f"Valid image: {Path(img_file).name}")
    
    return [str(f) for f in image_files]


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
    """Interactive mode - get input from user with input validation."""
    logger.info("Starting interactive mode")
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
    
    
    if len(image_paths) < 1:
        logger.error("No valid images provided")
        print("ERROR: No valid images provided!")
        return
    elif len(image_paths) < 2:
        logger.warning("Only one image provided")
        print("WARNING: Only 1 photo provided. You need at least 2 photos to compare luggage.")
        print("The system will still process this image for feature analysis.")
        
        continue_single = input("Continue with single image analysis? (y/n): ").strip().lower()
        if continue_single not in ['y', 'yes']:
            return
    
    # Ask for similarity threshold with validation
    print(f"\nWhat is the similarity threshold? (default: 75%)")
    print("Higher values = stricter matching, Lower values = more permissive matching")
    
    while True:
        threshold_input = input(f"Threshold value (0-100, default: {config.processing.similarity_threshold}): ").strip()
        
        if not threshold_input:
            threshold = config.processing.similarity_threshold
            logger.info(f"Using default threshold: {threshold}%")
            break
            
        try:
            threshold = float(threshold_input)
            if 0 <= threshold <= 100:
                logger.info(f"Using threshold: {threshold}%")
                break
            else:
                print("Please enter a value between 0 and 100")
        except ValueError:
            print("Please enter a valid number")
    
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
            while True:
                output_dir = input("Output folder (default: luggage_results): ").strip()
                if not output_dir:
                    output_dir = "luggage_results"
                
                # Expand user path
                output_dir = os.path.expanduser(output_dir)
                
                # Validate/create output directory
                if validate_directory(output_dir, create_if_missing=True):
                    logger.info(f"Saving results to: {output_dir}")
                    break
                else:
                    print(f"Cannot create or access directory: {output_dir}")
                    print("Please try another path")
            
            try:
                results = analyzer.save_results(output_dir)
                print(f"\nReports saved to {output_dir}/:")
                for key, path in results.items():
                    print(f"   - {Path(path).name}")
            except Exception as e:
                logger.error(f"Failed to save results: {e}")
                print(f"ERROR: Failed to save results: {e}")
        
        print("\nAnalysis complete!")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        print("\nAnalysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"ERROR: Analysis failed: {e}")
        print("Check the logs for more details")


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
        default=config.processing.similarity_threshold,
        help=f"Similarity threshold percentage (0-100, default: {config.processing.similarity_threshold})"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=config.output.default_output_dir,
        help=f"Output folder (default: {config.output.default_output_dir})"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save report, only print to console"
    )
    
# Ultra-precise mode is now the default - no flag needed
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.threshold < 0 or args.threshold > 100:
        logger.error(f"Invalid threshold: {args.threshold}")
        print(f"ERROR: Threshold must be between 0 and 100, got {args.threshold}")
        return
    
    if args.interactive:
        logger.info("Starting in interactive mode")
        interactive_mode()
        return
    
    if not args.folder and not args.files:
        logger.error("No photo source specified")
        print("ERROR: You must specify at least one photo source!")
        print("Help: python analyze_luggage.py --help")
        print("Interactive mode: python analyze_luggage.py --interactive")
        return
    
    # Prepare and validate photo list
    image_paths = []
    
    if args.folder:
        logger.info(f"Processing folder: {args.folder}")
        folder_path = os.path.expanduser(args.folder)
        
        if not validate_directory(folder_path):
            logger.error(f"Invalid folder: {folder_path}")
            print(f"ERROR: Folder not found or not accessible: {folder_path}")
            return
            
        folder_images = get_image_files_with_validation(folder_path)
        image_paths.extend(folder_images)
        
        if folder_images:
            logger.info(f"Found {len(folder_images)} valid photos from {args.folder}")
            print(f"Found {len(folder_images)} valid photos from {args.folder}")
        else:
            logger.warning(f"No valid images found in folder: {folder_path}")
    
    if args.files:
        logger.info(f"Processing {len(args.files)} individual files")
        valid_files = 0
        
        for file_path in args.files:
            expanded_path = os.path.expanduser(file_path)
            
            if validate_image_file(expanded_path):
                image_paths.append(expanded_path)
                valid_files += 1
                logger.debug(f"Valid file: {Path(expanded_path).name}")
            else:
                logger.warning(f"Invalid or missing file: {file_path}")
                print(f"WARNING: Invalid or missing file: {file_path}")
        
        if valid_files > 0:
            logger.info(f"Found {valid_files} valid files out of {len(args.files)} provided")
    
    if len(image_paths) < 1:
        logger.error("No valid images found")
        print("ERROR: No valid image files found!")
        return
    elif len(image_paths) < 2:
        logger.warning("Only one valid image found")
        print("WARNING: Only 1 valid photo found. You need at least 2 photos for comparison.")
        print("The system will still process this image for feature analysis.")
        
    logger.info(f"Total valid images to process: {len(image_paths)}")
    
    print(f"{len(image_paths)} photos will be analyzed with {args.threshold}% similarity threshold...")
    
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
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        print("\nAnalysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"ERROR: Analysis failed: {e}")
        print("Check the logs for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()