#!/usr/bin/env python3
"""
LUGGAGE ANALYSIS SYSTEM - Main Entry Point
Advanced luggage grouping system with AI-powered image analysis

Usage:
    python run_analysis.py                          # Basic analysis
    python run_analysis.py --folder input           # Specify input folder
    python run_analysis.py --threshold 90           # Set similarity threshold
    python run_analysis.py --output results         # Set output directory
    python run_analysis.py --config config.yaml     # Use configuration file
    python run_analysis.py --interactive            # Interactive mode
    python run_analysis.py --test                   # Run system tests
    python run_analysis.py --help                   # Show help
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List

from luggage_analyzer import LuggageAnalyzer
from luggage_comparator import LuggageComparator  
from utils import get_image_files, setup_logging, check_dependencies, validate_directory
from config import get_config_manager, get_config
from model_cache import get_model_cache

# Version info
__version__ = "2.0.0"
__author__ = "Luggage Analysis System"

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Advanced Luggage Analysis System - AI-powered luggage grouping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py                                    # Basic analysis
  python run_analysis.py --folder photos --threshold 85    # Custom settings
  python run_analysis.py --config my_config.yaml           # Use config file
  python run_analysis.py --interactive                     # Interactive mode
  python run_analysis.py --test                            # Run tests
  python run_analysis.py --cache-info                      # Show cache info
        """
    )
    
    # Input/Output options
    parser.add_argument("--folder", "-f", default="input", 
                       help="Input folder containing luggage images (default: input)")
    parser.add_argument("--output", "-o", default="output",
                       help="Output directory for results (default: output)")
    
    # Analysis options
    parser.add_argument("--threshold", "-t", type=float, default=None,
                       help="Similarity threshold (0-100, default: from config)")
    parser.add_argument("--use-filename-hints", action="store_true",
                       help="Use filename patterns for smarter grouping")
    
    # Configuration
    parser.add_argument("--config", "-c", type=str,
                       help="Configuration file path (YAML or JSON)")
    
    # Model options
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for computation")
    parser.add_argument("--sam-model", choices=["vit_b", "vit_l", "vit_h"],
                       help="SAM model type to use")
    
    # Operation modes
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--test", action="store_true",
                       help="Run system tests")
    parser.add_argument("--check-deps", action="store_true",
                       help="Check system dependencies")
    parser.add_argument("--cache-info", action="store_true",
                       help="Show model cache information")
    parser.add_argument("--cache-cleanup", action="store_true",
                       help="Clean up old cached models")
    
    # Logging options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress non-essential output")
    parser.add_argument("--log-file", type=str,
                       help="Log to file instead of console")
    
    # Utility options
    parser.add_argument("--version", action="version", 
                       version=f"Luggage Analysis System v{__version__}")
    
    return parser

def check_system_dependencies() -> bool:
    """Check if all required dependencies are available."""
    print("Checking system dependencies...")
    print("-" * 40)
    
    deps = check_dependencies()
    all_good = True
    
    critical_deps = ['torch', 'numpy', 'PIL', 'sklearn']
    optional_deps = ['transformers', 'segment_anything', 'cv2', 'faiss']
    
    print("Critical dependencies:")
    for dep in critical_deps:
        status = "[OK]" if deps[dep] else "[MISSING]"
        print(f"  {status} {dep}: {'Available' if deps[dep] else 'Missing'}")
        if not deps[dep]:
            all_good = False
    
    print("\nOptional dependencies:")
    for dep in optional_deps:
        status = "[OK]" if deps[dep] else "[OPTIONAL]"
        print(f"  {status} {dep}: {'Available' if deps[dep] else 'Missing (reduced functionality)'}")
    
    if not all_good:
        print("\n[ERROR] Some critical dependencies are missing!")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n[OK] All critical dependencies are available!")
        return True

def show_cache_info():
    """Show model cache information."""
    print("Model Cache Information")
    print("=" * 40)
    
    cache = get_model_cache()
    stats = cache.get_cache_stats()
    
    print(f"Cache directory: {cache.cache_dir}")
    print(f"Total models cached: {stats['total_models']}")
    print(f"Total cache size: {stats['total_size_formatted']}")
    
    if stats['models']:
        print("\nCached models:")
        print("-" * 40)
        for model_info in stats['models']:
            print(f"  {model_info['type']} ({model_info['hash'][:8]}...)")
            print(f"    Size: {model_info['size']}")
            print(f"    Access count: {model_info['access_count']}")
            print(f"    Last accessed: {model_info['last_access']}")
            print()
    else:
        print("\nNo models currently cached.")

def cleanup_cache():
    """Clean up old cached models."""
    print("Cleaning up model cache...")
    print("-" * 30)
    
    cache = get_model_cache()
    cache.cleanup_cache()
    print("Cache cleanup completed.")

def run_system_tests() -> bool:
    """Run comprehensive system tests."""
    print("Running System Tests")
    print("=" * 40)
    
    try:
        # Import and run tests
        from test_system import run_all_tests
        return run_all_tests()
    except ImportError:
        print("[ERROR] Test module not found!")
        return False

def interactive_mode():
    """Run the system in interactive mode."""
    print("LUGGAGE ANALYSIS SYSTEM - Interactive Mode")
    print("=" * 50)
    print("This mode allows you to configure and run analysis interactively.\n")
    
    # Get input folder
    while True:
        folder = input("Enter input folder path [input]: ").strip()
        if not folder:
            folder = "input"
        
        if validate_directory(folder):
            break
        else:
            print(f"[ERROR] Directory '{folder}' not found or invalid. Please try again.")
    
    # Get images
    image_files = [str(f) for f in get_image_files(folder)]
    if not image_files:
        print(f"[ERROR] No valid image files found in '{folder}'")
        return
    
    print(f"[OK] Found {len(image_files)} valid image files")
    
    # Get similarity threshold
    while True:
        try:
            threshold_str = input("Enter similarity threshold (60-95) [87]: ").strip()
            if not threshold_str:
                threshold = 87.0
                break
            threshold = float(threshold_str)
            if 60 <= threshold <= 95:
                break
            else:
                print("[ERROR] Threshold must be between 60 and 95")
        except ValueError:
            print("[ERROR] Please enter a valid number")
    
    # Ask about filename hints
    use_hints = input("Use filename patterns for grouping? (y/n) [n]: ").strip().lower()
    use_filename_hints = use_hints in ['y', 'yes', '1', 'true']
    
    # Get output directory
    output_dir = input("Enter output directory [output]: ").strip()
    if not output_dir:
        output_dir = "output"
    
    # Confirm settings
    print(f"\nAnalysis Configuration:")
    print(f"   Input folder: {folder}")
    print(f"   Images found: {len(image_files)}")
    print(f"   Similarity threshold: {threshold}%")
    print(f"   Use filename hints: {'Yes' if use_filename_hints else 'No'}")
    print(f"   Output directory: {output_dir}")
    
    confirm = input("\nProceed with analysis? (y/n) [y]: ").strip().lower()
    if confirm in ['n', 'no', '0', 'false']:
        print("Analysis cancelled.")
        return
    
    # Run analysis
    return run_analysis(folder, output_dir, threshold, use_filename_hints)

def run_analysis(
    input_folder: str,
    output_dir: str, 
    threshold: Optional[float] = None,
    use_filename_hints: bool = False,
    verbose: bool = False
) -> bool:
    """Run the main luggage analysis."""
    
    print("LUGGAGE ANALYSIS SYSTEM")
    print("=" * 40)
    
    # Get configuration
    config = get_config()
    if threshold is None:
        threshold = config.processing.similarity_threshold
    
    # Validate input directory
    if not validate_directory(input_folder):
        print(f"[ERROR] Input directory '{input_folder}' not found!")
        return False
    
    # Create output directory
    if not validate_directory(output_dir, create_if_missing=True):
        print(f"[ERROR] Could not create output directory '{output_dir}'")
        return False
    
    # Get image files
    image_files = [str(f) for f in get_image_files(input_folder)]
    if not image_files:
        print(f"[ERROR] No valid image files found in '{input_folder}'")
        print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp")
        return False
    
    print(f"Input folder: {input_folder}")
    print(f"Found {len(image_files)} images")
    print(f"Similarity threshold: {threshold}%")
    mode_text = "Smart Pattern-Assisted" if use_filename_hints else "Pure Visual Similarity"
    print(f"Analysis mode: {mode_text}")
    print(f"Output directory: {output_dir}")
    print()
    
    try:
        # Initialize analyzer
        print("Initializing analyzer...")
        analyzer = LuggageAnalyzer(
            similarity_threshold=threshold,
            use_filename_hints=use_filename_hints
        )
        print("[OK] Analyzer initialized successfully")
        
        # Run analysis
        print(f"\nProcessing {len(image_files)} images...")
        results = analyzer.analyze_images(image_files, threshold)
        print("[OK] Image processing completed")
        
        # Save results
        print("Saving results...")
        json_file, summary_file = analyzer.save_ultra_results(output_dir)
        print("[OK] Results saved successfully")
        
        # Display results
        print(f"\nANALYSIS RESULTS")
        print("-" * 20)
        print(f"Total images processed: {results['total_photos']}")
        print(f"Groups identified: {len(results['groups'])}")
        print()
        
        # Show group details
        for i, group in enumerate(results['groups'], 1):
            confidence = group['confidence']
            confidence_label = "[HIGH]" if confidence >= 90 else "[MEDIUM]" if confidence >= 75 else "[LOW]"
            print(f"Group {i}: {len(group['images'])} images {confidence_label} ({confidence:.1f}% confidence)")
            
            if verbose:
                # Show file names
                for img_id in group['images']:
                    img_path = results['processed_images'][img_id]['path']
                    img_name = Path(img_path).name
                    print(f"  - {img_name}")
                print()
        
        print(f"\nFILES CREATED:")
        print(f"   Detailed report: {json_file}")
        print(f"   Summary report: {summary_file}")
        
        print(f"\nAnalysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False

def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "WARNING" if args.quiet else "INFO"
    logger = setup_logging(level=log_level, log_file=args.log_file)
    
    # Load configuration
    if args.config:
        config_manager = get_config_manager(args.config)
    else:
        config_manager = get_config_manager()
    
    # Apply command line overrides
    config = config_manager.get_config()
    if args.threshold is not None:
        config.processing.similarity_threshold = args.threshold
    if args.device:
        config.model.device = args.device
    if args.sam_model:
        config.model.sam_model_type = args.sam_model
    
    # Handle different operation modes
    try:
        if args.check_deps:
            success = check_system_dependencies()
            sys.exit(0 if success else 1)
        
        elif args.cache_info:
            show_cache_info()
            sys.exit(0)
        
        elif args.cache_cleanup:
            cleanup_cache()
            sys.exit(0)
        
        elif args.test:
            success = run_system_tests()
            sys.exit(0 if success else 1)
        
        elif args.interactive:
            success = interactive_mode()
            sys.exit(0 if success else 1)
        
        else:
            # Normal analysis mode
            success = run_analysis(
                input_folder=args.folder,
                output_dir=args.output,
                threshold=args.threshold,
                use_filename_hints=args.use_filename_hints,
                verbose=args.verbose
            )
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()