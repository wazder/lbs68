#!/usr/bin/env python3
"""
Auto Luggage Analysis - Simple Drop & Analyze

Just put your luggage photos in the 'input' folder and run this script.
Results will automatically appear in the 'output' folder.
"""

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

try:
    from multi_luggage_analyzer import MultiLuggageAnalyzer
except ImportError:
    print("ERROR: Could not import analysis modules.")
    print("Make sure you have installed dependencies: pip install -r requirements.txt")
    sys.exit(1)


class AutoLuggageAnalyzer:
    """Automatic analysis of photos dropped in input folder."""
    
    def __init__(self, 
                 input_folder: str = "input", 
                 output_folder: str = "output",
                 similarity_threshold: float = 85.0):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.similarity_threshold = similarity_threshold
        
        # Create folders if they don't exist
        self.input_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
    
    def get_image_files(self) -> list:
        """Find all image files in input folder."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
        image_files = []
        
        if not self.input_folder.exists():
            return []
        
        for file_path in self.input_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))
        
        return sorted(image_files)
    
    def move_processed_files(self, image_files: list, session_id: str):
        """Move processed files to a processed subfolder."""
        processed_folder = self.output_folder / f"processed_{session_id}"
        processed_folder.mkdir(exist_ok=True)
        
        for image_file in image_files:
            src = Path(image_file)
            dst = processed_folder / src.name
            try:
                shutil.move(str(src), str(dst))
                print(f"Moved {src.name} to processed folder")
            except Exception as e:
                print(f"WARNING: Could not move {src.name}: {e}")
    
    def create_simple_summary(self, analyzer: MultiLuggageAnalyzer, session_id: str):
        """Create a simple, easy-to-read summary."""
        summary_path = self.output_folder / f"RESULTS_{session_id}.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("LUGGAGE ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Analysis Date: {timestamp}\n")
            f.write(f"Total Photos: {len(analyzer.processed_images)}\n")
            f.write(f"Similarity Threshold: {analyzer.similarity_threshold}%\n")
            f.write(f"Groups Found: {len(analyzer.groups)}\n\n")
            
            if not analyzer.groups:
                f.write("NO MATCHING LUGGAGE FOUND\n")
                f.write("All photos appear to show different luggage pieces.\n\n")
                
                f.write("INDIVIDUAL PHOTOS:\n")
                f.write("-" * 20 + "\n")
                for img_id, img_data in analyzer.processed_images.items():
                    img_name = Path(img_data['path']).name
                    features = img_data['features']
                    f.write(f"- {img_name}\n")
                    f.write(f"  Color: {features['color']['dominant_color']}\n")
                    f.write(f"  Size: {features['size']['size_category']}\n")
                    f.write(f"  Material: {features['brand']['material_type']}\n\n")
            else:
                f.write("MATCHING LUGGAGE GROUPS FOUND:\n\n")
                
                for i, group in enumerate(analyzer.groups, 1):
                    f.write(f"GROUP {i} - SAME LUGGAGE (Confidence: {group['confidence']}%)\n")
                    f.write("-" * 40 + "\n")
                    
                    # Common features
                    features = group['common_features']
                    f.write("FEATURES:\n")
                    f.write(f"  - Color: {features.get('dominant_color', 'Unknown')}\n")
                    f.write(f"  - Size: {features.get('size_category', 'Unknown')}\n")
                    f.write(f"  - Texture: {features.get('texture_type', 'Unknown')}\n")
                    f.write(f"  - Material: {features.get('material_type', 'Unknown')}\n\n")
                    
                    # Photos in this group
                    f.write("PHOTOS IN THIS GROUP:\n")
                    for img_id in group['images']:
                        img_path = analyzer.processed_images[img_id]['path']
                        img_name = Path(img_path).name
                        f.write(f"  - {img_name}\n")
                    
                    # Similarity details and explanation
                    if group['similarities']:
                        max_sim = max(group['similarities'].values())
                        min_sim = min(group['similarities'].values())
                        f.write(f"\nSimilarity Range: {min_sim}% - {max_sim}%\n")
                    
                    # Why are they grouped together?
                    if 'explanation' in group:
                        explanation = group['explanation']
                        f.write(f"\nWhy grouped together:\n")
                        for detail in explanation.get('details', []):
                            f.write(f"  - {detail}\n")
                    
                    f.write("\n" + "="*50 + "\n\n")
                
                # Individual photos (not in any group)
                grouped_images = set()
                for group in analyzer.groups:
                    grouped_images.update(group['images'])
                
                single_images = [img_id for img_id in analyzer.processed_images.keys() 
                               if img_id not in grouped_images]
                
                if single_images:
                    f.write("INDIVIDUAL PHOTOS (No matches found):\n")
                    f.write("-" * 35 + "\n")
                    for img_id in single_images:
                        img_path = analyzer.processed_images[img_id]['path']
                        img_name = Path(img_path).name
                        features = analyzer.processed_images[img_id]['features']
                        f.write(f"- {img_name}\n")
                        f.write(f"  Color: {features['color']['dominant_color']}\n")
                        f.write(f"  Size: {features['size']['size_category']}\n")
                        f.write(f"  Material: {features['brand']['material_type']}\n\n")
        
        print(f"Simple results saved: {summary_path}")
        return str(summary_path)
    
    def run_analysis(self, move_files: bool = True, keep_originals: bool = False):
        """Run complete analysis on input folder."""
        print("AUTO LUGGAGE ANALYSIS")
        print("=" * 30)
        
        # Check for images
        image_files = self.get_image_files()
        
        if not image_files:
            print(f"No image files found in '{self.input_folder}' folder!")
            print(f"Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp")
            print(f"Drop your luggage photos in the '{self.input_folder}' folder and run again.")
            return False
        
        print(f"Found {len(image_files)} photos in input folder:")
        for img_file in image_files:
            print(f"   - {Path(img_file).name}")
        
        if len(image_files) < 2:
            print("WARNING: Need at least 2 photos to compare!")
            return False
        
        print(f"\nStarting analysis (threshold: {self.similarity_threshold}%)...")
        print("STEP 1: Detecting luggage items...")
        print("STEP 2: Grouping similar luggage...")
        
        # Create analyzer
        analyzer = MultiLuggageAnalyzer(similarity_threshold=self.similarity_threshold)
        
        try:
            # Process images
            print("Processing images...")
            analyzer.process_images(image_files)
            
            # Group similar luggage
            print("Grouping similar luggage...")
            groups = analyzer.group_similar_luggage()
            
            # Create session ID for this analysis
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Show quick results
            print(f"\nANALYSIS COMPLETE!")
            print("-" * 25)
            
            if groups:
                print(f"Found {len(groups)} group(s) of matching luggage:")
                for i, group in enumerate(groups, 1):
                    print(f"   Group {i}: {len(group['images'])} photos (confidence: {group['confidence']}%)")
                
                # Show which photos match
                for i, group in enumerate(groups, 1):
                    print(f"\nGROUP {i} - These photos show the SAME luggage:")
                    for img_id in group['images']:
                        img_path = analyzer.processed_images[img_id]['path']
                        img_name = Path(img_path).name
                        print(f"   - {img_name}")
            else:
                print("No matching luggage found - all photos show different items")
            
            # Save detailed results
            print(f"\nSaving results to 'output' folder...")
            results = analyzer.save_results(str(self.output_folder / f"detailed_{session_id}"))
            
            # Create simple summary
            simple_summary = self.create_simple_summary(analyzer, session_id)
            
            # Move processed files
            if move_files:
                if keep_originals:
                    print("Copying processed files to output folder...")
                    processed_folder = self.output_folder / f"processed_{session_id}"
                    processed_folder.mkdir(exist_ok=True)
                    for image_file in image_files:
                        src = Path(image_file)
                        dst = processed_folder / src.name
                        shutil.copy2(str(src), str(dst))
                else:
                    print("Moving processed files to output folder...")
                    self.move_processed_files(image_files, session_id)
            
            print(f"\nANALYSIS COMPLETE!")
            print(f"Check results: {simple_summary}")
            print(f"Detailed reports: output/detailed_{session_id}/")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Error during analysis: {e}")
            return False


def main():
    """Main function for auto analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-analyze luggage photos from input folder")
    parser.add_argument("--threshold", "-t", type=float, default=85.0, 
                       help="Similarity threshold (0-100, default: 85.0)")
    parser.add_argument("--input", "-i", default="input", 
                       help="Input folder name (default: input)")
    parser.add_argument("--output", "-o", default="output", 
                       help="Output folder name (default: output)")
    parser.add_argument("--keep-files", "-k", action="store_true",
                       help="Keep original files in input folder (copy instead of move)")
    parser.add_argument("--no-move", action="store_true",
                       help="Don't move/copy files, leave them in input folder")
    
    args = parser.parse_args()
    
    print("LUGGAGE PHOTO AUTO-ANALYZER")
    print("=" * 35)
    print("Just drop photos in the 'input' folder and let the AI do the work!")
    print()
    
    # Create analyzer
    auto_analyzer = AutoLuggageAnalyzer(
        input_folder=args.input,
        output_folder=args.output, 
        similarity_threshold=args.threshold
    )
    
    # Run analysis
    success = auto_analyzer.run_analysis(
        move_files=not args.no_move,
        keep_originals=args.keep_files
    )
    
    if success:
        print(f"\nReady for next batch! Drop new photos in '{args.input}' folder.")
    else:
        print(f"\nDrop your luggage photos in '{args.input}' folder and try again.")


if __name__ == "__main__":
    main()