#!/usr/bin/env python3
"""
Folder Watcher - Automatically analyze when photos are added

This script monitors the input folder and automatically runs analysis
when new photos are added. Perfect for continuous monitoring.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    from auto_analyze import AutoLuggageAnalyzer
except ImportError:
    print("ERROR: Could not import auto analyzer.")
    sys.exit(1)


class FolderWatcher:
    """Watch input folder and auto-analyze when files are added."""
    
    def __init__(self, 
                 input_folder: str = "input",
                 output_folder: str = "output", 
                 similarity_threshold: float = 75.0,
                 check_interval: int = 5):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.similarity_threshold = similarity_threshold
        self.check_interval = check_interval
        
        # Create folders
        self.input_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        
        # Track folder state
        self.last_files = set()
        self.last_check = None
        
        # Auto analyzer
        self.analyzer = AutoLuggageAnalyzer(
            input_folder=str(self.input_folder),
            output_folder=str(self.output_folder),
            similarity_threshold=self.similarity_threshold
        )
    
    def get_current_files(self) -> set:
        """Get current image files in input folder."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
        current_files = set()
        
        if self.input_folder.exists():
            for file_path in self.input_folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    # Use (name, size, mtime) as unique identifier
                    stat = file_path.stat()
                    file_info = (file_path.name, stat.st_size, int(stat.st_mtime))
                    current_files.add(file_info)
        
        return current_files
    
    def has_folder_changed(self) -> tuple:
        """Check if folder has changed since last check."""
        current_files = self.get_current_files()
        
        if self.last_files is None:
            self.last_files = current_files
            return False, set(), set()
        
        added_files = current_files - self.last_files
        removed_files = self.last_files - current_files
        
        changed = len(added_files) > 0 or len(removed_files) > 0
        
        self.last_files = current_files
        return changed, added_files, removed_files
    
    def wait_for_stable_folder(self, timeout: int = 30) -> bool:
        """Wait for folder to be stable (no changes for a few seconds)."""
        print("Waiting for file operations to complete...")
        stable_time = 0
        last_files = None
        
        while stable_time < 3:  # Wait for 3 seconds of stability
            current_files = self.get_current_files()
            
            if current_files == last_files:
                stable_time += 1
            else:
                stable_time = 0
                last_files = current_files
            
            time.sleep(1)
            timeout -= 1
            
            if timeout <= 0:
                print("WARNING: Timeout waiting for folder stability, proceeding anyway...")
                return False
        
        return True
    
    def run_watch(self):
        """Start watching the input folder."""
        print("FOLDER WATCHER STARTED")
        print("=" * 30)
        print(f"Monitoring: {self.input_folder}")
        print(f"Output to: {self.output_folder}")
        print(f"Threshold: {self.similarity_threshold}%")
        print(f"Check every: {self.check_interval} seconds")
        print()
        print("Drop luggage photos in the input folder and they'll be analyzed automatically!")
        print("Press Ctrl+C to stop watching")
        print("-" * 50)
        
        # Initial scan
        self.last_files = self.get_current_files()
        if self.last_files:
            print(f"Found {len(self.last_files)} existing files in input folder")
        
        try:
            while True:
                changed, added, removed = self.has_folder_changed()
                
                if changed:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n[{timestamp}] Folder change detected!")
                    
                    if added:
                        print(f"Added files: {len(added)}")
                        for file_info in added:
                            print(f"   - {file_info[0]}")
                    
                    if removed:
                        print(f"Removed files: {len(removed)}")
                        for file_info in removed:
                            print(f"   - {file_info[0]}")
                    
                    # Wait for folder to stabilize
                    self.wait_for_stable_folder()
                    
                    # Check if we have enough files to analyze
                    current_image_files = self.analyzer.get_image_files()
                    
                    if len(current_image_files) >= 2:
                        print(f"Starting automatic analysis...")
                        success = self.analyzer.run_analysis(move_files=True, keep_originals=False)
                        
                        if success:
                            print("Analysis complete! Check output folder for results.")
                        else:
                            print("ERROR: Analysis failed. Check error messages above.")
                    
                    elif len(current_image_files) == 1:
                        print("Found 1 photo. Waiting for more photos to compare...")
                    
                    else:
                        print("No photos in input folder.")
                    
                    print(f"Continuing to watch for changes...")
                
                # Wait before next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print(f"\nFolder watching stopped.")
            print("You can restart watching anytime with: python watch_folder.py")


def main():
    """Main function for folder watcher."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Watch input folder and auto-analyze luggage photos")
    parser.add_argument("--threshold", "-t", type=float, default=75.0,
                       help="Similarity threshold (0-100, default: 75)")
    parser.add_argument("--input", "-i", default="input",
                       help="Input folder to watch (default: input)")
    parser.add_argument("--output", "-o", default="output",
                       help="Output folder (default: output)")
    parser.add_argument("--interval", type=int, default=5,
                       help="Check interval in seconds (default: 5)")
    
    args = parser.parse_args()
    
    # Create watcher
    watcher = FolderWatcher(
        input_folder=args.input,
        output_folder=args.output,
        similarity_threshold=args.threshold,
        check_interval=args.interval
    )
    
    # Start watching
    watcher.run_watch()


if __name__ == "__main__":
    main()