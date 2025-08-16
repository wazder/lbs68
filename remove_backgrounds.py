#!/usr/bin/env python3
"""
Offline Background Removal Script
Removes backgrounds from luggage images using rembg library.
"""

import os
import sys
from pathlib import Path
from rembg import remove
from PIL import Image
import io

def remove_backgrounds_offline(input_dir="input", output_dir="input_without_background"):
    """Remove backgrounds from all images in input directory using rembg."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ Input directory '{input_dir}' not found!")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions and f.is_file()]
    
    if not image_files:
        print(f"❌ No image files found in '{input_dir}'")
        return
    
    print(f"🔍 Found {len(image_files)} images to process...")
    
    processed = 0
    for img_file in sorted(image_files):
        try:
            print(f"🔄 Processing: {img_file.name}")
            
            # Read input image
            with open(img_file, 'rb') as input_file:
                input_data = input_file.read()
            
            # Remove background
            output_data = remove(input_data)
            
            # Save as PNG (supports transparency)
            output_file = output_path / f"{img_file.stem}_no_bg.png"
            with open(output_file, 'wb') as out_file:
                out_file.write(output_data)
            
            print(f"✅ Saved: {output_file.name}")
            processed += 1
            
        except Exception as e:
            print(f"❌ Error processing {img_file.name}: {e}")
    
    print(f"\n🎉 Background removal completed!")
    print(f"📊 Processed: {processed}/{len(image_files)} images")
    print(f"📁 Output directory: {output_dir}")

if __name__ == "__main__":
    # Check if we're in virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Please activate virtual environment first:")
        print("source venv/bin/activate")
        sys.exit(1)
    
    remove_backgrounds_offline()