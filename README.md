Luggage Photo Comparison System
=================================

A comprehensive system that uses SAM (Segment Anything Model) and CLIP to compare luggage photos and group identical ones with detailed feature analysis.

Quick Start
-----------

1. Easy Method - Just Drop Photos:
   Put your photos in the 'input' folder and run:
   python auto_analyze.py

2. Interactive Method:
   python analyze_luggage.py --interactive

3. Command Line Method:
   python analyze_luggage.py --folder your_photos/
   python analyze_luggage.py --files photo1.jpg photo2.jpg photo3.jpg

Folder Structure
----------------
lbs68/
├── input/                    Drop your photos here
├── output/                   Results will appear here
├── auto_analyze.py           Simple: analyzes input folder automatically
├── analyze_luggage.py        Full-featured analysis tool
├── multi_luggage_analyzer.py Core analysis engine
├── luggage_comparator.py     SAM + CLIP integration
└── requirements.txt          Install dependencies

Features
--------
- High Accuracy: Uses SAM for precise luggage segmentation + CLIP for semantic understanding
- Feature Analysis: Color, size, texture, material detection
- Smart Grouping: Groups identical luggage with confidence scores
- Multiple Formats: JSON, CSV, and text reports
- Easy to Use: Just drop photos in input folder

Installation
------------
Install dependencies:
pip install -r requirements.txt

For SAM model:
pip install segment-anything

SAM weights download automatically on first run

Example Output
--------------
LUGGAGE 1 - Confidence: 89.2%
• Color: black
• Size: medium
• Texture: smooth
• Material: hard_plastic_ABS
• Photos: suitcase_1.jpg, suitcase_5.jpg, suitcase_8.jpg
• Highest similarity: 94.5%

LUGGAGE 2 - Confidence: 76.8%
• Color: blue
• Size: large
• Texture: lightly_textured
• Material: fabric_soft
• Photos: bag_2.jpg, bag_7.jpg
• Highest similarity: 83.2%

Advanced Usage
--------------
Adjust similarity threshold: --threshold 80 (default: 75%)
Custom output folder: --output results/
No file saving: --no-save (console only)

How It Works
------------
1. SAM Segmentation: Isolates luggage from background/clothing
2. CLIP Embeddings: Generates semantic feature vectors
3. Cosine Similarity: Compares embeddings for similarity
4. Smart Grouping: Groups similar luggage with confidence scores
5. Feature Analysis: Extracts color, size, texture, material properties

Perfect for airports, hotels, or any scenario where you need to match luggage photos.