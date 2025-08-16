# Luggage Analysis System v3.0
## Advanced AI-Powered Luggage Owner Identification System

A sophisticated airport luggage owner identification system that uses pure visual clustering with CLIP embeddings. Features two-stage matching: intelligent clustering followed by detailed individual comparisons. Includes offline background removal capabilities for enhanced analysis precision.

## Quick Start

### 1. Luggage Owner Search (Main Use Case)
Put luggage photos in `input` folder and search photo in `search` folder:
```bash
python run_analysis.py --search --device cpu --sam-model vit_b
```

### 2. Background Removal (Enhanced Accuracy)
Remove backgrounds from luggage images for better matching:
```bash
# Virtual environment setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install rembg onnxruntime

# Remove backgrounds
python remove_backgrounds_offline.py

# Use clean images for search
python run_analysis.py --search --device cpu --sam-model vit_b --folder input_without_background --search-folder search_without_background
```

### 3. System Analysis (Group All Images)
For analyzing and grouping all images:
```bash
python run_analysis.py --folder input --device cpu --sam-model vit_b
```

### 4. System Check
Before first use, verify dependencies:
```bash
python run_analysis.py --check-deps
python run_analysis.py --test
```

## Project Structure
```
lbs68/
├── run_analysis.py                 Main entry point - START HERE
├── input/                          Database luggage photos
├── search/                         Search query photos  
├── input_without_background/       Clean luggage images (backgrounds removed)
├── search_without_background/      Clean search images (backgrounds removed)
├── output/                         Analysis results (JSON reports)
├── remove_backgrounds_offline.py   Offline background removal tool
├── venv/                          Virtual environment for background removal
├── model_cache/                   Cached AI models for faster loading
├── config.py                      Configuration management system
├── luggage_analyzer.py            Core visual clustering engine
├── luggage_comparator.py          CLIP embedding extraction
├── utils.py                       Utility functions and helpers
├── model_cache.py                 Intelligent model caching
└── requirements.txt               Install dependencies with this
```

## Features

- **Two-Stage Matching**: First clusters visually similar luggage, then performs detailed individual comparisons
- **Pure Visual Clustering**: Uses K-means and silhouette analysis for optimal clustering without filename dependencies
- **CLIP Embeddings**: Advanced semantic understanding of luggage features using transformer models
- **Background Removal**: Offline background removal using rembg for enhanced matching accuracy
- **Potential Match System**: Returns potential matches even when confidence is below threshold
- **Profile-Based Matching**: Analyzes color, texture, and shape profiles for comprehensive comparison
- **MacBook M2 Optimized**: CPU-optimized processing for 8GB RAM systems
- **Model Caching**: Intelligent caching system for faster subsequent runs
- **Rich JSON Output**: Detailed analysis results with confidence scores and reasoning

## Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python run_analysis.py --check-deps
```

### 3. Run Tests (Optional)
```bash
python run_analysis.py --test
```

**Note**: AI models (SAM, CLIP) download automatically on first use (~1-2GB)

## Complete Usage Examples

### Search Commands
```bash
# Basic luggage owner search
python run_analysis.py --search --device cpu --sam-model vit_b

# Search with custom folders
python run_analysis.py --search --device cpu --sam-model vit_b --folder my_luggage --search-folder my_queries

# Search with background-removed images
python run_analysis.py --search --device cpu --sam-model vit_b --folder input_without_background --search-folder search_without_background

# Verbose search with detailed logging
python run_analysis.py --search --device cpu --sam-model vit_b --verbose

# Help and all options
python run_analysis.py --help
```

### Background Removal Commands
```bash
# Remove backgrounds from input images
python remove_backgrounds_offline.py

# Remove backgrounds from custom directory
python -c "
from remove_backgrounds_offline import remove_backgrounds_offline
remove_backgrounds_offline('my_images', 'my_images_clean')
"
```

### System Management
```bash
# Check all dependencies
python run_analysis.py --check-deps

# Run system tests
python run_analysis.py --test

# Show cached models info
python run_analysis.py --cache-info

# Clean old cached models
python run_analysis.py --cache-cleanup
```

### Advanced Options
```bash
# Use larger SAM model for higher accuracy (requires more memory)
python run_analysis.py --search --device cpu --sam-model vit_l

# Use GPU if available (CUDA/MPS)
python run_analysis.py --search --device cuda --sam-model vit_b

# Custom output directory
python run_analysis.py --search --device cpu --sam-model vit_b --output results/search_2024

# Log to file
python run_analysis.py --search --device cpu --sam-model vit_b --log-file search.log
```

## Example Output

### Luggage Owner Search Result
```
LUGGAGE SEARCH SYSTEM
========================================
Input folder: input_without_background (24 images)
Search folder: search_without_background (1 image)

Step 1: Grouping 24 input images...
[OK] Found 6 clusters

Step 2: Searching for green_no_bg.png...

🎯 BEST MATCH FOUND!
📂 Best Cluster: Cluster 3 (86.2% match)
🔍 Individual Match: a03_no_bg.png (91.4% confidence)
   Visual: 89.2% | Profile: 94.1% | Embedding: 90.9%

🔍 POTENTIAL MATCHES in cluster:
   → a01_no_bg.png (87.3% confidence)  
   → a02_no_bg.png (85.7% confidence)
   → a04_no_bg.png (83.1% confidence)

📁 Results saved: output/search_results_20241216_235531.json
```

### Background Removal Output
```
🔍 Found 24 images to process...
🔄 Processing: a01.jpeg
✅ Saved: a01_no_bg.png
🔄 Processing: a02.jpeg  
✅ Saved: a02_no_bg.png
...
🎉 Background removal completed!
📊 Processed: 24/24 images
📁 Output directory: input_without_background
```

## How It Works

### Two-Stage Matching Process

**Stage 1: Visual Clustering**
1. **CLIP Embedding Extraction**: Generate semantic feature vectors for all luggage images
2. **K-means Clustering**: Group visually similar luggage using optimal cluster count (silhouette analysis)
3. **Cluster Selection**: Find the cluster with highest similarity to search query

**Stage 2: Individual Matching**
1. **Detailed Comparison**: Compare search query with each image in the best cluster
2. **Multi-Factor Analysis**: 
   - Visual similarity (60% weight)
   - Profile matching (30% weight) - color, texture, shape analysis
   - Embedding similarity (10% weight)
3. **Confidence Scoring**: Combine all factors for final confidence score
4. **Potential Matches**: Return matches even below main threshold for human review

### Background Removal Process
1. **rembg Integration**: Uses U2-Net model for precise background removal
2. **Offline Processing**: No internet required, fully local processing
3. **PNG Output**: Maintains transparency for clean luggage isolation

## Configuration

Create `config.yaml` for custom settings:

```yaml
model:
  sam_model_type: vit_h          # vit_b, vit_l, vit_h
  device: auto                   # auto, cpu, cuda, mps
  
processing:
  similarity_threshold: 75.0     # 60-95
  enable_segmentation: true
  
output:
  default_output_dir: output
  create_detailed_reports: true
  save_similarity_matrix: true
```

## Perfect For

- **Airport Lost & Found**: Match lost luggage with owner photos using two-stage identification
- **Hotel Concierge Services**: Identify guest luggage in storage areas
- **Cruise Ship Operations**: Organize and identify luggage during boarding/disembarkation
- **Conference & Event Management**: Match attendee luggage with registration photos
- **Transportation Hubs**: Identify suspicious or unclaimed luggage patterns
- **Baggage Handling**: Quality control and tracking in baggage processing systems

## Troubleshooting

- **Import errors**: Run `python run_analysis.py --check-deps`
- **Memory issues on MacBook M2**: Always use `--device cpu --sam-model vit_b` for 8GB RAM
- **Background removal issues**: Ensure virtual environment is activated: `source venv/bin/activate`
- **No search results**: Check that both input and search folders contain valid images (JPG, PNG)
- **Low matching accuracy**: Try background removal first, ensure good image quality and similar lighting
- **HEIF format errors**: Convert HEIF files to JPEG: `magick input.heic output.jpg`

## Support

Run the built-in diagnostics:
```bash
python run_analysis.py --check-deps
python run_analysis.py --test
```