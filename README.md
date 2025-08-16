# Luggage Analysis System v2.0
## Advanced AI-Powered Luggage Owner Identification System

A comprehensive airport luggage owner identification system that uses SAM (Segment Anything Model) and CLIP embeddings with pure visual clustering and two-stage matching to identify luggage owners with ultra-high precision.

## Quick Start

### 1. Basic Grouping (Recommended)
Put luggage photos in the `input` folder and run:
```bash
python run_analysis.py
```

### 2. Search Mode (Match Luggage to Owner)
Put luggage photos in `input` folder and owner photo in `search` folder:
```bash
python run_analysis.py --search
```

### 3. Interactive Mode
For step-by-step configuration:
```bash
python run_analysis.py --interactive
```

### 4. Custom Settings
```bash
python run_analysis.py --folder photos --threshold 75 --output results
python run_analysis.py --config my_config.yaml
```

### 5. System Check
Before first use, verify dependencies:
```bash
python run_analysis.py --check-deps
python run_analysis.py --test
```

## Project Structure
```
lbs68/
├── run_analysis.py          Main entry point - START HERE
├── input/                   Drop luggage photos here
├── input_without_background/ Background-removed luggage photos
├── search/                  Drop owner photos here for matching
├── search_without_background/ Background-removed owner photos
├── output/                  Analysis results (JSON + text reports)
├── model_cache/             Cached AI models for faster loading
├── config.py                Configuration management system
├── luggage_analyzer.py      Core analysis engine with two-stage matching
├── luggage_comparator.py    SAM + CLIP integration
├── remove_backgrounds.py    Background removal tool
├── utils.py                 Utility functions and helpers
├── model_cache.py           Intelligent model caching
├── test_system.py           System tests
├── test_imports.py          Basic import tests
└── requirements.txt         Install dependencies with this
```

## Features

- **Two-Stage Matching**: Pure visual clustering + detailed individual matching
- **Ultra-High Accuracy**: Uses SAM segmentation + CLIP embeddings for 90%+ precision
- **AI-Powered Analysis**: Deep learning models for color, texture, shape analysis
- **Background Removal**: Clean image analysis with background removal support
- **Smart Grouping**: Intelligent K-means clustering with confidence scores
- **Model Caching**: Fast loading with intelligent caching system
- **Rich Output**: JSON, text reports with detailed analysis
- **Easy Setup**: One command installation and execution
- **Configurable**: YAML/JSON config files + command line options
- **Interactive Mode**: Step-by-step guided analysis
- **Built-in Tests**: Comprehensive test suite included

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

### Basic Commands
```bash
# Simple analysis
python run_analysis.py

# Interactive mode with prompts
python run_analysis.py --interactive

# Custom folder and threshold
python run_analysis.py --folder my_photos --threshold 75

# Use configuration file
python run_analysis.py --config my_config.yaml

# Verbose output with detailed logging
python run_analysis.py --verbose

# Help and all options
python run_analysis.py --help
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
# Specify AI model settings
python run_analysis.py --device cuda --sam-model vit_h

# Use filename patterns for grouping
python run_analysis.py --use-filename-hints

# Custom output directory
python run_analysis.py --output results/analysis_2024

# Log to file
python run_analysis.py --log-file analysis.log
```

## Example Output

```
LUGGAGE ANALYSIS SYSTEM
========================================
Input folder: input
Found 12 images
Similarity threshold: 87.0%
Analysis mode: Pure Visual Similarity
Output directory: output

Initializing analyzer...
[OK] Analyzer initialized successfully

Processing 12 images...
[OK] Image processing completed

Saving results...
[OK] Results saved successfully

ANALYSIS RESULTS
--------------------
Total images processed: 12
Groups identified: 4

Group 1: 4 images [HIGH] (92.3% confidence)
Group 2: 3 images [HIGH] (89.7% confidence)  
Group 3: 3 images [MEDIUM] (78.2% confidence)
Group 4: 2 images [HIGH] (91.5% confidence)

FILES CREATED:
   Detailed report: output/ultra_precision_report_20241211_143022.json
   Summary report: output/ultra_precision_summary_20241211_143022.txt

Analysis completed successfully!
```

## How It Works

1. **Image Loading**: Validates and loads luggage photos
2. **Background Removal**: Optional SAM-based background removal for cleaner analysis
3. **SAM Segmentation**: Isolates luggage from background/people  
4. **CLIP Embeddings**: Generates semantic feature vectors
5. **Visual Clustering**: K-means clustering for pure visual grouping
6. **Two-Stage Search**: Cluster selection + individual detailed matching
7. **Feature Analysis**: Extracts color, shape, texture properties
8. **Similarity Calculation**: Multi-level similarity comparison with confidence scores
9. **Report Generation**: Creates JSON + human-readable summaries

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

- **Airports**: Match lost luggage with owner photos using two-stage visual matching
- **Hotels**: Identify guest luggage in storage with high precision
- **Cruise Ships**: Organize luggage during boarding with automated grouping
- **Event Management**: Sort and group luggage photos efficiently
- **Security**: Identify suspicious luggage patterns with visual clustering

## Troubleshooting

- **Import errors**: Run `python run_analysis.py --check-deps`
- **Memory issues**: Use `--device cpu` or smaller SAM model `--sam-model vit_b`
- **No results**: Check image formats (JPG, PNG supported)
- **Low accuracy**: Increase `--threshold` or ensure good image quality

## Support

Run the built-in diagnostics:
```bash
python run_analysis.py --check-deps
python run_analysis.py --test
```