# Luggage Analysis System v2.0
## Advanced AI-Powered Luggage Grouping System

A comprehensive system that uses SAM (Segment Anything Model) and CLIP to compare luggage photos and group identical ones with ultra-high precision using advanced computer vision techniques.

## Quick Start

### 1. Basic Analysis (Recommended)
Put your photos in the `input` folder and run:
```bash
python run_analysis.py
```

### 2. Interactive Mode
For step-by-step configuration:
```bash
python run_analysis.py --interactive
```

### 3. Custom Settings
```bash
python run_analysis.py --folder photos --threshold 85 --output results
python run_analysis.py --config my_config.yaml
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
├── run_analysis.py          Main entry point - START HERE
├── input/                   Drop your luggage photos here
├── output/                  Analysis results (JSON + text reports)
├── model_cache/             Cached AI models for faster loading
├── config.py                Configuration management system
├── luggage_analyzer.py      Core analysis engine
├── luggage_comparator.py    SAM + CLIP integration
├── utils.py                 Utility functions and helpers
├── model_cache.py           Intelligent model caching
├── test_system.py           System tests
├── test_imports.py          Basic import tests
└── requirements.txt         Install dependencies with this
```

## Features

- **Ultra-High Accuracy**: Uses SAM segmentation + CLIP embeddings for 90%+ precision
- **AI-Powered Analysis**: Deep learning models for color, texture, shape analysis
- **Smart Grouping**: Intelligent clustering with confidence scores
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
python run_analysis.py --folder my_photos --threshold 90

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
2. **SAM Segmentation**: Isolates luggage from background/people  
3. **CLIP Embeddings**: Generates semantic feature vectors
4. **Feature Analysis**: Extracts color, shape, texture properties
5. **Similarity Calculation**: Multi-level similarity comparison
6. **Smart Grouping**: Clusters identical luggage with confidence
7. **Report Generation**: Creates JSON + human-readable summaries

## Configuration

Create `config.yaml` for custom settings:

```yaml
model:
  sam_model_type: vit_h          # vit_b, vit_l, vit_h
  device: auto                   # auto, cpu, cuda, mps
  
processing:
  similarity_threshold: 87.0     # 60-95
  enable_segmentation: true
  
output:
  default_output_dir: output
  create_detailed_reports: true
  save_similarity_matrix: true
```

## Perfect For

- **Airports**: Match lost luggage with owner photos
- **Hotels**: Identify guest luggage in storage
- **Cruise Ships**: Organize luggage during boarding
- **Event Management**: Sort and group luggage photos
- **Security**: Identify suspicious luggage patterns

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