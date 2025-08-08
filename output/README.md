# Analysis Results Directory

This directory contains the results of luggage analysis operations.

## Directory Structure

- **archive/** - Archived results organized by date (automatically created)
- **Latest results** - Current analysis results appear directly here
- **Detailed reports** - JSON format with full analysis data
- **Summary reports** - Human-readable text summaries
- **Processed images** - Images that were analyzed (when save_processed_images=True)

## Archive Policy

Old results are automatically archived by date to keep this directory clean.
Results older than 7 days are moved to the archive folder automatically.

## File Naming Convention

- `RESULTS_YYYYMMDD_HHMMSS.txt` - Quick summary report
- `detailed_YYYYMMDD_HHMMSS/` - Full analysis folder containing:
  - `luggage_analysis_report_YYYYMMDD_HHMMSS.json` - Complete analysis data
  - `similarity_matrix_YYYYMMDD_HHMMSS.csv` - Similarity matrix
  - `summary_YYYYMMDD_HHMMSS.txt` - Human-readable summary
- `processed_YYYYMMDD_HHMMSS/` - Processed images folder (if enabled)

## Notes

- This directory should ONLY contain analysis outputs
- Input images should be placed in the `input/` directory at the project root
- Archive cleanup happens automatically after 7 days