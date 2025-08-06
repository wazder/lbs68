LUGGAGE PHOTOS INPUT FOLDER
==============================

DROP YOUR LUGGAGE PHOTOS HERE!

This folder is monitored for automatic analysis. Just drop your luggage photos here and the system will:

- Automatically detect new photos
- Compare and group identical luggage  
- Generate detailed similarity reports
- Move results to the 'output' folder

SUPPORTED FORMATS:
- .jpg / .jpeg
- .png  
- .bmp
- .tiff
- .webp
- .gif

HOW TO USE:
-----------
Option 1 - Auto Analysis (Recommended):
  python auto_analyze.py

Option 2 - Live Monitoring:
  python watch_folder.py

Option 3 - Manual Analysis:
  python analyze_luggage.py --folder input

TIPS:
- Drop at least 2 photos to compare
- Photos will be automatically moved to output folder after analysis
- Use --keep-files flag to keep original photos in this folder
- Check the 'output' folder for results

Happy analyzing!