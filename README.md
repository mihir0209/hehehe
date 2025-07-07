# Scenario-1: Earth Observation

This repository contains the solution for Q1: Spatial Reasoning & Data Filtering for the IIT Gandhinagar Sustainability Lab assignment.

## Directory Structure
- `Data/`: Contains input datasets (geojson files, worldcover_bbox_delhi_ncr_2021.tif, rgb/).
- `Visualizations/`: Contains output plots and reports.
- `Dataset/`: Contains processed datasets (e.g., filtered_images.csv).
- `scripts/`: Contains Python scripts (e.g., spatial_analysis.py).
- `requirements.txt`: Lists dependencies.

## Running the Script
1. Create virtual environment: `python -m venv env`
2. Activate the virtual env: `env/Scripts/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the script: `python scripts\spatial_analysis.py`
5. Outputs are saved in `Visualizations\` and `Dataset\`.

## Outputs
- `Visualizations\delhi_ncr_grid.pdf`: Delhi-NCR with 60x60 km grid.
- `Visualizations\delhi_ncr_satellite_grid.html`: Grid on satellite basemap.
- `Visualizations\delhi_ncr_grid_points.pdf`: Grid with corners and centers.
- `Visualizations\image_count_report.txt`: Image count report.
- `Dataset\filtered_images.csv`: Filtered image list.

## Notes
- Ensure PNG files in `Data/rgb/` follow the format `lat_lon.png` (e.g., `28.7041_77.1025.png`).