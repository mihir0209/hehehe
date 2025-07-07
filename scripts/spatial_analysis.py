import geopandas as gpd
import matplotlib.pyplot as plt
import shapely.geometry
import os
import pandas as pd
from pathlib import Path
import numpy as np

# Define paths based on project structure (Windows-compatible)
data_dir = Path("..\\Data")
delhi_ncr_geojson = data_dir / "delhi_ncr_region.geojson"
delhi_airshed_geojson = data_dir / "delhi_airshed.geojson"
land_cover_tif = data_dir / "worldcover_bbox_delhi_ncr_2021.tif"
image_dir = data_dir / "rgb"  # Contains Sentinel-2 PNG files
vis_dir = Path("..\\Visualizations")
dataset_dir = Path("..\\Dataset")

# Ensure output directories exist
vis_dir.mkdir(exist_ok=True)
dataset_dir.mkdir(exist_ok=True)

# Load GeoJSON files
delhi_ncr = gpd.read_file(delhi_ncr_geojson).to_crs(epsg=32643)  # UTM for accurate gridding
delhi_airshed = gpd.read_file(delhi_airshed_geojson).to_crs(epsg=32643)

# Q1.1: Plot Delhi-NCR shapefile with 60x60 km grid
fig, ax = plt.subplots(figsize=(10, 10))
delhi_ncr.plot(ax=ax, color='lightblue', edgecolor='black')

# Create 60x60 km grid
minx, miny, maxx, maxy = delhi_ncr.total_bounds
grid_size = 60000  # 60 km in meters
x = np.arange(minx, maxx, grid_size)
y = np.arange(miny, maxy, grid_size)
grid_cells = []

for i in range(len(x)-1):
    for j in range(len(y)-1):
        cell = shapely.geometry.box(x[i], y[j], x[i+1], y[j+1])
        grid_cells.append(cell)

grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:32643")
grid_gdf.plot(ax=ax, facecolor='none', edgecolor='red', alpha=0.5)
plt.title("Delhi-NCR with 60x60 km Grid")
plt.savefig(vis_dir / "delhi_ncr_grid.pdf")
plt.close()

# Q1.3: Mark four corners and center of each grid cell
fig, ax = plt.subplots(figsize=(10, 10))
delhi_ncr.plot(ax=ax, color='lightblue', edgecolor='black')
grid_gdf.plot(ax=ax, facecolor='none', edgecolor='red', alpha=0.5)

for cell in grid_gdf.geometry:
    # Corners
    minx, miny, maxx, maxy = cell.bounds
    corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    for corner in corners:
        ax.plot(corner[0], corner[1], 'bo', markersize=3)
    # Center
    center = cell.centroid
    ax.plot(center.x, center.y, 'ro', markersize=5)

plt.title("Delhi-NCR Grid with Corners and Centers")
plt.savefig(vis_dir / "delhi_ncr_grid_points.pdf")
plt.close()

# Q1.4 & Q1.5: Filter images by grid and count
image_files = list(image_dir.glob("*.png"))
image_coords = []  # List to store (filename, lat, lon)

if image_files:
    for img in image_files:
        fname = img.stem
        try:
            lat, lon = map(float, fname.split("_"))
            image_coords.append({"filename": str(img), "lat": lat, "lon": lon})
        except ValueError:
            continue  # Skip files with invalid names

    # Convert to GeoDataFrame
    image_gdf = gpd.GeoDataFrame(
        image_coords,  # Use the list of dictionaries directly
        geometry=[shapely.geometry.Point(coord["lon"], coord["lat"]) for coord in image_coords],  # Access values by key
        crs="EPSG:4326"
    ).to_crs(epsg=32643)

    # Filter images within grid
    images_in_grid = gpd.sjoin(image_gdf, grid_gdf, how="inner", predicate="within")

    # Count images
    total_images = len(image_files)
    filtered_images = len(images_in_grid)

    # Save report
    with open(vis_dir / "image_count_report.txt", "w") as f:
        f.write(f"Total images: {total_images}\n")
        f.write(f"Filtered images (within grid): {filtered_images}\n")

    # Save filtered image list
    images_in_grid[["filename", "lat", "lon"]].to_csv(dataset_dir / "filtered_images.csv", index=False)
else:
    with open(vis_dir / "image_count_report.txt", "w") as f:
        f.write("No Sentinel-2 images found in Data/rgb/. Please add the required PNG files.\n")