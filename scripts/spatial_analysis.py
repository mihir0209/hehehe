import geopandas as gpd
import matplotlib.pyplot as plt
import geemap
import shapely.geometry
import os
import pandas as pd
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchmetrics
from PIL import Image
import ee
from pyproj import Transformer

# ESA WorldCover 2021 class codes to standardized labels
esa_to_labels = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
    60: "Bare/sparse vegetation", 70: "Snow/ice", 80: "Permanent water", 90: "Herbaceous wetland",
    95: "Mangroves", 100: "Moss and lichen"
}

# Custom Dataset and CNN classes
class ImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        # Only use labels present in dataframe
        unique_labels = sorted(self.df['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["filename"]
        image = Image.open(img_path).convert("RGB")
        label = self.label_to_idx[self.df.iloc[idx]["label"]]
        if self.transform:
            image = self.transform(image)
        return image, label

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Earth Engine (ignore warnings about pkg_resources and IncompleteRead)
try:
    ee.Initialize(project='neurapods-pro')
except Exception as e:
    print("Earth Engine initialization warning:", e)

# Define directory paths
project_root = Path("D:/Project_Root")
data_dir = project_root / "Data"
delhi_ncr_geojson = data_dir / "delhi_ncr_region.geojson"
delhi_airshed_geojson = data_dir / "delhi_airshed.geojson"
land_cover_tif = data_dir / "worldcover_bbox_delhi_ncr_2021.tif"
image_dir = data_dir / "rgb"
vis_dir = project_root / "Visualizations"
dataset_dir = project_root / "Dataset"

vis_dir.mkdir(exist_ok=True)
dataset_dir.mkdir(exist_ok=True)

# Q1: Spatial Reasoning & Data Filtering
delhi_ncr = gpd.read_file(delhi_ncr_geojson).to_crs(epsg=32643)
delhi_airshed = gpd.read_file(delhi_airshed_geojson).to_crs(epsg=32643)

# Q1.1: Plot Delhi-NCR shapefile with 60x60 km grid
fig, ax = plt.subplots(figsize=(10, 10))
delhi_ncr.plot(ax=ax, color='lightblue', edgecolor='black')

minx, miny, maxx, maxy = delhi_ncr.total_bounds
grid_size = 60000
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

# Q1.2: Overlay grid on satellite basemap using geemap
m = geemap.Map()
m.add_basemap("SATELLITE")
m.add_gdf(delhi_ncr.to_crs(epsg=4326), layer_name="Delhi-NCR", style={"color": "#0000ff", "fillOpacity": 0.3})
grid_gdf_latlon = grid_gdf.to_crs(epsg=4326)
m.add_gdf(grid_gdf_latlon, layer_name="Grid", style={"color": "#ff0000", "fillOpacity": 0})
m.save(str(vis_dir / "delhi_ncr_satellite_grid.html"))

# Q1.3: Mark four corners and center of each grid cell
fig, ax = plt.subplots(figsize=(10, 10))
delhi_ncr.plot(ax=ax, color='lightblue', edgecolor='black')
grid_gdf.plot(ax=ax, facecolor='none', edgecolor='red', alpha=0.5)
for cell in grid_gdf.geometry:
    minx, miny, maxx, maxy = cell.bounds
    corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    for corner in corners:
        ax.plot(corner[0], corner[1], 'bo', markersize=3)
    center = cell.centroid
    ax.plot(center.x, center.y, 'ro', markersize=5)
plt.title("Delhi-NCR Grid with Corners and Centers")
plt.savefig(vis_dir / "delhi_ncr_grid_points.pdf")
plt.close()

# Q1.4 & Q1.5: Filter images by grid and count
image_files = list(image_dir.glob("*.png"))
image_coords = []
if image_files:
    for img in image_files:
        fname = img.stem
        try:
            lat, lon = map(float, fname.split("_"))
            image_coords.append({"filename": str(img), "lat": lat, "lon": lon})
        except ValueError:
            continue
    image_gdf = gpd.GeoDataFrame(
        image_coords,
        geometry=[shapely.geometry.Point(coord["lon"], coord["lat"]) for coord in image_coords],
        crs="EPSG:4326"
    ).to_crs(epsg=32643)
    images_in_grid = gpd.sjoin(image_gdf, grid_gdf, how="inner", predicate="within")
    total_images = len(image_files)
    filtered_images = len(images_in_grid)
    with open(vis_dir / "image_count_report.txt", "w") as f:
        f.write(f"Total images: {total_images}\n")
        f.write(f"Filtered images (within grid): {filtered_images}\n")
    images_in_grid[["filename", "lat", "lon"]].to_csv(dataset_dir / "filtered_images.csv", index=False)
else:
    with open(vis_dir / "image_count_report.txt", "w") as f:
        f.write("No Sentinel-2 images found in Data/rgb/. Please add the required PNG files.\n")

# Q2: Label Construction & Dataset Preparation
image_files = list(image_dir.glob("*.png"))
image_coords = []
for img in image_files:
    fname = img.stem
    try:
        lat, lon = map(float, fname.split("_"))
        image_coords.append({"filename": str(img), "lat": lat, "lon": lon})
    except ValueError:
        continue
image_gdf = gpd.GeoDataFrame(
    image_coords,
    geometry=[shapely.geometry.Point(coord["lon"], coord["lat"]) for coord in image_coords],
    crs="EPSG:4326"
)

with rasterio.open(land_cover_tif) as src:
    labels = []
    for idx, row in image_gdf.iterrows():
        lon, lat = row.geometry.x, row.geometry.y
        try:
            py, px = src.index(lon, lat)
            if (px >= 0 and py >= 0 and px < src.width and py < src.height):
                window = Window(max(0, px - 64), max(0, py - 64), min(128, src.width - max(0, px - 64)), min(128, src.height - max(0, py - 64)))
                patch = src.read(1, window=window)
                if np.any(patch == src.nodata) or patch.size == 0:
                    valid_patch = patch[patch != src.nodata]
                    if len(valid_patch) == 0:
                        label = "No-data"
                        label_code = -1
                    else:
                        label_code = np.bincount(valid_patch.ravel()).argmax()
                        label = esa_to_labels.get(label_code, "Mixed")
                else:
                    label_code = np.bincount(patch.ravel()).argmax()
                    label = esa_to_labels.get(label_code, "Mixed")
            else:
                label = "Out-of-bounds"
                label_code = -2
        except Exception as e:
            label = "Out-of-bounds"
            label_code = -2
        labels.append({"filename": row["filename"], "label": label, "label_code": label_code})
    label_df = pd.DataFrame(labels)
    train_df, test_df = train_test_split(label_df, train_size=0.6, random_state=42)
    train_df.to_csv(dataset_dir / "train_dataset.csv", index=False)
    test_df.to_csv(dataset_dir / "test_dataset.csv", index=False)
    class_counts = label_df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    class_counts.plot(kind="bar", ax=ax, color="teal")
    ax.set_title("Class Distribution in Dataset")
    ax.set_xlabel("Land Cover Class")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(vis_dir / "class_distribution.pdf")
    plt.close()
    with open(vis_dir / "class_distribution_report.txt", "w") as f:
        f.write("Class Distribution Analysis:\n")
        f.write("The dataset shows a distribution of land cover classes. ")
        if class_counts.max() / max(class_counts.min(), 1) > 3:
            f.write("There is an imbalance, with some classes (e.g., " + class_counts.idxmax() + ") dominating, which may bias model training. Oversampling or weighting could be considered.\n")
        else:
            f.write("The distribution appears relatively balanced, supporting fair model training.\n")

# Q3: Model Training & Supervised Evaluation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_df = train_df[~train_df["label"].isin(["No-data", "Out-of-bounds"])].reset_index(drop=True)
test_df = test_df[~test_df["label"].isin(["No-data", "Out-of-bounds"])].reset_index(drop=True)
train_df.to_csv(dataset_dir / "train_dataset.csv", index=False)
test_df.to_csv(dataset_dir / "test_dataset.csv", index=False)

print(f"Number of training samples: {len(train_df)}")
print(f"Number of test samples: {len(test_df)}")
if len(train_df) == 0 or len(test_df) == 0:
    print("Warning: No valid samples after filtering invalid labels. Check coordinate alignment with land_cover.tif.")
    with open(vis_dir / "training_warning.txt", "w") as f:
        f.write("No valid samples for training or testing. Please verify that image coordinates align with the land_cover.tif extent.")
    exit(0)

train_dataset = ImageDataset(train_df, image_dir, transform=transform)
test_dataset = ImageDataset(test_df, image_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(train_dataset.label_to_idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Multi-class F1 Score (sklearn and torchmetrics)
from sklearn.metrics import f1_score
custom_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"Custom F1 Score: {custom_f1}")

f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=len(train_dataset.label_to_idx), average="macro").to(device)
f1_torch = f1_metric(torch.tensor(all_preds, device=device), torch.tensor(all_labels, device=device)).item()
print(f"Torchmetrics F1 Score: {f1_torch}")

# Confusion Matrix (fix ticklabels mismatch)
cm = confusion_matrix(all_labels, all_preds)
labels_sorted = [train_dataset.idx_to_label[i] for i in range(len(train_dataset.idx_to_label))]
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(vis_dir / "confusion_matrix.pdf")
plt.close()

# Visualize predictions
def visualize_predictions(model, loader, idx_to_label, num_samples=5, correct=True):
    model.eval()
    preds, labels, imgs = [], [], []
    with torch.no_grad():
        for images, lbls in loader:
            images, lbls = images.to(device), lbls.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
            imgs.extend(images.cpu().numpy())
    matches = [i for i in range(len(preds)) if (preds[i] == labels[i]) == correct][:num_samples]
    if not matches:
        print("No samples to visualize.")
        return
    fig, axes = plt.subplots(1, len(matches), figsize=(len(matches) * 3, 3))
    if len(matches) == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        img = imgs[matches[idx]].transpose((1, 2, 0))
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        ax.imshow(img)
        pred_label = idx_to_label[preds[matches[idx]]]
        true_label = idx_to_label[labels[matches[idx]]]
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}")
        ax.axis("off")
    plt.savefig(vis_dir / f"{'correct' if correct else 'incorrect'}_predictions.pdf")
    plt.close()

visualize_predictions(model, test_loader, train_dataset.idx_to_label, correct=True)
visualize_predictions(model, test_loader, train_dataset.idx_to_label, correct=False)
