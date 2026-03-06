# GPU 3D Point Cloud Analysis

GPU-accelerated LiDAR point cloud processing combining **NVIDIA CuPy** with **ArcGIS Maps SDK for JavaScript 5.0**. Processes 500K synthetic LiDAR points over downtown Manhattan.

## Live Demo

**[View Live App](https://garridolecca.github.io/gpu-pointcloud-3d/)**

## GPU Analytics Pipeline

| Analysis | Method | Description |
|---|---|---|
| **Point Classification** | GPU Morphological | Ground/Building/Vegetation/Noise classification |
| **Building Footprints** | DBSCAN + Hull | Cluster extraction with convex hull boundaries |
| **DEM Surface** | GPU IDW | Digital Elevation Model from ground points |
| **Slope Analysis** | GPU Gradient | Terrain gradient computation |
| **Tree Detection** | GPU Local Max | Canopy Height Model local maxima detection |
| **Terrain Profile** | Cross Section | E-W elevation cross section |

## Tech Stack

- **GPU Compute**: NVIDIA RTX A4000 + CuPy (CUDA-accelerated NumPy)
- **Visualization**: ArcGIS Maps SDK for JavaScript 5.0 (Web Components)
- **Data**: 500K synthetic LiDAR points (LAZ format)
- **Region**: Downtown Manhattan (500m x 500m)

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_analytics.py
```

Then open `webapp/index.html` or deploy to GitHub Pages.

## Results

- **500,000 LiDAR Points** classified into 4 classes
- **Building Footprints** extracted via DBSCAN clustering
- **DEM/Slope** computed via GPU IDW interpolation
- **Tree Detection** via GPU local maxima on Canopy Height Model
