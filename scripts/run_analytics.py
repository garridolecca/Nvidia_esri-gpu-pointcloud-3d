"""GPU-Accelerated 3D Point Cloud Analysis Pipeline."""
import os, json, time, numpy as np, laspy
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

try:
    import cupy as cp; GPU = True; print(f"GPU: CuPy {cp.__version__}")
except ImportError: GPU = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output"); WEB = os.path.join(BASE, "webapp", "data")
os.makedirs(OUT, exist_ok=True); os.makedirs(WEB, exist_ok=True)

CX, CY = 583960.0, 4507523.0; CLON, CLAT = -74.006, 40.7128
MLON, MLAT = 85390.0, 111320.0

def to_ll(x, y): return CLON + (x-CX)/MLON, CLAT + (y-CY)/MLAT
def save(data, name):
    for d in [OUT, WEB]:
        with open(os.path.join(d, name), "w") as f: json.dump(data, f)

las = laspy.read(os.path.join(DATA, "manhattan_lidar.laz"))
x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)
cls = np.array(las.classification, dtype=np.uint8)
xl, yl = x - CX, y - CY
N = len(x)
print(f"Loaded {N:,} points, Z:[{z.min():.1f},{z.max():.1f}]")

# Building detection
print("\n=== Building Footprints ===")
t0 = time.time()
bm = cls == 6; bx, by, bz = xl[bm], yl[bm], z[bm]
sub = np.random.choice(len(bx), min(50000, len(bx)), replace=False)
db = DBSCAN(eps=8, min_samples=20).fit(np.column_stack([bx[sub], by[sub]]))
n_bld = len(set(db.labels_) - {-1})
bld_feats, heights = [], []
for lbl in range(n_bld):
    m = db.labels_ == lbl
    if m.sum() < 10: continue
    pts = np.column_stack([bx[sub][m], by[sub][m]])
    h = float(np.percentile(bz[sub][m], 95)); heights.append(h)
    try:
        hull = ConvexHull(pts)
        coords = [[float(lo), float(la)] for lo, la in [to_ll(p[0]+CX, p[1]+CY) for p in pts[hull.vertices]]]
        coords.append(coords[0])
        bld_feats.append({"type":"Feature","geometry":{"type":"Polygon","coordinates":[coords]},
                         "properties":{"id":int(lbl),"height_m":round(h,1),"area_m2":round(float(hull.volume),1),"points":int(m.sum())}})
    except: continue
save({"type":"FeatureCollection","features":bld_feats}, "building_footprints.geojson")
print(f"  {len(bld_feats)} buildings ({time.time()-t0:.1f}s)")

# DEM via GPU IDW
print("\n=== GPU DEM ===")
t0 = time.time()
gm = cls == 2; gnd_x, gnd_y, gnd_z = xl[gm], yl[gm], z[gm]
sub_g = np.random.choice(len(gnd_x), min(10000, len(gnd_x)), replace=False)
dem_n = 50; dem_x = np.linspace(-250, 250, dem_n); dem_y = np.linspace(-250, 250, dem_n)

if GPU:
    gxf = cp.asarray(np.tile(dem_x, dem_n).astype(np.float64))
    gyf = cp.asarray(np.repeat(dem_y, dem_n).astype(np.float64))
    kx, ky, kz = cp.asarray(gnd_x[sub_g]), cp.asarray(gnd_y[sub_g]), cp.asarray(gnd_z[sub_g])
    dx = gxf[:, None] - kx[None, :]; dy = gyf[:, None] - ky[None, :]
    dist = cp.maximum(cp.sqrt(dx**2 + dy**2), 0.1)
    w = 1.0 / dist**2
    dem = cp.asnumpy(cp.sum(w * kz[None, :], axis=1) / cp.sum(w, axis=1)).reshape(dem_n, dem_n)
    slope_y, slope_x = cp.gradient(cp.asarray(dem), 10.0)
    slope = cp.asnumpy(cp.degrees(cp.arctan(cp.sqrt(slope_x**2 + slope_y**2))))
else:
    gxf = np.tile(dem_x, dem_n); gyf = np.repeat(dem_y, dem_n)
    dx = gxf[:, None] - gnd_x[sub_g][None, :]; dy = gyf[:, None] - gnd_y[sub_g][None, :]
    dist = np.maximum(np.sqrt(dx**2 + dy**2), 0.1); w = 1.0 / dist**2
    dem = (np.sum(w * gnd_z[sub_g][None, :], axis=1) / np.sum(w, axis=1)).reshape(dem_n, dem_n)
    sy, sx = np.gradient(dem, 10.0); slope = np.degrees(np.arctan(np.sqrt(sx**2 + sy**2)))

print(f"  DEM: {dem_n}x{dem_n}, elev:[{dem.min():.1f},{dem.max():.1f}] ({time.time()-t0:.1f}s)")

# Export DEM + slope
dem_f, slope_f = [], []
for i in range(dem_n):
    for j in range(dem_n):
        lon, lat = to_ll(dem_x[j]+CX, dem_y[i]+CY)
        dem_f.append({"type":"Feature","geometry":{"type":"Point","coordinates":[float(lon),float(lat)]},
                     "properties":{"elevation_m":round(float(dem[i,j]),2)}})
        slope_f.append({"type":"Feature","geometry":{"type":"Point","coordinates":[float(lon),float(lat)]},
                       "properties":{"slope_deg":round(float(slope[i,j]),2),"elevation_m":round(float(dem[i,j]),2)}})
save({"type":"FeatureCollection","features":dem_f}, "dem_surface.geojson")
save({"type":"FeatureCollection","features":slope_f}, "slope_map.geojson")

# Tree detection
print("\n=== GPU Tree Detection ===")
t0 = time.time()
vm = cls == 3
# Simple: find high vegetation points and cluster them
veg_z = z[vm]; veg_xl, veg_yl = xl[vm], yl[vm]
high_v = veg_z > 8
if high_v.sum() > 100:
    v_sub = np.random.choice(np.where(high_v)[0], min(5000, high_v.sum()), replace=False)
    tree_db = DBSCAN(eps=5, min_samples=5).fit(np.column_stack([veg_xl[v_sub], veg_yl[v_sub]]))
    tree_feats = []
    for lbl in range(len(set(tree_db.labels_) - {-1})):
        m = tree_db.labels_ == lbl
        if m.sum() < 3: continue
        cx_t, cy_t = veg_xl[v_sub][m].mean(), veg_yl[v_sub][m].mean()
        lon, lat = to_ll(cx_t+CX, cy_t+CY)
        ch = float(veg_z[v_sub][m].max() - 2.0)
        tree_feats.append({"type":"Feature","geometry":{"type":"Point","coordinates":[float(lon),float(lat)]},
                          "properties":{"canopy_height_m":round(ch,1),"top_m":round(float(veg_z[v_sub][m].max()),1)}})
else:
    tree_feats = []
save({"type":"FeatureCollection","features":tree_feats}, "tree_locations.geojson")
print(f"  {len(tree_feats)} trees ({time.time()-t0:.1f}s)")

# Sampled classified points
print("\n=== Classified Points Export ===")
sample = np.random.choice(N, min(5000, N), replace=False)
names = {2:"Ground",3:"Vegetation",6:"Building",7:"Noise"}
pt_f = []
for i in sample:
    lon, lat = to_ll(x[i], y[i])
    pt_f.append({"type":"Feature","geometry":{"type":"Point","coordinates":[float(lon),float(lat)]},
                "properties":{"z":round(float(z[i]),1),"class":int(cls[i]),"class_name":names.get(int(cls[i]),"?")}})
save({"type":"FeatureCollection","features":pt_f}, "classified_points.geojson")

# Profile
profile = {"direction":"E-W","points":[]}
mid = dem_n // 2
for j in range(dem_n):
    lon, lat = to_ll(dem_x[j]+CX, dem_y[mid]+CY)
    profile["points"].append({"dist_m":round(float(dem_x[j]+250),1),"ground_m":round(float(dem[mid,j]),2),"lon":float(lon)})
save(profile, "terrain_profile.json")

summary = {"total_points":N,"buildings":len(bld_feats),"trees":len(tree_feats),
           "max_height_m":round(float(max(heights)) if heights else 0,1),
           "dem_grid":[dem_n,dem_n],"gpu":GPU,"area":"500m x 500m Manhattan"}
save(summary, "summary.json")
print(f"\nComplete! {len(bld_feats)} buildings, {len(tree_feats)} trees")
