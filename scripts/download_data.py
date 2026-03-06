"""Generate synthetic LiDAR point cloud for downtown Manhattan."""
import numpy as np, laspy, os

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA, exist_ok=True)
np.random.seed(42)

print("Generating synthetic LiDAR (500m x 500m, Manhattan)...")
CX, CY = 583960.0, 4507523.0; HALF = 250.0

# Ground
n_g = 100000
gx = np.random.uniform(-HALF, HALF, n_g); gy = np.random.uniform(-HALF, HALF, n_g)
gz = 2.0 + 0.5*np.sin(gx/80)*np.cos(gy/80) + np.random.normal(0, 0.15, n_g)
gz = np.clip(gz, 0, 5)

# Buildings
bld_x, bld_y, bld_z = [], [], []
for bx in np.arange(-HALF+30, HALF-30, 60):
    for by in np.arange(-HALF+30, HALF-30, 60):
        if np.random.random() < 0.65:
            w, d = np.random.uniform(15, 45), np.random.uniform(15, 45)
            h = np.random.uniform(20, 100) if np.sqrt(bx**2+by**2) < 150 else np.random.uniform(15, 60)
            n_b = 3000
            bld_x.extend(np.random.uniform(bx-w/2, bx+w/2, n_b))
            bld_y.extend(np.random.uniform(by-d/2, by+d/2, n_b))
            bld_z.extend(h + np.random.normal(0, 0.3, n_b))
bld_x, bld_y, bld_z = np.array(bld_x), np.array(bld_y), np.array(bld_z)

# Vegetation
vx, vy, vz = [], [], []
for _ in range(200):
    tx, ty = np.random.uniform(-HALF+10, HALF-10), np.random.uniform(-HALF+10, HALF-10)
    tr, th = np.random.uniform(3, 8), np.random.uniform(5, 15)
    n_t = 500
    theta = np.random.uniform(0, 2*np.pi, n_t); r = tr*np.random.uniform(0.3, 1, n_t)**0.5
    vx.extend(tx + r*np.cos(theta)); vy.extend(ty + r*np.sin(theta))
    vz.extend(th + tr*np.random.uniform(-0.5, 0.5, n_t))
vx, vy, vz = np.array(vx), np.array(vy), np.array(vz)

# Noise
n_n = 20000
nx = np.random.uniform(-HALF, HALF, n_n); ny = np.random.uniform(-HALF, HALF, n_n)
nz = np.random.uniform(-5, 150, n_n)

x = np.concatenate([gx, bld_x, vx, nx]) + CX
y = np.concatenate([gy, bld_y, vy, ny]) + CY
z = np.concatenate([gz, bld_z, vz, nz])
cls = np.concatenate([np.full(n_g,2,dtype=np.uint8), np.full(len(bld_x),6,dtype=np.uint8),
                      np.full(len(vx),3,dtype=np.uint8), np.full(n_n,7,dtype=np.uint8)])

header = laspy.LasHeader(point_format=1, version="1.4")
header.offsets = [CX-HALF, CY-HALF, 0]; header.scales = [0.001, 0.001, 0.001]
las = laspy.LasData(header)
las.x, las.y, las.z, las.classification = x, y, z, cls
las.intensity = np.random.randint(500, 5000, len(x)).astype(np.uint16)
las.return_number = np.ones(len(x), dtype=np.uint8)
las.number_of_returns = np.ones(len(x), dtype=np.uint8)

out = os.path.join(DATA, "manhattan_lidar.laz")
las.write(out)
print(f"  {len(x):,} points, {os.path.getsize(out)/1e6:.1f} MB")
print(f"  Ground:{n_g} Building:{len(bld_x)} Veg:{len(vx)} Noise:{n_n}")
print("Done!")
