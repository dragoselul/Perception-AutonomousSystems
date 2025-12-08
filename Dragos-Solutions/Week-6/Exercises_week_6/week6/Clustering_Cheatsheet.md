# K-means & Clustering - Exam Cheatsheet

## 🎯 What is K-means?
- Groups points into **K clusters**
- Each point assigned to nearest cluster center
- Iteratively updates centers until convergence
- Good for spherical clusters, fast

---

## ⚡ Quick Code Pattern

### Basic K-means
```python
from sklearn.cluster import KMeans

# Get point cloud data
xyz = np.asarray(pcl.points)

# Run K-means
km = KMeans(n_clusters=6, random_state=0)
labels = km.fit_predict(xyz)

# labels[i] tells you which cluster point i belongs to
```

### With Normals
```python
# Estimate normals first
pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
normals = np.asarray(pcl.normals)

# Combine xyz + normals
xyz_n = np.concatenate((xyz, normals), axis=1)

# K-means on combined features
km = KMeans(n_clusters=6, random_state=0)
labels = km.fit_predict(xyz_n)
```

### Weighted Features
```python
# Give normals more importance
weight = 5.0
normals_weighted = normals * weight
xyz_n = np.concatenate((xyz, normals_weighted), axis=1)

km = KMeans(n_clusters=6, random_state=0)
labels = km.fit_predict(xyz_n)
```

---

## 🎨 Feature Types

### 1. XYZ (Position)
```python
xyz = np.asarray(pcl.points)
```
- **Good for**: Spatially separated objects
- **Bad for**: Same location, different orientation

### 2. Normals (Surface Direction)
```python
pcl.estimate_normals(...)
normals = np.asarray(pcl.normals)
```
- **Good for**: Different surface orientations (cube faces)
- **Bad for**: Opposite faces have opposite normals

### 3. Colors
```python
colors = np.asarray(pcl.colors)
```
- **Good for**: Textured objects with different colors
- **Bad for**: Monochrome objects

### 4. Combined Features
```python
# Concatenate along axis=1 (columns)
combined = np.concatenate((xyz, normals, colors), axis=1)
```
- **Best**: Use multiple features for better separation

---

## 🆚 K-means vs DBSCAN

### K-means
```python
km = KMeans(n_clusters=6, random_state=0)
labels = km.fit_predict(xyz)
```
- ✅ Fast
- ✅ Simple
- ❌ Need to know K beforehand
- ❌ Only spherical clusters
- ❌ No noise handling

### DBSCAN
```python
labels = np.array(pcl.cluster_dbscan(
    eps=0.02,        # Max distance between neighbors
    min_points=10))  # Min points to form cluster
```
- ✅ Finds arbitrary shapes
- ✅ Handles noise (labels as -1)
- ✅ Auto-determines number of clusters
- ❌ Slower
- ❌ Need to tune eps and min_points

---

## 📊 Key Parameters

### K-means
| Parameter | What it does | Typical value |
|-----------|--------------|---------------|
| `n_clusters` | Number of clusters | Known from problem |
| `random_state` | Seed for reproducibility | 0 |
| `n_init` | Number of initializations | 10 |
| `max_iter` | Max iterations | 300 |

### DBSCAN
| Parameter | What it does | Typical value |
|-----------|--------------|---------------|
| `eps` | Max distance for neighbors | 0.02-0.1 |
| `min_points` | Min points per cluster | 10-50 |

---

## 💡 How K-means Works (4 Steps)

1. **Initialize**: Randomly place K cluster centers
2. **Assign**: Each point goes to nearest center
3. **Update**: Move centers to mean of assigned points
4. **Repeat**: Steps 2-3 until centers stop moving

---

## 🎓 Exam Questions

**Q: When to use K-means?**
A: When you know number of clusters and they're roughly spherical

**Q: When to use DBSCAN?**
A: For arbitrary shapes, noise handling, or unknown K

**Q: Why combine xyz + normals?**
A: Each gives different info - position + orientation = better separation

**Q: What does weighting do?**
A: Multiplying features by weight makes them more important in clustering

**Q: What is eps in DBSCAN?**
A: Maximum distance between points to be considered neighbors

**Q: What are labels?**
A: Array where labels[i] tells which cluster point i belongs to

---

## 🔧 Common Patterns

### Cube Face Segmentation
```python
# XYZ alone - doesn't work well
labels = KMeans(n_clusters=6).fit_predict(xyz)

# Normals alone - opposite faces same
normals = np.asarray(pcl.normals)
labels = KMeans(n_clusters=6).fit_predict(normals)

# XYZ + Normals - works!
xyz_n = np.concatenate((xyz, normals), axis=1)
labels = KMeans(n_clusters=6).fit_predict(xyz_n)
```

### Multiple Shapes
```python
# Spatially separated objects - use XYZ
labels = KMeans(n_clusters=6).fit_predict(xyz)
```

### Textured Objects
```python
# Color-based segmentation
colors = np.asarray(pcl.colors)
xyz_c = np.concatenate((xyz, colors * 3), axis=1)
labels = KMeans(n_clusters=5).fit_predict(xyz_c)
```

---

## 📐 Feature Concatenation

### The Pattern
```python
# np.concatenate along axis=1 (add columns)
feature1 = np.array([[1, 2], [3, 4]])  # Shape: (2, 2)
feature2 = np.array([[5, 6], [7, 8]])  # Shape: (2, 2)

combined = np.concatenate((feature1, feature2), axis=1)
# Result shape: (2, 4)
# [[1, 2, 5, 6],
#  [3, 4, 7, 8]]
```

### For Point Clouds
```python
xyz = np.asarray(pcl.points)        # (N, 3)
normals = np.asarray(pcl.normals)   # (N, 3)
colors = np.asarray(pcl.colors)     # (N, 3)

# Combine all
all_features = np.concatenate((xyz, normals, colors), axis=1)  # (N, 9)
```

---

## 🚨 Common Mistakes

❌ Wrong axis in concatenate
✅ Always use `axis=1` to add columns

❌ Forgetting to estimate normals
✅ Call `pcl.estimate_normals()` before getting normals

❌ Using same weight for all features
✅ Scale features based on importance (e.g., normals * 5)

❌ Wrong number of clusters
✅ Count objects/faces manually first

---

## ✅ Quick Checklist

- [ ] Import: `from sklearn.cluster import KMeans`
- [ ] Get data: `xyz = np.asarray(pcl.points)`
- [ ] Estimate normals: `pcl.estimate_normals(...)`
- [ ] Concatenate features: `np.concatenate((f1, f2), axis=1)`
- [ ] Run K-means: `KMeans(n_clusters=K).fit_predict(data)`
- [ ] Or DBSCAN: `pcl.cluster_dbscan(eps, min_points)`
- [ ] Visualize: Use provided `draw_labels_on_model()`

---

## 📝 Exam Template

```python
# 1. Load/create point cloud
pcl = o3d.geometry.TriangleMesh.create_box().sample_points_uniformly(10000)

# 2. Get xyz
xyz = np.asarray(pcl.points)

# 3. Estimate normals
pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
normals = np.asarray(pcl.normals)

# 4. Combine features
xyz_n = np.concatenate((xyz, normals * 5), axis=1)

# 5. Run K-means
km = KMeans(n_clusters=6, random_state=0)
labels = km.fit_predict(xyz_n)

# 6. Visualize
draw_labels_on_model(pcl, labels)
```

---

## 🎯 Feature Selection Guide

| Object Type | Best Features | Why |
|-------------|---------------|-----|
| Cube faces | XYZ + Normals | Position + orientation |
| Separate objects | XYZ | Spatial distance |
| Textured surfaces | XYZ + Colors | Color differences |
| Complex geometry | All 3 | Maximum info |

---

## 💭 Key Insights

**Why XYZ + Normals?**
- XYZ: Where is it?
- Normals: Which way does it face?
- Together: Unique signature per face

**Why Weighting?**
- Features have different scales
- Weighting controls importance
- Normal * 5 = normals 5x more important than xyz

**K-means Limitation**
- Assumes spherical clusters
- All clusters roughly same size
- Need to know K beforehand

**DBSCAN Advantage**
- Finds any shape
- Automatic K
- Handles outliers (noise)

---

## 🔑 Key Takeaways

1. **K-means**: Fast, simple, fixed K clusters
2. **DBSCAN**: Flexible shapes, auto K, noise handling
3. **Features matter**: XYZ, normals, colors all useful
4. **Combine features**: `np.concatenate(..., axis=1)`
5. **Weight features**: Multiply to control importance
6. **Estimate normals**: Always needed for surface info
7. **Tune parameters**: eps/min_points (DBSCAN), weights (K-means)

