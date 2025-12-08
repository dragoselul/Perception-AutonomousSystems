# ICP - Exam Cheatsheet (Ultra Simple)

## 🎯 What is ICP?
- **ICP** = Iterative Closest Point
- Aligns two 3D point clouds by minimizing distances
- Needs a **good starting position** (unlike RANSAC)

---

## ⚡ Quick Code Pattern

```python
# 1. Load point clouds
source = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)

# 2. Estimate normals (for Point-to-Plane only)
source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 3. Run ICP
result = o3d.pipelines.registration.registration_icp(
    source, target,
    threshold=0.02,                    # Max distance
    trans_init=np.eye(4),             # Initial guess
    TransformationEstimationPointToPoint())  # Or PointToPlane()

# 4. Check quality
print(f"Fitness: {result.fitness:.4f}")  # Higher = better
print(f"RMSE: {result.inlier_rmse:.4f}") # Lower = better

# 5. Apply transformation
source.transform(result.transformation)
```

---

## 📊 Key Parameters

| Parameter | What it does | Typical value |
|-----------|--------------|---------------|
| `threshold` | Max distance for matches | 0.02 (close) → 0.05 (far) |
| `trans_init` | Starting transformation | `np.eye(4)` (identity) |
| `max_iteration` | How many iterations | 30-100 |

---

## 🆚 Two Methods

### Point-to-Point (Simple)
```python
TransformationEstimationPointToPoint()
```
- ✅ No normals needed
- ✅ Faster
- ❌ Less accurate

### Point-to-Plane (Better)
```python
TransformationEstimationPointToPlane()
```
- ✅ More accurate
- ❌ Needs normals
- ❌ Slightly slower

**Always use Point-to-Plane when you can!**

---

## 💡 How ICP Works (4 Steps)

1. **Find closest points** in target for each source point
2. **Calculate transformation** to align them
3. **Move source** by that transformation
4. **Repeat** until it stops improving

---

## 🎓 Exam Answers

**Q: RANSAC vs ICP?**
- RANSAC = Find rough alignment from scratch
- ICP = Refine an existing alignment

**Q: When does ICP fail?**
- When point clouds are too far apart
- Solution: Increase threshold or use RANSAC first

**Q: What is fitness?**
- % of source points that match target (0 to 1)
- Good = > 0.8, Bad = < 0.5

**Q: What is RMSE?**
- Average error of aligned points
- Good = < 0.01, Bad = > 0.05

**Q: Why estimate normals?**
- Point-to-Plane needs surface directions for better accuracy

---

## 🏗️ 3D Reconstruction Pattern

```python
# Start with first frame
accumulated = first_point_cloud

# Add more frames
for frame in frames:
    new = load_frame(frame)
    
    # Estimate normals
    accumulated.estimate_normals(...)
    new.estimate_normals(...)
    
    # Align
    result = registration_icp(new, accumulated, 0.05, np.eye(4),
                             TransformationEstimationPointToPlane())
    
    # Merge
    new.transform(result.transformation)
    accumulated = accumulated + new
    
    # Downsample (every few frames)
    accumulated = accumulated.voxel_down_sample(0.01)
```

---

## ✅ Checklist

- [ ] Know the basic ICP code pattern
- [ ] Understand threshold (distance for matches)
- [ ] Know Point-to-Point vs Point-to-Plane
- [ ] Remember: estimate normals for Point-to-Plane
- [ ] Fitness > 0.8 = good, < 0.5 = bad
- [ ] RMSE lower = better
- [ ] ICP needs good initial alignment (unlike RANSAC)
- [ ] Combine point clouds with `+` operator
- [ ] Downsample with `voxel_down_sample()`

---

## 🚨 Common Mistakes

❌ Forgetting to estimate normals for Point-to-Plane
✅ Always call `estimate_normals()` first

❌ Using too small threshold for distant frames
✅ Increase to 0.05 or 0.1

❌ Trying ICP on completely misaligned clouds
✅ Use RANSAC first for initial alignment

❌ Not downsampling when merging many frames
✅ Downsample every 5-10 frames

---

## 📏 Good Values

```python
# Close frames (5 frame gap)
threshold = 0.02

# Medium frames (50 frame gap)
threshold = 0.05

# Distant frames (300 frame gap)
threshold = 0.1

# Normal estimation
radius = 0.1
max_nn = 30

# Downsampling
voxel_size = 0.01  # or 0.02
```

