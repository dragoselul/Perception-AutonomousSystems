# Perception for Autonomous Systems - Exam Theory Summary

## 1. Image Processing & Convolution

### Convolution Kernels
| Kernel Type | Effect | Example |
|-------------|--------|---------|
| **Identity** | No change | `[[0,0,0],[0,1,0],[0,0,0]]` |
| **Shift** | Moves image | `1` in off-center position shifts opposite direction |
| **Box/Mean** | Blur (averaging) | All same values, sum to 1 |
| **Gaussian** | Smooth blur | Bell-curve weights |
| **Sharpening** | Edge enhancement | Negative neighbors, large positive center |

### Key Concepts:
- **Convolution vs Correlation**: 
  - Convolution: kernel is **flipped** (180°) before sliding
  - Correlation: kernel slides directly (no flip)
  - **Symmetric kernels** → identical results for both operations
  - You CAN perform correlation using convolution by flipping the kernel

- **Linear smoothing filter** → outputs **average** of pixels
- **Non-linear smoothing filter** (e.g., median) → based on **ranking** pixel values

### Shift Kernel Rules:
```
Position of 1 in kernel → Image shifts OPPOSITE direction
[0,0,0]
[0,0,0]   → Image shifts LEFT (1 is on right)
[0,0,1]

[0,1,0]
[0,0,0]   → Image shifts DOWN (1 is on top)
[0,0,0]
```

---

## 2. Feature Detection

### Harris Corner Detector
- **Purpose**: Detect **corners** (NOT edges)
- Uses eigenvalues (λ₁, λ₂) of the **Structure Tensor M** (NOT Hessian!)

| Condition | Classification |
|-----------|---------------|
| λ₁ ≈ λ₂ and **both large** | **Corner** |
| λ₁ >> λ₂ or λ₂ >> λ₁ | **Edge** |
| λ₁ ≈ λ₂ ≈ 0 | **Flat region** |

### SIFT (Scale-Invariant Feature Transform)
| Step | Method |
|------|--------|
| **Detection** | Difference of Gaussians (DoG) |
| **Description** | Gradient orientations |
| **Descriptor size** | **128 values** (not 64!) |

Structure: 4×4 grid × 8 orientation bins = 128

### Hough Transform
- Used for detecting **lines** and **circles**
- Is a **model fitting algorithm**
- **CAN handle outliers** well
- NOT only used in computer vision
- Use **RANSAC** instead when feature dimension is high

---

## 3. Template Matching

### SAD (Sum of Absolute Differences)
```
SAD = Σ |template[i] - image[i]|
```
- **Lower SAD = better match** (0 = perfect match)
- Slide template over signal/image, compute SAD at each position

### SSD (Sum of Squared Differences)
```
SSD = Σ (template[i] - image[i])²
```

---

## 4. Optical Flow

### Sparse Optical Flow (Lucas-Kanade)
- Tracks **specific features/keypoints**
- Uses `cv2.calcOpticalFlowPyrLK()`
- Input: previous image, current image, previous points

### Dense Optical Flow (Farneback)
- Computes flow for **every pixel**
- Uses `cv2.calcOpticalFlowFarneback()`

---

## 5. Stereo Vision

### Rectified Stereo
| Statement | True/False |
|-----------|------------|
| Epipolar geometry applies to both rectified and unrectified | **TRUE** |
| Rectified is computationally simpler | **TRUE** |
| Can be achieved by mounting sensors on common plane | **TRUE** |
| Epipoles exist at infinity (parallel epipolar lines) | **TRUE** |
| Disparity grows with depth | **FALSE** (disparity INVERSELY proportional to depth!) |
| Need focal length and baseline for depth | **TRUE** |

### Depth Formula
```
depth = (focal_length × baseline) / disparity
```
**Higher disparity = closer object (smaller depth)**

### Stereo Matching
- **Local algorithms**: faster, but inferior quality
- **Global algorithms**: better quality, more computation
- **Bigger windows**: smoother results, lose fine detail, NOT always better!
- **Smaller windows**: better for fine texture, more noise
- **Convolution CAN be used** for similarity (symmetric kernel)

---

## 6. Camera Calibration & Projection

### Intrinsic vs Extrinsic Parameters
| Type | What it describes | Matrix |
|------|------------------|--------|
| **Intrinsic** | Camera internal (focal length, principal point) | K (3×3) |
| **Extrinsic** | Camera pose in world (rotation, translation) | [R\|t] (3×4) |

### Projection Matrix
```
P = K × [R|t]    (3×4 matrix)
```
- **Includes BOTH intrinsic AND extrinsic parameters**

### Key Matrices
| Matrix | Size | Contains |
|--------|------|----------|
| **Homography (H)** | 3×3 | Projects points on a **plane** (2D→2D) |
| **Fundamental (F)** | 3×3 | **Intrinsic + Extrinsic** |
| **Essential (E)** | 3×3 | **Only Extrinsic** (calibrated cameras) |

### Important Facts:
- **Homography**: Used for flat calibration patterns
- **Fundamental Matrix**: Projects point to epipolar **LINE** (not point!)
- **Epipoles**: Intersection of **baseline with image planes**
- **Lens distortion**: Modeled as **polynomial**

---

## 7. RANSAC

### Formula for Iterations
```
k = log(1 - p) / log(1 - w^n)
```
- k = iterations needed
- p = success probability (e.g., 0.98)
- w = inlier ratio
- n = min points for model (2 for line)

### Key Property:
**Iterations INDEPENDENT of dataset size!**
- Doubling dataset with same inlier ratio → **same iterations**

---

## 8. Point Cloud Registration

### ICP (Iterative Closest Point)
| Statement                                   | True/False |
|---------------------------------------------|------------|
| ICP is deterministic                        | **TRUE**   |
| Always converges regardless of initial pose | **FALSE**  |
| Kabsch IMPLEMENTS ICP                       | **TRUE**   |
| Transformations applied to same point cloud | **TRUE**   |
| Can use FPFH descriptors                    | **TRUE**   |
| Not robust to outliers                      | **TRUE**   |

### Alignment Order
**Global alignment FIRST → Local alignment SECOND**
(Statement "local first" is WRONG)

### Kabsch Algorithm
- Used for **Global alignment**
- Uses SVD for optimal rigid transformation

---

## 9. Clustering & Dimensionality Reduction

### K-Means
- **Elbow method**: Plot inertia vs K, find "elbow"
- **Inertia**: Sum of squared distances to centers

### PCA
- Choose components summing to desired variance (e.g., 95%)

### DBSCAN
- **DOES assign noise points** (points not in any cluster)
- Doesn't need number of clusters

---

## 10. State Estimation

### Histogram Filter vs Kalman Filter
| Property | Histogram Filter | Kalman Filter |
|----------|-----------------|---------------|
| State type | **Discrete** | **Continuous** |
| Distribution | **Multimodal** | **Unimodal (Gaussian)** |
| Measurement | Bayes rule | Kalman equations |
| Motion | Convolution | State prediction |

### Kalman Filter Variance:
- **Prediction** → Variance **INCREASES** ↑
- **Update/Measurement** → Variance **DECREASES** ↓

### Constant Velocity Model
```
State: [x, ẋ, y, ẏ]ᵀ

F = [1  dt  0   0 ]
    [0  1   0   0 ]
    [0  0   1   dt]
    [0  0   0   1 ]
```

---

## 11. Visual Odometry

### Methods Comparison
| Method | Uses | Accuracy |
|--------|------|----------|
| **2D-to-2D** | Essential matrix | Up to **scale only** |
| **3D-to-2D** | **PnP** | Full pose |
| **3D-to-3D** | Point matching | **Less accurate** |

### Key Facts:
| Statement | True/False                   |
|-----------|------------------------------|
| 3D-to-3D less accurate than 3D-to-2D | **TRUE**                     |
| Axis-angle has gimbal lock | **FALSE** (Euler angles do!) |
| Only frame-to-frame | **TRUE**                     |
| 2D-to-2D is "up to scale" | **TRUE**                     |
| 3D-to-2D uses PnP | **TRUE**                     |

---

## 12. SLAM

### Key Concepts:
| Statement | True/False |
|-----------|------------|
| Only identifies camera position | **FALSE** (also builds map!) |
| Graph: nodes=poses, edges=observations | **TRUE** |
| Goal: satisfy all constraints | **TRUE** |

### Optimization Methods
| Method | Problem |
|--------|---------|
| **Gradient Descent** | Slow convergence |
| **Gauss-Newton** | Overshooting |
| **Levenberg-Marquardt** | Combines both (best) |

---

## 13. Classification

### k-NN
- **Training**: Almost **ZERO** time (just stores data)
- **Testing**: **SLOW** (compares all samples)
- "Similar training/testing time" is **WRONG**

---

## 14. Quick Reference

### Depth from Stereo
```
Z = (f × B) / d
```

### Camera Projection
```
pixel = K × [R|t] × world_point
```

### Rotation Representations
| Type | Gimbal Lock? |
|------|-------------|
| Euler Angles | **YES** |
| Axis-Angle | **NO** |
| Quaternion | **NO** |

---

## ⚠️ Common Exam Traps

1. **Disparity is INVERSELY proportional to depth**
2. **SIFT = 128 values** (not 64)
3. **Harris uses Structure Tensor** (not Hessian)
4. **Axis-angle has NO gimbal lock** (Euler does!)
5. **k-NN: fast training, slow testing**
6. **RANSAC iterations independent of dataset size**
7. **Global alignment BEFORE local**
8. **Fundamental matrix: intrinsic + extrinsic**
9. **Essential matrix: only extrinsic**
10. **Kalman: prediction ↑variance, update ↓variance**

