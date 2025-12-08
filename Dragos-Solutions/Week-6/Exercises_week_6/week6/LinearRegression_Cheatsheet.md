# Linear Regression - Exam Cheatsheet

## 🎯 What is Linear Regression?
- Finds the **best-fit line** (or plane) through data points
- Minimizes the **squared errors** between predicted and actual values
- Formula: `y = β₀ + β₁*x` (2D) or `z = β₀ + β₁*x + β₂*y` (3D)

---

## ⚡ Quick Code Pattern

### 2D Line Fitting
```python
# Add column of 1s for intercept
X = np.column_stack([np.ones(len(x)), x])  # Shape: (N, 2)
y = y.reshape(-1, 1)  # Shape: (N, 1)

# OLS formula: β* = (X^T X)^-1 X^T y
beta = np.linalg.inv(X.T @ X) @ X.T @ y

# beta[0] = intercept, beta[1] = slope
print(f"y = {beta[0, 0]:.4f} + {beta[1, 0]:.4f}*x")
```

### 3D Plane Fitting
```python
# xy = (N, 2), z = (N, 1)
X = np.column_stack([np.ones(len(xy)), xy])  # Shape: (N, 3)

# OLS formula
beta = np.linalg.inv(X.T @ X) @ X.T @ z

# Plane: z = β₀ + β₁*x + β₂*y
print(f"z = {beta[0]:.4f} + {beta[1]:.4f}*x + {beta[2]:.4f}*y")

# Plot plane
x_range = np.linspace(x.min(), x.max(), 20)
y_range = np.linspace(y.min(), y.max(), 20)
xx, yy = np.meshgrid(x_range, y_range)
zz = beta[0] + beta[1]*xx + beta[2]*yy

ax.plot_surface(xx, yy, zz, alpha=0.3)
```

---

## 📐 OLS Formula (The Only One You Need)

### Matrix Formula
```
β* = (X^T X)^-1 X^T y
```

Where:
- `X` = data matrix with column of 1s added (for intercept)
- `y` = target values
- `β*` = optimal parameters [intercept, slope(s)]

### Alternative (2D only)
```python
β = Cov(X, Y) / Var(X)
α = mean(Y) - β * mean(X)
```

---

## 📊 Key Concepts

### What OLS Does
- **Minimizes**: Sum of squared errors (SSE)
- **Error**: Distance from point to line/plane
- **Best fit**: Line/plane with smallest total squared error

### Matrix Shapes
```
2D: X is (N, 2)    β is (2, 1)    y is (N, 1)
3D: X is (N, 3)    β is (3, 1)    z is (N, 1)
```

### The X Matrix
```
2D: X = [1  x₁]    First column = all 1s (intercept)
        [1  x₂]    Second column = x values
        [1  x₃]
        ...

3D: X = [1  x₁  y₁]    First column = all 1s
        [1  x₂  y₂]    Second = x values
        [1  x₃  y₃]    Third = y values
        ...
```

---

## 🎓 Exam Questions

**Q: What is OLS?**
A: Ordinary Least Squares - method that finds best-fit line by minimizing sum of squared errors

**Q: Why add column of 1s to X?**
A: For the intercept term (β₀). Without it, line passes through origin.

**Q: What does β contain?**
A: β[0] = intercept, β[1] = first slope, β[2] = second slope (if 3D)

**Q: How to check if fit is good?**
A: Calculate R² score (closer to 1 = better fit)

**Q: 2D vs 3D difference?**
A: 2D fits a line (y = β₀ + β₁*x), 3D fits a plane (z = β₀ + β₁*x + β₂*y)

---

## 💡 Common Patterns

### R² Score (Goodness of Fit)
```python
y_pred = X @ beta
ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
r2 = 1 - (ss_res / ss_tot)  # 1 = perfect, 0 = bad
```

### Predict New Values
```python
# 2D
new_x = np.array([[1, 5]])  # [1, x_value]
y_pred = new_x @ beta

# 3D
new_xy = np.array([[1, 5, 3]])  # [1, x_value, y_value]
z_pred = new_xy @ beta
```

### Compare with sklearn
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")
```

---

## 🚨 Common Mistakes

❌ Forgetting to add column of 1s
✅ Always: `X = np.column_stack([np.ones(len(x)), x])`

❌ Wrong shape for y
✅ Always: `y = y.reshape(-1, 1)`

❌ Transposing wrong matrix
✅ Remember: `X.T @ X` not `X @ X.T`

❌ Using wrong @ order
✅ Correct: `np.linalg.inv(X.T @ X) @ X.T @ y`

---

## ✅ Quick Checklist

- [ ] Know the OLS formula: `β* = (X^T X)^-1 X^T y`
- [ ] Add column of 1s for intercept
- [ ] Reshape y to (N, 1)
- [ ] For 3D: X has 3 columns [1s, x, y]
- [ ] β[0] = intercept, β[1] = slope (2D)
- [ ] β[0] = intercept, β[1] = x-slope, β[2] = y-slope (3D)
- [ ] Use meshgrid for plotting 3D planes
- [ ] R² score measures fit quality

---

## 📝 Exam Template (Copy This!)

```python
# 2D Linear Regression
X = np.column_stack([np.ones(len(x)), x])
y = y.reshape(-1, 1)
beta = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"Line: y = {beta[0, 0]:.4f} + {beta[1, 0]:.4f}*x")

# 3D Plane Fitting
X = np.column_stack([np.ones(len(xy)), xy])
z = z.reshape(-1, 1)
beta = np.linalg.inv(X.T @ X) @ X.T @ z
print(f"Plane: z = {beta[0]:.4f} + {beta[1]:.4f}*x + {beta[2]:.4f}*y")

# Plot 3D plane
x_range = np.linspace(x.min(), x.max(), 20)
y_range = np.linspace(y.min(), y.max(), 20)
xx, yy = np.meshgrid(x_range, y_range)
zz = beta[0] + beta[1]*xx + beta[2]*yy
ax.plot_surface(xx, yy, zz, alpha=0.3)
```

---

## 🧮 The Math (Simple Version)

**Goal**: Find line `y = β₀ + β₁*x` that minimizes errors

**Error**: `e = y_actual - y_predicted`

**Objective**: Minimize `Σ(e²)` = minimize sum of squared errors

**Solution**: `β* = (X^T X)^-1 X^T y` gives optimal β₀ and β₁

**Why squared?**: 
- Penalizes big errors more
- Makes math easier (no absolute values)
- Always positive

---

## 🎯 Key Takeaways

1. **OLS finds best-fit line/plane** by minimizing squared errors
2. **Always add column of 1s** to X for intercept
3. **Matrix formula works for any dimension**: 2D, 3D, 4D, etc.
4. **β[0] is always intercept**, rest are slopes
5. **R² measures quality**: 1 = perfect, 0 = bad
6. **Same formula for 2D and 3D**, just change X shape

