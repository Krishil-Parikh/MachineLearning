# Huber Regression (From Scratch)

## 1. Overview

Huber Regression is a **robust regression technique** designed to handle **outliers** better than standard Linear Regression.

It combines the strengths of:
- **Mean Squared Error (MSE)** for small errors (smooth optimization)
- **Mean Absolute Error (MAE)** for large errors (outlier resistance)

This makes Huber Regression especially useful when the dataset contains **noisy labels or extreme values**.

---

## 2. Linear Regression Recap

Standard Linear Regression models the relationship as:

\[
\hat{y} = Xw + b
\]

where:
- \(X\) = input features
- \(w\) = weights
- \(b\) = bias

### Loss Function (MSE)

\[
L_{MSE}(y, \hat{y}) = \frac{1}{n} \sum (y - \hat{y})^2
\]

❌ **Problem:** Squaring large errors makes the model highly sensitive to outliers.

---

## 3. Motivation for Huber Loss

Consider this dataset:

| x | y |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |
| 4 | 8 |
| 5 | 100 |

- MSE will aggressively try to fit the outlier
- Model becomes biased
- Poor generalization

👉 **Huber Loss limits the influence of large errors.**

---

## 4. Huber Loss Function

The Huber loss is defined as:

\[
L_\delta(e) =
\begin{cases}
\frac{1}{2}e^2 & \text{if } |e| \leq \delta \\
\delta(|e| - \frac{1}{2}\delta) & \text{if } |e| > \delta
\end{cases}
\]

Where:
- \(e = \hat{y} - y\)
- \(\delta\) is the threshold parameter

### Intuition
- Small errors → **Quadratic penalty (MSE)**
- Large errors → **Linear penalty (MAE)**

---

## 5. Gradient of Huber Loss

For gradient descent, we need:

\[
\frac{\partial L_\delta}{\partial e} =
\begin{cases}
e & |e| \leq \delta \\
\delta \cdot \text{sign}(e) & |e| > \delta
\end{cases}
\]

This ensures:
- Smooth optimization near the minimum
- Controlled updates for outliers

---

## 6. Optimization Algorithm

### Gradient Descent Update Rule

\[
w := w - \alpha \cdot \frac{1}{n} \sum g(e_i) \cdot x_i
\]

Where:
- \(\alpha\) = learning rate
- \(g(e)\) = Huber gradient

---

## 7. Feature Normalization

Before training, features are normalized:

\[
X_{norm} = \frac{X - \mu}{\sigma}
\]

Why?
- Faster convergence
- Stable gradients
- Prevents feature dominance

---

## 8. When to Use Huber Regression

✅ Dataset contains outliers  
✅ Labels are noisy  
✅ You want robustness without losing smooth optimization  
✅ Linear relationship still exists

---

## 9. When NOT to Use

❌ Extremely large datasets (slower than pure MSE)  
❌ Strong non-linear relationships  
❌ When outliers are meaningful and should be fitted  

---

## 10. Comparison with Other Losses

| Loss | Outlier Sensitivity | Smooth | Robust |
|----|----|----|----|
| MSE | ❌ High | ✅ | ❌ |
| MAE | ✅ Low | ❌ | ✅ |
| Huber | ✅ Medium | ✅ | ✅ |

---

## 11. Hyperparameter: Delta (δ)

- Small δ → closer to MAE (more robust)
- Large δ → closer to MSE (less robust)

Typical values:
- `δ = 1.0 – 2.0`
- `δ = 1.35` (common default)

---

## 12. Implementation Notes

- Bias term added manually
- Gradient descent implemented from scratch
- No external ML libraries used
- Robust gradient handling for outliers

---

## 13. Key Takeaways

✔ Huber Regression is a **balanced loss function**  
✔ It reduces outlier impact without sacrificing smooth learning  
✔ It is widely used in **robust statistics and real-world ML systems**

---

## 14. References

- Peter J. Huber (1964), *Robust Estimation of a Location Parameter*
- scikit-learn documentation (conceptual reference only)