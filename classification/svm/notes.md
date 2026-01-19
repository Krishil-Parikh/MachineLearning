# Support Vector Machine (SVM) — From Scratch

## 1. Overview

Support Vector Machine (SVM) is a **margin-based classifier** that finds the optimal hyperplane
which separates data points of different classes with the **maximum margin**.

This implementation focuses on:
- Linear SVM
- Hinge loss
- L2 regularization
- Gradient descent optimization

---

## 2. Binary Classification Setting

Given:
- Features: \( X \in \mathbb{R}^{n \times d} \)
- Labels: \( y \in \{-1, +1\} \)

Goal:
Find a hyperplane:
\[
w^T x - b = 0
\]

---

## 3. Margin Concept

The margin is defined as the distance between:
- The decision boundary
- The closest data points (support vectors)

SVM **maximizes this margin**.

---

## 4. Optimization Objective (Primal Form)

\[
\min_{w, b} \quad \lambda \|w\|^2 + \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i (w^T x_i - b))
\]

Where:
- First term → L2 regularization
- Second term → Hinge loss

---

## 5. Hinge Loss

\[
L(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})
\]

Properties:
- Zero loss for correctly classified points outside margin
- Linear penalty for margin violations

---

## 6. Gradient Updates

### If correctly classified with margin:
\[
w := w - \alpha (2\lambda w)
\]

### If margin violated:
\[
w := w - \alpha (2\lambda w - yx)
\]
\[
b := b - \alpha y
\]

---

## 7. Feature Normalization

SVM is sensitive to feature scale.

Normalization ensures:
- Faster convergence
- Balanced margin computation

---

## 8. Decision Function

\[
f(x) = w^T x - b
\]

Prediction:
\[
\hat{y} = \text{sign}(f(x))
\]

---

## 9. Evaluation Metric

- Accuracy
- Margin visualization (for 2D data)

---

## 10. When to Use SVM

✅ Binary classification  
✅ High-dimensional data  
✅ Clear margin separation  
✅ Small to medium datasets  

---

## 11. Limitations

❌ Not scalable to very large datasets  
❌ Sensitive to feature scaling  
❌ Kernel methods increase complexity  

---

## 12. Implementation Notes (This Repo)

- Linear SVM only
- Gradient descent optimization
- Pure NumPy implementation
- No kernel trick (yet)

---

## 13. Key Takeaways

✔ SVM maximizes margin, not probability  
✔ Hinge loss enforces strong decision boundaries  
✔ Regularization controls overfitting  

---

## 14. References

- Vapnik, *Statistical Learning Theory*
- Andrew Ng – Machine Learning