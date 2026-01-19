# Logistic Regression (From Scratch)

## 1. Overview

Logistic Regression is a **linear classification algorithm** used for **binary classification** problems.

Despite its name, Logistic Regression is a **classification model**, not a regression model.

It predicts the **probability** that a data point belongs to a class.

---

## 2. Problem Setting

Given:
- Input features \(X \in \mathbb{R}^{n \times d}\)
- Binary target \(y \in \{0, 1\}\)

Goal:
Predict:
\[
P(y = 1 \mid X)
\]

---

## 3. Model Formulation

### Linear Combination
\[
z = Xw + b
\]

### Sigmoid Function
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

The sigmoid maps real values into the range **(0, 1)**.

\[
\hat{y} = \sigma(Xw + b)
\]

---

## 4. Decision Boundary

Prediction rule:
\[
\hat{y} =
\begin{cases}
1 & \text{if } \hat{y} \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
\]

The decision boundary is **linear** in feature space.

---

## 5. Loss Function (Binary Cross-Entropy)

Logistic Regression uses **log loss**, not MSE.

\[
L(y, \hat{y}) = -\frac{1}{n} \sum \left[
y \log(\hat{y}) + (1-y)\log(1-\hat{y})
\right]
\]

### Why not MSE?
- Non-convex with sigmoid
- Poor gradients
- Slower convergence

---

## 6. Gradient of Loss

Gradient w.r.t weights:

\[
\frac{\partial L}{\partial w}
= \frac{1}{n} X^T (\hat{y} - y)
\]

This simple form makes Logistic Regression efficient.

---

## 7. Optimization

### Gradient Descent Update
\[
w := w - \alpha \cdot \frac{1}{n} X^T (\hat{y} - y)
\]

Where:
- \(\alpha\) = learning rate

---

## 8. Feature Normalization

Features are normalized to:
\[
X_{norm} = \frac{X - \mu}{\sigma}
\]

Benefits:
- Faster convergence
- Stable gradients

---

## 9. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Log Loss

(Accuracy is most commonly used for binary classification.)

---

## 10. When to Use Logistic Regression

✅ Binary classification  
✅ Linearly separable data  
✅ Interpretable coefficients  
✅ Probabilistic output needed  

---

## 11. When NOT to Use

❌ Non-linear decision boundaries  
❌ Many classes (use softmax)  
❌ Very complex patterns  

---

## 12. Advantages

✔ Simple and fast  
✔ Interpretable  
✔ Convex loss (global minimum)  
✔ Strong baseline classifier  

---

## 13. Limitations

❌ Linear decision boundary  
❌ Sensitive to outliers  
❌ Requires feature engineering  

---

## 14. Implementation Notes (This Repo)

- Implemented using pure NumPy
- Manual sigmoid and loss
- Gradient descent from scratch
- No external ML libraries

---

## 15. Key Takeaways

✔ Logistic Regression is a probabilistic classifier  
✔ Uses sigmoid + log loss  
✔ Convex optimization guarantees convergence  
✔ Excellent baseline model

---

## 16. References

- Andrew Ng – Machine Learning
- Bishop – Pattern Recognition and Machine Learning