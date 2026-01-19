# Multiple Linear Regression (From Scratch)

## 1. Overview

Multiple Linear Regression is an extension of Simple Linear Regression where the target variable depends on **multiple input features**.

It models the relationship between one continuous output variable and **two or more independent variables** using a linear equation.

---

## 2. Model Formulation

For a dataset with \(n\) samples and \(d\) features:

\[
\hat{y} = w_0 + w_1x_1 + w_2x_2 + \dots + w_dx_d
\]

In vectorized form:

\[
\hat{y} = Xw
\]

Where:
- \(X \in \mathbb{R}^{n \times (d+1)}\) (including bias column)
- \(w \in \mathbb{R}^{(d+1)}\)
- \(\hat{y} \in \mathbb{R}^{n}\)

---

## 3. Assumptions of Multiple Linear Regression

Multiple Linear Regression relies on the following assumptions:

1. **Linearity**  
   The relationship between features and target is linear.

2. **Independence of Errors**  
   Residuals are independent of each other.

3. **Homoscedasticity**  
   Constant variance of residuals.

4. **No Multicollinearity**  
   Features are not highly correlated with each other.

5. **Normality of Errors** (optional but useful for inference)

---

## 4. Loss Function (Mean Squared Error)

The objective is to minimize Mean Squared Error (MSE):

\[
L(w) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

MSE penalizes larger errors more strongly, encouraging better overall fit.

---

## 5. Gradient Descent Optimization

### Gradient of MSE

\[
\frac{\partial L}{\partial w} = \frac{2}{n} X^T (Xw - y)
\]

### Update Rule

\[
w := w - \alpha \cdot \frac{2}{n} X^T (Xw - y)
\]

Where:
- \(\alpha\) = learning rate

---

## 6. Bias Term Handling

To include the bias term \(w_0\), a column of ones is appended to the feature matrix:

\[
X = 
\begin{bmatrix}
1 & x_{11} & x_{12} & \dots \\
1 & x_{21} & x_{22} & \dots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix}
\]

This allows the bias to be learned as part of the weight vector.

---

## 7. Feature Normalization

Features are normalized before training:

\[
X_{norm} = \frac{X - \mu}{\sigma}
\]

### Why Normalize?
- Faster convergence
- Stable gradients
- Prevents features with large scales from dominating

---

## 8. Batch vs Mini-Batch Gradient Descent

### Full-Batch Gradient Descent
- Uses entire dataset per update
- Stable but slow

### Mini-Batch Gradient Descent
- Uses subsets of data
- Faster and more scalable
- Common in practice

---

## 9. Evaluation Metrics

### Mean Squared Error (MSE)

\[
MSE = \frac{1}{n} \sum (y - \hat{y})^2
\]

### R² Score (Coefficient of Determination)

\[
R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}
\]

- \(R^2 = 1\): perfect fit  
- \(R^2 = 0\): predicts mean  
- \(R^2 < 0\): worse than baseline  

---

## 10. Regularization (Optional)

### Ridge Regression (L2)

\[
L = MSE + \lambda \sum w^2
\]

Purpose:
- Prevent overfitting
- Reduce coefficient magnitude
- Handle multicollinearity

---

## 11. When to Use Multiple Linear Regression

✅ Predicting continuous values  
✅ Relationship is approximately linear  
✅ Model interpretability is important  
✅ Dataset size is small to medium  

---

## 12. When NOT to Use

❌ Strong non-linear relationships  
❌ High multicollinearity  
❌ Many irrelevant features  
❌ Large number of categorical variables (without encoding)

---

## 13. Advantages

✔ Simple and interpretable  
✔ Fast training  
✔ Works well with small datasets  
✔ Strong baseline model  

---

## 14. Limitations

❌ Sensitive to outliers  
❌ Cannot capture non-linearity  
❌ Assumptions may not hold in real data  

---

## 15. Implementation Notes (This Repo)

- Implemented using pure NumPy
- Manual bias handling
- Gradient descent from scratch
- Optional regularization
- No external ML libraries used

---

## 16. Key Takeaways

✔ Multiple Linear Regression generalizes simple linear regression  
✔ Gradient descent enables scalable optimization  
✔ Feature normalization is critical  
✔ Serves as a strong baseline model

---

## 17. References

- Trevor Hastie, Robert Tibshirani – *The Elements of Statistical Learning*
- Andrew Ng – Machine Learning (Coursera)