import numpy as np
from model import LinearRegression

# Toy dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression(
    lr=0.05,
    epochs=1000,
    l2_lambda=0.01
)

model.fit(X, y)

preds = model.predict(X)

print("Predictions:", preds)
print("MSE:", model.mse(y, preds))
print("R2:", model.r2_score(y, preds))