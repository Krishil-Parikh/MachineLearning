import numpy as np
from model import HuberRegression

X = np.array([[1], [2], [3], [4], [100]])
y = np.array([2, 4, 6, 8, 200])  # outlier

model = HuberRegression(lr=0.01, epochs=2000, delta=1.35)
model.fit(X, y)

preds = model.predict(X)

print("Predictions:", preds)
print("MSE:", model.mse(y, preds))
print("R2:", model.r2_score(y, preds))