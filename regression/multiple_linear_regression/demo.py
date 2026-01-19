import numpy as np
from model import MultipleLinearRegression

# ===================== TOY DATASET ===================== #
# y = 3*x1 + 5*x2 + 10

X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 5]
])

y = np.array([23, 21, 37, 37, 50])

# ===================== TRAIN MODEL ===================== #

model = MultipleLinearRegression(
    lr=0.05,
    epochs=2000,
    l2_lambda=0.01
)

model.fit(X, y)

# ===================== PREDICTIONS ===================== #

predictions = model.predict(X)

print("Predictions:")
print(predictions)

print("\nMSE:", model.mse(y, predictions))
print("R2 Score:", model.r2_score(y, predictions))

# ===================== OPTIONAL: PLOT LOSS ===================== #
try:
    import matplotlib.pyplot as plt

    plt.plot(model.loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.3)
    plt.show()
except ImportError:
    pass