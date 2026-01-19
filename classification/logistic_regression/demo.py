import numpy as np
from model import LogisticRegression

# ===================== TOY DATASET ===================== #
# AND gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# ===================== TRAIN ===================== #

model = LogisticRegression(lr=0.5, epochs=2000)
model.fit(X, y)

# ===================== PREDICT ===================== #

probs = model.predict_proba(X)
preds = model.predict(X)

print("Probabilities:", probs)
print("Predictions:", preds)
print("Accuracy:", model.accuracy(y, preds))

# ===================== OPTIONAL: LOSS CURVE ===================== #
try:
    import matplotlib.pyplot as plt

    plt.plot(model.loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")
    plt.title("Logistic Regression Training Curve")
    plt.grid(alpha=0.3)
    plt.show()
except ImportError:
    pass