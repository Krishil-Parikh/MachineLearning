import numpy as np
import matplotlib.pyplot as plt
from model import SVM

# ===================== TOY DATA ===================== #
X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [2, 1],
    [3, 2],
    [4, 2]
])

y = np.array([-1, -1, -1, 1, 1, 1])

# ===================== TRAIN ===================== #
model = SVM(lr=0.01, lambda_param=0.01, epochs=2000)
model.fit(X, y)

preds = model.predict(X)

print("Predictions:", preds)
print("Accuracy:", model.accuracy(y, preds))

# ===================== VISUALIZE (2D ONLY) ===================== #
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    plt.contour(xx, yy, Z, levels=[-1, 1], linestyles='dashed', colors='gray')

    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='b', label='Class +1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='r', label='Class -1')

    plt.legend()
    plt.title("Linear SVM Decision Boundary")
    plt.grid(alpha=0.3)
    plt.show()

plot_decision_boundary(model, X, y)