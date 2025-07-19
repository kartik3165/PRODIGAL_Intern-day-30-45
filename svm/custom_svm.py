import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class CustomSVM:
    def __init__(self, C=1.0, learning_rate=0.01, n_iterations=1000):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_encoded = np.where(y == 0, -1, 1)  # Convert 0 to -1 for SVM
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_encoded[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * self.w
                else:
                    self.w -= self.learning_rate * (self.w - self.C * y_encoded[idx] * x_i)
                    self.b += self.learning_rate * self.C * y_encoded[idx]
        return self

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(np.sign(linear_output) == 1, 1, 0)

# ------------------------------------
# Test on Binary Classification
# ------------------------------------
# Load Iris dataset (only 2 classes for binary SVM)
data = load_iris()
X = data.data[data.target != 2]     # Only class 0 and 1
y = data.target[data.target != 2]   # Only class 0 and 1

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Custom SVM
svm = CustomSVM(C=1.0, learning_rate=0.001, n_iterations=1000)
svm.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Custom SVM Accuracy: {accuracy:.4f}")
