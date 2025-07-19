import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class SVMAnalysis:
    def __init__(self):
        self.datasets = {}
        self.results = {}

    def load_datasets(self):
        iris = load_iris()
        wine = load_wine()
        cancer = load_breast_cancer()
        self.datasets = {
            'iris': {'X': iris.data, 'y': iris.target},
            'wine': {'X': wine.data, 'y': wine.target},
            'cancer': {'X': cancer.data, 'y': cancer.target}
        }
        return self.datasets

    def svm_kernel_comparison(self, dataset_name='iris'):
        X = self.datasets[dataset_name]['X']
        y = self.datasets[dataset_name]['y']

        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Kernel testing
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        kernel_results = []
        for kernel in kernels:
            svm = SVC(kernel=kernel, random_state=42)
            svm.fit(X_train_scaled, y_train)
            y_pred = svm.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            kernel_results.append({
                'Kernel': kernel,
                'Accuracy': accuracy,
                'Support_Vectors': sum(svm.n_support_)
            })

        self.results[dataset_name] = kernel_results
        return kernel_results

    def print_results(self, dataset_name):
        print(f"\nSVM Kernel Comparison for '{dataset_name}' Dataset:")
        print("-" * 50)
        for result in self.results[dataset_name]:
            print(f"Kernel: {result['Kernel']:8} | Accuracy: {result['Accuracy']:.4f} | Support Vectors: {result['Support_Vectors']}")
        print("-" * 50)

# ----------------------------
# Usage
# ----------------------------
if __name__ == "__main__":
    analysis = SVMAnalysis()
    analysis.load_datasets()

    # Test on Iris dataset
    analysis.svm_kernel_comparison('iris')
    analysis.print_results('iris')

    # Test on Wine dataset
    analysis.svm_kernel_comparison('wine')
    analysis.print_results('wine')

    # Test on Breast Cancer dataset
    analysis.svm_kernel_comparison('cancer')
    analysis.print_results('cancer')
