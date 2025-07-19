import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from mlxtend.evaluate import bias_variance_decomp

class BiasVarianceAnalysis:
    def __init__(self, degrees, n_trials=100):
        self.degrees = degrees
        self.n_trials = n_trials
        self.results = {}

    def analyze_synthetic(self, n_samples=100, noise=0.1):
        X = np.random.uniform(-1, 1, (n_samples, 1))
        y = X[:, 0]**3 + np.random.normal(0, noise, n_samples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        bias_values, variance_values, mse_values = [], [], []
        for degree in self.degrees:
            poly_model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])
            mse, bias, var = bias_variance_decomp(
                poly_model, X_train, y_train, X_test, y_test,
                loss='mse', num_rounds=self.n_trials, random_seed=42
            )
            bias_values.append(bias)
            variance_values.append(var)
            mse_values.append(mse)
        self.results['synthetic'] = {
            'degrees': list(self.degrees),
            'bias': bias_values,
            'variance': variance_values,
            'mse': mse_values
        }
        return self.results