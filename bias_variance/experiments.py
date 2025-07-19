from analysis import BiasVarianceAnalysis
from visualization import plot_bias_variance

degrees = range(1, 11)
bva = BiasVarianceAnalysis(degrees)
results = bva.analyze_synthetic()
plot_bias_variance(results)