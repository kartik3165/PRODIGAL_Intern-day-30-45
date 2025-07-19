import matplotlib.pyplot as plt

def plot_bias_variance(results, dataset_name='synthetic'):
    degrees = results[dataset_name]['degrees']
    bias = results[dataset_name]['bias']
    variance = results[dataset_name]['variance']
    mse = results[dataset_name]['mse']
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, bias, label='Bias^2')
    plt.plot(degrees, variance, label='Variance')
    plt.plot(degrees, mse, label='MSE')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error')
    plt.title(f'Bias-Variance Trade-off for {dataset_name}')
    plt.legend()
    plt.savefig('bv_plot.png')
    plt.close()