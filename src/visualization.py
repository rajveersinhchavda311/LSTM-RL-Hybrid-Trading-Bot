import matplotlib.pyplot as plt

def plot_signals(data, fib_levels, kmeans_levels=None):
    plt.figure(figsize=(14,7))
    plt.plot(data['Date'], data['Close'], label='Close')
    plt.plot(data['Date'], data['bb_upper'], linestyle='--', label='BB Upper')
    plt.plot(data['Date'], data['bb_lower'], linestyle='--', label='BB Lower')
    plt.scatter(data[data['signal']=='Buy']['Date'], data[data['signal']=='Buy']['Close'], marker='^', color='g', label='Buy Signal')
    plt.scatter(data[data['signal']=='Sell']['Date'], data[data['signal']=='Sell']['Close'], marker='v', color='r', label='Sell Signal')
    plt.axhline(fib_levels['38.2%'], color='purple', linestyle=':', label='Fib 38.2%')
    plt.axhline(fib_levels['61.8%'], color='orange', linestyle=':', label='Fib 61.8%')
    # Plot support and resistance as lines
    plt.plot(data['Date'], data['support'], color='blue', linestyle=':', alpha=0.5, label='Support')
    plt.plot(data['Date'], data['resistance'], color='red', linestyle=':', alpha=0.5, label='Resistance')
    if 'fract_support' in data.columns and 'fract_resistance' in data.columns:
        plt.plot(data['Date'], data['fract_support'], color='cyan', linestyle='--', alpha=0.5, label='Fractal Support')
        plt.plot(data['Date'], data['fract_resistance'], color='magenta', linestyle='--', alpha=0.5, label='Fractal Resistance')
    # Plot K-means S/R levels if provided
    if kmeans_levels is not None:
        for level in kmeans_levels:
            plt.axhline(level, color='grey', linestyle='-.', alpha=0.7, label='KMeans S/R')
    plt.legend()
    plt.show()