import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df,title,ylabel,xlabel="Number of Items",bins=50):
    plt.figure()
    plt.hist(df, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_heatmap(series,title):
    sns.set_theme()
    plt.figure(figsize=(15,1))
    sns.heatmap(series.describe(percentiles=[0.5,0.75,0.9,0.95]).to_frame("Value").T, annot=True, fmt=".2f", cmap="Blues")
    plt.title(title)
    plt.show()

def plot_line_chart(index, values, title=None, xlabel=None, ylabel=None):
    plt.figure()
    plt.plot(index, values)
    plt.title(t)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()