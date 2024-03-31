import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import math as math

def mean(arr):
    avg = sum(arr) / len(arr)
    return avg

def Pearson(x1, y1):
    # Calculate Pearson linear correlation
    sumX = 0
    sumY = 0
    numerator = 0
    denominatorX = 0
    denominatorY = 0
    
    meanX = mean(x1)
    meanY = mean(y1)
    
 
    for i in range(len(x1)):
        numerator += (x1[i] - meanX) * (y1[i] - meanY)
        denominatorX += (x1[i] - meanX) * (x1[i] - meanX)
        denominatorY += (y1[i] - meanY) * (y1[i] - meanY)
    
    a = numerator / denominatorX
    
    # Calculate linear regression equation
    correlationDenominator = math.sqrt(denominatorX * denominatorY)
    r = numerator / correlationDenominator
    b = meanY - (numerator / denominatorX) * meanX
    return r, a, b

def plot(x1, y1, x_label, y_label):
    correlation, a, b = Pearson(x1, y1)

    if b >= 0:
        sign_b = '+'
    else:
        sign_b = '-'

    regression_equation = f"r={correlation:.2f}; y={a:.1f}x {sign_b} {abs(b):.1f}"

    #plt.figure(figsize=(8, 6))
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=x1, y=y1, s=70)
    plt.plot(x1, b + a * x1, color='red')
    plt.title(regression_equation, fontsize=15)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.ylim(min(y1) - 0.25, max(y1) + 0.25)
    plt.xticks(np.arange(math.floor(min(x)), math.ceil(max(x)) + 1, 1))
    plt.xlim(math.floor(min(x)) - 0.1, math.ceil(max(x)) + 0.1)
    plt.show()

def load_file(path):
    try:
        data = np.loadtxt(path, delimiter=',')
        return data.tolist()
    except FileNotFoundError:
        print(f"File '{path}' does not exist.")
        return None

# Load data
features = ["Sepal Length (cm)", "Sepal Width (cm)",
            "Petal Length (cm)", "Petal Width (cm)"]

data_list = load_file('data/data.csv')

# Calculate correlation, regression, and create plots for each pair of features
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        x = np.array(data_list)[:, i]
        y = np.array(data_list)[:, j]
        plot(x, y, features[i], features[j])
