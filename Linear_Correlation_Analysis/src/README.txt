# Linear Correlation Analysis Program

This Python program is designed for conducting linear correlation analysis using the Pearson correlation coefficient. It enables users to explore and visualize relationships between different features within a dataset.

## Purpose

The primary objective of this program is to analyze the linear correlation between pairs of features in a dataset. It provides insights into how two variables are related to each other and helps identify patterns or trends in the data.

## Dependencies

- `numpy`: For numerical computations and array manipulation.
- `matplotlib.pyplot`: For creating visualizations and plots.
- `seaborn`: For enhancing the aesthetics of plots.
- `scipy.stats.linregress`: For computing linear regression statistics.
- `math`: For mathematical operations.

## Features

### `mean(arr)`

Calculates the mean (average) of an array.

### `Pearson(x1, y1)`

Computes the Pearson linear correlation coefficient between two arrays (`x1` and `y1`). Additionally, it determines the parameters of the linear regression equation.

### `plot(x1, y1, x_label, y_label)`

Generates a scatter plot of the data points, overlays it with a linear regression line, and displays the Pearson correlation coefficient along with the regression equation.

### `load_file(path)`

Loads data from a CSV file specified by `path` using `numpy` and returns it as a list.

## Usage

1. Define the features you want to analyze by specifying them in the `features` list.
2. Provide the path to your dataset CSV file in the `data_list` variable.
3. Execute the script to compute correlations, regressions, and generate plots for each pair of features.

