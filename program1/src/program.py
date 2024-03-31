from tabulate import tabulate
import math
import matplotlib.pyplot as plt
import numpy as np

def create_histogram(data, feature_name, min, max):
    plt.hist(data, bins=int((max-min)*2), edgecolor='k', range=(min, max))  # Creating a histogram with range from min to max
    if feature_name[0] == 'D':
        plt.xlabel('Length (cm)', fontsize=22)
    else:
        plt.xlabel('Width (cm)', fontsize=22)
    plt.ylabel('Frequency', fontsize=22)
    plt.title(feature_name, fontsize=22)
    plt.xticks(np.arange(min, max+0.5, 0.5), fontsize=12)  # Setting steps on X-axis
    plt.yticks(fontsize=12)
    #plt.ylim(0, 35)  # Setting the range on Y-axis
    plt.show()

def create_box_plot(data_list, index):
    # Creating a dictionary to store statistical data for each species
    statistics = {0: [], 1: [], 2: []}

    # Processing data and assigning them to respective species
    for row in data_list:
        species = row[-1]
        statistics[species].append(row[index])  # Sepal length (or any other statistic)

    # Converting data to a list for box plot
    data_for_plot = [statistics[species] for species in statistics]

    # Creating box plot
    plt.boxplot(data_for_plot, labels=["setosa", "versicolor", "virginica"])
    plt.xlabel("Species", fontsize=22)
    if index == 0 or index == 2:
        plt.ylabel("Length (cm)", fontsize=22)
    else:
        plt.ylabel("Width (cm)", fontsize=22)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=12)
    
    plt.show()

def calculate_mean(data_list):
    # Ensure the list is not empty
    if not data_list:
        return None
    
    # Calculate the sum of all numbers in the list
    total_sum = sum(data_list)
    
    # Calculate the arithmetic mean
    mean = total_sum / len(data_list)
    
    return mean

def calculate_median(data_list):
    # Ensure the list is not empty
    if not data_list:
        return None
    
    # Sort the list
    sorted_numbers = sorted(data_list)
    
    # Check if the list has an even number of elements
    if len(sorted_numbers) % 2 == 0:
        # If the number of elements is even, the median is the average of the two middle numbers
        middle1 = sorted_numbers[len(sorted_numbers) // 2 - 1]
        middle2 = sorted_numbers[len(sorted_numbers) // 2]
        median = (middle1 + middle2) / 2
    else:
        # If the number of elements is odd, the median is the middle element
        median = sorted_numbers[len(sorted_numbers) // 2]
    
    return median

def calculate_standard_deviation(data_list):
    # Ensure the list is not empty
    if not data_list:
        return None
    
    # Calculate the arithmetic mean
    mean = sum(data_list) / len(data_list)
    
    # Calculate the squares of differences between each value and the mean
    squares_of_difference = [(x - mean) ** 2 for x in data_list]
    
    # Calculate the arithmetic mean of squares of differences
    mean_of_squares = sum(squares_of_difference) / (len(data_list)-1)
    
    # Calculate the square root of the arithmetic mean of squares of differences
    standard_deviation = math.sqrt(mean_of_squares)
    
    return standard_deviation

def calculate_q1(data_list):
    # Ensure the list is not empty
    if not data_list:
        return None
    
    # Sort the data
    sorted_data = sorted(data_list)
    
    # Find the median of the lower half of the dataset
    half_index = len(sorted_data) // 4  # Calculate Q1 index as 25% of the data
    if len(sorted_data) % 4 == 0:
         # If the number of data is divisible by 4, Q1 is the average of two middle values
        q1 = (sorted_data[half_index - 1] + sorted_data[half_index]) / 2
    else:
        # Otherwise, Q1 is just the middle value
        q1 = sorted_data[half_index]
    
    return q1

def calculate_q3(data_list):
    # Ensure the list is not empty
    if not data_list:
        return None
    
    # Sort the data
    sorted_data = sorted(data_list)
    
    # Find the median of the upper half of the dataset
    half_index = len(sorted_data) // 4 * 3  # Calculate Q3 index as 75% of the data
    if len(sorted_data) % 2 == 0:
        # If the number of data is divisible by 4, Q3 is the average of two middle values
        q3 = (sorted_data[half_index] + sorted_data[half_index + 1]) / 2
    else:
        # Otherwise, Q3 is just the middle value
        q3 = sorted_data[half_index + 1]
    
    return q3

def round_to_nearest_half(number):
    return round(number * 2) / 2

def load_file(path):
    
    data_list = []
    try:
        f = open(path, 'r')

        for line in f:
            elements = line.strip().split(',')
                
            # Adding the last element from the line (without separator)
            last_element = elements.pop()
            elements.append(last_element)
                
            data_list.append(elements)

        f.close()

        # Convert strings to numbers in the two-dimensional array
        for i in range(len(data_list)):
            for j in range(len(data_list[i])):
                # Convert string to float
                data_list[i][j] = float(data_list[i][j])

        return data_list
    
    except FileNotFoundError:
        # Handling error if the file doesn't exist
        print(f"File '{path}' does not exist.")
        return None

def task1(table):
    # Initialize variables for three species and the total count
    setosa = 0
    versicolor = 0
    virginica = 0
    total = 0

    # Loop through each row in the table
    for row in table:
        total += 1  # Count the number of rows in the table
        if row[4] == 0:
            setosa += 1
        elif row[4] == 1:
            versicolor += 1
        elif row[4] == 2:
            virginica += 1

    # Create data for the table, including counts and percentage shares
    data = [
        ["Setosa", str(setosa) + " " + str(round(setosa / total * 100, 2)) + "%"],
        ["Versicolor", str(versicolor) + " " + str(round(versicolor / total * 100, 2)) + "%"],
        ["Virginica", str(virginica) + " " + str(round(virginica / total * 100, 2)) + "%"],
        ["Total", str(total) + " " + str(round(total / total * 100, 2)) + "%"]
    ]

    headers = ["Species", "Count (%)"]

    # Generate and print the table
    table = tabulate(data, headers, tablefmt="pretty")
    print(table)

def task2(table):
    # Initialize empty lists for different features
    sepal_length = []
    sepal_width = []
    petal_length = []
    petal_width = []

    # Loop processing input data
    for row in table:
        # Adding respective values to corresponding lists
        sepal_length.append(row[0])
        sepal_width.append(row[1])
        petal_length.append(row[2])
        petal_width.append(row[3])
    
    # Headers for the resulting table
    headers = ["Feature", "Minimum", "Mean (± std. dev.)", "Median (Q1 - Q3)", "Maximum"]

    # Data in the form of list of lists
    data = [
        ["Sepal Length (cm)", "{:.2f}".format(round(float(min(sepal_length)), 2)),
         "{:.2f}".format(round(float(calculate_mean(sepal_length)), 2)) + " (" +
         "{:.2f}".format(round(float(calculate_standard_deviation(sepal_length)), 2)) + ")",
         "{:.2f}".format(round(float(calculate_median(sepal_length)), 2)) + " (" +
         "{:.2f}".format(round(float(calculate_q1(sepal_length)), 2)) + " - " +
         "{:.2f}".format(round(float(calculate_q3(sepal_length)), 2)) + ")",
         "{:.2f}".format(round(float(max(sepal_length)), 2))],
        ["Sepal Width (cm)", "{:.2f}".format(round(float(min(sepal_width)), 2)),
         "{:.2f}".format(round(float(calculate_mean(sepal_width)), 2)) + " (" +
         "{:.2f}".format(round(float(calculate_standard_deviation(sepal_width)), 2)) + ")",
         "{:.2f}".format(round(float(calculate_median(sepal_width)), 2)) + " (" +
         "{:.2f}".format(round(float(calculate_q1(sepal_width)), 2)) + " - " +
         "{:.2f}".format(round(float(calculate_q3(sepal_width)), 2)) + ")",
         "{:.2f}".format(round(float(max(sepal_width)), 2))],
        ["Petal Length (cm)", "{:.2f}".format(round(float(min(petal_length)), 2)),
         "{:.2f}".format(round(float(calculate_mean(petal_length)), 2)) + " (" +
         "{:.2f}".format(round(float(calculate_standard_deviation(petal_length)), 2)) + ")",
         "{:.2f}".format(round(float(calculate_median(petal_length)), 2)) + " (" +
         "{:.2f}".format(round(float(calculate_q1(petal_length)), 2)) + " - " +
         "{:.2f}".format(round(float(calculate_q3(petal_length)), 2)) + ")",
         "{:.2f}".format(round(float(max(petal_length)), 2))],
        ["Petal Width (cm)", "{:.2f}".format(round(float(min(petal_width)), 2)),
         "{:.2f}".format(round(float(calculate_mean(petal_width)), 2)) + " (" +
         "{:.2f}".format(round(float(calculate_standard_deviation(petal_width)), 2)) + ")",
         "{:.2f}".format(round(float(calculate_median(petal_width)), 2)) + " (" +
         "{:.2f}".format(round(float(calculate_q1(petal_width)), 2)) + " - " +
         "{:.2f}".format(round(float(calculate_q3(petal_width)), 2)) + ")",
         "{:.2f}".format(round(float(max(petal_width)), 2))]
    ]

    # Generate and print the table
    table = tabulate(data, headers, tablefmt="pretty")
    print(table)

    # Creating histograms
    create_histogram(sepal_length, 'Sepal Length', math.floor(min(sepal_length)), math.ceil(max(sepal_length)))
    create_histogram(sepal_width, 'Sepal Width', math.floor(min(sepal_width)), round_to_nearest_half(max(sepal_width)))
    create_histogram(petal_length, 'Petal Length', math.floor(min(petal_length)), math.ceil(max(petal_length)))
    create_histogram(petal_width, 'Petal Width', math.floor(min(petal_width)), math.ceil(max(petal_width)))


data_list = load_file('data/data.csv')
task1(data_list)
task2(data_list)
create_box_plot(data_list, 0)
create_box_plot(data_list, 1)
create_box_plot(data_list, 2)
create_box_plot(data_list, 3)
