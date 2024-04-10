import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from collections import Counter, defaultdict

# Function calculating Euclidean distance between two points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def initialize_centroids(data, k, random_state=42):
    np.random.seed(random_state)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    return centroids

# Function assigning points to the nearest centroid
def assign_to_nearest_centroid(data, centroids):
    return np.array([np.argmin([euclidean_distance(point, centroid) for centroid in centroids]) for point in data])

def update_centroids(data, cluster_assignment, k, min_points=2):
    new_centroids = []
    for i in range(k):
        cluster_points = data[cluster_assignment == i]
        if len(cluster_points) >= min_points:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            # If the cluster has too few points, use a random point as a new centroid
            new_centroids.append(data[np.random.choice(data.shape[0])])
    return np.array(new_centroids)

# Function calculating WCSS
def calculate_wcss(data, centroids, cluster_assignment):
    wcss = sum(np.sum((data[cluster_assignment == i] - centroids[i])**2) for i in range(len(centroids)))
    return wcss

# Function classifying using k-nearest neighbors (KNN)
def knn(train_data, train_labels, test_point, k):
    distances = []
    
    # Calculate distances between test_point and each point in train_data
    for i, train_point in enumerate(train_data):
        dist = euclidean_distance(train_point, test_point)  
        distances.append((i, dist)) 

    # Sort distances from smallest to largest
    distances.sort(key=lambda x: x[1])

    # Choose k nearest neighbors considering weights
    neighbors = distances[:k]
    label_count = defaultdict(float)

    label_occurrences = {'0':0,'1':0,'2':0}

    # Count labels considering weights
    for neighbor_index, dist in neighbors:
        label = train_labels[neighbor_index]  
        label_count[label] += 1/dist if dist != 0 else float('inf')
        label_occurrences[str(label)] +=1
    
   # Find key with the highest value
    max_value = max(label_occurrences.values())
    max_labels = [label for label, count in label_occurrences.items() if count == max_value]

    if len(max_labels) == 1:
        return int(max_labels[0])
    else:
        # Set value 0 for labels in label_count that are not in max_labels
        for label in label_count:
            if str(label) not in max_labels:
                label_count[label] = 0

        max_label_with_highest_value = None
        highest_value = float('-inf')

        for label in max_labels:
            if label_count[label] > highest_value:
                highest_value = label_count[label]
                max_label_with_highest_value = label
                
        return int(max_label_with_highest_value)

# Load training and test data
data = pd.read_csv('data.csv', header=None)
train = pd.read_csv('data_train.csv', header=None)
test = pd.read_csv('data_test.csv', header=None)

# Split data into features and labels
data_features = data.iloc[:, :-1]

train_features = train.iloc[:, :-1] 
train_labels = train.iloc[:, -1]  

test_features = test.iloc[:, :-1]  
test_labels = test.iloc[:, -1]  

feature_combinations = list(itertools.combinations(range(4), 2))

features_names = ["Sepal Length (cm)", "Sepal Width (cm)",
         "Petal Length (cm)", "Petal Width (cm)"]

# Visualize clusters for different feature combinations
for k_num_clusters in range(2, 11):

    for i, combination in enumerate(feature_combinations):
        features_subset = data_features.iloc[:, list(combination)].values

        centroids = initialize_centroids(features_subset, k_num_clusters)

        
        while True:
            cluster_assignment = assign_to_nearest_centroid(features_subset, centroids)
            new_centroids = update_centroids(features_subset, cluster_assignment, k_num_clusters)

            if np.all(new_centroids == centroids):
                break
            
            centroids = new_centroids
        if(k_num_clusters==3):
            plt.figure(figsize=(8, 6))  
            for cluster_index in range(k_num_clusters):
                cluster_points = features_subset[cluster_assignment == cluster_index]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.5, edgecolors=[plt.cm.viridis(cluster_index / 3)],
                    c='none')
                plt.scatter(centroids[cluster_index, 0], centroids[cluster_index, 1], marker='d', s=100,
                    c=[plt.cm.viridis(cluster_index / 3)])

            # Labels and plot configuration
            plt.xlabel(features_names[combination[0]])  
            plt.ylabel(features_names[combination[1]])  
            plt.show()  


kmeans_features = data.iloc[:, :-1].values
wcss_values = []
iteration_values = []

k_range = range(2, 11)

for k_num_clusters in k_range:
    centroids = initialize_centroids(kmeans_features, k_num_clusters)
    iteration = 0
    max_iterations = 100  

    while iteration < max_iterations:
        cluster_assignment = assign_to_nearest_centroid(kmeans_features, centroids)
        new_centroids = update_centroids(kmeans_features, cluster_assignment, k_num_clusters)
        iteration += 1

        if np.all(np.abs(new_centroids - centroids) < 1e-5):
            break

        centroids = new_centroids

    wcss = calculate_wcss(kmeans_features, centroids, cluster_assignment)
    
    wcss_values.append(wcss)
    iteration_values.append(iteration)
print(iteration_values)
print(wcss_values)
print("-------------------------")


plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), iteration_values, marker='o')
plt.title('Number of iterations depending on the number of clusters (k)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Number of iterations')
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), wcss_values, marker='o')
plt.title('WCSS depending on the number of clusters (k)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
      

min_vals_train = train_features.min()
max_vals_train = train_features.max()

train_features_scaled = (train_features - min_vals_train) / (max_vals_train - min_vals_train)

test_features_scaled = (test_features - min_vals_train) / (max_vals_train - min_vals_train)


accuracy = []

for k_value in range(1, 16):
    correct_predictions = 0

    for index, test_point in test_features_scaled.iterrows():
        correct_label = test_labels[index]  

        predicted_label = knn(train_features_scaled.values, train_labels.values, test_point.values, k_value)
        if predicted_label == correct_label:
            correct_predictions += 1
    
    accuracy.append(correct_predictions)


max_index = accuracy.index(max(accuracy))
k_value = max_index + 1  


plt.figure(figsize=(8, 6))
bars = plt.bar(range(1, 16), accuracy, color='blue')
bars[max_index].set_color('red')  
plt.title('Number of correct matches for different k values')
plt.xlabel('Value of k')
plt.ylabel('Number of correct matches')
plt.xticks(range(1, 16))
plt.yticks(range(0, max(accuracy) + 1, 5))  
plt.grid(which='major', axis='y', linestyle='-') 
plt.show()


confusion_matrix = np.zeros((3, 3), dtype=int)

for index, test_point in test_features_scaled.iterrows():
    correct_label = test_labels[index]  
        
    predicted_label = knn(train_features_scaled.values, train_labels.values, test_point.values, k_value)
    
    labels_map = {0: 0, 1: 1, 2: 2}
    
    confusion_matrix[labels_map[correct_label]][labels_map[predicted_label]] += 1
    

print("Confusion matrix:")
print(confusion_matrix)


accuracy_results = []
for combination in feature_combinations:
    combination_accuracy = []
    for k_value in range(1, 16):
        train_subset = train_features_scaled.loc[:, combination]
        test_subset = test_features_scaled.loc[:, combination]
        correct_predictions = 0

        for index, test_point in test_subset.iterrows():
            correct_label = test_labels[index]  
            
            predicted_label = knn(train_subset.values, train_labels.values, test_point.values, k_value)
            if predicted_label == correct_label:
                correct_predictions += 1
        
        combination_accuracy.append(correct_predictions)
    accuracy_results.append(combination_accuracy)

for idx, combination_accuracy in enumerate(accuracy_results):
    plt.figure(figsize=(8, 6)) 
    plt.bar(range(1, 16), combination_accuracy, color='blue')
    plt.title(f'Feature combination: {feature_combinations[idx]}')
    plt.xlabel('Value of k')
    plt.ylabel('Number of correct matches')
    plt.xticks(range(1, 16))
    plt.yticks(range(0, 50, 5))
    plt.grid(axis='y')

    max_idx = np.argmax(combination_accuracy)
    plt.bar(max_idx + 1, combination_accuracy[max_idx], color='red')

    plt.show() 


for idx, combination_accuracy in enumerate(accuracy_results):
    max_idx = np.argmax(combination_accuracy)
    combination = feature_combinations[idx]
    k_value = max_idx + 1
    
    train_subset = train_features_scaled.loc[:, combination]
    test_subset = test_features_scaled.loc[:, combination]
    
    correct_predictions = 0
    confusion_matrix = np.zeros((3, 3), dtype=int)

    for index, test_point in test_subset.iterrows():
        correct_label = test_labels[index]
        predicted_label = knn(train_subset.values, train_labels.values, test_point.values, k_value)
        
        if predicted_label == correct_label:
            correct_predictions += 1
        confusion_matrix[correct_label][predicted_label] += 1

    print(f"Confusion matrix for feature combination {combination} and k={k_value}:")
    print(confusion_matrix)
