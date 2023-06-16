import numpy as np
import pandas as pd
import math

# Load the dataset
data = pd.read_csv("kaggle_bot_accounts.csv", nrows=1000)
data = data.dropna()
data = data.drop_duplicates()

# Split the dataset into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Set the random seed for reproducibility
np.random.seed(48)

# Shuffle the indices
indices = np.random.permutation(len(X))

# Define the ratio for train-test split
train_ratio = 0.8
test_ratio = 0.2

# Calculate the split index
train_split = math.floor(len(X) * train_ratio)

# Split the dataset
X_train = X[indices[:train_split]]
y_train = y[indices[:train_split]]
X_test = X[indices[train_split:]]
y_test = y[indices[train_split:]]


# Calculate the prior probabilities for each class
def calculate_prior_probabilities(labels):
    class_counts = {}
    total_count = len(labels)

    for label in labels:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    prior_probs = {}
    for label, count in class_counts.items():
        prior_probs[label] = count / total_count

    return prior_probs


# Calculate the likelihood probabilities for each feature
def calculate_likelihood_probabilities(features, labels):
    likelihood_probs = {}
    class_features = {}

    for i in range(len(features)):
        label = labels[i]
        feature = features[i]

        if label in class_features:
            class_features[label].append(feature)
        else:
            class_features[label] = [feature]

    for label, features in class_features.items():
        likelihood_probs[label] = []

        for i in range(len(features[0])):
            feature_values = [feature[i] for feature in features]
            unique_values = set(feature_values)
            value_probs = {}

            for value in unique_values:
                count = feature_values.count(value)
                value_probs[value] = count / len(features)

            likelihood_probs[label].append(value_probs)

    return likelihood_probs


# Train the Naive Bayes classifier
def train(X_train, y_train):
    prior_probs = calculate_prior_probabilities(y_train)
    likelihood_probs = calculate_likelihood_probabilities(X_train, y_train)

    return prior_probs, likelihood_probs


# Classify a new instance
def classify(instance, prior_probs, likelihood_probs):
    class_scores = {}

    for label in prior_probs:
        class_scores[label] = math.log(prior_probs[label])

        for i in range(len(instance)):
            feature_value = instance[i]

            if feature_value in likelihood_probs[label][i]:
                likelihood = likelihood_probs[label][i][feature_value]
                class_scores[label] += math.log(likelihood)

    return max(class_scores, key=class_scores.get)


# Train the Naive Bayes classifier
prior_probs, likelihood_probs = train(X_train, y_train)

# Classify the test instances
predictions = []
for instance in X_test:
    predictions.append(classify(instance, prior_probs, likelihood_probs))

# Classify the test instances
correct_predictions = 0
total_predictions = len(y_test)

for i in range(total_predictions):
    instance = X_test[i]
    true_label = y_test[i]
    predicted_label = classify(instance, prior_probs, likelihood_probs)

    if predicted_label == true_label:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
print("Accuracy:", accuracy)
