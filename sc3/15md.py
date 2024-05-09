import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define evaluation function
def evaluate_features(features):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train[:, features], y_train)
    y_pred = clf.predict(X_test[:, features])
    return accuracy_score(y_test, y_pred)

# Genetic Algorithm
def genetic_algorithm(population_size, num_generations, mutation_rate):
    # Initialize population
    population = np.random.randint(2, size=(population_size, X.shape[1]))

    for generation in range(num_generations):
        # Evaluate fitness
        fitness_scores = [evaluate_features(features) for features in population]

        # Select parents
        parents = population[np.argsort(fitness_scores)[-2:]]

        # Crossover
        crossover_point = np.random.randint(1, X.shape[1])
        children = np.hstack((parents[0][:crossover_point], parents[1][crossover_point:]))

        # Mutation
        for idx in range(len(children)):
            if np.random.random() < mutation_rate:
                mutation_point = np.random.randint(0, X.shape[1])
                children[mutation_point] = 1 - children[mutation_point]

        # Replace old population with children
        population[:-2] = children

    # Return the best feature set
    best_features = population[np.argmax(fitness_scores)]
    return best_features

# Example usage
best_feature_set = genetic_algorithm(population_size=10, num_generations=5, mutation_rate=0.1)
print("Best feature set:", best_feature_set)
