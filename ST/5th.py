import random

# Define the fitness function (our objective function to maximize)
def fitness_function(individual):
    x = sum(individual)
    return -x**2 + 6*x + 9

# Initialize the population
def initialize_population(pop_size, gene_length, lower_bound, upper_bound):
    return [[random.uniform(lower_bound, upper_bound) for _ in range(gene_length)] for _ in range(pop_size)]

# Select parents based on their fitness
def select_parents(population):
    total_fitness = sum(fitness_function(individual) for individual in population)
    roulette_wheel = [fitness_function(individual) / total_fitness for individual in population]
    parent1 = random.choices(population, weights=roulette_wheel)[0]
    parent2 = random.choices(population, weights=roulette_wheel)[0]
    return parent1, parent2

# Perform crossover to create a new generation
def crossover(parent1, parent2, crossover_prob=0.7):
    if random.random() < crossover_prob:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        return parent1, parent2

# Perform mutation in the population
def mutate(individual, mutation_prob=0.01):
    mutated_individual = individual
    for i in range(len(individual)):
        if random.random() < mutation_prob:
            mutated_individual[i] += random.uniform(-0.1, 0.1) # Small modification to the gene value
    return mutated_individual

# Genetic Algorithm
def genetic_algorithm(generations, pop_size, gene_length, lower_bound, upper_bound):
    population = initialize_population(pop_size, gene_length, lower_bound, upper_bound)
    for gen in range(generations):
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = new_population
        best_individual = max(population, key=fitness_function)
        print(f"Generation {gen + 1}: Best individual - {best_individual}, Fitness - {fitness_function(best_individual)}")
    return max(population, key=fitness_function)

if __name__ == "__main__":
    generations = 5
    pop_size = 100
    gene_length = 5  # Define the length of each individual's genes
    lower_bound = -10
    upper_bound = 10
    best_solution = genetic_algorithm(generations, pop_size, gene_length, lower_bound, upper_bound)
    print(f"Best solution found: {best_solution}, Fitness: {fitness_function(best_solution)}")
