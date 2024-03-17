import random
import math

# Define the fitness function (three-input nonlinear function)
def fitness_function(x, y, z):
    return -(x**2 + y**2 + z**2) + 10 * (math.cos(2*math.pi*x) + math.cos(2*math.pi*y) + math.cos(2*math.pi*z))

# Initialize the population
def initialize_population(pop_size, lower_bound, upper_bound):
    return [(random.uniform(lower_bound, upper_bound), random.uniform(lower_bound, upper_bound), random.uniform(lower_bound, upper_bound)) for _ in range(pop_size)]

# Select parents based on their fitness
def select_parents(population):
    total_fitness = sum(fitness_function(x, y, z) for x, y, z in population)
    roulette_wheel = [fitness_function(x, y, z) / total_fitness for x, y, z in population]
    parent1 = random.choices(population, weights=roulette_wheel)[0]
    parent2 = random.choices(population, weights=roulette_wheel)[0]
    return parent1, parent2

# Perform crossover to create a new generation
def crossover(parent1, parent2, crossover_prob=0.7):
    if random.random() < crossover_prob:
        crossover_point1 = random.uniform(0, 1)
        crossover_point2 = random.uniform(0, 1)
        child1 = (
            crossover_point1 * parent1[0] + (1 - crossover_point1) * parent2[0],
            crossover_point1 * parent1[1] + (1 - crossover_point1) * parent2[1],
            crossover_point1 * parent1[2] + (1 - crossover_point1) * parent2[2]
        )
        child2 = (
            crossover_point2 * parent1[0] + (1 - crossover_point2) * parent2[0],
            crossover_point2 * parent1[1] + (1 - crossover_point2) * parent2[1],
            crossover_point2 * parent1[2] + (1 - crossover_point2) * parent2[2]
        )
        return child1, child2
    else:
        return parent1, parent2

# Perform mutation in the population
def mutate(individual, mutation_prob=0.01):
    x, y, z = individual
    if random.random() < mutation_prob:
        x += random.uniform(-0.1, 0.1)
    if random.random() < mutation_prob:
        y += random.uniform(-0.1, 0.1)
    if random.random() < mutation_prob:
        z += random.uniform(-0.1, 0.1)
    return x, y, z

# Genetic Algorithm
def genetic_algorithm(generations, pop_size, lower_bound, upper_bound):
    population = initialize_population(pop_size, lower_bound, upper_bound)
    for gen in range(generations):
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = new_population
        best_individual = max(population, key=lambda ind: fitness_function(*ind))
        print(f"Generation {gen+1}: Best individual - {best_individual}, Fitness - {fitness_function(*best_individual)}")
    return max(population, key=lambda ind: fitness_function(*ind))

if __name__ == "__main__":
    generations = 5
    pop_size = 100
    lower_bound = -1
    upper_bound = 1
    best_solution = genetic_algorithm(generations, pop_size, lower_bound, upper_bound)
    print(f"Best solution found: {best_solution}, Fitness: {fitness_function(*best_solution)}")

