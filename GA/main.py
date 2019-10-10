import GA
import numpy as np

# the mean of population equals to solution
# to maximizing follow equation: y = w1x1 + w2x2 + w3x3 + w4x4  + w5x5 + w6x6 

equation_input = [4, -2, 3.5, 5, -11, -4.7]# inputs of the equation
num_weights = 6# number of weight we are looking to optimize
sol_per_pop = 8# define the population size, it meas that there are 8 individuals in a population, each individual in the population will definitely have 6 genes
pop_size = (sol_per_pop, num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)

num_generations = 5# number of generation
num_parents_mating = 4

for i in range(num_generations):
    # measing the fitness of each chromosome in the population
    fitness = GA.cal_pop_fitness(equation_input, new_population)
    # selecting the best parents in the population for mating
    parents  = GA.select_mating_pool(new_population, fitness, num_parents_mating)
    # generating next generation using crossover
    offspring_crossover = GA.crossover(parents, (pop_size[0]-parents.shape[0], num_weights))
    # adding some variations to the offspring using mutation
    offspring_mutation = GA.mutation(offspring_crossover)# 由于offspring_crossover是array, 所以是按引用传递，就地修改了offspring_crossover
    # creating new population based on parents and offspring
    new_population[0: parents.shape[0]] = parents
    new_population[parents.shape[0]: ] = offspring_mutation

fitness = GA.cal_pop_fitness(equation_input, new_population)
best_match_idx = np.argmax(fitness)

print("best solution: ", new_population[best_match_idx, :])
print('best solution fitness: ', fitness[best_match_idx])