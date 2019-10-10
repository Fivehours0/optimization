import numpy as np

def cal_pop_fitness(equation_inputs, pop):
    return np.sum(pop*equation_inputs, axis=1)

def select_mating_pool(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    for parent in range(num_parents):
        max_fitness_idx = np.argmax(fitness)
        parents[parent, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999
    return parents
 
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # the point at which crossover takes place between two parents. Usually, it is at the center
    crossover_point = np.uint8(offspring_size[1] / 2)

    for i in range(offspring_size[0]):
        # index of first parent to mate
        parents_idx1 = i % parents.shape[0]
        # index of second parent to mate
        parents_idx2 = (i+1) % parents.shape[0]
        # the new offspring will have its first half of its gene taken from the first parent
        offspring[i, 0:crossover_point] = parents[parents_idx1, 0:crossover_point]
        # the new offspring will have its second half of its gene taken from the second parent
        offspring[i, crossover_point: ] = parents[parents_idx2, crossover_point]
    return offspring

def mutation(offspring_crossover):
    for idx in range(offspring_crossover.shape[0]):
        random_value = np.random.uniform(low=-1.0, high=1.0, size = 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    return offspring_crossover

