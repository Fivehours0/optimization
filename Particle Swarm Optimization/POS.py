import numpy as np

def cal_fitness(position):
    fitness_arr = np.sum(np.square(position), axis=1)
    return fitness_arr

def speed_updata(current_position, old_v, pbest, gbest, w, c1, c2):
    new_v = w * old_v + c1 * np.random.uniform(size=1) * (pbest - current_position) + c2 * np.random.uniform(size=1) * (gbest - current_position)
    return new_v

def position_bound(position):
    position = np.where(position>10, 10, position)
    position = np.where(position<-10, -10, position)
    return position
