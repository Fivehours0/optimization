import numpy as np
import matplotlib.pyplot as plt
import POS

num_individual = 100
num_variable = 10
num_iteration = 100
variable_max = 10
variable_min = -10

fitness_record = []
position = np.random.uniform(low=variable_min, high=variable_max, size=(num_individual, num_variable))
speed_arr = np.random.uniform(low=-5, high=5, size=(num_individual, num_variable))
state = {'pbest': position, 'gbest': np.array([0, 0])}

plt.figure()
plt.ion()
for  i in range(num_iteration):
    # show points' position
    plt.scatter(position[:, 0], position[:, 1])
    plt.pause(0.5)
    # calculate adaptablity
    fitness = POS.cal_fitness(position)
    # updata the gbest
    min_fitness = np.min(fitness)
    state['gbest'] = position[np.argmin(fitness)]
    # updata the speed
    speed_arr = POS.speed_updata(position, speed_arr, state['pbest'], state['gbest'], 0.5, 2, 2)
    # updata the position
    position += speed_arr
    position = POS.position_bound(position)
    state['pbest'] = position
    # record best fitness
    fitness_record.append(min_fitness)
    plt.clf()
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])

plt.ioff()

plt.figure()
plt.plot(fitness)
plt.show()
