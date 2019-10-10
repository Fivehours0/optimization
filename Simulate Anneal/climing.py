import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# number of workers
n = 100

# 初始化方法的创建
def solution_init():
    # working hours
    dij = np.random.uniform(low=0, high=3, size=(n, n))
    # 产生随机初始
    mission = np.random.permutation(np.arange(100))
    # 独热编码，方便后面的矩阵点乘
    one_hot_encoder = OneHotEncoder(sparse=False)
    mission_encoder = one_hot_encoder.fit_transform(mission.reshape(-1, 1))
    # 计算总的工作时间
    work_time = np.sum(mission_encoder * dij)
    return {'dij': dij, 'solution': mission_encoder, 'work_time': work_time}

# 随机交换两个员工的工作任务, 返回新的solution
def choose_one_solution(state):
    exchange = np.random.randint(low=0, high=100, size=2)
    new_solution = state['solution']
    temp = np.array(new_solution[exchange[0]])
    new_solution[exchange[0]] = new_solution[exchange[1]]
    new_solution[exchange[1]] = temp
    return new_solution

# 计算当前solution下的总工作时间,并更新方法,用于爬山算法
def cal_work_time_climb(state, new_solution):
    new_work_time = np.sum(new_solution * state['dij'])
    if new_work_time < state['work_time']: 
        state['solution'] = new_solution
        state['work_time'] = new_work_time
    return state

# 计算当前solution下的总工作时间,并更新方法，用于SA
def cal_work_time_SA(state, new_solution, TK):
    new_work_time = np.sum(new_solution * state['dij'])
    if new_work_time < state['work_time']: 
        state['solution'] = new_solution
        state['work_time'] = new_work_time
    else:
        random_number = np.random.uniform(low=0, high=1, size=1)
        f_sub = new_work_time - state['work_time']
        value = np.exp(-f_sub/TK)
        if(value>random_number):
            state['solution'] = new_solution
            state['work_time'] = new_work_time
    return state

########### 爬山算法 ############
state_SA = state_climb = solution_init()
state_record_climb = []
for i in range(20):
    state_record_climb.append(state_climb['work_time'])# 记录工作时间
    new_solution = choose_one_solution(state_climb)# 生成新方法
    state_climb = cal_work_time_climb(state_climb, new_solution)# 计算新方法的工作时间并更新方法

########### SA ############
T_INITIAL = 100# 初始温度
T_STOP = 20# 终止温度
T_INTERVAL = 20# 间隔温度
INNER_LOOP = 5# 内循环次数
OUTTER_LOOP = 4# 外循环次数

state_record_SA = []
best_state = dict(state_SA)# 记录过程中最小工作时间的方法

for i in range(OUTTER_LOOP):# 外循环
    for j in range(INNER_LOOP):# 内循环
        state_record_SA.append(state_SA['work_time'])# 记录工作时间
        new_solution = choose_one_solution(state_SA)# 生成新方法
        state_SA = cal_work_time_SA(state_SA, new_solution, T_INITIAL)# 计算新方法的工作时间并更新方法
        if(best_state['work_time']>state_SA['work_time']):
            best_state = dict(state_SA)
    T_INITIAL = T_INITIAL - T_INTERVAL# generate new TK
    
print(best_state['work_time'])
plt.figure()
plt.plot(state_record_climb, c='r')
plt.plot(state_record_SA, c='b')
plt.legend(('climb', 'SA'))
plt.show()