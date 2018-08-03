
import numpy as np
datapath_set = ['D:/database/action1-person1-white/','D:/database/action1-person4-white/','D:/database/action1-person5-white/']

flag = 0
datapath = 'D:/database/action1-person145-white/'
for i in datapath_set:
    if flag == 0:
        x = np.load(i+'prednetflames12.npy')
        y = np.load(i+'next_one_flame12.npy')
        flag = 1
    else:
        this_x = np.load(i + 'prednetflames12.npy')
        this_y = np.load(i + 'next_one_flame12.npy')
        x = np.concatenate((x,this_x))
        y = np.concatenate((y,this_y))

np.save(datapath+'prednetflames12.npy',x)
np.save(datapath+'next_one_flame12.npy',y)
