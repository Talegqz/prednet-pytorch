import numpy as np


data_set = ['D:/database/action1-person1-white/','D:/database/action1-person4-white/','D:/database/action1-person5-white/',
            'D:/database/action1-person7-white/']

data_set = ['D:/database/action1-person6-white/']

data_set = ['D:/database/action2-person6-white/']


data_all = []
test_all = []
step = 3
flames_num = 11
for datapath in data_set:
    data = np.load(datapath+'this_data_noreshape.npy')
    data = np.transpose(data,(0,2,3,1))
    for i in range(len(data)-200):
        pic = []
        for k in range(flames_num):
            pic.append(data[i+k*step])
        data_all.append(pic)
    for i in range(len(data)-200,len(data)-3*flames_num):
        pic = []
        for k in range(flames_num):
            pic.append(data[i+k*step])
        test_all.append(pic)


data_all = np.array(data_all)
test_all = np.array(test_all)

np.random.shuffle(data_all)
np.random.shuffle(test_all)

np.save('ourdata/act2_train_6.npy',data_all)
np.save('ourdata/act2_test_6.npy',test_all)



