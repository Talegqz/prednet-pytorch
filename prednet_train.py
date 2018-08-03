

import prednet
import torch
import numpy as np




if __name__ == '__main__':
    stack_sizes = [3,48,96,192]

    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)
    lamda_l = (1,0.1,0.1,0.1)
    lamda_t = np.ones((12))
    print(lamda_t.shape)

    lamda_t[0] = 0
    datapath_set = ['D:/database/action1-person1-white/','D:/database/action1-person4-white/','D:/database/action1-person5-white/']

    # for datapath in datapath_set:
    #     net = prednet.Prednet(stack_sizes,R_stack_sizes,A_filt_sizes,Ahat_filt_sizes,R_filt_sizes,lamda_l,lamda_t,batch_size=4,n_epochs=40)
    #
    #     net.fit(datapath,part=True,numb_st=0,numb_end=1200)
    #
    #     torch.save(net,'save_model/first_test'+datapath[-14:-7])
    #
    # datapath_ = 'D:/database/action1-person145-white/'
    #
    # net = prednet.Prednet(stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes, lamda_l, lamda_t,
    #                       batch_size=4, n_epochs=4)
    # for all in range(0,10):
    #     for data in range(0,12):
    #         for i in range(0,3):
    #             print("all_npoch_index_%d person_index %d,data_index_ %d_%d "%(all,i,data*100,data*100+100))
    #             net.fit(datapath_set[i], part=True, numb_st=data*100, numb_end=data*100+100,loss_path='all'+str(i)+'/')
    #
    #     torch.save(net, 'save_model/small_er' + datapath_[-16:-7]+'_wiht_eporch'+str(all))
    datapath = 'ourdata/train_145.npy'
    net = prednet.Prednet(stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes, lamda_l, lamda_t,
                          batch_size=4, n_epochs=7)

    net.fit(datapath, part=True, numb_st=0, numb_end=2700,print_every=100,loss_path='save_loss/ourdata_jump_0_3000')
    torch.save(net,'save_model/ourdata_jump_3_')