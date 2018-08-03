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
    # print(lamda_t.shape)
    datapath = 'ourdata/train_145.npy'
    lamda_t[0] = 0
    # net = prednet.Prednet(stack_sizes,R_stack_sizes,A_filt_sizes,Ahat_filt_sizes,R_filt_sizes,lamda_l,lamda_t,batch_size=4,n_epochs=1000)

    net = torch.load('save_model/ourdata_jump_3_')
    # # net.fit(datapath,True,0,20)
    # net.test(datapath,'person145','person4',True,1201,1400)
    net.test(datapath,'ourdata_145','for_pix2pix_person_145',True,0,2000)
    pass