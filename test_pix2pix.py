import pix2pix
import numpy as np
import torch.utils.data as Data
import torch
import visdom
from PIL import Image
import os

datapath = 'save_result/use_net_ourdata_145____use_data_person_6/'
use_net = 'ourdata_145'
use_data = 'person_6'
x = np.load(datapath+'input.npy')
y = np.load(datapath+'pred.npy')
z = np.load(datapath+'true.npy')




x = torch.from_numpy(x)
x = x.float()/255
x = (x-0.5)/0.5

y = torch.from_numpy(y)
y = y.float()/255
y = (y-0.5)/0.5

z = torch.from_numpy(z)
z = z.float()/255
z = (z-0.5)/0.5

print('finishdata')
mydata = Data.TensorDataset(x,y,z)


loader = Data.DataLoader(
    mydata,
    batch_size=1,
    shuffle=False
)


save_path = ''
def save_pic(data, name):
    data = data.cpu().detach().numpy()
    data = ((data[0])/2+0.5)*255
    data = np.transpose(data, (1, 2, 0))
    data = np.uint8(data)
    im = Image.fromarray(data)

    im.save(save_path + '/' + name + '.png')
    return im
def mse(pic1,pic2):
    pic1 = np.array(pic1)
    pic2 = np.array(pic2)
    cha = np.fabs(pic1-pic2)
    return np.mean(cha)

pixnet = pix2pix.Pix2Pix(11*3,3,11*3)
pixnet.load_state_dict(torch.load('save_model/pix2pix/params_20.pkl'))
for t,data in enumerate(loader):

    save_path = 'save_result/pix2pix/usenet' + use_net + '____use_data_' + use_data + '/data' + str(t)
    if os.path.exists(save_path):
            pass
    else:
        os.makedirs(save_path)
    pixnet.set_input(data)
    pixnet.forward()
    pred = data[1]
    result = pixnet.fake
    true = data[2]
    true2 = pixnet.true
    loss = pixnet.cal_g1_loss()

    pred = save_pic(pred,'source_pred')
    result = save_pic(result,'repair_pred')
    true = save_pic(true,'ground_truth')
    true2  =save_pic(true2,'maybegroundtruth')
    print(loss)
    print(mse(pred, true2))
    print(mse(pred,true),'pred___true')
    print(mse(true, result), 'repair___true')





