import numpy as np
from PIL import Image
import os


datapath = 'save_result/use_net_ourdata_145____use_data_person_6'


all_input = []
all_pred=[]
all_true = []
for i in list(os.listdir(datapath)):
    this = datapath+'/'+i
    input = []
    for num in range(10):
        image = Image.open(this+'/'+"input_t_"+str(num)+'.png')
        im = np.array(image)
        im = np.transpose(im,(2,0,1))

        input.append(im)
    input = np.concatenate(input,0)
    all_input.append(input)
    # print(input.shape)
    image = Image.open(this+'/'+'pred_t_10'+'.png')
    im = np.array(image)
    im = np.transpose(im, (2, 0, 1))
    all_pred.append(im)


    image = Image.open(this+'/'+'true_next_flame'+'.png')
    im = np.array(image)
    im = np.transpose(im, (2, 0, 1))
    all_true.append(im)

all_true = np.array(all_true)
print(all_true.shape)
all_input = np.array(all_input)
print(all_input.shape)
all_pred = np.array(all_pred)
print(all_pred.shape)

np.save(datapath+'/'+'input.npy',all_input)
np.save(datapath+'/'+'pred.npy',all_pred)
np.save(datapath+'/'+'true.npy',all_true)
