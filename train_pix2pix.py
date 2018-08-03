import pix2pix
import numpy as np
import torch.utils.data as Data
import torch
import visdom
import time
from options.train_options import TrainOptions
from util.visualizer import Visualizer
import torchvision.transforms
import sklearn.preprocessing
from PIL import Image
import os
save_path = ''

def save_pic(data, name):
    data = data.cpu().detach().numpy()
    data = ((data[0])/2+0.5)*255
    data = np.transpose(data, (1, 2, 0))
    data = np.uint8(data)
    im = Image.fromarray(data)

    im.save(save_path + '/' + name + '.png')
    return im

part = 1300
datapath = 'D:/database/action1-person1-white/'

x = np.load(datapath+'this_data_noreshape.npy')
y = np.load(datapath+'just_scale_one_point_map.npy')
# z = np.load(datapath+'true.npy')

def process(x):

    x = x[0:part]
    x = torch.from_numpy(x)
    x = x.float() / 255
    x = (x - 0.5) / 0.5
    return x

x = process(x)
# y = np.transpose(y,(0,2,3,1))
y = y[0:part]
y =torch.from_numpy(y)
# z = z[0:part]


# x = x.astype(dtype='float32')
# y = y.astype(dtype='float32')
# z = z.astype(dtype='float32')
#
#
# for i in range(len(x)):
#     for j in range(len(x[i])):
#         x[i][j] = x[i][j]*1.0
#         x[i][j] = (x[i][j]-np.mean(x[i][j]))/np.std(x[i][j])
#
# for i in range(len(y)):
#     for j in range(len(y[i])):
#         y[i][j]=y[i][j]*1.0
#         y[i][j] = (y[i][j]-np.mean(y[i][j]))/np.std(y[i][j])
#
# for i in range(len(z)):
#     for j in range(len(z[i])):
#         z[i][j]=z[i][j]*1.0
#         z[i][j] = (z[i][j]-np.mean(z[i][j]))/np.std(z[i][j])




# y = torch.from_numpy(y)
# y = y.float()/255
# y = (y-0.5)/0.5
#
# z = torch.from_numpy(z)
# z = z.float()/255
# z = (z-0.5)/0.5


print('finishdata')
mydata = Data.TensorDataset(x, y)

pixnet = pix2pix.Pix2Pix(18, 3, 18+3)
opt = TrainOptions().parse()
total_steps = 0
opt.batchSize = 4
opt.niter = 10000
opt.print_freq = 10
opt.display_freq = 10
opt.save_epoch_freq = 20
loader = Data.DataLoader(
    mydata,
    batch_size=opt.batchSize,
    shuffle=True
)
dataset_size = part



visualizer = Visualizer(opt)
loss_G_GAN = []
loss_G_L1 = []
loss_D_real = []
loss_D_fake = []

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(loader):

        iter_start_time = time.time()
        if total_steps % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        pixnet.set_input(data)
        pixnet.optimize_parameters()
        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(pixnet.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            losses = pixnet.new_get_current_losses()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_losses(epoch, epoch_iter, losses[0], t, t_data)
            if opt.display_id > 0:
                visualizer.new_plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, losses)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            # pixnet.save_networks('latest')

        iter_data_time = time.time()
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        # pixnet.save_networks('latest')
        # pixnet.save_networks(epoch)
        torch.save(pixnet.state_dict(),'save_model/pix2pix/point_map_epoch_%d.pkl'%epoch)
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    pixnet.update_lr(0.95)
    #     if t%100==0:
    #         loss = pixnet.get_current_losses()
    #         print(str(t),loss)
    #
    # pixnet.update_lr(0.95)
    #
    # loss = pixnet.get_current_losses()
    # loss_G_GAN.append(loss['G_GAN'])
    # loss_G_L1.append(loss['G_L1'])
    # loss_D_real.append(loss['D_real'])
    # loss_D_fake.append(loss['D_fake'])

    #
    # if e%100==0:
    #     save_path = 'save_result/pix2pix/just11111__lamda_10000' + '/data' + str(e)
    #     if os.path.exists(save_path):
    #         pass
    #     else:
    #         os.makedirs(save_path)
    #     pixnet.set_input(data)
    #     pixnet.forward()
    #     pred = data[1]
    #     result = pixnet.fake
    #     true = data[2]
    #     true2 = pixnet.true
    #     loss = pixnet.cal_g1_loss()
    #
    #     pred = save_pic(pred, 'source_pred')
    #     result = save_pic(result, 'repair_pred')
    #     true = save_pic(true, 'ground_truth')
    #
    #     print('print')
    #     name = ['G_GAN','D_real','D_fake']
    #     # vis.line(np.array(loss_G_GAN), opts={
    #     # 'title':'loss_G_GAN_%d'%e ,
    #     #
    #     # 'xlabel': 'epoch',
    #     # 'ylabel': 'loss'})
    #     all_loss = []
    #     all_loss.append(np.array(loss_G_GAN))
    #     all_loss.append( np.array(loss_D_real))
    #     all_loss.append(np.array(loss_D_fake))
    #
    #     all_loss = np.array(all_loss)
    #     all_loss = np.transpose(all_loss,(1,0))
    #     vis.line(all_loss,opts={
    #     'title':'loss_D_real_%d'%e ,
    #     'legend':name,
    #     'xlabel': 'epoch',
    #     'ylabel': 'loss'})
    #
    #     vis.line(np.array(loss_G_L1),opts={
    #     'title':'loss_G_L1_%d'%e ,

        # 'xlabel': 'epoch',
        # 'ylabel': 'loss'})
        # vis.line(np.array(loss_D_fake),opts={
        # 'title':'loss_D_fake_%d'%e ,
        #
        # 'xlabel': 'epoch',
        # 'ylabel': 'loss'})
        # np.save(all_loss)
        # loss_G_GAN = []
        # loss_G_L1 = []
        # loss_D_real = []
        # loss_D_fake = []

        # torch.save(pixnet.state_dict(),'save_model/pix2pix/params_%d.pkl'%e)



