import visdom
import numpy as np

vis = visdom.Visdom()

loss = []
start_epoch = 0
end_epoch = 10
for i in range(start_epoch, end_epoch):
    this_loss = np.load('save_loss/epoch' + str(i) + '.npy')
    loss = np.concatenate((loss, this_loss))

loss = np.array(loss)

vis.line(loss, opts={
    'title': str(start_epoch+1)+'---'+str(end_epoch)+'  enpoch',

    'xlabel': 'epoch',
    'ylabel': 'loss'}, )
