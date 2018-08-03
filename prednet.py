import torch

import torch.nn as nn
import functools
import torch.optim as optim
import numpy as np
import torch.utils.data as Data
from PIL import Image
from torch.nn import init
import visdom
import os


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)


class Prednet(nn.Module):
    def __init__(self,
                 stack_sizes, R_stack_sizes,
                 A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,

                 lamda_l,
                 lamda_t,
                 pixel_max=1,
                 batch_size=4,
                 n_epochs=10,
                 lr=0.003,
                 optimizer='Adam',

                 use_cuda=True):

        super(Prednet, self).__init__()

        self.batch_size = batch_size
        self.stack_sizes = stack_sizes
        self.nb_layers = len(stack_sizes)
        self.n_epochs = n_epochs
        assert len(R_stack_sizes) == self.nb_layers, 'len(R_stack_sizes) must equal len(stack_sizes)'
        self.R_stack_sizes = R_stack_sizes
        assert len(A_filt_sizes) == (self.nb_layers - 1), 'len(A_filt_sizes) must equal len(stack_sizes) - 1'
        self.A_filt_sizes = A_filt_sizes
        assert len(Ahat_filt_sizes) == self.nb_layers, 'len(Ahat_filt_sizes) must equal len(stack_sizes)'
        self.Ahat_filt_sizes = Ahat_filt_sizes
        assert len(R_filt_sizes) == (self.nb_layers), 'len(R_filt_sizes) must equal len(stack_sizes)'
        self.R_filt_sizes = R_filt_sizes

        self.pixel_max = pixel_max

        self.conv_layers = {c: [] for c in ['i', 'f', 'c', 'o', 'a', 'ahat']}

        parms = []

        for l in range(self.nb_layers):
            for c in ['i', 'o', 'f', 'c']:
                if l == self.nb_layers - 1:
                    input_size = self.stack_sizes[l] * 3
                else:
                    input_size = self.stack_sizes[l] * 3 + self.stack_sizes[l + 1]
                if c == 'c':
                    self.conv_layers[c].append(nn.Sequential(*[nn.Conv2d(input_size, self.R_stack_sizes[l],
                                                                         stride=1,
                                                                         kernel_size=self.R_filt_sizes[l],
                                                                         padding=(-1 + self.R_filt_sizes[l]) / 2
                                                                         ),
                                                               nn.Tanh()]))

                else:

                    self.conv_layers[c].append(nn.Sequential(*[nn.Conv2d(input_size, self.R_stack_sizes[l],
                                                                         kernel_size=self.R_filt_sizes[l],
                                                                         stride=1,
                                                                         padding=(-1 + self.R_filt_sizes[l]) / 2),
                                                               nn.Hardtanh(min_val=0, max_val=1)]))
                parmsdict = dict()
                parmsdict['params'] = self.conv_layers[c][l].parameters()
                parms.append(parmsdict)
            self.conv_layers['ahat'].append(nn.Sequential(*[nn.Conv2d(self.R_stack_sizes[l], self.R_stack_sizes[l],
                                                                      stride=1,
                                                                      kernel_size=self.Ahat_filt_sizes[l],
                                                                      padding=(-1 + self.Ahat_filt_sizes[l]) / 2
                                                                      ),
                                                            nn.ReLU(True)
                                                            ]))
            parmsdict = dict()
            parmsdict['params'] = self.conv_layers['ahat'][l].parameters()

            parms.append(parmsdict)

            if l < self.nb_layers - 1:
                self.conv_layers['a'].append(nn.Sequential(*[nn.Conv2d(2 * self.stack_sizes[l], self.stack_sizes[l + 1],
                                                                       kernel_size=self.A_filt_sizes[l],
                                                                       stride=1,
                                                                       padding=(-1 + self.A_filt_sizes[l]) / 2),
                                                             nn.MaxPool2d(2, 2)]))
                parmsdict = dict()
                parmsdict['params'] = self.conv_layers['a'][l].parameters()
                parms.append(parmsdict)

        self.lamda_l = lamda_l
        self.lamda_t = lamda_t

        self.upsample = nn.Upsample(scale_factor=2)
        self.LSTMrelu = nn.ReLU(True).cuda()
        self.Errorrelu = nn.ReLU(True).cuda()
        # for l in range(0,layer_num):
        #     self.LSTMS.append(nn.LSTM())
        self.states = []
        self.last_state = []
        self.nb_flames = 0
        self.loss = 0
        self.use_cuda = use_cuda
        if self.use_cuda:

            for l in range(self.nb_layers):
                for c in ['i', 'o', 'f', 'c', 'ahat']:
                    self.conv_layers[c][l].cuda()
                if l < self.nb_layers - 1:
                    self.conv_layers['a'][l].cuda()

            self.cuda()
            self.dtype = torch.cuda.FloatTensor

        self.apply(init_weights)
        self.parameters()
        self.initflag = False
        # print(self.parameters())
        # for i in self.parameters():
        #     print(i)
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(parms, lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(parms, lr=lr)
        else:
            raise ValueError('not a recongized optimizer')
        # for i in range(self.nb_layers):
        #     for c in ['i', 'f', 'c', 'o', 'a', 'ahat']:
        #         if c=='ahat' and l==0:
        #             pass
        #         else:
        #             self.optimizer.add_param_group(self.conv_layers[c][l].parameters())

    def forward(self, x):
        self.batch_size = len(x)
        if self.use_cuda:
            x = x.cuda()
        x = torch.transpose(x, 0, 1)

        self.nb_flames = len(x)-1

        # first, make A 0t = xt

        for t in range(self.nb_flames + 1):  # additional t is used  to predict
            if self.initflag == False:
                self.states.append([])
            else:
                pass
            for l in range(self.nb_layers):
                if self.initflag == False:
                    self.states[t].append(dict())
                else:
                    pass
            if t < self.nb_flames:
                self.states[t][0]['a'] = x[t]
                self.states[t][0]['a'].cuda()
                # E0l,R0l is zero
        self.initflag = True
        for l in range(self.nb_layers):
            picshape = x.shape[-2:]

            # picshape[0] = picshape[0]/(2**l)
            # picshape[1] = picshape[1] / (2 ** l)

            self.states[0][l]['e'] = torch.zeros(
                [self.batch_size, 2 * self.stack_sizes[l], picshape[0] / (2 ** l), picshape[1] / (2 ** l)]).cuda()
            self.states[0][l]['r'] = torch.zeros(
                [self.batch_size, self.stack_sizes[l], picshape[0] / (2 ** l), picshape[1] / (2 ** l)]).cuda()
            self.states[0][l]['c'] = torch.zeros(
                [self.batch_size, self.stack_sizes[l], picshape[0] / (2 ** l), picshape[1] / (2 ** l)]).cuda()

        # calculatation prednet states

        for t in range(1, self.nb_flames):
            for l in reversed((range(self.nb_layers))):
                input = torch.cat((self.states[t - 1][l]['r'], self.states[t - 1][l]['e']), dim=1)
                if l < self.nb_layers - 1:
                    r_up = self.upsample(self.states[t][l + 1]['r'])

                    input = torch.cat((input, r_up), dim=1)
                # cal r t l  convlstm
                i = self.conv_layers['i'][l](input)
                f = self.conv_layers['f'][l](input)
                o = self.conv_layers['o'][l](input)
                self.states[t][l]['c'] = f * self.states[t - 1][l]['c'] + i * self.conv_layers['c'][l](input)
                self.states[t][l]['r'] = o * self.LSTMrelu(self.states[t][l]['c'])

            for l in range(self.nb_layers):

                self.states[t][l]['ahat'] = self.conv_layers['ahat'][l](self.states[t][l]['r'])
                if l == 0:
                    self.states[t][l]['ahat'] = torch.clamp(self.states[t][l]['ahat'], 0, self.pixel_max)

                e_up = self.Errorrelu(self.states[t][l]['ahat'] - self.states[t][l]['a'])
                e_down = self.Errorrelu(self.states[t][l]['a'] - self.states[t][l]['ahat'])

                self.states[t][l]['e'] = torch.cat((e_up, e_down), dim=1)

                if l < self.nb_layers - 1:
                    self.states[t][l + 1]['a'] = self.conv_layers['a'][l](self.states[t][l]['e'])
        # prediction

        for l in reversed((range(self.nb_layers))):
            input = torch.cat((self.states[self.nb_flames - 1][l]['r'], self.states[self.nb_flames - 1][l]['e']), dim=1)
            if l < self.nb_layers - 1:
                r_up = self.upsample(self.states[self.nb_flames][l + 1]['r'])

                input = torch.cat((input, r_up), dim=1)
            # cal r t l  convlstm
            i = self.conv_layers['i'][l](input)
            f = self.conv_layers['f'][l](input)
            o = self.conv_layers['o'][l](input)
            self.states[self.nb_flames][l]['c'] = f * self.states[self.nb_flames - 1][l]['c'] + i * \
                                                  self.conv_layers['c'][l](input)
            self.states[self.nb_flames][l]['r'] = o * self.LSTMrelu(self.states[self.nb_flames][l]['c'])

        for l in range(self.nb_layers):

            self.states[self.nb_flames][l]['ahat'] = self.conv_layers['ahat'][l](self.states[self.nb_flames][l]['r'])
            # print(self.states[self.nb_flames][l]['ahat'])
            if l == 0:
                self.states[self.nb_flames][l]['ahat'] = torch.clamp(self.states[self.nb_flames][l]['ahat'], 0,
                                                                     self.pixel_max)
                # print(self.states[self.nb_flames][l]['ahat'])

    def compute_loss(self, data_patch):

        self.batch_size = len(data_patch)

        loss = torch.tensor(0.0, requires_grad=True).cuda()
        self(data_patch[0])
        for t in range(self.nb_flames):
            this_layer_loss = torch.tensor(0.0, requires_grad=True).cuda()
            for l in range(self.nb_layers):
                here_loss = torch.mean(self.states[t][l]['e'])
                this_layer_loss = this_layer_loss + self.lamda_l[l] * here_loss

            loss = loss + this_layer_loss * self.lamda_t[t]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

        pass

    def test(self, X, use_net, use_data, part=False, numb_st=0, numb_end=10, print_every=10):
        if os.path.exists('save_result/use_net_' + use_net + '____use_data_' + use_data):
            pass
        else:
            os.makedirs('save_result/use_net_' + use_net + '____use_data_' + use_data)

        self.part = part

        X = self.test_create_dataloader(X, part, numb_st, numb_end)
        self.train()
        start = numb_st

        for t, data in enumerate(X):
            start += 1
            input = data[0]
            self(input)

            if os.path.exists('save_result/use_net_' + use_net + '____use_data_' + use_data + '/data' + str(start)):
                pass
            else:
                os.makedirs('save_result/use_net_' + use_net + '____use_data_' + use_data + '/data' + str(start))
            save_path = 'save_result/use_net_' + use_net + '____use_data_' + use_data + '/data' + str(start)

            def save_pic(data, name):
                data = data.cpu().detach().numpy()
                data = data[0] * 255
                data = np.transpose(data, (1, 2, 0))
                data = np.uint8(data)
                im = Image.fromarray(data)

                im.save(save_path + '/' + name + '.png')

            for t in range(1, self.nb_flames + 1):
                pred = self.states[t][0]['ahat']
                save_pic(pred, 'pred_t_' + str(t))

                ground_truth = self.states[t - 1][0]['a']
                save_pic(ground_truth, 'input_t_' + str(t - 1))

                latent = self.states[t][0]['r']

                save_pic(latent, 'latent_t_' + str(t))

            the_true = data[0][0][self.nb_flames]

            the_true = the_true*255

            the_true = np.transpose(the_true,(1,2,0))

            the_true = np.uint8(the_true)

            the_true_im = Image.fromarray(the_true)

            the_true_im.save(save_path+'/true_next_flame.png')

            # source = self.states[self.nb_flames - 1][0]['a']
            #
            # source = source.cpu().detach().numpy()
            # source = source[0] * 255
            #
            # source = np.transpose(source, (1, 2, 0))
            # source = np.uint8(source)
            # this_image = Image.fromarray(source, 'RGB')
            # # this_image = this_image.convert('RGB')
            # this_image.save('save_result/source' + str(start) + '.png')
            #
            # pred = self.states[self.nb_flames][0]['ahat']
            #
            # pred = pred.cpu().detach().numpy()
            # pred = pred[0] * 255
            # pred = np.transpose(pred, (1, 2, 0))
            # pred = np.uint8(pred)
            # this_image = Image.fromarray(pred, 'RGB')
            # # this_image = this_image.convert('RGB')
            # this_image.save('save_result/pred' + str(start) + '.png')
            #
            # latent = self.states[self.nb_flames][0]['r']
            #
            # latent = latent.cpu().detach().numpy()
            # latent = latent[0] * 255
            # latent = np.transpose(latent, (1, 2, 0))
            # latent = np.uint8(latent)
            # this_image = Image.fromarray(latent, "RGB")
            # # this_image = this_image.convert('RGB')
            # this_image.save('save_result/latent' + str(start) + '.png')

    def _train(self, data, epoch, print_every=10):
        pass

    def test_create_dataloader(self, datapath, part=False, numb_st=0, numb_end=20):
        x = np.load(datapath)
        x = np.transpose(x, (0, 1, 4, 2, 3))
        if part:
            x = x[numb_st:numb_end]


        x = torch.from_numpy(x)
        x = x.float() / 255



        mydata = Data.TensorDataset(x)

        loader = Data.DataLoader(
            mydata,
            batch_size=1,
            shuffle=False
        )

        return loader

    def create_dataloader(self, datapath, part=False, numb_st=0, numb_end=20):
        flag = 0
        x = np.load(datapath)
        x = np.transpose(x,(0,1,4,2,3))
        if part:
            x = x[numb_st:numb_end]


        x = torch.from_numpy(x)
        x = x.float() / 255

        mydata = Data.TensorDataset(x)
        loader = Data.DataLoader(
            mydata,
            batch_size=self.batch_size,
            shuffle=True
        )
        return loader

    def fit(self, datapath, part=False, numb_st=0, numb_end=10, print_every=100,loss_path=''):
        # X = Variable(torch.from_numpy(X))
        print('train_data%s,start_num%d,end_num_%d'%(datapath,numb_st,numb_end))


        self.part = part

        X = self.create_dataloader(datapath, part, numb_st, numb_end)

        if os.path.exists(loss_path):
            pass
        else:
            os.makedirs(loss_path)
        save_loss_path = loss_path

        self.train()

        for epoch in range(self.n_epochs):
            loss_data = []
            # if epoch % 2 == 0:
            #     torch.save(self.state_dict(), 'save_model/epoch' + str(epoch) + ',pkl')
            vis = visdom.Visdom()

            print('epoch %d'%epoch)
            for t, data in enumerate(X):

                loss = self.compute_loss(data)
                loss = loss.cpu().detach().numpy()
                loss_data.append(loss)
                pass
                # if self.part:
                #     print('Batch %d, save_loss = %.4f' % (t + 1, loss))

                if (t + 1) % print_every == 0:
                    print('Batch %d, save_loss = %.4f' % (t + 1, loss))

            print('loss = %.4f' % loss)
            loss_data = np.array(loss_data)
            np.save(save_loss_path+'/'+str(epoch)+'loss.npy', loss_data)
            vis.line(loss_data)
