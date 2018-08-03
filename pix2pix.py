import torch
import torch
from collections import OrderedDict
import torch.nn as nn
import functools
import torch.optim as optim
from models import networks
import os
from util.image_pool import ImagePool

class Pix2Pix(nn.Module):
    def __init__(self,
                input_G,
                output_G,
                 input_D,
                 isTarin = True,
                 batch_size=4,
                 n_epochs=10,
                 lr=0.0002,
                 ngf = 64,
                 which_model_netG = 'resnet_6blocks',
                 optimizer='Adam',
                 no_dropout = True,
                 init_type = 'normal',
                 norm = 'batch',
                 init_gain = 0.02,
                 gpu_ids = [0],
                 which_model_netD = 'basic',
                 pool_size = 50,
                 lambda_L1 = 100,

                 beta1 = 0.5,
                 use_cuda=True):
        super(Pix2Pix, self).__init__()
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        self.isTrain = isTarin
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.loss_names_gan = ['G_GAN', 'D_real', 'D_fake']
        self.loss_names_g1 = ['G_L1']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['fake', 'true']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.lambda_L1 = lambda_L1
        self.netG = networks.define_G(input_G,output_G, ngf,
                                      which_model_netG, norm, not no_dropout,init_type, init_gain,
                                      gpu_ids)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(18, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 60 * 60, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc.to(self.device)
        self.localization.to(self.device)
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        if self.isTrain:
            self.netD = networks.define_D(input_D, ngf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[0])
        self.optimizers = []
        self.optimizer_fcloc = torch.optim.Adam(self.fc_loc.parameters(),
                                            lr=lr, betas=(beta1, 0.999))

        self.optimizer_localization = torch.optim.Adam(self.fc_loc.parameters(),
                                                lr=lr, betas=(beta1, 0.999))

        self.optimizers.append(self.optimizer_fcloc)
        self.optimizers.append(self.optimizer_localization)

        self.loss_mse = torch.nn.MSELoss()
        if self.isTrain:
            self.fake_AB_pool = ImagePool(pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=True).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=lr, betas=(beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=lr, betas=(beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        # AtoB = self.opt.which_direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

        #
        # self.input_to_G  =torch.cat((input[0],input[1]),1).to(self.device)
        # self.input = input[0].to(self.device)
        # self.true = input[2].to(self.device)
        # self.source_pred = input[1].to(self.device)

        self.input_to_G = input[1].to(self.device)
        self.input_true =input[1].to(self.device).clone()
        self.input = input[1].to(self.device)
        self.true = input[0].to(self.device)


    def forward(self):
        self.input_af_stn = self.stn(self.input_to_G)
        self.fake = self.netG(self.input_af_stn)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.input, self.fake), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.input, self.true), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.input, self.fake), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake,self.true) * self.lambda_L1

        self.loss_shape = self.loss_mse(self.input_af_stn, self.input_to_G)*100000000

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 +self.loss_shape




        self.loss_G.backward()
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 60 * 60)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = torch.nn.functional.affine_grid(theta, x.size())


        x = torch.nn.functional.grid_sample(x, grid)
        return x
    def cal_g1_loss(self):
        self.loss_G_L1 = self.criterionL1(self.fake, self.true) * self.lambda_L1
        return self.loss_G_L1

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()

        # self.optimizer_fcloc.zero_grad()
        #
        # self.optimizer_localization.zero_grad()

        self.backward_D()
        self.optimizer_D.step()
        # self.optimizer_fcloc.step()

        # self.optimizer_localization.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()

        self.optimizer_fcloc.zero_grad()

        self.optimizer_localization.zero_grad()


        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_fcloc.step()

        self.optimizer_localization.step()
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def new_get_current_losses(self):
        errors_g1 = OrderedDict()
        errors_gan = OrderedDict()
        for name in self.loss_names_g1:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_g1[name] = float(getattr(self, 'loss_' + name))
        for name in self.loss_names_gan:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_gan[name] = float(getattr(self, 'loss_' + name))

        return [errors_g1,errors_gan]

    def update_lr(self,decay_rate):
        for op in self.optimizers:
            for param_group in op.param_groups:
                param_group['lr'] = param_group['lr']* decay_rate

    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)


    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret