import time
import datetime
import itertools
import torch
import re
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam
import os
from util.data.dataset import SteelyDataset, get_dataset, MixedSourceDataset
import torch.nn as nn
import torchvision as tv
from torchsummary import summary
from torchnet.meter import MovingAverageValueMeter
from networks.musegan import MuseDiscriminator, MuseGenerator, GANLoss
from networks.MST import Discriminator, Generator
from process.config import Config
from util.image_pool import ImagePool


class CycleGAN(object):
    def __init__(self):
        self.opt = Config()

        self.name = self.opt.name

        self.genreA = self.opt.genreA
        self.genreB = self.opt.genreB

        self.save_path = self.opt.save_path
        self.model_path = self.opt.model_path
        self.checkpoint_path = self.save_path + '/checkpoints'
        self.test_path = self.save_path + '/test_results'

        self.G_A2B_save_path = self.opt.G_A2B_save_path
        self.G_B2A_save_path = self.opt.G_B2A_save_path
        self.D_A_save_path = self.opt.D_A_save_path
        self.D_B_save_path = self.opt.D_B_save_path
        self.D_A_all_save_path = self.opt.D_A_all_save_path
        self.D_B_all_save_path = self.opt.D_B_all_save_path

        self.data_shape = self.opt.data_shape
        self.input_shape = self.opt.input_shape

        self.gpu = self.opt.gpu
        self.device = torch.device('cuda') if self.opt.gpu else torch.device('cpu')

        self.beta1 = self.opt.beta1
        self.beta2 = self.opt.beta2

        self.phase = self.opt.phase
        self.lr = self.opt.lr
        self.batch_size = self.opt.batch_size
        self.max_epoch = self.opt.max_epoch
        self.epoch_step = self.opt.epoch_step
        self.start_epoch = self.opt.start_epoch

        self.plot_every = 100 # iteration

        self.save_every = 5 # epochs

        self.model = self.opt.model

        self.pool = ImagePool(self.opt.image_pool_max_size)

        self.continue_train = self.opt.continue_train

        self._build_model()

    def _build_model(self):

        self.generator_A2B = Generator()
        self.generator_B2A = Generator()

        self.discriminator_A = Discriminator()
        self.discriminator_B = Discriminator()

        self.discriminator_A_all = None
        self.discriminator_B_all = None

        if self.model != 'base':
            self.discriminator_A_all = Discriminator()
            self.discriminator_B_all = Discriminator()


        if self.gpu:
            self.generator_A2B.to(self.device)
            summary(self.generator_A2B, input_size=self.input_shape)
            self.generator_B2A.to(self.device)

            self.discriminator_A.to(self.device)
            summary(self.discriminator_A, input_size=self.input_shape)
            self.discriminator_B.to(self.device)

            if self.model != 'base':
                self.discriminator_A_all.to(self.device)
                self.discriminator_B_all.to(self.device)

        decay_lr = lambda epoch: self.lr if epoch < self.epoch_step else self.lr * (self.max_epoch - epoch) / (self.max_epoch - self.epoch_step)

        self.DA_optimizer = Adam(params=self.discriminator_A.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.DB_optimizer = Adam(params=self.discriminator_B.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.GA2B_optimizer = Adam(params=self.generator_A2B.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.GB2A_optimizer = Adam(params=self.generator_B2A.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        '''
        self.DA_scheduler = lr_scheduler.LambdaLR(self.DA_optimizer, lr_lambda=decay_lr)
        self.DB_scheduler = lr_scheduler.LambdaLR(self.DB_optimizer, lr_lambda=decay_lr)
        self.GA2B_scheduler = lr_scheduler.LambdaLR(self.GA2B_optimizer, lr_lambda=decay_lr)
        self.GB2A_scheduler = lr_scheduler.LambdaLR(self.GB2A_optimizer, lr_lambda=decay_lr)
        '''

        self.DA_scheduler = lr_scheduler.StepLR(self.DA_optimizer, step_size=5, gamma=0.8)
        self.DB_scheduler = lr_scheduler.StepLR(self.DB_optimizer, step_size=5, gamma=0.8)
        self.GA2B_scheduler = lr_scheduler.StepLR(self.GA2B_optimizer, step_size=5, gamma=0.8)
        self.GB2A_scheduler = lr_scheduler.StepLR(self.GB2A_optimizer, step_size=5, gamma=0.8)


        if self.model != 'base':
            self.DA_all_optimizer = torch.optim.Adam(params=self.discriminator_A_all.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
            self.DB_all_optimizer = torch.optim.Adam(params=self.discriminator_B_all.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

            self.DA_all_scheduler = lr_scheduler.StepLR(self.DA_all_optimizer, step_size=5, gamma=0.8)
            self.DB_all_scheduler = lr_scheduler.StepLR(self.DB_all_optimizer, step_size=5, gamma=0.8)

    def train(self):
        torch.cuda.empty_cache()

        if self.model == 'base':
            dataset = SteelyDataset(self.genreA, self.genreB, 'train', use_mix=False)
            dataset_size = len(dataset)

        else:
            dataset = SteelyDataset(self.genreA, self.genreB, 'train', use_mix=True)
            dataset_size = len(dataset)

        if self.continue_train:
            latest_checked_epoch = self.find_latest_checkpoint()
            self.start_epoch = latest_checked_epoch

            G_A2B_path = self.G_A2B_save_path + 'steely_gan_G_A2B_' + str(latest_checked_epoch) + '.pth'
            G_B2A_path = self.G_B2A_save_path + 'steely_gan_G_B2A_' + str(latest_checked_epoch) + '.pth'
            D_A_path = self.D_A_save_path + 'steely_gan_D_A_' + str(latest_checked_epoch) + '.pth'
            D_B_path = self.D_B_save_path + 'steely_gan_D_B_' + str(latest_checked_epoch) + '.pth'

            self.generator_A2B.load_state_dict(torch.load(G_A2B_path))
            self.generator_B2A.load_state_dict(torch.load(G_B2A_path))
            self.discriminator_A.load_state_dict(torch.load(D_A_path))
            self.discriminator_B.load_state_dict(torch.load(D_B_path))

            if self.model != 'base':
                D_A_all_path = self.D_A_all_save_path + 'steely_gan_D_A_all_' + str(latest_checked_epoch) + '.pth'
                D_B_all_path = self.D_B_all_save_path + 'steely_gan_D_B_all_' + str(latest_checked_epoch) + '.pth'

                self.discriminator_A_all.load_state_dict(torch.load(D_A_all_path))
                self.discriminator_B_all.load_state_dict(torch.load(D_B_all_path))

            print(f'Loaded model from epoch {self.start_epoch}')

        iter_num = int(dataset_size / self.batch_size)

        print(f'loaded {dataset_size} images for training')


        # optimizers = [optimizer_g, optimizer_d]

        lambda_A = 10.0  # weight for cycle loss (A -> B -> A^)
        lambda_B = 10.0  # weight for cycle loss (B -> A -> B^)

        L1_lambda = 10.0
        lambda_identity = 0.5

        # it's a MSELoss() when initialized, only calculate later during iteration
        # criterionGAN = nn.MSELoss().to(device)
        criterionGAN = GANLoss(gan_mode='lsgan')

        # cycle loss
        criterionCycle = nn.L1Loss()

        # identical loss
        criterionIdt = nn.L1Loss()

        loss_A_meter = MovingAverageValueMeter(self.plot_every)
        loss_B_meter = MovingAverageValueMeter(self.plot_every)
        score_DA_real_B = MovingAverageValueMeter(self.plot_every)
        score_DA_fake_B = MovingAverageValueMeter(self.plot_every)

        # loss meters
        losses = {}
        scores = {}


        for epoch in range(self.start_epoch+1, self.max_epoch):
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=True)
            epoch_start_time = time.time()

            for i, data in enumerate(loader):


                real_A = torch.unsqueeze(data[:, 0, :, :], 1).to(self.device, dtype=torch.float)
                real_B = torch.unsqueeze(data[:, 1, :, :], 1).to(self.device, dtype=torch.float)

                gaussian_noise = torch.abs(torch.normal(mean=torch.zeros(self.data_shape), std=0.01)).to(self.device, dtype=torch.float)

                ######################
                # A -> B' -> A^ cycle
                ######################

                self.GA2B_optimizer.zero_grad()   # set g_x and g_y gradients to zero

                fake_B = self.generator_A2B(real_A)       # X -> Y'
                DB_fake = self.discriminator_B(fake_B + gaussian_noise)  #netD_x provide feedback to netG_x
                '''
                to_binary
                '''
                loss_G_A2B = criterionGAN(DB_fake, True)

                # cycle_consistence
                cycle_A = self.generator_B2A(fake_B)         # Y' -> X^
                # Forward cycle loss x^ = || G_y(G_x(real_x)) ||
                loss_cycle_A2B = criterionCycle(cycle_A, real_A) * lambda_A

                # identity loss
                if lambda_identity > 0:
                    # netG_x should be identity if real_y is fed: ||netG_x(real_y) - real_y||
                    idt_A = self.generator_A2B(real_B)
                    loss_idt_A = criterionIdt(idt_A, real_B) * lambda_A * lambda_identity
                else:
                    loss_idt_A = 0.

                loss_A = loss_G_A2B + 10. * loss_cycle_A2B + loss_idt_A
                loss_A.backward(retain_graph=True)
                self.GA2B_optimizer.step()

                loss_A_meter.add(loss_A.item())


                ######################
                # B -> A' -> B^ cycle
                ######################

                self.GB2A_optimizer.zero_grad()   # set g_x and g_y gradients to zero

                fake_A = self.generator_B2A(real_B)   # Y -> X'
                DA_fake = self.discriminator_A(fake_A + gaussian_noise)
                loss_G_B2A = criterionGAN(DA_fake, True)
                # print(f'loss_G_Y = {round(float(loss_G_Y), 3)}')

                cycle_B = self.generator_A2B(fake_A)   # Y -> X' -> Y^
                # Forward cycle loss y^ = || G_x(G_y(real_y)) ||
                loss_cycle_B2A = criterionCycle(cycle_B, real_B) * lambda_B

                # identity loss
                if lambda_identity > 0:
                # netG_y should be identiy if real_x is fed: ||netG_y(real_x) - real_x||
                    idt_B = self.generator_B2A(real_A)
                    loss_idt_B = criterionIdt(idt_B, real_A) * lambda_A * lambda_identity
                else:
                    loss_idt_B = 0.

                loss_B = loss_G_B2A + 10. * loss_cycle_B2A + loss_idt_B
                loss_B.backward(retain_graph=True)
                self.GB2A_optimizer.step()

                loss_B_meter.add(loss_B.item())


                ######################
                # sample
                ######################

                [fake_A_sample, fake_B_sample] = self.pool([fake_A, fake_B])

                DB_fake_sample = self.discriminator_B(fake_B_sample + gaussian_noise)

                ######################
                # netD_A
                ######################

                self.DA_optimizer.zero_grad()

                # loss_real
                DA_real = self.discriminator_A(real_A + gaussian_noise)
                loss_DA_real = criterionGAN(DA_real, True)
                score_DA_real_B.add(float(DA_real.data.mean()))

                # loss fake
                DA_fake_sample = self.discriminator_A(fake_A_sample + gaussian_noise)
                loss_DA_fake = criterionGAN(DA_fake_sample, False)
                score_DA_fake_B.add(float(DA_fake_sample.data.mean()))

                # loss_DA_fake = criterionGAN(DA_fake, False)
                # score_DA_fake_B.add(float(DA_fake.data.mean()))

                # loss and backward
                loss_DA = (loss_DA_real + loss_DA_fake) * 0.5

                loss_DA.backward()
                self.DA_optimizer.step()


                ######################
                # netD_B
                ######################

                self.DB_optimizer.zero_grad()

                # loss_real
                DB_real = self.discriminator_B(real_B + gaussian_noise)
                loss_DB_real = criterionGAN(DB_real, True)

                # loss_fake
                DB_fake_sample = self.discriminator_B(fake_B_sample + gaussian_noise)
                loss_DB_fake = criterionGAN(DB_fake_sample, False)
                # loss_DB_fake = criterionGAN(DB_fake, False)

                # loss and backward
                loss_DB = (loss_DB_real + loss_DB_fake) * 0.5

                loss_DB.backward()
                self.DB_optimizer.step()


                if self.model != 'base':
                    real_mixed = torch.unsqueeze(data[:, 2, :, :], 1).to(self.device, dtype=torch.float)

                    ######################
                    # netD_A_all
                    ######################

                    self.DA_all_optimizer.zero_grad()

                    # loss_real
                    DA_real_all = self.discriminator_A_all(real_mixed + gaussian_noise)
                    loss_DA_all_real = criterionGAN(DA_real_all, True)

                    # loss fake
                    DA_fake_sample_all = self.discriminator_A_all(fake_A_sample + gaussian_noise)
                    loss_DA_all_fake = criterionGAN(DA_fake_sample_all, False)

                    loss_DA_all = (loss_DA_all_real + loss_DA_all_fake) * 0.5
                    loss_DA_all.backward()
                    self.DA_all_optimizer.step()

                    ######################
                    # netD_B_all
                    ######################

                    self.DB_all_optimizer.zero_grad()

                    # loss_real
                    DB_real_all = self.discriminator_B_all(real_mixed + gaussian_noise)
                    loss_DB_all_real = criterionGAN(DB_real_all, True)

                    # loss fake
                    DB_fake_sample_all = self.discriminator_B_all(fake_B_sample + gaussian_noise)
                    loss_DB_all_fake = criterionGAN(DB_fake_sample_all, False)

                    loss_DB_all = (loss_DB_all_real + loss_DB_all_fake) * 0.5
                    loss_DB_all.backward()
                    self.DB_all_optimizer.step()


                # save snapshot
                if i % self.plot_every == 0:
                    file_name = self.name + '_snap_%03d_%05d.png' % (epoch, i,)
                    test_path = os.path.join(self.checkpoint_path, file_name)
                    tv.utils.save_image(fake_B, test_path, normalize=True)
                    print(f'{file_name} saved.')
                    losses['loss_A'] = loss_A_meter.value()[0]
                    losses['loss_B'] = loss_B_meter.value()[0]
                    scores['score_DA_real_B'] = score_DA_real_B.value()[0]
                    scores['score_DA_fake_B'] = score_DA_fake_B.value()[0]
                    print(losses)
                    print(scores)
                    print('Epoch {} progress: {:.2%}\n'.format(epoch, i / iter_num))

            # save model
            if epoch % self.save_every == 0 or epoch == self.max_epoch - 1:
                G_A2B_filename = f'{self.name}_G_A2B_{epoch}.pth'
                G_B2A_filename = f'{self.name}_G_B2A_{epoch}.pth'
                D_A_filename = f'{self.name}_D_A_{epoch}.pth'
                D_B_filename = f'{self.name}_D_B_{epoch}.pth'


                G_A2B_filepath = os.path.join(self.G_A2B_save_path, G_A2B_filename)
                G_B2A_filepath = os.path.join(self.G_B2A_save_path, G_B2A_filename)
                D_A_filepath = os.path.join(self.D_A_save_path, D_A_filename)
                D_B_filepath = os.path.join(self.D_B_save_path, D_B_filename)

                torch.save(self.generator_A2B.state_dict(), G_A2B_filepath)
                torch.save(self.generator_B2A.state_dict(), G_B2A_filepath)
                torch.save(self.discriminator_A.state_dict(), D_A_filepath)
                torch.save(self.discriminator_B.state_dict(), D_B_filepath)

                if self.model != 'base':
                    D_A_all_filename = f'{self.name}_D_A_all_{epoch}.pth'
                    D_B_all_filename = f'{self.name}_D_B_all_{epoch}.pth'

                    D_A_all_filepath = os.path.join(self.D_A_all_save_path, D_A_all_filename)
                    D_B_all_filepath = os.path.join(self.D_B_all_save_path, D_B_all_filename)

                    torch.save(self.discriminator_A_all.state_dict(), D_A_all_filepath)
                    torch.save(self.discriminator_B_all.state_dict(), D_B_all_filepath)

                print(f'model saved')


            self.DA_scheduler.step(epoch)
            self.DB_scheduler.step(epoch)
            self.GA2B_scheduler.step(epoch)
            self.GB2A_scheduler.step(epoch)


            epoch_time = int(time.time() - epoch_start_time)

            print_options(self.opt, epoch_log=True, epoch=epoch, time=epoch_time, losses=losses, scores=scores)


    def test(self):
        opt = Config()
        opt.phase = 'test'
        opt.num_threads = 1
        opt.batch_size = 1
        opt.serial_batches = True
        opt.no_flip = True
        opt.no_dropout = True
        opt.mode = 'test'

        device = torch.device('cuda') if opt.gpu else torch.device('cpu')

        dataset = get_dataset(opt)
        dataset_size = len(dataset)
        print(f'loaded {dataset_size} images for test.')

        netG_x = MuseGenerator(opt)
        netG_x.to(device)
        print(netG_x)
        summary(netG_x, opt.data_shape)

        models = sorted(os.listdir(opt.model_path))
        assert len(models) > 0, 'no models found!'
        latest_model = models[-1]
        model_path = os.path.join(opt.model_path, latest_model)
        print(f'loading trained model {model_path}')

        map_location = lambda storage, loc: storage
        state_dict = torch.load(model_path, map_location=map_location)

        # for model trained on pytorch < 0.4
        # for key in list(state_dict.keys()):
        #     print(key, '--', key.split('.'))
        #     keys = key.split('.')
        #     __patch_instance_norm_state_dict(state_dict, netG_x, keys)
        netG_x.load_state_dict(state_dict)

        for i, data in enumerate(dataset):
            real_x = data['A'].to(device)

            with torch.no_grad():
                # fake_y = netG_x.forward(real_x)
                fake_y = netG_x(real_x)
                filename = opt.name + '_fake_%05d.png' % (i,)
                test_path = os.path.join(opt.test_path, filename)
                tv.utils.save_image(fake_y, test_path, normalize=True)
                print(f'{filename} saved')


    def find_latest_checkpoint(self):
        path = self.D_B_save_path
        file_list = os.listdir(path)
        match_str = r'\d+'
        epoch_list = sorted([int(re.findall(match_str, file)[0]) for file in file_list])
        latest_num = epoch_list[-1]
        return latest_num



def print_options(opt, epoch_log = False, epoch=0, time=0, losses=None, scores=None):
    file_name = os.path.join(opt.save_path, 'options.txt')
    if epoch_log:
        with open(file_name, 'a+') as opt_file:
            print(f'epoch {epoch} finished, cost time {time}s.')
            print(losses)
            print(scores)
            if epoch == 0:
                opt_file.write(f'Each epoch cost about {time}s.\n')
            opt_file.write(f'epoch {epoch} ')
            opt_file.write(str(losses)+' ')
            opt_file.write(str(scores)+'\n')
        return

    var_opt = vars(opt)
    message = f'\nTraining start time: {datetime.datetime.now()} \n\n'
    message += '----------------- Options ---------------\n'
    for key, value in var_opt.items():
        message += '{:>20}: {:<30}\n'.format(str(key), str(value))
    message += '----------------- End -------------------'
    message += '\n'
    print(message)
    # save to the disk
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def test_lr():
    model = Generator()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    lr = 0.002
    epoch_step = 10
    whole_epoch = 20
    decay_lr = lambda epoch : lr if epoch < epoch_step else lr * (whole_epoch-epoch) / (whole_epoch-epoch_step)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=decay_lr)
    lr_list = []
    for epoch in range(100):
        scheduler.step(epoch)
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    plt.plot(range(100), lr_list)
    plt.show()

def load_model_test():
    path1 = 'D:/checkpoints/steely_gan/models/steely_gan_netG_5.pth'
    path2 = 'D:/checkpoints/steely_gan/models/steely_gan_netG_0.pth'
    generator1 = Generator()
    generator2 = Generator()
    generator1.load_state_dict(torch.load(path1))
    generator2.load_state_dict(torch.load(path2))

    params1 = generator1.state_dict()
    params2 = generator2.state_dict()
    print(params1['cnet1.1.weight'])
    print(params2['cnet1.1.weight'])

if __name__ == '__main__':
    cyclegan = CycleGAN()
    cyclegan.train()