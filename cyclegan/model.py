import time
import torch
import re
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam
import os
from util.data.dataset import SteelyDataset, get_dataset
import torch.nn as nn
import torchvision as tv
from torchsummary import summary
from torchnet.meter import MovingAverageValueMeter
from networks.musegan import GANLoss
import shutil
# from networks.SMGT import Generator
from networks.SteelyGAN import Discriminator, Generator
from cyclegan.config import Config
from util.toolkit import generate_midi_segment_from_tensor, generate_data_from_midi, generate_whole_midi_from_tensor, evaluate_tonal_scale, get_md5_of
from util.image_pool import ImagePool
import logging
import colorlog
import json
from cyclegan.error import CyganException


class CycleGAN(object):
    def __init__(self):
        self.opt = Config()

        self.device = torch.device('cuda') if self.opt.gpu else torch.device('cpu')
        self.pool = ImagePool(self.opt.image_pool_max_size)

        self.set_up_terminal_logger()

        self._build_model()

    def _build_model(self):

        self.generator_A2B = Generator()
        self.generator_B2A = Generator()

        self.discriminator_A = Discriminator()
        self.discriminator_B = Discriminator()

        self.discriminator_A_all = None
        self.discriminator_B_all = None

        if self.opt.model != 'base':
            self.discriminator_A_all = Discriminator()
            self.discriminator_B_all = Discriminator()

        if self.opt.gpu:
            self.generator_A2B.to(self.device)
            summary(self.generator_A2B, input_size=self.opt.input_shape)
            self.generator_B2A.to(self.device)

            self.discriminator_A.to(self.device)
            summary(self.discriminator_A, input_size=self.opt.input_shape)
            self.discriminator_B.to(self.device)

            if self.opt.model != 'base':
                self.discriminator_A_all.to(self.device)
                self.discriminator_B_all.to(self.device)

        decay_lr = lambda epoch: self.opt.lr if epoch < self.opt.epoch_step else self.opt.lr * (self.opt.max_epoch - epoch) / (
                    self.opt.max_epoch - self.opt.epoch_step)

        self.DA_optimizer = Adam(params=self.discriminator_A.parameters(), lr=self.opt.lr,
                                 betas=(self.opt.beta1, self.opt.beta2))
        self.DB_optimizer = Adam(params=self.discriminator_B.parameters(), lr=self.opt.lr,
                                 betas=(self.opt.beta1, self.opt.beta2))
        self.GA2B_optimizer = Adam(params=self.generator_A2B.parameters(), lr=self.opt.lr,
                                   betas=(self.opt.beta1, self.opt.beta2))
        self.GB2A_optimizer = Adam(params=self.generator_B2A.parameters(), lr=self.opt.lr,
                                   betas=(self.opt.beta1, self.opt.beta2))

        self.DA_scheduler = lr_scheduler.StepLR(self.DA_optimizer, step_size=5, gamma=0.2)
        self.DB_scheduler = lr_scheduler.StepLR(self.DB_optimizer, step_size=5, gamma=0.2)
        self.GA2B_scheduler = lr_scheduler.StepLR(self.GA2B_optimizer, step_size=5, gamma=0.2)
        self.GB2A_scheduler = lr_scheduler.StepLR(self.GB2A_optimizer, step_size=5, gamma=0.2)

        if self.opt.model != 'base':
            self.DA_all_optimizer = torch.optim.Adam(params=self.discriminator_A_all.parameters(), lr=self.opt.lr,
                                                     betas=(self.opt.beta1, self.opt.beta2))
            self.DB_all_optimizer = torch.optim.Adam(params=self.discriminator_B_all.parameters(), lr=self.opt.lr,
                                                     betas=(self.opt.beta1, self.opt.beta2))

            self.DA_all_scheduler = lr_scheduler.StepLR(self.DA_all_optimizer, step_size=5, gamma=0.8)
            self.DB_all_scheduler = lr_scheduler.StepLR(self.DB_all_optimizer, step_size=5, gamma=0.8)

    def continue_from_latest_checkpoint(self):
        latest_checked_epoch = self.find_latest_checkpoint()
        self.opt.start_epoch = latest_checked_epoch + 1

        G_A2B_path = self.opt.G_A2B_save_path + 'steely_gan_G_A2B_' + str(latest_checked_epoch) + '.pth'
        G_B2A_path = self.opt.G_B2A_save_path + 'steely_gan_G_B2A_' + str(latest_checked_epoch) + '.pth'
        D_A_path = self.opt.D_A_save_path + 'steely_gan_D_A_' + str(latest_checked_epoch) + '.pth'
        D_B_path = self.opt.D_B_save_path + 'steely_gan_D_B_' + str(latest_checked_epoch) + '.pth'

        self.generator_A2B.load_state_dict(torch.load(G_A2B_path))
        self.generator_B2A.load_state_dict(torch.load(G_B2A_path))
        self.discriminator_A.load_state_dict(torch.load(D_A_path))
        self.discriminator_B.load_state_dict(torch.load(D_B_path))

        if self.opt.model != 'base':
            D_A_all_path = self.opt.D_A_all_save_path + 'steely_gan_D_A_all_' + str(latest_checked_epoch) + '.pth'
            D_B_all_path = self.opt.D_B_all_save_path + 'steely_gan_D_B_all_' + str(latest_checked_epoch) + '.pth'

            self.discriminator_A_all.load_state_dict(torch.load(D_A_all_path))
            self.discriminator_B_all.load_state_dict(torch.load(D_B_all_path))

        print(f'Loaded model from epoch {self.opt.start_epoch-1}')

    def reset_save(self):
        import shutil
        if os.path.exists(self.opt.save_path):
            shutil.rmtree(self.opt.save_path)

        os.makedirs(self.opt.save_path, exist_ok=True)
        os.makedirs(self.opt.model_path, exist_ok=True)
        os.makedirs(self.opt.checkpoint_path, exist_ok=True)
        os.makedirs(self.opt.test_path, exist_ok=True)

        os.makedirs(self.opt.G_A2B_save_path, exist_ok=True)
        os.makedirs(self.opt.G_B2A_save_path, exist_ok=True)
        os.makedirs(self.opt.D_A_save_path, exist_ok=True)
        os.makedirs(self.opt.D_B_save_path, exist_ok=True)
        os.makedirs(self.opt.D_A_all_save_path, exist_ok=True)
        os.makedirs(self.opt.D_B_all_save_path, exist_ok=True)

    def set_up_terminal_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        ch = colorlog.StreamHandler()
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(fg_cyan)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )

        ch.setFormatter(color_formatter)
        self.logger.addHandler(ch)

    def add_file_logger(self):
        fh = logging.FileHandler(filename=self.opt.log_path, mode='a')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(fh)

    def save_model(self, epoch):
        G_A2B_filename = f'{self.opt.name}_G_A2B_{epoch}.pth'
        G_B2A_filename = f'{self.opt.name}_G_B2A_{epoch}.pth'
        D_A_filename = f'{self.opt.name}_D_A_{epoch}.pth'
        D_B_filename = f'{self.opt.name}_D_B_{epoch}.pth'

        G_A2B_filepath = os.path.join(self.opt.G_A2B_save_path, G_A2B_filename)
        G_B2A_filepath = os.path.join(self.opt.G_B2A_save_path, G_B2A_filename)
        D_A_filepath = os.path.join(self.opt.D_A_save_path, D_A_filename)
        D_B_filepath = os.path.join(self.opt.D_B_save_path, D_B_filename)

        torch.save(self.generator_A2B.state_dict(), G_A2B_filepath)
        torch.save(self.generator_B2A.state_dict(), G_B2A_filepath)
        torch.save(self.discriminator_A.state_dict(), D_A_filepath)
        torch.save(self.discriminator_B.state_dict(), D_B_filepath)

        if self.opt.model != 'base':
            D_A_all_filename = f'{self.opt.name}_D_A_all_{epoch}.pth'
            D_B_all_filename = f'{self.opt.name}_D_B_all_{epoch}.pth'

            D_A_all_filepath = os.path.join(self.opt.D_A_all_save_path, D_A_all_filename)
            D_B_all_filepath = os.path.join(self.opt.D_B_all_save_path, D_B_all_filename)

            torch.save(self.discriminator_A_all.state_dict(), D_A_all_filepath)
            torch.save(self.discriminator_B_all.state_dict(), D_B_all_filepath)

        self.logger.info(f'model saved')

    def train(self):
        torch.cuda.empty_cache()

        ######################
        # Save / Load model
        ######################

        if self.opt.continue_train:
            try:
                self.continue_from_latest_checkpoint()
            except CyganException as e:
                self.logger.error(e)
                self.opt.continue_train = False
                self.reset_save()

        else:
            self.reset_save()

        self.add_file_logger()

        ######################
        # Dataset
        ######################

        if self.opt.model == 'base':
            dataset = SteelyDataset(self.opt.genreA, self.opt.genreB, self.opt.phase, use_mix=False)
        else:
            dataset = SteelyDataset(self.opt.genreA, self.opt.genreB, self.opt.phase, use_mix=True)

        dataset_size = len(dataset)
        iter_num = int(dataset_size / self.opt.batch_size)

        self.logger.info(f'Dataset loaded, genreA: {self.opt.genreA}, genreB: {self.opt.genreB}, total size: {dataset_size}.')

        ######################
        # Initiate
        ######################

        lambda_A = 10.0  # weight for cycle loss (A -> B -> A^)
        lambda_B = 10.0  # weight for cycle loss (B -> A -> B^)

        lambda_identity = 0.5

        criterionGAN = GANLoss(gan_mode='lsgan')

        criterionCycle = nn.L1Loss()

        criterionIdt = nn.L1Loss()

        GLoss_meter = MovingAverageValueMeter(self.opt.plot_every)
        DLoss_meter = MovingAverageValueMeter(self.opt.plot_every)
        CycleLoss_meter = MovingAverageValueMeter(self.opt.plot_every)

        # loss meters
        losses = {}
        scores = {}

        losses_dict = {
            'loss_G': [],
            'loss_D': [],
            'loss_C': [],
            'epoch': []
        }

        ######################
        # Start Training
        ######################

        for epoch in range(self.opt.start_epoch, self.opt.max_epoch):
            loader = DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_threads, drop_last=True)
            epoch_start_time = time.time()

            for i, data in enumerate(loader):

                real_A = torch.unsqueeze(data[:, 0, :, :], 1).to(self.device, dtype=torch.float)
                real_B = torch.unsqueeze(data[:, 1, :, :], 1).to(self.device, dtype=torch.float)

                gaussian_noise = torch.abs(torch.normal(mean=torch.zeros(self.opt.data_shape), std=self.opt.gaussian_std)).to(self.device, dtype=torch.float)

                if self.opt.model == 'base':

                    ######################
                    # Generator
                    ######################

                    fake_B = self.generator_A2B(real_A)  # X -> Y'
                    fake_A = self.generator_B2A(real_B)  # Y -> X'

                    fake_B_copy = copy.copy(fake_B.detach())
                    fake_A_copy = copy.copy(fake_A.detach())

                    DB_fake = self.discriminator_B(fake_B + gaussian_noise)  # netD_x provide feedback to netG_x
                    DA_fake = self.discriminator_A(fake_A + gaussian_noise)

                    loss_G_A2B = criterionGAN(DB_fake, True)
                    loss_G_B2A = criterionGAN(DA_fake, True)

                    # cycle_consistence
                    cycle_A = self.generator_B2A(fake_B)  # Y' -> X^
                    cycle_B = self.generator_A2B(fake_A)  # Y -> X' -> Y^

                    loss_cycle_A2B = criterionCycle(cycle_A, real_A) * lambda_A
                    loss_cycle_B2A = criterionCycle(cycle_B, real_B) * lambda_B

                    # identity loss
                    if lambda_identity > 0:
                        # netG_x should be identity if real_y is fed: ||netG_x(real_y) - real_y||
                        idt_A = self.generator_A2B(real_B)
                        idt_B = self.generator_B2A(real_A)
                        loss_idt_A = criterionIdt(idt_A, real_B) * lambda_A * lambda_identity
                        loss_idt_B = criterionIdt(idt_B, real_A) * lambda_A * lambda_identity

                    else:
                        loss_idt_A = 0.
                        loss_idt_B = 0.

                    loss_idt = loss_idt_A + loss_idt_B

                    self.GA2B_optimizer.zero_grad()  # set g_x and g_y gradients to zero
                    loss_A2B = loss_G_A2B + loss_cycle_A2B + loss_idt_A
                    loss_A2B.backward(retain_graph=True)
                    self.GA2B_optimizer.step()

                    self.GB2A_optimizer.zero_grad()  # set g_x and g_y gradients to zero
                    loss_B2A = loss_G_B2A + loss_cycle_B2A + loss_idt_B
                    loss_B2A.backward(retain_graph=True)
                    self.GB2A_optimizer.step()

                    cycle_loss = loss_cycle_A2B + loss_cycle_B2A
                    CycleLoss_meter.add(cycle_loss.item())

                    loss_G = loss_G_A2B + loss_G_B2A + loss_idt
                    GLoss_meter.add(loss_G.item())

                    ######################
                    # Sample
                    ######################
                    fake_A_sample, fake_B_sample = (None, None)
                    if self.opt.use_image_pool:
                        [fake_A_sample, fake_B_sample] = self.pool([fake_A_copy, fake_B_copy])

                    ######################
                    # Discriminator
                    ######################

                    # loss_real
                    DA_real = self.discriminator_A(real_A + gaussian_noise)
                    DB_real = self.discriminator_B(real_B + gaussian_noise)

                    loss_DA_real = criterionGAN(DA_real, True)
                    loss_DB_real = criterionGAN(DB_real, True)

                    # loss fake
                    if self.opt.use_image_pool:
                        DA_fake_sample = self.discriminator_A(fake_A_sample + gaussian_noise)
                        DB_fake_sample = self.discriminator_B(fake_B_sample + gaussian_noise)

                        loss_DA_fake = criterionGAN(DA_fake_sample, False)
                        loss_DB_fake = criterionGAN(DB_fake_sample, False)

                    else:
                        loss_DA_fake = criterionGAN(DA_fake, False)
                        loss_DB_fake = criterionGAN(DB_fake, False)

                    # loss and backward
                    self.DA_optimizer.zero_grad()
                    loss_DA = (loss_DA_real + loss_DA_fake) * 0.5
                    loss_DA.backward()
                    self.DA_optimizer.step()

                    self.DB_optimizer.zero_grad()
                    loss_DB = (loss_DB_real + loss_DB_fake) * 0.5
                    loss_DB.backward()
                    self.DB_optimizer.step()

                    loss_D = loss_DA + loss_DB
                    DLoss_meter.add(loss_D.item())


                else:
                    real_mixed = torch.unsqueeze(data[:, 2, :, :], 1).to(self.device, dtype=torch.float)

                    ######################
                    # Generator
                    ######################

                    fake_B = self.generator_A2B(real_A)  # X -> Y'
                    fake_A = self.generator_B2A(real_B)  # Y -> X'

                    fake_B_copy = copy.copy(fake_B)
                    fake_A_copy = copy.copy(fake_A)

                    DB_fake = self.discriminator_B(fake_B + gaussian_noise)  # netD_x provide feedback to netG_x
                    DA_fake = self.discriminator_A(fake_A + gaussian_noise)

                    loss_G_A2B = criterionGAN(DB_fake, True)
                    loss_G_B2A = criterionGAN(DA_fake, True)

                    # cycle_consistence
                    cycle_A = self.generator_B2A(fake_B)  # Y' -> X^
                    cycle_B = self.generator_A2B(fake_A)  # Y -> X' -> Y^

                    loss_cycle_A2B = criterionCycle(cycle_A, real_A) * lambda_A
                    loss_cycle_B2A = criterionCycle(cycle_B, real_B) * lambda_B

                    # identity loss
                    if lambda_identity > 0:
                        # netG_x should be identity if real_y is fed: ||netG_x(real_y) - real_y||
                        idt_A = self.generator_A2B(real_B)
                        idt_B = self.generator_B2A(real_A)
                        loss_idt_A = criterionIdt(idt_A, real_B) * lambda_A * lambda_identity
                        loss_idt_B = criterionIdt(idt_B, real_A) * lambda_A * lambda_identity

                    else:
                        loss_idt_A = 0.
                        loss_idt_B = 0.

                    loss_idt = loss_idt_A + loss_idt_B

                    self.GA2B_optimizer.zero_grad()  # set g_x and g_y gradients to zero
                    loss_A2B = loss_G_A2B + loss_cycle_A2B + loss_idt_A
                    loss_A2B.backward(retain_graph=True)
                    self.GA2B_optimizer.step()

                    self.GB2A_optimizer.zero_grad()  # set g_x and g_y gradients to zero
                    loss_B2A = loss_G_B2A + loss_cycle_B2A + loss_idt_B
                    loss_B2A.backward(retain_graph=True)
                    self.GB2A_optimizer.step()

                    cycle_loss = loss_cycle_A2B + loss_cycle_B2A
                    CycleLoss_meter.add(cycle_loss.item())

                    loss_G = loss_G_A2B + loss_G_B2A + loss_idt
                    GLoss_meter.add(loss_G.item())

                    ######################
                    # Sample
                    ######################
                    fake_A_sample, fake_B_sample = (None, None)
                    if self.opt.use_image_pool:
                        [fake_A_sample, fake_B_sample] = self.pool([fake_A_copy, fake_B_copy])

                    ######################
                    # Discriminator
                    ######################

                    # loss_real
                    DA_real = self.discriminator_A(real_A + gaussian_noise)
                    DB_real = self.discriminator_B(real_B + gaussian_noise)

                    DA_real_all = self.discriminator_A_all(real_mixed + gaussian_noise)
                    DB_real_all = self.discriminator_B_all(real_mixed + gaussian_noise)

                    loss_DA_real = criterionGAN(DA_real, True)
                    loss_DB_real = criterionGAN(DB_real, True)

                    loss_DA_all_real = criterionGAN(DA_real_all, True)
                    loss_DB_all_real = criterionGAN(DB_real_all, True)

                    # loss fake
                    if self.opt.use_image_pool:
                        DA_fake_sample = self.discriminator_A(fake_A_sample + gaussian_noise)
                        DB_fake_sample = self.discriminator_B(fake_B_sample + gaussian_noise)

                        DA_fake_sample_all = self.discriminator_A_all(fake_A_sample + gaussian_noise)
                        DB_fake_sample_all = self.discriminator_B_all(fake_B_sample + gaussian_noise)

                        loss_DA_all_fake = criterionGAN(DA_fake_sample_all, False)
                        loss_DB_all_fake = criterionGAN(DB_fake_sample_all, False)

                        loss_DA_fake = criterionGAN(DA_fake_sample, False)
                        loss_DB_fake = criterionGAN(DB_fake_sample, False)

                    else:
                        DA_fake_all = self.discriminator_A_all(fake_A_copy + gaussian_noise)
                        DB_fake_all = self.discriminator_B_all(fake_B_copy + gaussian_noise)

                        loss_DA_all_fake = criterionGAN(DA_fake_all, False)
                        loss_DB_all_fake = criterionGAN(DB_fake_all, False)

                        loss_DA_fake = criterionGAN(DA_fake, False)
                        loss_DB_fake = criterionGAN(DB_fake, False)

                    # loss and backward
                    self.DA_optimizer.zero_grad()
                    loss_DA = (loss_DA_real + loss_DA_fake) * 0.5
                    loss_DA.backward()
                    self.DA_optimizer.step()

                    self.DB_optimizer.zero_grad()
                    loss_DB = (loss_DB_real + loss_DB_fake) * 0.5
                    loss_DB.backward()
                    self.DB_optimizer.step()

                    self.DA_all_optimizer.zero_grad()
                    loss_DA_all = (loss_DA_all_real + loss_DA_all_fake) * 0.5
                    loss_DA_all.backward()
                    self.DA_all_optimizer.step()

                    self.DB_all_optimizer.zero_grad()
                    loss_DB_all = (loss_DB_all_real + loss_DB_all_fake) * 0.5
                    loss_DB_all.backward()
                    self.DB_all_optimizer.step()

                    loss_D = loss_DA + loss_DB + loss_DB_all + loss_DA_all
                    DLoss_meter.add(loss_D.item())

                ######################
                # Snapshot
                ######################

                if i % self.opt.plot_every == 0:
                    file_name = self.opt.name + '_snap_%03d_%05d.png' % (epoch, i,)
                    test_path = os.path.join(self.opt.checkpoint_path, file_name)
                    tv.utils.save_image(fake_B, test_path, normalize=True)
                    self.logger.info(f'Snapshot {file_name} saved.')

                    losses['loss_C'] = float(CycleLoss_meter.value()[0])
                    losses['loss_G'] = float(GLoss_meter.value()[0])
                    losses['loss_D'] = float(DLoss_meter.value()[0])

                    self.logger.info(str(losses))
                    self.logger.info('Epoch {} progress: {:.2%}\n'.format(epoch, i / iter_num))

            # save model
            if epoch % self.opt.save_every == 0 or epoch == self.opt.max_epoch - 1:
                self.save_model(epoch)

            ######################
            # lr_scheduler
            ######################

            self.GA2B_scheduler.step(epoch)
            self.GB2A_scheduler.step(epoch)
            self.DA_scheduler.step(epoch)
            self.DB_scheduler.step(epoch)

            if self.opt.model != 'base':
                self.DA_all_scheduler.step(epoch)
                self.DB_all_scheduler.step(epoch)

            epoch_time = int(time.time() - epoch_start_time)

            ######################
            # Logging
            ######################

            self.logger.info(f'Epoch {epoch} finished, cost time {epoch_time}\n')
            self.logger.info(str(losses) + '\n\n')

            ######################
            # Loss_Dict
            ######################

            losses_dict['loss_C'].append(losses['loss_C'])
            losses_dict['loss_G'].append(losses['loss_G'])
            losses_dict['loss_D'].append(losses['loss_D'])
            losses_dict['epoch'].append(epoch)

            with open(self.opt.loss_save_path, 'w') as f:
                json.dump(losses_dict, f)

    def test(self):
        torch.cuda.empty_cache()

        ######################
        # Save paths
        ######################

        os.makedirs(self.opt.test_save_path, exist_ok=True)
        npy_save_dir = self.opt.test_save_path + '/npy'
        midi_save_dir = self.opt.test_save_path + '/midi'

        '''
        os.makedirs(npy_save_dir, exist_ok=True)
        os.makedirs(npy_save_dir + '/origin', exist_ok=True)
        os.makedirs(npy_save_dir + '/transfer', exist_ok=True)
        os.makedirs(npy_save_dir + '/cycle', exist_ok=True)
        '''

        os.makedirs(midi_save_dir, exist_ok=True)
        os.makedirs(midi_save_dir + '/origin', exist_ok=True)
        os.makedirs(midi_save_dir + '/transfer', exist_ok=True)
        os.makedirs(midi_save_dir + '/cycle', exist_ok=True)

        ######################
        # Dataset
        ######################

        if self.opt.model == 'base':
            dataset = SteelyDataset(self.opt.genreA, self.opt.genreB, self.opt.phase, use_mix=False)

        else:
            dataset = SteelyDataset(self.opt.genreA, self.opt.genreB, self.opt.phase, use_mix=True)

        dataset_size = len(dataset)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)
        self.logger.info(f'Dataset loaded, genreA: {self.opt.genreA}, genreB: {self.opt.genreB}, total size: {dataset_size}.')

        ######################
        # Load model
        ######################

        try:
            self.continue_from_latest_checkpoint()
        except CyganException as e:
            self.logger.error(e)
            return

        ######################
        # Test
        ######################

        for i, data in enumerate(loader):
            if self.opt.direction == 'AtoB':
                origin = torch.unsqueeze(data[:, 0, :, :], 1).to(self.device, dtype=torch.float)
                transfer = self.generator_A2B(origin)
                cycle = self.generator_B2A(transfer)

            else:
                origin = torch.unsqueeze(data[:, 1, :, :], 1).to(self.device, dtype=torch.float)
                transfer = self.generator_B2A(origin)
                cycle = self.generator_A2B(transfer)

            generate_midi_segment_from_tensor(origin.cpu().detach().numpy()[0, 0, :, :], midi_save_dir + '/origin/' + str(i+1) + '.mid')
            generate_midi_segment_from_tensor(transfer.cpu().detach().numpy()[0, 0, :, :], midi_save_dir + '/transfer/' + str(i + 1) + '.mid')
            generate_midi_segment_from_tensor(cycle.cpu().detach().numpy()[0, 0, :, :], midi_save_dir + '/cycle/' + str(i + 1) + '.mid')

    def find_latest_checkpoint(self):
        path = self.opt.D_B_save_path
        file_list = os.listdir(path)
        match_str = r'\d+'
        epoch_list = sorted([int(re.findall(match_str, file)[0]) for file in file_list])
        if len(epoch_list) == 0:
            raise CyganException('No model to load.')
        latest_num = epoch_list[-1]
        return latest_num


def test_lr():
    model = Generator()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    lr = 0.002
    epoch_step = 10
    whole_epoch = 20
    decay_lr = lambda epoch: lr if epoch < epoch_step else lr * (whole_epoch - epoch) / (whole_epoch - epoch_step)
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


def test_sample_song_old():
    dataset = SteelyDataset('rock', 'jazz', 'test', False)

    cyclegan = CycleGAN()
    cyclegan.continue_from_latest_checkpoint()

    converted_dir = '../data/converted_midi'

    for index in range(10):
        data = dataset[index + 2000]
        dataA, dataB = data[0, :, :], data[1, :, :]
        # print(torch.unsqueeze(torch.from_numpy(dataA), 0).shape)
        dataA2B = cyclegan.generator_A2B(
            torch.unsqueeze(torch.unsqueeze(torch.from_numpy(dataA), 0), 0).to(device='cuda',
                                                                               dtype=torch.float)).cpu().detach().numpy()[
                  0, 0, :, :]
        dataB2A = cyclegan.generator_B2A(
            torch.unsqueeze(torch.unsqueeze(torch.from_numpy(dataB), 0), 0).to(device='cuda',
                                                                               dtype=torch.float)).cpu().detach().numpy()[
                  0, 0, :, :]
        midi_A_path = converted_dir + '/midi_A_' + str(index) + '.mid'
        midi_A2B_path = converted_dir + '/midi_A2B_' + str(index) + '.mid'

        midi_B_path = converted_dir + '/midi_B_' + str(index) + '.mid'
        midi_B2A_path = converted_dir + '/midi_B2A_' + str(index) + '.mid'

        tonality_A = evaluate_tonal_scale(dataA)
        tonality_A2B = evaluate_tonal_scale(dataA2B)

        tonality_B = evaluate_tonal_scale(dataB)
        tonality_B2A = evaluate_tonal_scale(dataB2A)

        print(tonality_A, tonality_A2B)
        print(tonality_B, tonality_B2A)

        # plot_data(dataA)
        # plot_data(dataA2B)

        generate_midi_segment_from_tensor(dataA, midi_A_path)
        generate_midi_segment_from_tensor(dataA2B, midi_A2B_path)
        generate_midi_segment_from_tensor(dataB, midi_B_path)
        generate_midi_segment_from_tensor(dataB2A, midi_B2A_path)


def test_whole_song(performer='Bill Evans', song='Autumn Leaves', genre='jazz'):
    root_dir = 'E:/free_midi_library/merged_midi'
    try:
        md5 = get_md5_of(performer, song, genre)
        original_path = root_dir + '/' + genre + '/' + md5 + '.mid'
    except Exception as e:
        print(e)
        return

    cyclegan = CycleGAN()
    cyclegan.continue_from_latest_checkpoint()

    # direction = 'AtoB'
    direction = 'BtoA'

    transformed_path = '../data/converted_midi/' + song + ' - ' + performer + '.mid'
    copy_path = '../data/original_midi/' + song + ' - ' + performer + '.mid'

    ori_data = generate_data_from_midi(original_path)

    if direction == 'AtoB':
        transformed_data = cyclegan.generator_A2B(
            torch.unsqueeze(torch.from_numpy(ori_data), 1).to(
                device='cuda',  dtype=torch.float)).cpu().detach().numpy()
    else:
        transformed_data = cyclegan.generator_B2A(
            torch.unsqueeze(torch.from_numpy(ori_data), 1).to(
                device='cuda', dtype=torch.float)).cpu().detach().numpy()
    print(transformed_data.shape)
    generate_whole_midi_from_tensor(transformed_data, transformed_path)
    shutil.copyfile(original_path, copy_path)


def remove_dir_test():
    path = 'D:/checkpoints/steely_gan/base'
    shutil.rmtree(path)


def run():
    cyclegan = CycleGAN()
    if cyclegan.opt.phase == 'train':
        cyclegan.train()
    elif cyclegan.opt.phase == 'test':
        cyclegan.test()


if __name__ == '__main__':
    test_whole_song()
