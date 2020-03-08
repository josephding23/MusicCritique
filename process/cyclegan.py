import time
import datetime
import itertools
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
from util.data.dataset import SteelyDataset, get_dataset
import torch.nn as nn
import torchvision as tv
from torchsummary import summary
from torchnet.meter import MovingAverageValueMeter
from networks.musegan import MuseDiscriminator, MuseGenerator, GANLoss
from networks.MST import Discriminator, Generator
from process.config import Config

def train():
    torch.cuda.empty_cache()

    opt = Config()

    device = torch.device('cuda') if opt.gpu else torch.device('cpu')

    dataset = SteelyDataset(opt.genreA, opt.genreB, 'train')
    dataset_size = len(dataset)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    iter_per_epoch = int(dataset_size / opt.batch_size)

    print(f'loaded {dataset_size} images for training')

    model_names = ['netG_x', 'netG_y', 'netD_x', 'netD_y']

    generator_A2B = Generator()
    generator_B2A = Generator()

    discriminator_A = Discriminator()
    discriminator_B = Discriminator()

    if opt.model != 'base':
        discriminator_A_all = Discriminator()
        discriminator_B_all = Discriminator()

    if opt.gpu:
        generator_A2B.to(device)
        summary(generator_A2B, input_size=opt.data_shape)
        generator_B2A.to(device)

        discriminator_A.to(device)
        summary(discriminator_A, input_size=opt.data_shape)
        discriminator_B.to(device)


    DA_optimizer = torch.optim.Adam(params=discriminator_A.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))
    DB_optimizer = torch.optim.Adam(params=discriminator_B.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))
    GA2B_optimizer = torch.optim.Adam(params=generator_A2B.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))
    GB2A_optimizer = torch.optim.Adam(params=generator_B2A.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))

    if opt.model != 'base':
        optimizer_g = torch.optim.Adam(params=generator_A2B.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))
        optimizer_d = torch.optim.Adam(params=discriminator_A.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))


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

    loss_A_meter = MovingAverageValueMeter(opt.plot_every)
    loss_B_meter = MovingAverageValueMeter(opt.plot_every)
    score_DA_real_B = MovingAverageValueMeter(opt.plot_every)
    score_DA_fake_B = MovingAverageValueMeter(opt.plot_every)

    # loss meters
    losses = {}
    scores = {}

    lr = opt.lr

    for epoch in range(opt.epoch):

        epoch_start_time = time.time()

        lr = lr if epoch < opt.epoch_step else lr * (opt.epoch-epoch) / (opt.epoch-opt.epoch_step)

        for i, data in enumerate(loader):

            real_A = torch.unsqueeze(data[:, 0, :, :], 0).to(device, dtype=torch.float)
            real_B = torch.unsqueeze(data[:, 1, :, :], 0).to(device, dtype=torch.float)

            gaussian_noise = (torch.randn(opt.data_shape) * 1. + 0).to(device, dtype=torch.float)
            # gaussian_noise = torch.random.(loc=0, scale=opt.sigma_d, size=opt.data_shape).to(device)

            ######################
            # X -> Y' -> X^ cycle
            ######################

            GA2B_optimizer.zero_grad()   # set g_x and g_y gradients to zero

            fake_B = generator_A2B(real_A)       # X -> Y'
            prediction = discriminator_A(fake_B + gaussian_noise)  #netD_x provide feedback to netG_x
            '''
            to_binary
            '''
            loss_G_A = criterionGAN(prediction, True)

            # cycle_consistence
            cycle_A = generator_B2A(fake_B)         # Y' -> X^
            # Forward cycle loss x^ = || G_y(G_x(real_x)) ||
            loss_cycle_A = criterionCycle(cycle_A, real_A) * lambda_A

            # identity loss
            if lambda_identity > 0:
                # netG_x should be identity if real_y is fed: ||netG_x(real_y) - real_y||
                idt_A = generator_A2B(real_B)
                loss_idt_A = criterionIdt(idt_A, real_B) * lambda_A * lambda_identity
            else:
                loss_idt_A = 0.

            loss_A = loss_G_A + loss_cycle_A + loss_idt_A
            loss_A.backward(retain_graph=True)
            GA2B_optimizer.step()

            loss_A_meter.add(loss_A.item())


            ######################
            # Y -> X' -> Y^ cycle
            ######################

            GB2A_optimizer.zero_grad()   # set g_x and g_y gradients to zero

            fake_A = generator_B2A(real_B)   # Y -> X'
            prediction = discriminator_B(fake_A + gaussian_noise)
            loss_G_B = criterionGAN(prediction, True)
            # print(f'loss_G_Y = {round(float(loss_G_Y), 3)}')

            y_hat = generator_A2B(fake_A)   # Y -> X' -> Y^
            # Forward cycle loss y^ = || G_x(G_y(real_y)) ||
            loss_cycle_B = criterionCycle(y_hat, real_B) * lambda_B

            # identity loss
            if lambda_identity > 0:
            # netG_y should be identiy if real_x is fed: ||netG_y(real_x) - real_x||
                idt_B = generator_B2A(real_A)
                loss_idt_B = criterionIdt(idt_B, real_A) * lambda_A * lambda_identity
            else:
                loss_idt_B = 0.

            loss_B = loss_G_B + loss_cycle_B + loss_idt_B
            loss_B.backward(retain_graph=True)
            GB2A_optimizer.step()

            loss_B_meter.add(loss_B.item())


            ######################
            # netD_x
            ######################

            DA_optimizer.zero_grad()

            # loss_real
            pred_real = discriminator_A(real_B + gaussian_noise)
            loss_D_A_real = criterionGAN(pred_real, True)
            score_DA_real_B.add(float(pred_real.data.mean()))

            # loss fake
            pred_fake = discriminator_A(fake_B + gaussian_noise)
            loss_D_A_fake = criterionGAN(pred_fake, False)
            score_DA_fake_B.add(float(pred_fake.data.mean()))

            # loss and backward
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

            loss_D_A.backward()
            DA_optimizer.step()


            ######################
            # netD_y
            ######################

            DB_optimizer.zero_grad()

            # loss_real
            pred_real = discriminator_B(real_A + gaussian_noise)
            loss_D_B_real = criterionGAN(pred_real, True)

            # loss_fake
            pred_fake = discriminator_B(fake_A + gaussian_noise)
            loss_D_B_fake = criterionGAN(pred_fake, False)

            # loss and backward
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

            loss_D_B.backward()
            DB_optimizer.step()


            # save snapshot
            if i % opt.plot_every == 0:
                file_name = opt.name + '_snap_%03d_%05d.png' % (epoch, i,)
                test_path = os.path.join(opt.checkpoint_path, file_name)
                tv.utils.save_image(fake_B, test_path, normalize=True)
                print(f'{file_name} saved.')

                losses['loss_A'] = loss_A_meter.value()[0]
                losses['loss_B'] = loss_B_meter.value()[0]
                scores['score_DA_real_B'] = score_DA_real_B.value()[0]
                scores['score_DA_fake_B'] = score_DA_fake_B.value()[0]
                print(losses)
                print(scores)

            print(f'iteration {i} finished')

        # save model
        if epoch % opt.save_every == 0 or epoch == opt.epoch - 1:
            save_filename = f'{opt.name}_netG_{epoch}.pth'
            save_filepath = os.path.join(opt.model_path, save_filename)
            torch.save(generator_A2B.state_dict(), save_filepath)
            print(f'model saved as {save_filename}')


        epoch_time = int(time.time() - epoch_start_time)

        print_options(opt, epoch_log=True, epoch=epoch, time=epoch_time, losses=losses, scores=scores)


def test():
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


if __name__ == '__main__':
    train()