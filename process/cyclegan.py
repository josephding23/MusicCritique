import time
import datetime
import itertools
import numpy as np
import torch
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

    dataset = SteelyDataset(opt)
    dataset_size = len(dataset)
    iter_per_epoch = int(dataset_size / opt.batch_size)

    print(f'loaded {dataset_size} images for training')

    model_names = ['netG_x', 'netG_y', 'netD_x', 'netD_y']

    netG_x = Generator()
    netG_y = Generator()

    netD_x = Discriminator()
    netD_y = Discriminator()

    if opt.gpu:
        netG_x.to(device)
        summary(netG_x, input_size=opt.data_shape)
        netG_y.to(device)

        netD_x.to(device)
        summary(netD_x, input_size=opt.data_shape)
        netD_y.to(device)

    '''
    optimizer_GA2B = torch.optim.Adam(params=netG_x.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))
    optimizer_GB2A = torch.optim.Adam(params=netG_y.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))
    optimizer_DA = torch.optim.Adam(params=netG_y.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))
    optimizer_DB = torch.optim.Adam(params=netG_y.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))
    '''
    optimizer_g = torch.optim.Adam(params=netG_x.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))
    optimizer_d = torch.optim.Adam(params=netD_x.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))

    # optimizers = [optimizer_g, optimizer_d]

    lambda_X = 10.0  # weight for cycle loss (A -> B -> A^)
    lambda_Y = 10.0  # weight for cycle loss (B -> A -> B^)
    lambda_identity = 0.5

    # it's a MSELoss() when initialized, only calculate later during iteration
    # criterionGAN = nn.MSELoss().to(device)
    criterionGAN = GANLoss(gan_mode='lsgan')

    # cycle loss
    criterionCycle = nn.L1Loss()

    # identical loss
    criterionIdt = nn.L1Loss()

    loss_X_meter = MovingAverageValueMeter(opt.plot_every)
    loss_Y_meter = MovingAverageValueMeter(opt.plot_every)
    score_Dx_real_y = MovingAverageValueMeter(opt.plot_every)
    score_Dx_fake_y = MovingAverageValueMeter(opt.plot_every)

    # loss meters
    losses = {}
    scores = {}

    for epoch in range(opt.max_epochs):
        epoch_start_time = time.time()

        for i, data in enumerate(dataset):

            real_x = data['A'].to(device)
            real_y = data['B'].to(device)

            np.random.shuffle(real_x)
            np.random.shuffle(real_y)

            gaussian_noise = np.random.normal(loc=0, scale=opt.sigma_d, size=opt.data_shape)

            ######################
            # X -> Y' -> X^ cycle
            ######################

            optimizer_g.zero_grad()   # set g_x and g_y gradients to zero

            fake_y = netG_x(real_x)       # X -> Y'
            prediction = netD_x(fake_y)  #netD_x provide feedback to netG_x
            '''
            to_binary
            '''
            loss_G_X = criterionGAN(prediction, True)

            # cycle_consistence
            x_hat = netG_y(fake_y)         # Y' -> X^
            # Forward cycle loss x^ = || G_y(G_x(real_x)) ||
            loss_cycle_X = criterionCycle(x_hat, real_x) * lambda_X

            # identity loss
            if lambda_identity > 0:
                # netG_x should be identity if real_y is fed: ||netG_x(real_y) - real_y||
                idt_x = netG_x(real_y)
                loss_idt_x = criterionIdt(idt_x, real_y) * lambda_Y * lambda_identity
            else:
                loss_idt_x = 0.

            loss_X = loss_G_X + loss_cycle_X + loss_idt_x
            loss_X.backward(retain_graph=True)
            optimizer_g.step()

            loss_X_meter.add(loss_X.item())


            ######################
            # Y -> X' -> Y^ cycle
            ######################

            optimizer_g.zero_grad()   # set g_x and g_y gradients to zero

            fake_x = netG_y(real_y)   # Y -> X'
            prediction = netD_y(fake_y)
            loss_G_Y = criterionGAN(prediction, True)
            # print(f'loss_G_Y = {round(float(loss_G_Y), 3)}')

            y_hat = netG_x(fake_x)   # Y -> X' -> Y^
            # Forward cycle loss y^ = || G_x(G_y(real_y)) ||
            loss_cycle_Y = criterionCycle(y_hat, real_y) * lambda_Y

            # identity loss
            if lambda_identity > 0:
            # netG_y should be identiy if real_x is fed: ||netG_y(real_x) - real_x||
                idt_y = netG_y(real_x)
                loss_idt_y = criterionIdt(idt_y, real_x) * lambda_X * lambda_identity
            else:
                loss_idt_y = 0.

            loss_Y = loss_G_Y + loss_cycle_Y + loss_idt_y
            loss_Y.backward(retain_graph=True)
            optimizer_g.step()

            loss_Y_meter.add(loss_Y.item())


            ######################
            # netD_x
            ######################

            optimizer_d.zero_grad()

            # loss_real
            pred_real = netD_x(real_y)
            loss_D_x_real = criterionGAN(pred_real, True)
            score_Dx_real_y.add(float(pred_real.data.mean()))

            # loss fake
            pred_fake = netD_x(fake_y)
            loss_D_x_fake = criterionGAN(pred_fake, False)
            score_Dx_fake_y.add(float(pred_fake.data.mean()))

            # loss and backward
            loss_D_x = (loss_D_x_real + loss_D_x_fake) * 0.5

            loss_D_x.backward()
            optimizer_d.step()


            ######################
            # netD_y
            ######################

            optimizer_d.zero_grad()

            # loss_real
            pred_real = netD_y(real_x)
            loss_D_y_real = criterionGAN(pred_real, True)

            # loss_fake
            pred_fake = netD_y(fake_x)
            loss_D_y_fake = criterionGAN(pred_fake, False)

            # loss and backward
            loss_D_y = (loss_D_y_real + loss_D_y_fake) * 0.5

            loss_D_y.backward()
            optimizer_d.step()


            # save snapshot
            if i % opt.plot_every == 0:
                file_name = opt.name + '_snap_%03d_%05d.png' % (epoch, i,)
                test_path = os.path.join(opt.checkpoint_path, file_name)
                tv.utils.save_image(fake_y, test_path, normalize=True)
                print(f'{file_name} saved.')

                losses['loss_X'] = loss_X_meter.value()[0]
                losses['loss_Y'] = loss_Y_meter.value()[0]
                scores['score_Dx_real_y'] = score_Dx_real_y.value()[0]
                scores['score_Dx_fake_y'] = score_Dx_fake_y.value()[0]
                print(losses)
                print(scores)

            # print(f'iteration {i} finished')

        # save model
        if epoch % opt.save_every == 0 or epoch == opt.max_epochs - 1:
            save_filename = f'{opt.name}_netG_{epoch}.pth'
            save_filepath = os.path.join(opt.model_path, save_filename)
            torch.save(netG_x.state_dict(), save_filepath)
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