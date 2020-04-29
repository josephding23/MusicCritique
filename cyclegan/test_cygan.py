import torch
from util.data.dataset import SteelyDataset, get_dataset
import shutil
from util.toolkit import *
from util.analysis.tonality import evaluate_tonal_scale_of_data, get_md5_of
from cyclegan.cygan_model import CycleGAN
from classify.classify_model import Classify
from networks.SteelyGAN import Discriminator, Generator
import matplotlib.pyplot as plt
import numpy as np
from util.analysis.tonality import *
from torch.optim import lr_scheduler, Adam



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

        tonality_A = evaluate_tonal_scale_of_data(dataA)
        tonality_A2B = evaluate_tonal_scale_of_data(dataA2B)

        tonality_B = evaluate_tonal_scale_of_data(dataB)
        tonality_B2A = evaluate_tonal_scale_of_data(dataB2A)

        print(tonality_A, tonality_A2B)
        print(tonality_B, tonality_B2A)

        # plot_data(dataA)
        # plot_data(dataA2B)

        generate_midi_segment_from_tensor(dataA, midi_A_path)
        generate_midi_segment_from_tensor(dataA2B, midi_A2B_path)
        generate_midi_segment_from_tensor(dataB, midi_B_path)
        generate_midi_segment_from_tensor(dataB2A, midi_B2A_path)

def test_whole_song(performer='Nirvana', song='In Bloom', genre='rock'):
# def test_whole_song(performer='Frank Sinatra', song='Fly Me To The Moon', genre='jazz'):
    root_dir = 'E:/free_midi_library/merged_midi'
    try:
        md5 = get_md5_of(performer, song, genre)
        original_path = root_dir + '/' + genre + '/' + md5 + '.mid'
        print(original_path)
    except Exception as e:
        print(e)
        return
    transformed_path = '../data/converted_midi/' + song + ' - ' + performer + '.mid'
    copy_path = '../data/original_midi/' + song + ' - ' + performer + '.mid'

    ori_data = generate_data_from_midi(original_path)
    ori_data = np.expand_dims(ori_data, 1)


    cyclegan = CycleGAN()
    cyclegan.continue_from_latest_checkpoint()

    # direction = 'AtoB'
    direction = 'BtoA'

    if direction == 'AtoB':
        transformed_data = cyclegan.generator_A2B(torch.from_numpy(ori_data.copy()).to(
                device='cuda',  dtype=torch.float)).cpu().detach().numpy()
    else:
        transformed_data = cyclegan.generator_B2A(torch.from_numpy(ori_data.copy()).to(
                device='cuda', dtype=torch.float)).cpu().detach().numpy()

    save_midis(ori_data, copy_path)
    save_midis(transformed_data, transformed_path)
    print(round(evaluate_tonal_scale_of_file()),
          round(evaluate_tonal_scale_of_data(transformed_data[:, 0, :, :]), 3))



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

if __name__ == '__main__':
    test_whole_song()
