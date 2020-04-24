import torch
from util.data.dataset import SteelyDataset, get_dataset
import shutil
from util.toolkit import generate_midi_segment_from_tensor, generate_data_from_midi, generate_whole_midi_from_tensor
from util.analysis.tonality import evaluate_tonal_scale_of_data, get_md5_of
from cyclegan.cygan_model import CycleGAN
from classify.classify_model import Classify
from networks.SteelyGAN import Discriminator, Generator
import matplotlib.pyplot as plt
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
