import os


class Config(object):
    def __init__(self):

        ##########################
        # Info

        self.name = 'classifier'
        self.dataset_name = 'free_midi_library'

        self.genre_group = 3

        if self.genre_group == 1:
            self.genreA = 'metal'
            self.genreB = 'country'

        elif self.genre_group == 2:
            self.genreA = 'punk'
            self.genreB = 'classical'

        else:
            self.genreA = 'rock'
            self.genreB = 'jazz'

        self.dataset_mode = 'unaligned'
        self.track_merged = False

        self.time_step = 120
        self.bar_length = 4
        self.note_valid_range = (24, 108)
        self.note_valid_length = 84
        self.instr_num = 5

        self.phase = 'train'
        self.continue_train = False

        ###########################

        ##########################
        # Train

        self.gaussian_std = 1

        self.sigma_c = 0.01
        self.sigma_d = 1.0

        self.gpu = True

        self.beta1 = 0.9                     # Adam optimizer beta1 & 2
        self.beta2 = 0.999

        self.lr = 0.0002

        self.weight_decay = 0.

        self.no_flip = True
        self.num_threads = 0
        self.batch_size = 32
        self.max_epoch = 100
        self.epoch_step = 5

        self.data_shape = (self.batch_size, 1, 64, 84)
        self.input_shape = (1, 64, 84)

        self.plot_every = 200                # iterations
        self.save_every = 5                  # epochs

        self.start_epoch = 0

        ##########################

        ##########################
        # Save Paths

        self.save_path = 'd:/checkpoints/' + '{}_{}2{}'.format(self.name, self.genreA, self.genreB)
        self.model_path = self.save_path + '/models/'
        self.checkpoint_path = self.save_path + '/checkpoints/'

        self.log_path = self.save_path + '/info.log'
        self.loss_save_path = self.save_path + '/losses.json'

        self.test_path = self.save_path + '/test_results'

        ##########################


if __name__ == '__main__':
    config = Config()
    print(config.save_path)

