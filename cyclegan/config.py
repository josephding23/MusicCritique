import os


class Config(object):
    def __init__(self):

        ##########################
        # Info

        self.name = 'steely_gan'

        self.dataset_name = 'free_midi_library'
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

        self.direction = 'AtoB'

        ###########################

        ###########################
        # Structure

        self.model = 'full'  # three different models, base, partial, full

        self.use_image_pool = True
        self.image_pool_info = 'pooled' if self.use_image_pool else 'not_pooled'
        self.image_pool_max_size = 20

        ##########################

        ##########################
        # Train

        self.gaussian_std = 1

        self.sigma_c = 1.0
        self.sigma_d = 1.0

        self.gpu = True

        self.beta1 = 0.5                     # Adam optimizer beta1 & 2
        self.beta2 = 0.999

        self.lr = 0.0001

        self.weight_decay = 0.2

        self.no_flip = True
        self.num_threads = 1
        self.batch_size = 16
        self.max_epoch = 30
        self.epoch_step = 5

        self.data_shape = (self.batch_size, 1, 64, 84)
        self.input_shape = (1, 64, 84)

        self.plot_every = 100                # iterations
        self.save_every = 5                  # epochs

        self.start_epoch = 0

        ##########################

        ##########################
        # Save Paths

        self.save_path = 'd:/checkpoints/' + '{}_{}2{}_{}_{}_gn{}_lr{}_wd{}'.format(self.name, self.genreA, self.genreB,
                                                                          self.model, self.image_pool_info,
                                                                          self.gaussian_std, self.lr, self.weight_decay)
        self.model_path = self.save_path + '/models'
        self.checkpoint_path = self.save_path + '/checkpoints'

        self.log_path = self.save_path + '/info.log'
        self.loss_save_path = self.save_path + '/losses.json'

        self.test_path = self.save_path + '/test_results'
        self.test_save_path = self.test_path + '/' + self.direction

        self.G_A2B_save_path = self.model_path + '/G_A2B/'
        self.G_B2A_save_path = self.model_path + '/G_B2A/'
        self.D_A_save_path = self.model_path + '/D_A/'
        self.D_B_save_path = self.model_path + '/D_B/'

        self.D_A_all_save_path = self.model_path + '/D_A_all/'
        self.D_B_all_save_path = self.model_path + '/D_B_all/'

        ##########################


if __name__ == '__main__':
    config = Config()
    print(config.save_path)

